import contextlib
import datetime
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import common as b
from admission import admitted

ROOT=b.ROOT

def rows(label):
    return [json.loads(line) for line in (ROOT/label/'measurements.jsonl').read_text().splitlines()]

def ours(pid):
    for _ in range(16):
        if pid==os.getpid():return True
        try:line=(Path('/proc')/str(pid)/'stat').read_text()
        except FileNotFoundError:return True  # process exited after the NVML sample
        pid=int(line[line.rfind(')')+2:].split()[1])
        if pid<=1:return False
    return False

@contextlib.contextmanager
def monitoring():
    stop=threading.Event()
    def observe():
        with (ROOT/'compute-processes.jsonl').open('w') as stream:
            while not stop.is_set():
                try:
                    apps=b.capture(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'])
                    foreign=[int(line.split(',')[0]) for line in apps.splitlines() if line.strip() and not ours(int(line.split(',')[0]))]
                    stream.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),apps=apps,foreign=foreign))+'\n');stream.flush()
                    if foreign:b.ABORT.set();return
                except BaseException as error:
                    stream.write(json.dumps(dict(error=repr(error)))+'\n');stream.flush();b.ABORT.set();return
                stop.wait(3)
    with (ROOT/'telemetry.csv').open('w') as stream:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
            '--format=csv','--loop-ms=1000'],stdout=stream,stderr=subprocess.STDOUT)
        thread=threading.Thread(target=observe,daemon=True);thread.start()
        try:yield
        finally:
            stop.set();thread.join(timeout=15);monitor.terminate();monitor.wait(timeout=10)

def validate(label, workload, engine):
    data=rows(label)
    setup=next(r for r in data if r['kind']=='setup')
    assert setup['prompt_tokens']==len(workload['prompt_ids']) and setup['decode_steps']==len(workload['continuation_ids'])
    assert json.loads((ROOT/label/'prompt-ids.json').read_text())==workload['prompt_ids']
    assert all(r['finite_logit_values']==(setup['decode_steps']+1)*setup['vocab'] for r in data if r['kind']=='completion')
    if engine=='llama':
        assert setup['gpu_count']==2 and setup['devices']==['CUDA0','CUDA1'] and setup['split_mode']=='layer'
        assert json.loads((ROOT/label/'continuation-ids.json').read_text())==workload['continuation_ids']
        # GGUF metadata can be truncated in the middle of a UTF-8 sequence.
        # Keep the raw log intact and tolerate replacement only when parsing
        # its ASCII diagnostics; numeric JSON measurements remain strict.
        log=(ROOT/(label+'.log')).read_text(encoding='utf-8',errors='replace')
        for device in [0,1]:
            match=re.search(rf'CUDA{device} model buffer size\s*=\s*([\d.]+) MiB',log)
            assert match and float(match[1])>60000, ('GPU residency absent',label,device)
    else:
        assert json.loads((ROOT/label/'target-ids.json').read_text())==workload['target_ids']
        assert b.sha(ROOT/label/'all-logits.f32')==workload['nt_reference_logits_sha256']
        for r in data:
            if r.get('kind')=='phase':
                assert r['peer_fallback_calls']==r['routed_cpu_calls']==0
                assert r['peer_execute_calls']==r['routed_gpu_calls']>0
    for r in data:
        if r.get('kind')=='phase':
            assert all(r[k]>=0 for k in ['major_page_faults','minor_page_faults','input_blocks','output_blocks'])
    return data

def llama(label, workload_name, config, repeats, required=True):
    b.ENV=b.BASE_ENV | {'GGML_OP_OFFLOAD_MIN_BATCH':'32' if config['op_offload'] else '1000000'}
    code=b.command(label,['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'bench-llama',b.MODEL,
        ROOT/('workload-'+workload_name+'.json'),ROOT/label,repeats,config['threads'],config['cpu_moe'],512,2],required=required)
    if code:
        log=(ROOT/(label+'.log')).read_text(encoding='utf-8',errors='replace').lower()
        if not any(token in log for token in ['cuda error: out of memory','cuda out of memory','cudamalloc failed','failed to allocate cuda']):
            raise RuntimeError('non-OOM pilot failure: '+label)
        return None
    workload=json.loads((ROOT/('workload-'+workload_name+'.json')).read_text())
    return validate(label,workload,'llama')

def main():
    with admitted(b), monitoring():
        b.verify()
        plan=json.loads((ROOT/'plan.json').read_text())
        pilot=[]
        for cpu in plan['llama_pilot']['cpu_moe_layers']:
            for threads in plan['llama_pilot']['threads']:
                for offload in plan['llama_pilot']['operation_offload']:
                    config=dict(cpu_moe=cpu,threads=threads,op_offload=offload)
                    label=f'pilot-cpu{cpu}-t{threads}-offload{int(offload)}'
                    data=llama(label,'pilot',config,2,required=False)
                    if data is None:pilot.append(dict(label=label,config=config,feasible=False));continue
                    phases={(r['round'],r['phase']):r for r in data if r['kind']=='phase'}
                    score=statistics.median(phases[i,'prefill']['seconds']+4*phases[i,'decode']['seconds'] for i in [1,2])
                    pilot.append(dict(label=label,config=config,feasible=True,score_seconds=score))
        candidates=[r for r in pilot if r['feasible']]
        if not candidates:raise RuntimeError('no two-GPU llama placement fit')
        best=min(candidates,key=lambda r:r['score_seconds'])
        b.save('selection.json',dict(selected=best,pilots=pilot,rule=plan['llama_pilot']['score']))
        for launch in range(plan['independent_launches']):
            for index,workload_name in enumerate(plan['workloads']):
                workload=json.loads((ROOT/('workload-'+workload_name+'.json')).read_text())
                engines=list(plan['main_engines'])
                if workload_name=='p512':engines.append('nt-lru')
                if (launch+index)%2:engines.reverse()
                for engine in engines:
                    b.verify()
                    label=f'{engine}-{workload_name}-r{launch}'
                    if engine=='llama':llama(label,workload_name,best['config'],4)
                    else:
                        b.ENV=b.BASE_ENV | dict(DS41_GPU='1',DS41_GPU_ATTN='1',DS41_PROFILE='1',
                            DS41_Q8_COOPERATIVE='1',DS41_PEER_MODE='execute',DS41_MXFP4_COMPUTE='fp32',
                            DS41_VRAM_GB='62',DS41_PEER_VRAM_GB='62',DS41_BENCH_ALL_LOGITS='1',
                            DS41_BENCH_TEACHER_FORCING='1',DS41_CACHE_POLICY=engine.removeprefix('nt-'),
                            DS41_CACHE_PROTECTED_PERCENT='80',DS41_CACHE_TRACE='0')
                        b.command(label,['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'bench-nt',b.MODEL,
                            ROOT/workload['corpus'],len(workload['prompt_ids']),64,4,ROOT/label,label])
                        validate(label,workload,engine)
        if b.ABORT.is_set():raise RuntimeError('foreign workload interrupted measurement')
    b.verify();b.status('measurements-complete')

if __name__=='__main__':main()
