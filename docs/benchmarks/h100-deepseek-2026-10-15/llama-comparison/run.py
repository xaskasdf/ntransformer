import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
import time

ROOT=Path('/home/ubuntu/ntransformer-h100/llama-bench-20261015')
HOST=ROOT.parent
PREFIX=HOST/'env'
STAGES=Path('/lambda/nfs/btiny/brandon-mini/stages')
MODEL=HOST/'models/deepseek-v41-flash-q8/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf'
CUBLAS=Path('/home/ubuntu/brandon-mini/.venv/lib/python3.12/site-packages/nvidia/cu13')
ENV={k:v for k,v in os.environ.items() if not k.startswith(('DS41_','GGML_','LLAMA_'))}
ENV.update(CUDA_VISIBLE_DEVICES='1',PATH=str(PREFIX/'bin')+':'+ENV['PATH'],
    LD_LIBRARY_PATH=':'.join(map(str,[ROOT/'build/bin',PREFIX/'targets/x86_64-linux/lib',PREFIX/'lib',CUBLAS/'lib'])))

def save(name,value): (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def status(state,**extra):
    save('status.json',dict(state=state,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
def capture(args): return subprocess.check_output(list(map(str,args)),env=ENV,text=True)
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def command(label,args,timeout=1800,required=True):
    args=list(map(str,args));status('running',case=label)
    start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if (ROOT/(label+'.log')).exists() and not (ROOT/(label+'-first.log')).exists():
        shutil.copy2(ROOT/(label+'.log'),ROOT/(label+'-first.log'))
    with (ROOT/(label+'.log')).open('w') as log:
        p=subprocess.Popen(args,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=p.wait(timeout=timeout)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    save(label+'-command.json',dict(command=args,returncode=code,utc_start=utc,wall_seconds=time.monotonic()-start,
        process_group=p.pid,environment={k:v for k,v in ENV.items() if k.startswith(('DS41_','GGML_','LLAMA_')) or k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']}))
    print(label,code,round(time.monotonic()-start,3),flush=True)
    if code and required: raise RuntimeError(label+' failed')
    return code

@contextlib.contextmanager
def admitted():
    deadline=time.monotonic()+3600
    while True:
        locks=contextlib.ExitStack()
        try:
            paths=sorted(list(STAGES.glob('*/pipeline.lock'))+[HOST/'exclusive-training.lock',HOST/'gpu1-correctness.lock'])
            for path in paths:
                f=locks.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def check():
                states={p.parent.name:json.loads(p.read_text()) for p in STAGES.glob('*/status.json')}
                for name,d in states.items():
                    if d.get('state')=='complete':continue
                    if d.get('state')=='failed' and d.get('pid') and not Path('/proc',str(d['pid'])).exists():continue
                    raise BlockingIOError('active stage '+name)
                if capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip():raise BlockingIOError('GPU busy')
                return states
            check();time.sleep(2);save('admission.json',check());break
        except BlockingIOError as e:
            locks.close()
            if time.monotonic()>deadline:raise
            status('waiting',reason=str(e));time.sleep(10)
        except BaseException:locks.close();raise
    try:yield
    finally:locks.close()

def main():
    deadline=time.monotonic()+3000
    while True:
        prep=json.loads((ROOT/'prepare-status.json').read_text())
        if prep['state']=='complete':break
        if prep['state']=='failed':raise RuntimeError(prep)
        if time.monotonic()>deadline:raise TimeoutError('CUDA build')
        status('waiting-build',case=prep.get('case'));time.sleep(10)
    source=ROOT/'source'
    command('compile-harness',[PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor'),
        ROOT/'benchmark_llama_ds41.cpp','-L'+str(ROOT/'build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu',
        '-L'+str(PREFIX/'targets/x86_64-linux/lib/stubs'),'-lcuda','-lpthread','-o',ROOT/'benchmark_llama_ds41'])
    workloads=json.loads((ROOT/'workloads.json').read_text())
    for key,value in workloads.items():save('workload-p'+key+'.json',value)
    pilot=dict(workloads['32']);pilot['continuation_ids']=pilot['continuation_ids'][:8];pilot['expected_prediction_ids']=pilot['expected_prediction_ids'][:9]
    save('workload-pilot.json',pilot)
    save('environment.json',dict(gpu=capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv']),
        cpu=capture(['lscpu']),memory=capture(['free','-b']),toolkit=capture([PREFIX/'bin/nvcc','--version']),
        compiler=capture([PREFIX/'bin/x86_64-conda-linux-gnu-g++','--version']),
        source_commit='5210c7c5ed61dddaee6ed476623abf4b63093d16',
        binary_sha256=sha(ROOT/'benchmark_llama_ds41'),
        library_sha256={p.name:sha(p) for p in sorted((ROOT/'build/bin').glob('*.so*')) if p.is_file() and not p.is_symlink()},
        cublas_sha256=sha(CUBLAS/'lib/libcublas.so.13'),cublas_library=str(CUBLAS/'lib/libcublas.so.13'),
        workload_sha256=sha(ROOT/'workloads.json')))
    with admitted():
        stop=threading.Event()
        with (ROOT/'telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            def observe():
                with (ROOT/'compute-processes.jsonl').open('w') as output:
                    while not stop.is_set():
                        apps=capture(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'])
                        output.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),apps=apps))+'\n');output.flush();stop.wait(3)
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            try:
                def run_case(label,workload,repeats,threads,cpu_moe=31,ubatch=512,required=True):
                    return command(label,['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'benchmark_llama_ds41',
                        MODEL,ROOT/workload,ROOT/label,repeats,threads,cpu_moe,ubatch],required=required)
                # Default physical-core count and all logical CPUs get the same
                # short pilot. Select on warmed decode time, before main cases.
                run_case('pilot-t26','workload-pilot.json',1,26)
                run_case('pilot-t52','workload-pilot.json',1,52)
                candidates={}
                for count in [26,52]:
                    rows=[json.loads(l) for l in (ROOT/f'pilot-t{count}/measurements.jsonl').read_text().splitlines()]
                    candidates[count]=next(r['seconds'] for r in rows if r.get('phase')=='decode' and r['round']==1)
                chosen=min(candidates,key=candidates.get)
                save('selection.json',dict(threads=chosen,cpu_moe_layers=31,gpu_moe_layers=9,n_ubatch=512,pilot_decode_seconds=candidates))
                for n in [32,128,512]:run_case(f'p{n}-d64','workload-p'+str(n)+'.json',3,chosen)
            finally:
                stop.set();thread.join(timeout=10);monitor.terminate();monitor.wait(timeout=10)
    save('after.json',dict(compute_apps=capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])))
    status('complete')

if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise
