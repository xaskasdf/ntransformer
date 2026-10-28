"""Synthetic artifact validation; fabricated timings are never benchmark evidence."""
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fair_analysis',HERE/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def write(path,value):path.write_text(json.dumps(value)+'\n')

def main():
    with tempfile.TemporaryDirectory(dir=HERE,prefix='analysis-fixture-') as directory:
        root=Path(directory).resolve();assert root.is_relative_to(HERE.resolve());a.ROOT=root
        plan=json.loads((HERE/'plan.json').read_text());write(root/'plan.json',plan)
        selected=dict(threads=52,cpu_moe=20,op_offload=False)
        write(root/'selection.json',dict(selected=dict(config=selected)))
        (root/'compute-processes.jsonl').write_text('{"foreign":[]}\n')
        telemetry=['timestamp,index,memory.used [MiB]']
        base=datetime.datetime(2026,10,16,tzinfo=datetime.timezone.utc);index=0
        for name in plan['workloads']:
            n=int(name.split('p')[-1]);prompt=[i%8 for i in range(n)];targets=[i%8 for i in range(65)]
            logits=np.arange(65*8,dtype=np.float32).reshape(65,8)/100
            workload=dict(prompt_ids=prompt,target_ids=targets,continuation_ids=targets[:-1])
            engines=['nt-slru','llama']+(['nt-lru'] if name=='p512' else [])
            for engine in engines:
                for launch in range(2):
                    label=f'{engine}-{name}-r{launch}';case=root/label;case.mkdir()
                    actual=logits if engine!='llama' else logits+np.sin(logits)*.1
                    actual.astype('<f4').tofile(case/'all-logits.f32')
                    if engine!='llama':workload['nt_reference_logits_sha256']=a.sha(case/'all-logits.f32')
                    write(case/'prompt-ids.json',prompt)
                    write(case/('continuation-ids.json' if engine=='llama' else 'target-ids.json'),targets[:-1] if engine=='llama' else targets)
                    setup=dict(kind='setup',teacher_forcing=True,prompt_tokens=n,decode_steps=64,vocab=8,
                        gpu_count=2,devices=['CUDA0','CUDA1'],split_mode='layer',threads=52,cpu_moe_layers=20)
                    rows=[setup]
                    for round in range(5):
                        for phase in ['prefill','decode']:
                            rows.append(dict(kind='phase',round=round,phase=phase,tokens=64 if phase=='decode' else n,
                                seconds=2 if engine=='llama' else 4,major_page_faults=0,minor_page_faults=0,input_blocks=0,output_blocks=0,
                                peer_execute_calls=2560,routed_gpu_calls=2560,peer_fallback_calls=0,routed_cpu_calls=0))
                        rows.append(dict(kind='completion',round=round,finite_logit_values=520,repeat_bitwise_equal=True))
                    (case/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    env={'CUDA_VISIBLE_DEVICES':'1,0','GGML_OP_OFFLOAD_MIN_BATCH':'1000000'}
                    if engine!='llama':env.update(DS41_PEER_MODE='execute',DS41_VRAM_GB='62',DS41_PEER_VRAM_GB='62',
                        DS41_MXFP4_COMPUTE='fp32',DS41_Q8_COOPERATIVE='1',DS41_CACHE_POLICY=engine.removeprefix('nt-'))
                    start=base+datetime.timedelta(seconds=index*10);index+=1
                    write(root/(label+'-command.json'),dict(returncode=0,environment=env,utc_start=start.isoformat(),wall_seconds=5))
                    for gpu in [0,1]:telemetry.append(f"{(start+datetime.timedelta(seconds=1)).strftime('%Y/%m/%d %H:%M:%S.%f')},{gpu},70000 MiB")
                    (root/(label+'-resources.txt')).write_text('Maximum resident set size (kbytes): 102400\n')
            write(root/('workload-'+name+'.json'),workload)
        (root/'telemetry.csv').write_text('\n'.join(telemetry)+'\n')
        a.main();summary=json.loads((root/'summary.json').read_text())
        assert all(r['llama_over_nt_decode']==2 for r in summary['comparisons'].values())
        assert all(r['warm_without_observed_major_faults'] for r in summary['comparisons'].values())
        target=root/'llama-p32-r0/continuation-ids.json';original=target.read_text()
        altered=json.loads(original);altered[0]=(altered[0]+1)%8;write(target,altered)
        try:a.main();raise RuntimeError('token mismatch accepted')
        except AssertionError:pass
        target.write_text(original)
        target=root/'llama-p32-r0/measurements.jsonl';original=target.read_text()
        rows=[json.loads(line) for line in original.splitlines()]
        next(r for r in rows if r.get('phase')=='decode' and r['round']==2)['major_page_faults']=7
        target.write_text(''.join(json.dumps(r)+'\n' for r in rows));a.main()
        assert not json.loads((root/'summary.json').read_text())['comparisons']['p32']['warm_without_observed_major_faults']
        target.write_text(original)
        data=np.fromfile(root/'llama-p32-r0/all-logits.f32',dtype='<f4');data[0]=np.nan
        data.tofile(root/'llama-p32-r0/all-logits.f32')
        try:a.main();raise RuntimeError('NaN accepted')
        except AssertionError:pass
    print('PASS synthetic analysis: expected ratio, mismatched tokens rejected, warm paging flagged, NaN rejected')

if __name__=='__main__':main()
