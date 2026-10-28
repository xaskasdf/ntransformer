import array
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import statistics
import sys

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
baseline=root/'baseline'
old=json.loads((baseline/'summary.json').read_text())
inputs=json.loads((root/'workloads.json').read_text())
selection=json.loads((root/'selection.json').read_text())
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_floats(path):
    result=array.array('f');result.frombytes(path.read_bytes())
    if sys.byteorder!='little':result.byteswap()
    assert all(math.isfinite(v) for v in result)
    return result
def percentile(data,q):
    data=sorted(data);p=(len(data)-1)*q;i=int(p)
    return data[i]+(data[math.ceil(p)]-data[i])*(p-i)
cases=[]
for n in [32,128,512]:
    label=f'p{n}-d64';path=root/label
    rows=[json.loads(line) for line in (path/'measurements.jsonl').read_text().splitlines()]
    setup=rows[0];phases=[r for r in rows if r['kind']=='phase'];completions=[r for r in rows if r['kind']=='completion']
    assert setup['log_level']=='info'
    assert len(phases)==8 and len(completions)==4
    assert setup['prompt_tokens']==n and setup['decode_steps']==64 and setup['vocab']==129280
    assert setup['threads']==selection['threads'] and setup['cpu_moe_layers']==selection['cpu_moe_layers']
    assert setup['n_ubatch']==selection['n_ubatch'] and setup['kv_requested']=='f32'
    assert setup['device'].startswith('CUDA')
    workload=inputs[str(n)]
    assert workload['prompt_ids']==json.loads((baseline/label/'prompt-ids.json').read_text())
    nt_rows=[json.loads(l) for l in (baseline/label/'measurements.jsonl').read_text().splitlines()]
    nt_completion=next(r for r in nt_rows if r['kind']=='completion')
    assert workload['expected_prediction_ids']==nt_completion['predicted_ids']
    assert workload['continuation_ids']==nt_completion['predicted_ids'][:64]
    command=json.loads((root/(label+'-command.json')).read_text());assert command['returncode']==0
    # Model metadata logging can truncate a UTF-8 vocabulary string mid-byte.
    # Preserve raw logs; replacement decoding is only for ASCII marker parsing.
    log=(root/(label+'.log')).read_text(encoding='utf-8',errors='replace')
    assert 'FAIL' not in log
    resource=(root/(label+'-resources.txt')).read_text()
    rss=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resource).group(1))
    c=dict(label=label,setup=setup,cold={},reused={},max_rss_kib=rss,
           command_wall_seconds=command['wall_seconds'],model_buffer_log=[l for l in log.splitlines() if 'model buffer size' in l],
           kv_buffer_log=[l for l in log.splitlines() if 'KV buffer size' in l])
    for phase in ['prefill','decode']:
        group=[r for r in phases if r['phase']==phase]
        assert [r['round'] for r in group]==[0,1,2,3]
        assert all(r['tokens']==(n if phase=='prefill' else 64) for r in group)
        c['cold'][phase]=dict(seconds=group[0]['seconds'],tokens_per_second=group[0]['tokens']/group[0]['seconds'])
        warm=group[1:];durations=[r['seconds'] for r in warm]
        c['reused'][phase]=dict(mean_seconds=statistics.mean(durations),min_seconds=min(durations),max_seconds=max(durations),
            tokens_per_second=sum(r['tokens'] for r in warm)/sum(durations))
        if phase=='decode':
            samples=[t for r in warm for t in r['token_seconds']]
            assert len(samples)==192 and all(t>0 for t in samples)
            for r in group:assert math.isclose(r['seconds'],sum(r['token_seconds']),rel_tol=1e-12)
            c['reused'][phase].update(p50_token_ms=percentile(samples,.5)*1000,p95_token_ms=percentile(samples,.95)*1000)
        else:c['reused'][phase]['ttft_mean_seconds']=statistics.mean(r['ttft_seconds'] for r in warm)
    assert all(r['finite_logit_values']==65*129280 for r in completions)
    for r in completions:
        assert len(r['predicted_ids'])==65
        assert r['agreement_with_ntransformer']==sum(a==b for a,b in zip(r['predicted_ids'],workload['expected_prediction_ids']))
    c['repeat_bitwise_equal']=all(r['repeat_bitwise_equal'] for r in completions[1:])
    c['repeat_top1_equal']=all(r['repeat_top1_equal'] for r in completions[1:])
    c['repeat_max_abs']=max(r['repeat_max_abs'] for r in completions)
    c['repeat_relative_l2']=max(r['repeat_relative_l2'] for r in completions)
    c['greedy_agreement_by_round']=[r['agreement_with_ntransformer'] for r in completions]
    ref=read_floats(baseline/label/'final-logits.f32')
    observed=read_floats(path/'final-logits-round0.f32')
    assert len(ref)==len(observed)==129280
    difference=[float(a)-b for a,b in zip(observed,ref)]
    c['final_logits_vs_ntransformer']=dict(max_abs=max(map(abs,difference)),
        relative_l2=math.sqrt(math.fsum(d*d for d in difference)/math.fsum(float(x)*x for x in ref)),
        cosine=math.fsum(float(x)*y for x,y in zip(observed,ref))/math.sqrt(math.fsum(float(x)*x for x in ref)*math.fsum(float(x)*x for x in observed)),
        reference_top8=heapq.nlargest(8,range(len(ref)),key=ref.__getitem__),
        llama_top8=heapq.nlargest(8,range(len(observed)),key=observed.__getitem__),
        reference_sha256=digest(baseline/label/'final-logits.f32'),llama_sha256=digest(path/'final-logits-round0.f32'))
    for r in range(4):
        name=f'final-logits-round{r}.f32'
        assert (path/name).read_bytes()==(root/'initial-debug'/label/name).read_bytes()
    nt=next(r for r in old['cases'] if r['label']==label)
    c['ntransformer_reused']=nt['reused']
    c['llama_over_ntransformer_throughput']={phase:c['reused'][phase]['tokens_per_second']/nt['reused'][phase]['tokens_per_second'] for phase in ['prefill','decode']}
    cases.append(c)

with (root/'telemetry.csv').open() as file:telemetry=list(csv.DictReader(file,skipinitialspace=True))
gpu={}
for index in ['0','1']:
    samples=[r for r in telemetry if r['index'].strip()==index]
    def field(key):return [float(r[key].split()[0]) for r in samples]
    gpu[index]=dict(samples=len(samples),max_memory_mib=max(field('memory.used [MiB]')),
        mean_utilization_pct=statistics.mean(field('utilization.gpu [%]')),mean_power_w=statistics.mean(field('power.draw [W]')))
processes=[json.loads(l) for l in (root/'compute-processes.jsonl').read_text().splitlines()]
for r in processes:
    apps=[l for l in r['apps'].splitlines() if l.strip()]
    assert len(apps)<=1
    for app in apps:assert 'benchmark_llama_ds41' in app and 'GPU-67e3ec29-9bbc-da80-baf1-8f982193ee6c' in app
serial_path=root/'serial-p128-d64'
serial_rows=[json.loads(line) for line in (serial_path/'measurements.jsonl').read_text().splitlines()]
assert serial_rows[0]['n_ubatch']==1 and serial_rows[0]['log_level']=='info'
serial_completions=[r for r in serial_rows if r['kind']=='completion']
assert len(serial_completions)==2 and all(r['repeat_bitwise_equal'] for r in serial_completions)
serial_logits=read_floats(serial_path/'final-logits-round0.f32')
nt_logits=read_floats(baseline/'p128-d64/final-logits.f32')
native_logits=read_floats(root/'p128-d64/final-logits-round0.f32')
def discrepancy(values,reference):
    return dict(max_abs=max(abs(float(a)-b) for a,b in zip(values,reference)),
        relative_l2=math.sqrt(math.fsum((float(a)-b)**2 for a,b in zip(values,reference))/math.fsum(float(b)*b for b in reference)))
serial=dict(greedy_agreement_by_round=[r['agreement_with_ntransformer'] for r in serial_completions],
            versus_ntransformer=discrepancy(serial_logits,nt_logits),versus_native_prefill=discrepancy(serial_logits,native_logits),
            warm_seconds={p:next(r['seconds'] for r in serial_rows if r.get('phase')==p and r['round']==1) for p in ['prefill','decode']})
summary=dict(cases=cases,selection=selection,telemetry=gpu,compute_process_observations=len(processes),serial_prefill_control=serial,
    equal_prompt_and_consumed_continuation_ids=True,observed_foreign_gpu_processes=0)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({c['label']:{p:c['reused'][p]['tokens_per_second'] for p in ['prefill','decode']} for c in cases},indent=2))
