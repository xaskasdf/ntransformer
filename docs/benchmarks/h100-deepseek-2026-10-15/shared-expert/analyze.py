import hashlib, json, re, statistics
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-shared-20261015')
assert json.loads((root/'status.json').read_text())['state']=='complete'
rows=[]
pattern=re.compile(r'expert cache: (\d+) lookups, (\d+) hits .*?, (\d+) evictions')
labels=['cpu1','gpu1','gpu2','cpu2']
for label in labels:
    previous=(0,0,0);pending=None
    text=(root/(label+'.log')).read_text()
    assert 'PASS warm repeats exact' in text and not any(word in text for word in ['FAIL','resident fail','prologue failed','copy failed','allocation failed'])
    for line in text.splitlines():
        if line.startswith('{'):
            pending=json.loads(line);pending.update(run=label,arm='cpu' if label.startswith('cpu') else 'gpu')
        elif match:=pattern.search(line):
            current=tuple(map(int,match.groups()))
            if pending is not None:
                pending['expert_lookups']=current[0]-previous[0]
                pending['expert_hits']=current[1]-previous[1]
                pending['expert_evictions']=current[2]-previous[2]
                assert pending['gpu_calls']==pending['tokens']*40 and pending['cpu_calls']==0,pending
                assert pending['overlapped_calls']==(pending['gpu_calls'] if pending['arm']=='cpu' else 0),pending
                assert pending['tail_calls']==pending['tokens']*40,pending
                assert pending['grouped_tail_calls']==pending['tail_calls'],pending
                assert pending['tail_gemv_launches']==pending['tail_calls']*2,pending
                assert pending['moe_calls']==pending['gpu_calls']
                assert pending['batched_calls']==pending['moe_calls']
                assert pending['shared_calls']==pending['shared_gpu_calls']==(pending['moe_calls'] if pending['arm']=='gpu' else 0)
                assert pending['shared_cpu_calls']==(pending['moe_calls'] if pending['arm']=='cpu' else 0)
                assert pending['shared_launches']==3*pending['shared_calls']
                assert pending['moe_launches']==pending['moe_calls']*4
                assert pending['traced_calls']==pending['trace_failures']==0
                if pending['round']>=0:
                    assert pending['uploaded_bytes']==pending['shared_weight_bytes']==0
                    assert pending['expert_hits']==pending['expert_lookups']==pending['tokens']*720,pending
                    assert pending['expert_evictions']==0
                rows.append(pending);pending=None
            previous=current
summary={}
for phase in ['prefill','decode']:
    item={}
    for arm in ['cpu','gpu']:
        selected=[r for r in rows if r['phase']==phase and r['round']>=0 and r['arm']==arm]
        times=[r['seconds'] for r in selected]
        item[arm]=dict(samples=len(times),processes=len({r['run'] for r in selected}),mean_seconds=statistics.mean(times),
            min_seconds=min(times),max_seconds=max(times),tokens_per_second=selected[0]['tokens']/statistics.mean(times),
            phases={k:statistics.mean(r[k] for r in selected) for k in ['moe','attention','engram','hc','other','router','routed','shared','qkv','compression','attention_loop','attention_tail']})
    item['speedup']=item['cpu']['mean_seconds']/item['gpu']['mean_seconds']
    item['throughput_gain_percent']=(item['speedup']-1)*100
    item['latency_reduction_percent']=(1-1/item['speedup'])*100
    summary[phase]=item
from compare import compare
numerics=json.loads((root/'numerics.json').read_text())
assert numerics['controls_bitwise_exact'] and numerics['gpu_runs_bitwise_repeatable']
assert numerics['contexts']['vectors']==20
for label,passed in [('tiny-prefill',1),('gpu-tests',11)]:
    log=(root/(label+'.log')).read_text()
    assert f'100% tests passed out of {passed}' in log and '***Skipped' not in log
assert (root/'real-state.log').read_text().count('PASS schedule=')==2
paths=['CMakeLists.txt','src/model/ds41/ds41_model.cpp','src/model/ds41/ds41_forward.cpp','src/model/ds41/ds41_forward.h',
    'src/model/ds41/ds41_gpu.cu','src/model/ds41/ds41_gpu.h','src/model/ds41/ds41_hc.h','tests/test_ds41_backbone.cpp','tests/test_ds41_moe_overlap.cpp','tests/test_ds41_prefill.cpp',
    'src/model/ds41/ds41_shared.cu','src/model/ds41/ds41_shared.h','tests/test_ds41_shared.cpp']
trace={}
for label in ['trace-cpu','trace-gpu']:
    records=[json.loads(line) for line in (root/(label+'.log')).read_text().splitlines() if line.startswith('{')]
    for r in records:
        assert r['gpu_calls']==r['moe_calls']==r['traced_calls']==r['tokens']*40
        assert r['cpu_calls']==r['trace_failures']==0
        assert r['moe_launches']==r['gpu_calls']*4
        assert r['shared_calls']==r['shared_gpu_calls']==(r['gpu_calls'] if label=='trace-gpu' else 0)
        assert r['shared_cpu_calls']==(r['gpu_calls'] if label=='trace-cpu' else 0)
        if r['round']>=0:assert r['uploaded_bytes']==r['shared_weight_bytes']==0
    trace[label]=dict(rows=records)
    for phase in ['prefill','decode']:
        selected=[r for r in records if r['round']>=0 and r['phase']==phase]
        trace[label][phase]={k:statistics.mean(r[k] for r in selected) for k in selected[0] if isinstance(selected[0][k],(int,float))}
result=dict(summary=summary,rows=rows,trace=trace,numerics=numerics,
    cpu_exact=json.loads((root/'cpu-exact.json').read_text()),
    source_sha256={name:hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() for name in paths},
    shared_resident_bytes=40*3*12533760,shared_scratch_bytes=(3*2304+5120)*4,
    additional_readback_bytes_per_token=40*5120*4)
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(json.dumps({name:{phase:data[phase] for phase in ['prefill','decode']} for name,data in trace.items()},indent=2))
