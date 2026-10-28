import hashlib, json, re, statistics
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-hc-20261015')
assert json.loads((root/'status.json').read_text())['state']=='complete'
rows=[]
pattern=re.compile(r'expert cache: (\d+) lookups, (\d+) hits .*?, (\d+) evictions')
labels=['pool1','direct1','direct2','pool2']
for label in labels:
    previous=(0,0,0);pending=None
    text=(root/(label+'.log')).read_text()
    assert 'PASS warm repeats exact' in text and not any(word in text for word in ['FAIL','resident fail','prologue failed','copy failed','allocation failed'])
    for line in text.splitlines():
        if line.startswith('{'):
            pending=json.loads(line);pending.update(run=label,arm='pool' if label.startswith('pool') else 'direct')
        elif match:=pattern.search(line):
            current=tuple(map(int,match.groups()))
            if pending is not None:
                pending['expert_lookups']=current[0]-previous[0]
                pending['expert_hits']=current[1]-previous[1]
                pending['expert_evictions']=current[2]-previous[2]
                assert pending['gpu_calls']==pending['tokens']*40 and pending['cpu_calls']==0,pending
                assert pending['overlapped_calls']==pending['gpu_calls'],pending
                if pending['round']>=0:
                    assert pending['expert_hits']==pending['expert_lookups']==pending['tokens']*720,pending
                    assert pending['expert_evictions']==0
                rows.append(pending);pending=None
            previous=current
summary={}
for phase in ['prefill','decode']:
    item={}
    for arm in ['pool','direct']:
        selected=[r for r in rows if r['phase']==phase and r['round']>=0 and r['arm']==arm]
        times=[r['seconds'] for r in selected]
        item[arm]=dict(samples=len(times),processes=len({r['run'] for r in selected}),mean_seconds=statistics.mean(times),
            min_seconds=min(times),max_seconds=max(times),tokens_per_second=selected[0]['tokens']/statistics.mean(times),
            phases={k:statistics.mean(r[k] for r in selected) for k in ['moe','attention','engram','hc','other','router','routed','shared']})
    item['speedup']=item['pool']['mean_seconds']/item['direct']['mean_seconds']
    item['throughput_gain_percent']=(item['speedup']-1)*100
    item['latency_reduction_percent']=(1-1/item['speedup'])*100
    summary[phase]=item
names=['previous']+labels
data=[(root/(name+'.bin')).read_bytes() for name in names]
assert len(data[0])==5*129280*4 and all(d==data[0] for d in data)
for label,passed in [('tiny-prefill',1),('cpu-tests',6)]:
    text=(root/(label+'.log')).read_text()
    assert f'100% tests passed out of {passed}' in text and '***Skipped' not in text
assert (root/'real-state.log').read_text().count('PASS schedule=')==2
assert 'PASS HC direct/pool/scalar' in (root/'hc-final-test.log').read_text()
paths=['CMakeLists.txt','src/model/ds41/ds41_model.cpp','src/model/ds41/ds41_forward.cpp','src/model/ds41/ds41_forward.h',
    'src/model/ds41/ds41_gpu.cu','src/model/ds41/ds41_gpu.h','src/model/ds41/ds41_hc.h','tests/test_ds41_hc.cpp','tests/test_ds41_parallel.cpp','tests/test_ds41_prefill.cpp']
result=dict(summary=summary,rows=rows,exact=json.loads((root/'exact.json').read_text()),
    cpu_exact=json.loads((root/'cpu-exact.json').read_text()),
    source_sha256={name:hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() for name in paths},
    eliminated_pool_dispatches_per_prompt=16*40*4+1,eliminated_pool_dispatches_per_decode_window=4*(40*4+1),
    microbench=[json.loads(line) for line in (root/'microbench.log').read_text().splitlines() if line.startswith('{')])
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
