import hashlib, json, re, statistics
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-schedule-20261015')
assert json.loads((root/'experiment-status.json').read_text())['state']=='complete'
rows=[]
pattern=re.compile(r'expert cache: (\d+) lookups, (\d+) hits .*?, (\d+) evictions')
for label in ['a1','b1','b2','a2','control-grain1']:
    previous=(0,0,0);pending=None
    text=(root/(label+'.log')).read_text()
    assert 'PASS warm repeats exact' in text and 'FAIL' not in text
    assert not any(word in text for word in ['prologue failed', 'resident fail', 'copy failed', 'allocation failed'])
    for line in text.splitlines():
        if line.startswith('{'):
            pending=json.loads(line);pending.update(run=label,arm='before' if label.startswith('a') else 'after')
        elif match:=pattern.search(line):
            current=tuple(map(int,match.groups()))
            if pending is not None:
                pending['expert_lookups']=current[0]-previous[0]
                pending['expert_hits']=current[1]-previous[1]
                pending['expert_evictions']=current[2]-previous[2]
                if pending['round']>=0:
                    assert pending['expert_hits']==pending['expert_lookups']==pending['tokens']*720,pending
                    assert pending['expert_evictions']==0
                rows.append(pending);pending=None
            previous=current
summary={}
for phase in ['prefill','decode']:
    item={}
    for arm in ['before','after']:
        selected=[r for r in rows if r['phase']==phase and r['round']>=0 and r['arm']==arm and r['run']!='control-grain1']
        times=[r['seconds'] for r in selected]
        item[arm]=dict(samples=len(times),processes=len({r['run'] for r in selected}),mean_seconds=statistics.mean(times),
            min_seconds=min(times),max_seconds=max(times),tokens_per_second=selected[0]['tokens']/statistics.mean(times),
            phases={k:statistics.mean(r[k] for r in selected) for k in ['moe','attention','engram','hc','other','qkv','compression','attention_loop','attention_tail']})
    item['speedup']=item['before']['mean_seconds']/item['after']['mean_seconds']
    item['throughput_gain_percent']=(item['speedup']-1)*100
    item['latency_reduction_percent']=(1-1/item['speedup'])*100
    summary[phase]=item
paths=['CMakeLists.txt','src/model/ds41/ds41_model.cpp','src/model/ds41/ds41_forward.cpp','src/model/ds41/ds41_forward.h','src/model/ds41/ds41_gpu.cu','tests/test_ds41_parallel.cpp','tests/test_ds41_prefill.cpp']
result=dict(summary=summary,rows=rows,exact=json.loads((root/'exact.json').read_text()),
    source_sha256={name:hashlib.sha256((root/'source'/name).read_bytes()).hexdigest() for name in paths},
    baseline_model_cpp_sha256=hashlib.sha256((root.parent/'ds41-prefill-20261015/source/src/model/ds41/ds41_model.cpp').read_bytes()).hexdigest())
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
