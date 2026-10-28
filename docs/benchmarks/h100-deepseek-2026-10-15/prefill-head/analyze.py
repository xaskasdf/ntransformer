import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import numpy as np

root=Path('/home/ubuntu/ntransformer-h100/ds41-prefill-20261015')
assert json.loads((root/'status.json').read_text())['state']=='complete'
commands={p.name:json.loads(p.read_text()) for p in root.glob('*-command.json')}
assert all(c['returncode']==0 for c in commands.values())

def compare(a,b):
    x=np.fromfile(root/(a+'.bin'),dtype='<f4').astype(np.float64)
    y=np.fromfile(root/(b+'.bin'),dtype='<f4').astype(np.float64)
    assert x.size==y.size==129280 and np.isfinite(x).all() and np.isfinite(y).all()
    return dict(max_abs=float(np.abs(x-y).max()),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),
        bitwise=(root/(a+'.bin')).read_bytes()==(root/(b+'.bin')).read_bytes(),
        top1=[int(x.argmax()),int(y.argmax())],top8_identical=np.array_equal(np.argsort(x)[-8:],np.argsort(y)[-8:]))

log=(root/'real-prefill.log').read_text()
rows=[json.loads(line) for line in log.splitlines() if line.startswith('{"kind":"prefill"')]
assert len(rows)==8 and all(r['exact'] and r['heads']==(16 if r['all_logits'] else 1) for r in rows)
pairs=[]
for n in range(4):
    full=next(r for r in rows if r['pair']==n and r['all_logits'])
    last=next(r for r in rows if r['pair']==n and not r['all_logits'])
    pairs.append(dict(pair=n,all_seconds=full['seconds'],last_seconds=last['seconds'],
        speedup_percent=(full['seconds']/last['seconds']-1)*100))
mean_full=statistics.mean(r['all_seconds'] for r in pairs)
mean_last=statistics.mean(r['last_seconds'] for r in pairs)
cpu_gpu=compare('cpu-attn','gpu-attn')
assert cpu_gpu['max_abs']<1e-3 and cpu_gpu['relative_l2']<1e-4 and cpu_gpu['top1'][0]==cpu_gpu['top1'][1]
cli=compare('cli-all','cli-last')
assert cli['bitwise']
cli_data={}
for name in ['cli-all','cli-last']:
    text=(root/(name+'.log')).read_text()
    match=re.search(r'\[prefill\] tokens=(\d+) heads=(\d+) mode=(\w+) seconds=([\d.]+)',text)
    assert match
    generation=re.search(r'^ (.*?)ds41-gpu: expert cache:',text,re.MULTILINE)
    assert generation
    cli_data[name]=dict(tokens=int(match[1]),heads=int(match[2]),mode=match[3],seconds=float(match[4]),
        generated_text=generation[1],generated_count=int(re.search(r'generated (\d+) tokens',text)[1]))
assert cli_data['cli-all']['heads']==6 and cli_data['cli-last']['heads']==1
assert cli_data['cli-all']['generated_text']==cli_data['cli-last']['generated_text']

result=dict(note='Warm in-process paired latency. Not full model startup or autoregressive decode throughput.',
    pairs=pairs,mean_all_seconds=mean_full,mean_last_seconds=mean_last,
    all_tokens_per_second=16/mean_full,last_tokens_per_second=16/mean_last,
    mean_speedup_percent=(mean_full/mean_last-1)*100,
    saved_ms_per_prompt=(mean_full-mean_last)*1000,
    removed_heads=15,logical_head_weight_bytes_avoided=15*129280*5120*2,
    logits_d2h_bytes_avoided=15*129280*4,
    cpu_vs_gpu_attention_single_token=cpu_gpu,cli_logits=cli,cli=cli_data,
    real_prompt=16,tiny_prompt=18,continuation=4,
    real_schedules_passed=log.count('PASS schedule='),
    tiny_schedules_passed=(root/'tiny-prefill.log').read_text().count('PASS schedule='),
    gpu_prologue_failures=sum((root/(n+'.log')).read_text().count('[prologue] resident fail') for n in ['gpu-attn','real-prefill','cli-all','cli-last']),
    source_sha256={str(p.relative_to(root/'source')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root/'source/src/model/ds41/ds41_forward.cpp',root/'source/src/model/ds41/ds41_forward.h',
                  root/'source/src/model/ds41/ds41_gpu.cu',root/'source/tests/test_ds41_prefill.cpp',root/'source/CMakeLists.txt']},
    fixture_sha256=hashlib.sha256(Path('/home/ubuntu/tiny-deepseek41.gguf').read_bytes()).hexdigest())
assert result['gpu_prologue_failures']==0 and result['real_schedules_passed']==result['tiny_schedules_passed']==2
(root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
observations=dict(threshold_status=json.loads(Path('/lambda/nfs/btiny/brandon-mini/stages/threshold-v1/status.json').read_text()),
    processes=subprocess.check_output(['ps','-eo','pid,lstart,args'],text=True))
observations['processes']='\n'.join(line for line in observations['processes'].splitlines() if 'train_mini_native.py' in line)
(root/'after-observation.json').write_text(json.dumps(observations,indent=2)+'\n')
print(json.dumps(result,indent=2))
