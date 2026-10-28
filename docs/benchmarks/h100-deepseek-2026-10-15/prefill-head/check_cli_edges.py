import json
import os
from pathlib import Path
import re
import subprocess

root=Path('/home/ubuntu/ntransformer-h100/ds41-prefill-20261015')
env=dict(os.environ,DS41_GPU='0',DS41_GPU_ATTN='0',DS41_PROFILE='1',DS41_PREFILL_ALL_LOGITS='0')
env['LD_LIBRARY_PATH']='/home/ubuntu/ntransformer-h100/env/targets/x86_64-linux/lib:/home/ubuntu/ntransformer-h100/env/lib'
env.pop('DS41_DUMP_STAGES',None)
cases=[('single',['--token-ids','1'],{},1,1,'last'),
       ('empty',['--prompt',''],{},1,1,'last'),
       ('stages',['--token-ids','1,40,77'],{'DS41_DUMP_STAGES':'1'},3,3,'all')]
results=[]
for name,args,extra,tokens,heads,mode in cases:
    command=[str(root/'build/ntransformer'),'-m','/home/ubuntu/tiny-deepseek41.gguf','-n','0',*args]
    p=subprocess.run(command,env=dict(env,**extra),capture_output=True,text=True,timeout=30)
    text=p.stdout+p.stderr
    (root/('edge-'+name+'.log')).write_text(text)
    match=re.search(r'\[prefill\] tokens=(\d+) heads=(\d+) mode=(\w+)',text)
    assert p.returncode==0 and match and (int(match[1]),int(match[2]),match[3])==(tokens,heads,mode)
    results.append(dict(name=name,command=command,returncode=p.returncode,tokens=tokens,heads=heads,mode=mode))
env.pop('DS41_TEST_MODEL',None)
p=subprocess.run([str(root/'build/test_ds41_prefill')],env=env,capture_output=True,text=True,timeout=10)
assert p.returncode==77
results.append(dict(name='missing fixture',returncode=p.returncode,stdout=p.stdout))
(root/'cli-edges.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
