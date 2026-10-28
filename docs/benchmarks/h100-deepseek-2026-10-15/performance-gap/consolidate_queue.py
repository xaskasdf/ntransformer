from pathlib import Path
import os,signal,json
root=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015')
expected={3082521:'run_dump.py',3111237:'smoke.py',3112784:'run.py'}
checked=[]
for pid,name in expected.items():
    cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    assert str(root/name).encode() in cmd,(pid,cmd)
    checked.append(dict(pid=pid,script=name))
for item in checked:os.kill(item['pid'],signal.SIGTERM)
(root/'superseded-waiters.json').write_text(json.dumps(checked,indent=2)+'\n')
print(checked)
