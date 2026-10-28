import json
from pathlib import Path
import subprocess
stages=Path('/lambda/nfs/btiny/brandon-mini/stages')
for p in sorted(stages.glob('*/status.json')):
    d=json.loads(p.read_text())
    if d.get('state')!='complete':print(p.parent.name,json.dumps(d))
for line in subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True).splitlines():
    if any(k in line for k in ['3152430','3152655','ds41-flow','pipeline.py']):print(line)
