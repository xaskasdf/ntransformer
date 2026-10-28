from pathlib import Path
import json,shutil,subprocess,sys
r=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015')
dest=r/'initial-cupti13-failure';dest.mkdir(exist_ok=True)
for name in ['nt-trace-p32-command.json','nt-trace-p32.log']:
    p=r/name
    if p.exists():shutil.copy2(p,dest/name)
if (r/'nt-trace-p32').exists():shutil.copytree(r/'nt-trace-p32',dest/'nt-trace-p32',dirs_exist_ok=True)
(r/'initial-failure.json').write_text(json.dumps(dict(cupti_api_version=130001,nt_trace_returncode=-11,usable_timeline=False),indent=2)+'\n')
with (r/'supervisor.log').open('w') as f:
    p=subprocess.Popen([sys.executable,'-u',str(r/'supervise.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('supervisor PID',p.pid)
(r/'launch.json').write_text(json.dumps(dict(pid=p.pid,script='supervise.py'),indent=2)+'\n')
