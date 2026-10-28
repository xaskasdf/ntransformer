import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parent
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-gap-20261015')
def state(value,**extra):
    (ROOT/'queue-status.json').write_text(json.dumps(dict(state=value,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    for script in ['run.py','run_dump.py','analyze.py','report.py']:
        state('running',script=script)
        subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,check=True)
    state('sealing')
    shutil.copy2(ROOT/'cupti-13.1/package.json',ROOT/'cupti-package.json')
    exclude={'operator-weights','cupti-13.1','cupti-include','cupti131-include','__pycache__'}
    excluded_files={'queue-status.json','supervisor.log','status.json','archive.json','gap-results.tar.gz','sha256.json'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
           and not any(part in exclude for part in p.relative_to(ROOT).parts)
           and p.name not in excluded_files and p.suffix not in ['.o']]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (ROOT/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(ROOT/'gap-results.tar.gz','w:gz') as t:
        for p in files+[ROOT/'sha256.json']:t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    info=dict(sha256=sha(ROOT/'gap-results.tar.gz'),bytes=(ROOT/'gap-results.tar.gz').stat().st_size,payload_files=len(files))
    (ROOT/'archive.json').write_text(json.dumps(info,indent=2)+'\n')
    PERSIST.mkdir(exist_ok=True,parents=True)
    for p in files+[ROOT/'sha256.json',ROOT/'archive.json',ROOT/'gap-results.tar.gz']:
        dest=PERSIST/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        assert sha(dest)==sha(p)
    state('complete',archive=info,persistent_directory=str(PERSIST))
    shutil.copy2(ROOT/'queue-status.json',PERSIST/'queue-status.json')
if __name__=='__main__':
    try:main()
    except BaseException as e:state('failed',error=repr(e));raise
