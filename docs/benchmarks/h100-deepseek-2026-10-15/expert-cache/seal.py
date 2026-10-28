import hashlib
import json
from pathlib import Path
import shutil
import tarfile
ROOT=Path(__file__).resolve().parent
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training')/ROOT.name.replace('ds41-','deepseek-',1)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    state=json.loads((ROOT/'status.json').read_text())['state']
    if state not in ['initial-complete','complete']:raise RuntimeError('measurement job still active')
    # No model weights, compiler intermediates or executable binaries needed
    # in the result payload; their hashes and build commands are recorded.
    exclude={'build','__pycache__'}
    names={'queue.log','queue-status.json','queue-status.tmp','trace.log','cache-sanitized','replay','replay-selftest-bin','preparation.tar.gz','bench','bench-initial','bench-pilot','sha256.json','flow-results.tar.gz','archive.json','seal.log','run.log','followup.log','prepare.log','finish.log','resume.log'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and not any(part in exclude for part in p.relative_to(ROOT).parts) and p.name not in names]
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (ROOT/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(ROOT/'flow-results.tar.gz','w:gz') as t:
        for p in files+[ROOT/'sha256.json']:t.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    for p in files:
        assert sha(p) == manifest[p.relative_to(ROOT).as_posix()], 'payload changed while archiving: '+str(p)
    info=dict(sha256=sha(ROOT/'flow-results.tar.gz'),bytes=(ROOT/'flow-results.tar.gz').stat().st_size,payload_files=len(files))
    (ROOT/'archive.json').write_text(json.dumps(info,indent=2)+'\n')
    PERSIST.mkdir(parents=True,exist_ok=True)
    for p in files+[ROOT/'sha256.json',ROOT/'archive.json',ROOT/'flow-results.tar.gz']:
        target=PERSIST/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert sha(target)==sha(p)
    print(json.dumps(info),flush=True)
if __name__=='__main__':main()
