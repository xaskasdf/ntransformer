import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

root = Path(__file__).resolve().parent
deadline = time.monotonic()+1800
while True:
    state = json.loads((root/'status.json').read_text())
    if state['state']=='failed': raise RuntimeError(state)
    if state['state']=='complete': break
    if time.monotonic()>deadline: raise TimeoutError('benchmark did not finish')
    time.sleep(5)
for script in ['analyze.py','report.py']:
    subprocess.run([sys.executable,root/script],check=True)
shutil.copy2(root.parent/'ds41-shared-20261015/shared-compile-flags.log',root/'shared-compile-flags.log')
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
excluded = {'benchmark-results.tar.gz','archive.json','sha256.json','local-verification.json','seal.log'}
payload = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts)
hashes = {p.relative_to(root).as_posix():sha(p) for p in payload}
(root/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
archive = root/'benchmark-results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for path in payload+[root/'sha256.json']:
        tar.add(path,arcname=path.relative_to(root).as_posix())
record = dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(payload),
              persistent='/lambda/nfs/btiny/ntransformer-training/deepseek-benchmark-20261015')
(root/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
target=Path(record['persistent']);target.mkdir(parents=True,exist_ok=True)
for path in payload+[root/'sha256.json',archive,root/'archive.json']:
    dst=target/path.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    assert sha(path)==sha(dst), str(dst)
print(json.dumps(record),flush=True)
