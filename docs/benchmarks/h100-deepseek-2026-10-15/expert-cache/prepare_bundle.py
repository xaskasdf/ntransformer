import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-cache-20261015/preparation')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
frozen=json.loads((ROOT/'frozen-inputs.json').read_text())
names=[name for name in frozen if name not in ['bench','build/libntransformer_lib.a']]
names+=['frozen-inputs.json','cache-cpu-test.log','cache-sanitized.log',
    'compile-cache-sanitized-command.json','cache-sanitized-command.json',
    'replay-selftest-0.log','replay-selftest-80.log','replay-reject-truncated.log',
    'replay-selftest-0-command.json','replay-selftest-80-command.json','replay-reject-truncated-command.json']
names=sorted(set(names))
manifest={name:sha(ROOT/name) for name in names}
(ROOT/'preparation-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(ROOT/'preparation.tar.gz','w:gz') as package:
    for name in names+['preparation-sha256.json']:package.add(ROOT/name,arcname=name,recursive=False)
for name,digest in manifest.items():assert sha(ROOT/name)==digest,name
info=dict(sha256=sha(ROOT/'preparation.tar.gz'),bytes=(ROOT/'preparation.tar.gz').stat().st_size,
          payload_files=len(manifest),state='compiled-and-CPU-validated; GPU validation and benchmarks pending')
(ROOT/'preparation.json').write_text(json.dumps(info,indent=2)+'\n')
PERSIST.mkdir(parents=True,exist_ok=True)
for name in names+['preparation-sha256.json','preparation.tar.gz','preparation.json']:
    target=PERSIST/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/name,target);assert sha(target)==sha(ROOT/name)
print(json.dumps(info),flush=True)
