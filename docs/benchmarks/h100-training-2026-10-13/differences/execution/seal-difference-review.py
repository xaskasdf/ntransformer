import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

root=Path('/home/ubuntu/ntransformer-h100')
out=root/'training/difference-review-20261013'
assert json.loads((out/'status.json').read_text())['state']=='complete'
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()
for name in ('probe-real-attention.py','seal-difference-review.py'):
    shutil.copyfile(root/name,out/'execution'/name)
provenance=json.loads((out/'provenance.json').read_text())
provenance['library_sha256']=sha(root/'build-training/libntransformer_training.so')
provenance['checkpoint_sha256']={name:sha(out/name/'last.pt')
    for name in ('reference-repeat','deterministic-reference','deterministic-native','deterministic-checkpoint','deterministic-peer')}
provenance['gpu_processes_at_completion']=subprocess.check_output(
    ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
with tarfile.open(out/'results.tar.gz','w:gz') as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.py','.cu','.txt'):
            archive.add(path,arcname=str(path.relative_to(out)))
print(json.dumps(dict(archive_sha256=sha(out/'results.tar.gz'),gpu_processes=provenance['gpu_processes_at_completion'])))
