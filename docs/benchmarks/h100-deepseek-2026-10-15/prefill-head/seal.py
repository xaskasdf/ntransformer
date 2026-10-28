import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

root=Path('/home/ubuntu/ntransformer-h100/ds41-prefill-20261015')
assert json.loads((root/'status.json').read_text())['state']=='complete'
analysis=json.loads((root/'analysis.json').read_text())
for name,expected in analysis['source_sha256'].items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,name
environment={}
for name,command in [
    ('cuda',['/home/ubuntu/ntransformer-h100/env/bin/nvcc','--version']),
    ('compiler',['/home/ubuntu/ntransformer-h100/env/bin/x86_64-conda-linux-gnu-g++','--version']),
    ('gpu',['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv']),
    ('reference',['git','-C','/home/ubuntu/llama-ds41','rev-parse','HEAD'])]:
    environment[name]=subprocess.check_output(command,text=True)
(root/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
files=sorted(p for p in root.iterdir() if p.is_file() and
             (p.suffix in {'.py','.json','.log','.bin','.md'} or p.name=='source.tar'))
files=[p for p in files if p.name not in ['archive.json','local-verification.json']]
manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files)
(root/'SHA256SUMS').write_text(manifest)
files.append(root/'SHA256SUMS')
archive=root/'prefill-head.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in files: t.add(p,arcname=p.name,recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
dest=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-prefill-20261015')
dest.mkdir(parents=True,exist_ok=True)
for name in ['prefill-head.tar.gz','SHA256SUMS','RESULTS.md','OPTIMIZATION.md','analysis.json','environment.json']:
    shutil.copy2(root/name,dest/name)
assert hashlib.sha256((dest/archive.name).read_bytes()).hexdigest()==digest
record=dict(sha256=digest,archive=archive.name,files=len(files),bytes=archive.stat().st_size,
            persistent=str(dest))
for path in [root,dest]: (path/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
