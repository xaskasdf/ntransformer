import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path('/home/ubuntu/ntransformer-h100/ds41-audit-20261015')
shutil.copy2('/home/ubuntu/ntransformer-h100/gguf_scan.py',root/'gguf_scan.py')
files=[p for p in root.iterdir() if p.is_file() and
       (p.suffix in {'.py','.json','.log','.cpp','.md'} or p.name=='cmake-source.tar')]
files += [p for folder in ['source','evidence'] for p in (root/folder).rglob('*') if p.is_file()]
files=sorted(set(files))
manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(root).as_posix()+'\n' for p in files)
(root/'SHA256SUMS').write_text(manifest)
files.append(root/'SHA256SUMS')
archive=root/'audit.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in files: t.add(p,arcname=p.relative_to(root),recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
dest=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-audit-20261015')
dest.mkdir(parents=True,exist_ok=True)
for p in [archive,root/'SHA256SUMS',root/'RESULTS.md',root/'OPTIMIZATION.md',root/'layout.json',root/'checks.json',root/'cmake-checks.json']:
    shutil.copy2(p,dest/p.name)
assert hashlib.sha256((dest/archive.name).read_bytes()).hexdigest()==digest
summary=dict(archive=archive.name,sha256=digest,files=len(files),bytes=archive.stat().st_size,
    persistent_directory=str(dest))
for d in [root,dest]: (d/'archive.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
