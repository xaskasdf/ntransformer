import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+1200
while True:
    state=json.loads((root/'status.json').read_text())
    if state['state']=='complete':break
    if state['state']=='failed':raise RuntimeError(state)
    if time.monotonic()>deadline:raise TimeoutError('final run')
    time.sleep(5)
for script in ['analyze.py','report.py']:
    subprocess.run([sys.executable,root/script],check=True)
def sha(path):
    with path.open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
runtime=root/'runtime';runtime.mkdir(exist_ok=True)
libraries={}
# Store regular files at their SONAME paths so Windows extraction needs no
# symlink privileges. Runtime CUDA/cuBLAS and system libraries stay external.
for path in sorted((root/'build/bin').glob('*.so.0')):
    shutil.copy2(path,runtime/path.name)
    libraries[path.name]=dict(original_file=path.resolve().name,sha256=sha(runtime/path.name))
(root/'runtime-libraries.json').write_text(json.dumps(libraries,indent=2)+'\n')
for source,target in [
    (root/'build/CMakeCache.txt',root/'CMakeCache.txt'),
    (root/'build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/flags.make',root/'cuda-compile-flags.txt'),
    (root/'build/src/CMakeFiles/llama.dir/flags.make',root/'llama-compile-flags.txt')]:
    shutil.copy2(source,target)
exclude_dirs={'source','build','cublas-include','__pycache__'}
exclude_files={'harness.o','llama-comparison.tar.gz','archive.json','sha256.json','local-verification.json'}
payload=sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
    and not any(part in exclude_dirs for part in p.relative_to(root).parts)
    and not (p.parent==root and p.name in exclude_files))
hashes={p.relative_to(root).as_posix():sha(p) for p in payload}
(root/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
archive=root/'llama-comparison.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for path in payload+[root/'sha256.json']:tar.add(path,arcname=path.relative_to(root).as_posix())
record=dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),payload_files=len(payload),
    persistent='/lambda/nfs/btiny/ntransformer-training/deepseek-llama-comparison-20261015')
(root/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
destination=Path(record['persistent']);destination.mkdir(parents=True,exist_ok=True)
for path in payload+[archive,root/'archive.json',root/'sha256.json']:
    target=destination/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    assert sha(path)==sha(target),str(target)
print(json.dumps(record),flush=True)
