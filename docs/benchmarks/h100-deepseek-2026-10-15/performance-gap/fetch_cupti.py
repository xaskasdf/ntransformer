import json,urllib.request,zipfile,hashlib
from pathlib import Path
r=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015/cupti-13.1');r.mkdir(exist_ok=True)
u='https://pypi.org/pypi/nvidia-cuda-cupti/json'
data=json.load(urllib.request.urlopen(u))
versions=sorted((v for v in data['releases'] if v.startswith('13.1.')),key=lambda v:tuple(map(int,v.split('.'))))
print('versions',versions,flush=True)
version=versions[-1]
files=[f for f in data['releases'][version] if 'manylinux' in f['filename'] and 'x86_64' in f['filename']]
assert len(files)==1
f=files[0];dest=r/f['filename']
if not dest.exists():urllib.request.urlretrieve(f['url'],dest)
assert hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==f['digests']['sha256']
with zipfile.ZipFile(dest) as z:
    for info in z.infolist():
        if info.is_dir():continue
        p=(r/info.filename).resolve();assert p.is_relative_to(r.resolve())
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(info))
(r/'package.json').write_text(json.dumps(dict(version=version,file=f),indent=2)+'\n')
print('downloaded',f['filename'],dest.stat().st_size)

