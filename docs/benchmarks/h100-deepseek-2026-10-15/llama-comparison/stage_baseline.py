import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root=Path(__file__).resolve().parent
source=root.parent/'ds41-bench-20261015'
record=json.loads((source/'archive.json').read_text())
with (source/record['name']).open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
destination=root/'baseline';destination.mkdir(exist_ok=True)
with tarfile.open(source/record['name']) as archive:archive.extractall(destination,filter='data')
shutil.copy2(source/'archive.json',destination/'archive.json')
print('baseline staged',record['sha256'])
