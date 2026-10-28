import json
from pathlib import Path
import shutil
import tarfile
import common as b

ROOT=b.ROOT
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016')

def main():
    assert json.loads((ROOT/'status.json').read_text())['state']=='complete'
    excluded_dirs={'llama-lib','llama-source','__pycache__'}
    excluded_names={'bench-nt','bench-llama','libntransformer_lib.a','queue.log','queue-status.json',
        'queue-status.json.tmp','sha256.json','archive.json','results.tar.gz','preparation.tar.gz','prepare.log','launch.log'}
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
        and not any(part in excluded_dirs for part in p.relative_to(ROOT).parts) and p.name not in excluded_names and not p.name.endswith('.tmp')]
    manifest={p.relative_to(ROOT).as_posix():b.sha(p) for p in files}
    b.save('sha256.json',manifest)
    with tarfile.open(ROOT/'results.tar.gz','w:gz') as archive:
        for p in files+[ROOT/'sha256.json']:archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    assert all(b.sha(ROOT/name)==digest for name,digest in manifest.items())
    b.save('archive.json',dict(sha256=b.sha(ROOT/'results.tar.gz'),bytes=(ROOT/'results.tar.gz').stat().st_size,payload_files=len(files)))
    PERSIST.mkdir(parents=True,exist_ok=True)
    for p in files+[ROOT/'sha256.json',ROOT/'archive.json',ROOT/'results.tar.gz']:
        target=PERSIST/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert b.sha(target)==b.sha(p)
    print((ROOT/'archive.json').read_text(),flush=True)

if __name__=='__main__':main()
