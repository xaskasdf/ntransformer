"""Prepare a fresh native rebuild from frozen inputs, without a runtime cache."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(destination, frozen_repo, seed):
    expected=json.loads((frozen_repo/'evidence/seed_files.json').read_text())
    for name,digest in expected.items():
        source=seed/name
        if not source.resolve().is_relative_to(seed.resolve()) or not source.is_file() or sha(source)!=digest:
            raise ValueError('Seed verification failed: '+name)
    destination.mkdir(parents=True,exist_ok=False)
    excluded={'evidence/cache.txt','evidence/cache_files.json','evidence/seed.txt'}
    copied={}
    for p in sorted(frozen_repo.rglob('*')):
        if not p.is_file():continue
        if p.is_symlink():raise ValueError('Symlink in frozen sources')
        name=p.relative_to(frozen_repo).as_posix()
        if name in excluded:continue
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);copied[name]=sha(target)
        if copied[name]!=sha(p):raise ValueError('Copy verification failed: '+name)
    (destination/'evidence/seed.txt').write_text(str(seed.resolve())+'\n')
    record=dict(status='FRESH_NATIVE_REBUILD_PREPARED',seed_files_verified=len(expected),
                sources=copied,compiled=False,cache_required=False)
    (destination/'REBUILD.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('destination',type=Path)
    ap.add_argument('--frozen-repo',type=Path,required=True)
    ap.add_argument('--seed',type=Path,required=True)
    a=ap.parse_args();r=prepare(a.destination.resolve(),a.frozen_repo.resolve(),a.seed.resolve())
    print(json.dumps({k:v for k,v in r.items() if k!='sources'}))
