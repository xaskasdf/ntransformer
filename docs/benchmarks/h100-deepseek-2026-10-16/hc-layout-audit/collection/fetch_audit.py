import hashlib
import importlib.util
import json
import math
from pathlib import Path,PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
DEST=WORKSPACE/'docs/benchmarks/h100-deepseek-2026-10-16/hc-layout-audit'
FAIR=DEST.parent/'fair-comparison'
REMOTE='/home/ubuntu/ntransformer-h100/ds41-audit-20261016/results'
HOST='ubuntu@<h100-host>'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-q',*[HOST+':'+REMOTE+'/'+n for n in ['sha256.json','archive.json']],str(DEST)],check=True,timeout=90)
    manifest=json.loads((DEST/'sha256.json').read_text());info=json.loads((DEST/'archive.json').read_text())
    assert len(manifest)==info['payload_files']
    local={}
    for root in [HERE,DEST]:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix not in ['.gz','.tar','.pyc']:local[sha(p)]=p
    for name in manifest:
        if name.startswith('source/'):
            p=WORKSPACE/name.removeprefix('source/')
        elif name.startswith('baseline-source/'):
            p=FAIR/'nt-source'/name.removeprefix('baseline-source/')
        else:continue
        if p.is_file():local[sha(p)]=p
    missing={}
    for name,digest in manifest.items():
        relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
        target=(DEST/name).resolve();assert target.is_relative_to(DEST.resolve())
        if digest in local:
            target.parent.mkdir(parents=True,exist_ok=True)
            if local[digest].resolve()!=target:shutil.copy2(local[digest],target)
        elif digest not in missing:missing[digest]=name
    if missing:
        names=list(missing.values());print('Downloading',len(names),'unique missing payloads',flush=True)
        command='tar -czf - -C '+shlex.quote(REMOTE)+' -- '+' '.join(map(shlex.quote,names))
        with (HERE/'delivery.tar.gz').open('wb') as output:
            subprocess.run(['ssh',HOST,command],stdout=output,check=True,timeout=1800)
        with tarfile.open(HERE/'delivery.tar.gz') as archive:
            for member in archive:
                assert member.isfile() and member.name in names
                target=(DEST/member.name).resolve();assert target.is_relative_to(DEST.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
                assert sha(target)==manifest[member.name]
                local[manifest[member.name]]=target
        for name,digest in manifest.items():
            target=DEST/name
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(local[digest],target)
    assert all(sha(DEST/name)==digest for name,digest in manifest.items())
    summary=(DEST/'audit-summary.json').read_bytes()
    spec=importlib.util.spec_from_file_location('audit',DEST/'analyze_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=DEST;module.FAIR=FAIR
    try:
        module.main()
        assert equivalent(json.loads(summary),json.loads((DEST/'audit-summary.json').read_text()))
    finally:(DEST/'audit-summary.json').write_bytes(summary)
    assert all(sha(DEST/name)==digest for name,digest in manifest.items())
    proof=dict(payload_files_verified=len(manifest),analysis_reproduced=True,numerical_tolerance=1e-12,
        remote_archive_sha256=info['sha256'],remote_archive_bytes_verified_locally=False,
        delivery='Verified individual payloads; reused identical local control files',
        workspace_matches={name:sha(WORKSPACE/name)==sha(DEST/'source'/name) for name in
            ['src/model/ds41/ds41_hc.h','src/model/ds41/ds41_forward.cpp','tests/test_ds41_hc.cpp']})
    (DEST/'local-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    subprocess.run(['scp','-q',str(DEST/'local-verification.json'),HOST+':/lambda/nfs/btiny/ntransformer-training/deepseek-audit-20261016/results/'],check=True,timeout=90)
    print('Verified',len(manifest),'payload hashes and reproduced full-logit analysis locally.',flush=True)

if __name__=='__main__':main()
