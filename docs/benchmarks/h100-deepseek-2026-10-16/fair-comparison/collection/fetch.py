import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
DEST=WORKSPACE/'docs/benchmarks/h100-deepseek-2026-10-16/fair-comparison'
REMOTE='/home/ubuntu/ntransformer-h100/ds41-fair-20261016'
HOST='ubuntu@<h100-host>'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b

def restore_source_support(environment):
    # seal.py excludes __pycache__ from the result archive, while the frozen
    # source snapshot includes it. Restore those exact bytes from the already
    # downloaded preparation archive instead of weakening source verification.
    required={f'nt-source/{name}':digest for name,digest in environment['nt_source_sha256'].items()
        if not (DEST/'nt-source'/name).exists()}
    if not required:return {}
    frozen=json.loads((DEST/'frozen-inputs.json').read_text())
    preparation=json.loads((DEST/'preparation.json').read_text())
    package_path=HERE/'preparation.tar.gz'
    assert package_path.stat().st_size==preparation['bytes']
    assert sha(package_path)==preparation['sha256']
    with tarfile.open(package_path) as package:
        for name,digest in required.items():
            assert frozen.get(name)==digest,('unfrozen source support',name)
            relative=PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            target=(DEST/name).resolve();assert target.is_relative_to(DEST.resolve())
            member=package.getmember(name);assert member.isfile()
            payload=package.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest()==digest,name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    return required

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-q',HOST+':'+REMOTE+'/sha256.json',HOST+':'+REMOTE+'/archive.json',str(DEST)],check=True,timeout=90)
    manifest=json.loads((DEST/'sha256.json').read_text());archive=json.loads((DEST/'archive.json').read_text())
    assert len(manifest)==archive['payload_files']
    representatives={};copies={}
    for name,digest in manifest.items():
        relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
        assert (DEST/name).resolve().is_relative_to(DEST.resolve())
        if digest not in representatives:representatives[digest]=name
        else:copies[name]=representatives[digest]
    missing=[name for digest,name in representatives.items() if not (DEST/name).exists() or sha(DEST/name)!=digest]
    if missing:
        print(f'Downloading {len(missing)} unique payloads; reconstructing {len(copies)} identical copies.',flush=True)
        command='tar -czf - -C '+shlex.quote(REMOTE)+' -- '+' '.join(map(shlex.quote,missing))
        with (HERE/'delivery.tar.gz').open('wb') as stream:
            subprocess.run(['ssh',HOST,command],stdout=stream,check=True,timeout=1800)
        with tarfile.open(HERE/'delivery.tar.gz') as package:
            for member in package:
                assert member.isfile() and member.name in missing
                target=(DEST/member.name).resolve();assert target.is_relative_to(DEST.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(package.extractfile(member).read())
    for name,source in copies.items():
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DEST/source,target)
    assert all(sha(DEST/name)==digest for name,digest in manifest.items())
    print(f'All {len(manifest)} payload hashes verified.',flush=True)
    original_summary=(DEST/'summary.json').read_bytes();original_report=(DEST/'REPORT.md').read_bytes()
    try:
        subprocess.run([sys.executable,str(DEST/'analyze.py')],check=True,timeout=180)
        reproduced=equivalent(json.loads(original_summary),json.loads((DEST/'summary.json').read_text()))
        report_identical=original_report==(DEST/'REPORT.md').read_bytes()
        assert reproduced,'local numerical analysis differs beyond 1e-12 tolerance'
    finally:
        (DEST/'summary.json').write_bytes(original_summary);(DEST/'REPORT.md').write_bytes(original_report)
    assert all(sha(DEST/name)==digest for name,digest in manifest.items())
    environment=json.loads((DEST/'environment.json').read_text())
    restored=restore_source_support(environment)
    assert all(sha(DEST/'nt-source'/name)==digest for name,digest in environment['nt_source_sha256'].items())
    proof=dict(payload_files_verified=len(manifest),archive_sha256=archive['sha256'],archive_verified_locally=False,
        delivery='unique payload download plus verified identical copies',analysis_reproduced=True,
        numerical_reproduction_tolerance=1e-12,report_text_identical=report_identical,source_snapshot_verified=True,
        source_support_restored_from_preparation=restored,
        workspace_matches_at_collection={name:sha(WORKSPACE/'tools'/name)==sha(DEST/name)
            for name in ['benchmark_ds41.cpp','benchmark_llama_ds41.cpp','benchmark_resources.h']})
    (DEST/'local-verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8',newline='\n')
    subprocess.run(['scp','-q',str(DEST/'local-verification.json'),HOST+':/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016/'],check=True,timeout=90)
    print('Analysis reproduced locally; original payload hashes preserved.',flush=True)

if __name__=='__main__':main()
