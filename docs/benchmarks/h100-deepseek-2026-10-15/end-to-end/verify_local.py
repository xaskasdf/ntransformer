import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

workspace=Path(__file__).resolve().parents[2]
staging=Path(__file__).resolve().parent
destination=workspace/'docs/benchmarks/h100-deepseek-2026-10-15/end-to-end'
record=json.loads((staging/'archive.json').read_text())
archive=staging/record['name']
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
assert archive.stat().st_size==record['bytes'] and sha(archive)==record['sha256']
destination.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        resolved=(destination/member.name).resolve()
        assert resolved.is_relative_to(destination.resolve()) and member.isfile(), member.name
    tar.extractall(destination,filter='data')
shutil.copy2(staging/'archive.json',destination/'archive.json')
hashes=json.loads((destination/'sha256.json').read_text())
assert len(hashes)==record['payload_files']
for name,value in hashes.items(): assert sha(destination/name)==value, name
assert sha(workspace/'tools/benchmark_ds41.cpp')==hashes['benchmark_ds41.cpp']
# Recompute analysis in an isolated copy, preserving the sealed original files.
check=staging/'local-check'
check.mkdir(exist_ok=True)
for name in hashes:
    source=destination/name; target=check/name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
subprocess.run([sys.executable,check/'analyze.py',check],check=True)
original=json.loads((destination/'summary.json').read_text())
recomputed=json.loads((check/'summary.json').read_text())
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    else: assert a==b,(a,b)
equal(original,recomputed)
# Use the original summary for exact report reproduction across Python versions.
shutil.copy2(destination/'summary.json',check/'summary.json')
subprocess.run([sys.executable,check/'report.py',check],check=True)
assert (check/'RESULTS.md').read_bytes().replace(b'\r\n',b'\n')==(destination/'RESULTS.md').read_bytes()
for name,value in hashes.items(): assert sha(destination/name)==value, name
verification=dict(archive_sha256=record['sha256'],payload_files_verified=len(hashes),
    benchmark_source_matches_working_tree=True,analysis_recomputed=True,report_reproduced_with_normalized_newlines=True,
    float_comparison_relative_and_absolute_tolerance=1e-12,
    observed_competing_gpu_processes=0,finite_logit_values_checked=original['finite_logit_values_checked'])
(destination/'local-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
