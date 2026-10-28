import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root/'training/gradient-buckets-20261015-r2'
assert json.loads((out/'status.json').read_text())['state'] == 'complete'
assert json.loads((out/'gradient-bucket-tests.json').read_text())['status'] == 'passed'
sys.path.insert(0,str(out/'source-snapshot/scripts'))
from analyze_training_trace import analyze

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

results = {}
for label in ['preflight-control','preflight-batched','preflight-buckets25','preflight-buckets512']+[x[0] for x in json.loads((out/'cases.json').read_text())]:
    results[label] = json.loads((out/label/'summary.json').read_text())
    results[label]['checkpoint_sha256'] = sha(out/label/'last.pt')
    comparison = out/(label+'-comparison.json')
    if comparison.exists():
        results[label]['audit'] = json.loads(comparison.read_text())
        assert results[label]['audit']['status'] == 'passed',label
for trace in sorted(out.glob('profile-*/trace-rank*.json')):
    trace.with_name(trace.stem.replace('trace-','trace-summary-')+'.json').write_text(
        json.dumps(analyze(trace),indent=2)+'\n')
    with trace.open('rb') as source,gzip.open(str(trace)+'.gz','wb',compresslevel=1) as target:
        shutil.copyfileobj(source,target)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
checkpoint = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt')
(out/'provenance.json').write_text(json.dumps(dict(
    library_sha256=sha(out/'libntransformer_training.so'),source_checkpoint=str(checkpoint),
    source_checkpoint_sha256=sha(checkpoint),
    throughput_scope='100 updates, offsets 10..99; continuous window includes data, audit, communication and logging',
    profile_scope='Separate 13-update strict runs, active offsets 11 and 12; not throughput benchmarks',
    tests_scope='Exact graph/bucket gradients, tied weights, two updates with accumulation and optimizer state, four distributed preflights; not a new SFT quality evaluation',
    memory_scope='Allocator allocated and reserved counters; graph pool reservation must not be treated as free memory'),indent=2)+'\n')
shutil.copy2(__file__,out/'seal-gradient-buckets.py')
for name in ('gradient-buckets-20261015',):
    shutil.copytree(root/'training'/name,out/'rejected-probes'/name,
                    ignore=shutil.ignore_patterns('*.so','__pycache__'),dirs_exist_ok=True)
portable = [p for p in sorted(out.rglob('*')) if p.is_file()
            and p.suffix in ('.json','.jsonl','.log','.py','.txt','.gz','.yaml','.md','.cu')
            and p.name not in ('results.tar.gz','archive_ready.json','SHA256SUMS.json')
            and not (p.name.startswith('trace-rank') and p.suffix == '.json')]
(out/'SHA256SUMS.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in portable},indent=2)+'\n')
with tarfile.open(out/'results.tar.gz','w:gz') as archive:
    for path in [*portable,out/'SHA256SUMS.json']:
        archive.add(path,arcname=path.relative_to(out).as_posix())
ready = dict(archive_sha256=sha(out/'results.tar.gz'),files=len(portable)+1)
(out/'archive_ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print(json.dumps(ready),flush=True)
