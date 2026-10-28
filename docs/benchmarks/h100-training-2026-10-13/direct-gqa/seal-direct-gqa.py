import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root/'training/direct-gqa-20261014'
assert json.loads((out/'status.json').read_text())['state'] == 'complete'
assert json.loads((out/'attention-tests.json').read_text())['status'] == 'passed'
sys.path.insert(0, str(out/'source-snapshot/scripts'))
from analyze_training_trace import analyze

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

results = {}
for label, mode, flags in json.loads((out/'cases.json').read_text()):
    results[label] = json.loads((out/label/'summary.json').read_text())
    results[label]['checkpoint_sha256'] = sha(out/label/'last.pt')
    comparison = out/(label+'-comparison.json')
    if comparison.exists():
        results[label]['audit'] = json.loads(comparison.read_text())
        assert results[label]['audit']['status'] == 'passed'
for trace in sorted(out.glob('profile-*/trace-rank*.json')):
    trace.with_name(trace.stem.replace('trace-', 'trace-summary-')+'.json').write_text(
        json.dumps(analyze(trace), indent=2)+'\n')
    with trace.open('rb') as source, gzip.open(str(trace)+'.gz', 'wb', compresslevel=1) as target:
        shutil.copyfileobj(source, target)
(out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
checkpoint = Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt')
(out/'provenance.json').write_text(json.dumps(dict(
    library_sha256=sha(root/'build-training/libntransformer_training.so'),
    source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha(checkpoint),
    throughput_scope='100 updates, offsets 10..99; continuous window includes data, audit, communication and logging',
    profile_scope='Separate 13-update runs, active offsets 11 and 12; not throughput benchmarks',
    tests_scope='Full pretraining batch and partially masked pretraining targets; not a new SFT quality evaluation',
    attention_probe='attention-probe-20261014: frozen real layer 0 and 29 tensors and upstream gradients; hashes in probe results'), indent=2)+'\n')
shutil.copy2(__file__, out/'seal-direct-gqa.py')
probe=root/'training/attention-probe-20261014'
(out/'attention-probe').mkdir(exist_ok=True)
for src,dest in ((probe/'gqa/results.json','results.json'),(probe/'probe_training_attention.py','probe_training_attention.py'),(probe/'probe.log','probe.log')):
    shutil.copy2(src,out/'attention-probe'/dest)
portable = [p for p in sorted(out.rglob('*')) if p.is_file()
            and p.suffix in ('.json', '.jsonl', '.log', '.py', '.txt', '.gz', '.yaml', '.md')
            and p.name not in ('results.tar.gz', 'archive_ready.json', 'SHA256SUMS.json')
            and not (p.name.startswith('trace-rank') and p.suffix == '.json')]
(out/'SHA256SUMS.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in portable}, indent=2)+'\n')
with tarfile.open(out/'results.tar.gz', 'w:gz') as archive:
    for path in [*portable, out/'SHA256SUMS.json']:
        archive.add(path, arcname=path.relative_to(out).as_posix())
ready = dict(archive_sha256=sha(out/'results.tar.gz'), files=len(portable)+1)
(out/'archive_ready.json').write_text(json.dumps(ready, indent=2)+'\n')
print(json.dumps(ready), flush=True)
