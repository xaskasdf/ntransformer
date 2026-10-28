import hashlib
import json
import shutil
import tarfile
from pathlib import Path

root = Path('/home/ubuntu/ntransformer-h100')
out = root / 'training/improve-20261013'
assert json.loads((out / 'status.json').read_text())['state'] == 'complete'
reference = root / 'training/instructions-20261013/reference'
provenance = {}
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def preserve(source, relative):
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    provenance[str(source)] = dict(local_record=str(relative), sha256=sha(source))
for name in ('summary.json', 'metrics.jsonl', 'experiment.json', 'batches-rank0.json', 'batches-rank1.json'):
    preserve(reference / name, Path('comparison-reference') / name)
for name in ('config.yaml', 'inputs.json'):
    preserve(reference.parent / name, Path('comparison-reference') / name)
for path in [root / 'source/CMakeLists.txt', *sorted((root / 'source/python/ntransformer_training').glob('*.py')),
             root / 'source/src/training/ops.cu']:
    preserve(path, Path('source-snapshot') / path.relative_to(root / 'source'))
for name in ('train_brandon_experiment.py', 'train_verified_instructions.py', 'analyze_verified_instructions.py',
             'compare_training_experiment.py'):
    preserve(root / 'source/scripts' / name, Path('source-snapshot/scripts') / name)
for name in ('test_training_rms_compatible.py', 'test_training_ops.py'):
    preserve(root / 'source/tests' / name, Path('source-snapshot/tests') / name)
for name in ('run-improvement.py', 'improve-build.sh', 'probe-rms.py', 'seal-improvement.py'):
    preserve(root / name, Path('execution') / name)
for path in (reference / 'last.pt', out / 'sft-compatible/last.pt'):
    provenance[str(path)] = dict(sha256=sha(path), storage='Remote checkpoint; optimizer and RNG included')
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
with tarfile.open(out / 'results.tar.gz', 'w:gz') as archive:
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.yaml', '.txt', '.py', '.cu', '.sh'):
            archive.add(path, arcname=str(path.relative_to(out)))
print('Portable archive sealed')
