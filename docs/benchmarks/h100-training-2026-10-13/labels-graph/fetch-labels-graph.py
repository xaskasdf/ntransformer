import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

root = next(p for p in Path(__file__).resolve().parents
            if (p/'scripts/analyze_training_trace.py').is_file()
            and (p/'python/ntransformer_training').is_dir())
dest = root/'docs/benchmarks/h100-training-2026-10-13/labels-graph'
dest.mkdir(exist_ok=True)
remote = 'ubuntu@<h100-host>:/home/ubuntu/ntransformer-h100/training/labels-graph-20261014/'
for name in ('archive_ready.json', 'results.tar.gz'):
    subprocess.run(['scp', remote+name, str(dest/name)], check=True, timeout=180)
with (dest/'results.tar.gz').open('rb') as handle:
    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
assert actual == json.loads((dest/'archive_ready.json').read_text())['archive_sha256']
with tarfile.open(dest/'results.tar.gz') as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        assert not (member.issym() or member.islnk() or path.is_absolute() or '..' in path.parts
                    or '\\' in member.name or ':' in member.name)
    archive.extractall(dest, filter='data')
manifest = json.loads((dest/'SHA256SUMS.json').read_text())
for name, expected in manifest.items():
    with (dest/name).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, name
for path in ('python/ntransformer_training/graphs.py','python/ntransformer_training/loss.py',
             'python/ntransformer_training/brandon.py','scripts/train_brandon_experiment.py',
             'tests/test_training_loss.py','tests/test_training_loss_labels.py',
             'tests/test_training_graph_storage.py','src/training/cross_entropy.cu'):
    assert (root/path).read_bytes() == (dest/'source-snapshot'/path).read_bytes(), path
print(len(manifest), 'downloaded files and current executed sources verified', flush=True)
