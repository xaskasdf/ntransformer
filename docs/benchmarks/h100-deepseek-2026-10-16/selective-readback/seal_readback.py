import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
FAIR = ROOT.parent / 'ds41-fair-20261016'
AUDIT = ROOT.parent / 'ds41-audit-20261016'
PERSIST = Path('/lambda/nfs/btiny/ntransformer-training/deepseek-readback-20261016')

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'complete'
    assert json.loads((ROOT / 'post-validation/validation.json').read_text())['passed']
    import analyze_readback
    analyze_readback.main()
    result = ROOT / 'results'
    result.mkdir(exist_ok=False)
    def copy(source, relative):
        target = result / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in ['.json', '.jsonl', '.csv', '.py', '.log', '.txt', '.md']:
            copy(path, path.name)
    for run in sorted(ROOT.glob('nt-selective*-r[01]')):
        for path in run.iterdir():
            if path.is_file(): copy(path, path.relative_to(ROOT))
    for path in (ROOT / 'post-validation').iterdir():
        if path.is_file() and path.suffix in ['.json', '.jsonl', '.csv', '.log']:
            copy(path, path.relative_to(ROOT))
    copy(ROOT / 'test_ds41_backbone_initial.cpp', 'source/tests/test_ds41_backbone_initial.cpp')
    for name in ['src/model/ds41/ds41_gpu.cu', 'src/model/ds41/ds41_gpu.h', 'src/model/ds41/ds41_forward.cpp',
                 'src/model/ds41/ds41_hc.h', 'tests/test_ds41_backbone.cpp', 'tests/test_ds41_attention.cpp', 'tests/test_ds41_hc.cpp']:
        copy(ROOT / 'source' / name, 'source/' + name)
        copy(AUDIT / 'hc-source' / name, 'baseline-source/' + name)
    for name in ['benchmark_ds41.cpp', 'benchmark_resources.h']:
        copy(ROOT / name, 'source/tools/' + name)
    copy(FAIR / 'environment.json', 'baseline-environment.json')
    copy(AUDIT / 'hc-build.json', 'baseline-build.json')
    for case in ['p32', 'p128', 'p512', 'heldout-p512']:
        copy(FAIR / ('workload-' + case + '.json'), 'inputs/workload-' + case + '.json')
        workload = json.loads((FAIR / ('workload-' + case + '.json')).read_text())
        copy(FAIR / workload['corpus'], 'inputs/' + workload['corpus'])
    # The input manifests retain historical pre-HC predictions for provenance;
    # only corpus tokens are used. Quality compares the corrected HC full logits.
    files = [p for p in sorted(result.rglob('*')) if p.is_file()]
    manifest = {p.relative_to(result).as_posix(): sha(p) for p in files}
    (result / 'sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive_path = ROOT / 'readback-results.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in [*files, result / 'sha256.json']:
            archive.add(path, arcname=path.relative_to(result), recursive=False)
    info = dict(sha256=sha(archive_path), bytes=archive_path.stat().st_size, payload_files=len(files))
    (result / 'archive.json').write_text(json.dumps(info, indent=2) + '\n')
    shutil.copytree(result, PERSIST / 'results', dirs_exist_ok=False)
    for name, digest in manifest.items(): assert sha(PERSIST / 'results' / name) == digest
    shutil.copy2(archive_path, PERSIST / archive_path.name)
    assert sha(PERSIST / archive_path.name) == info['sha256']
    # Keep the measured executable and patched library durable as well.
    for name in ['bench-nt', 'libntransformer-readback.a']:
        shutil.copy2(ROOT / name, PERSIST / name)
        assert sha(PERSIST / name) == sha(ROOT / name)
    print(json.dumps(info), flush=True)

if __name__ == '__main__': main()
