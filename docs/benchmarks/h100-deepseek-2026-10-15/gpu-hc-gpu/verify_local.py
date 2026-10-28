import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
DEST = WORKSPACE / 'docs/benchmarks/h100-deepseek-2026-10-15/gpu-hc-gpu'
SCRATCH = WORKSPACE / 'build/ds41-hc-gpu/local-check'
SCRATCH.mkdir(parents=True, exist_ok=True)
for p in DEST.iterdir():
    if p.is_file() and (p.suffix in ['.json', '.py', '.csv'] or p.name.endswith('-resources.txt')):
        shutil.copy2(p, SCRATCH / p.name)
    if p.is_dir() and (p / 'measurements.jsonl').exists():
        (SCRATCH / p.name).mkdir(exist_ok=True)
        for name in ['measurements.jsonl', 'all-logits.f32', 'final-logits.f32']:
            shutil.copy2(p / name, SCRATCH / p.name / name)
subprocess.run([sys.executable, str(SCRATCH / 'analyze.py')], check=True, capture_output=True)
assert (SCRATCH / 'RESULTS.md').read_bytes() == (DEST / 'RESULTS.md').read_bytes()
assert json.loads((SCRATCH / 'summary.json').read_text()) == json.loads((DEST / 'summary.json').read_text())
env = json.loads((DEST / 'environment.json').read_text())
files = ['src/model/ds41/' + n for n in ['ds41_gpu.cu', 'ds41_gpu.h', 'ds41_shared.cu', 'ds41_forward.cpp', 'ds41_forward.h']]
files += ['tests/test_ds41_router.cpp']
for name in files:
    assert hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() == env['source_sha256'][name], name
assert (WORKSPACE / 'tools/benchmark_ds41.cpp').read_bytes() == (DEST / 'benchmark_ds41.cpp').read_bytes()
proof = json.loads((DEST / 'local-verification.json').read_text())
proof.update(analysis_recomputed=True, report_reproduced_exactly=True, local_implementation_matches_measured_source=files)
(DEST / 'local-verification.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8', newline='\n')
subprocess.run(['scp', '-q', str(DEST / 'local-verification.json'),
               'ubuntu@<h100-host>:/lambda/nfs/btiny/ntransformer-training/deepseek-hc-gpu-20261015/local-verification.json'], check=True)
print(json.dumps(proof, indent=2))
