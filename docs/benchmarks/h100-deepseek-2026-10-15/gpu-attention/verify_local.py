import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
DEST = WORKSPACE / 'docs/benchmarks/h100-deepseek-2026-10-15/gpu-attention'
SCRATCH = WORKSPACE / 'build/ds41-attention/local-check'
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
largest_drift = 0.

def compare(a, b, path=''):
    global largest_drift
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: compare(a[k], b[k], path + '/' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '/' + str(i))
    elif isinstance(a, float):
        # NumPy/BLAS/libm differ across Windows and Linux. The report is rounded;
        # raw numerical diagnostics must agree within an explicit FP64 allowance.
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=5e-15), (path, a, b)
        largest_drift = max(largest_drift, abs(a - b))
    else:
        assert a == b, path

summary = json.loads((DEST / 'summary.json').read_text())
compare(json.loads((SCRATCH / 'summary.json').read_text()), summary)
assert summary['cases']['defaults-p32']['all_logits_sha256'] == summary['cases']['resident-p32']['all_logits_sha256']
test_log = (DEST / 'ctest-final.log').read_text(encoding='utf-8')
assert re.search(r'100% tests passed(?:, 0 tests failed)? out of 20\b', test_log)
assert json.loads((DEST / 'ctest-final-command.json').read_text())['returncode'] == 0
assert not any(word in test_log for word in ['FAIL', 'SKIP', 'Not Run'])
fixture = re.search(r'PASS resident attention: steps=(\d+) values=(\d+) max_abs=([^;]+);', test_log)
assert fixture
for name, c in summary['cases'].items():
    assert c['all_repeats_bitwise_equal']
    d = c['comparison']
    assert d['aligned_vectors'] == d['argmax_matches'] == 65, name
    assert d['consumed_ids_equal'] and d['expert_upload_bytes_equal'], name
    assert d['relative_l2'] < 1e-5, name
env = json.loads((DEST / 'environment.json').read_text())
files = ['src/model/ds41/' + n for n in ['ds41_gpu.cu', 'ds41_gpu.h', 'ds41_attention.cu', 'ds41_attention.h',
                                        'ds41_shared.cu', 'ds41_forward.cpp', 'ds41_forward.h']]
files += ['tests/test_ds41_attention.cpp', 'tests/test_ds41_backbone.cpp', 'CMakeLists.txt']
for name in files:
    assert hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() == env['source_sha256'][name], name
assert (WORKSPACE / 'tools/benchmark_ds41.cpp').read_bytes() == (DEST / 'benchmark_ds41.cpp').read_bytes()
proof = json.loads((DEST / 'local-verification.json').read_text())
proof.update(analysis_recomputed=True, analysis_fp64_absolute_tolerance=5e-15, analysis_fp64_relative_tolerance=1e-12,
             maximum_analysis_drift=largest_drift, report_reproduced_exactly=True,
             final_default_matches_resident_bitwise=True,
             final_ctest_passed=20,
             attention_fixture=dict(steps=int(fixture[1]), values=int(fixture[2]), max_abs=float(fixture[3])),
             local_implementation_matches_measured_source=files)
(DEST / 'local-verification.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8', newline='\n')
subprocess.run(['scp', '-q', str(DEST / 'local-verification.json'),
               'ubuntu@<h100-host>:/lambda/nfs/btiny/ntransformer-training/deepseek-attention-20261015/local-verification.json'], check=True)
doc = WORKSPACE / 'docs/DEEPSEEK_INFERENCE_OPTIMIZATION.md'
body = doc.read_text(encoding='utf-8')
body = re.sub(r'shared-expert, MoE and prefill behavior\. The attention fixture exercises \d[\d,]*\nsteps and [\d,]+ comparisons',
              f'shared-expert, MoE and prefill behavior. The attention fixture exercises {int(fixture[1]):,}\nsteps and {int(fixture[2]):,} comparisons', body)
body = re.sub(r'absolute error against its independent CPU reference is [\d.eE+-]+\. Prologue',
              f'absolute error against its independent CPU reference is {float(fixture[3]):.3g}. Prologue', body)
start = '<!-- ds41-attention-final -->'
end = '<!-- /ds41-attention-final -->'
rate = summary['cases']['defaults-p32']['decode']['tokens_per_second']
block = f'''{start}
Resident attention is enabled by default; `DS41_ATTENTION_GPU=0` restores the
CPU attention control, and detailed stage dumping retains that control path.
The final build passed all 20 CTest cases without skips, including zero-RoPE
attention, isolated reset and query recovery after an unsupported tail. With
the flag unset, its prompt-32 verification measured {rate:.4f} decode tok/s;
all 8,403,200 logits matched the explicit resident run bitwise. The six paired
cases use the archived initial source; the final build additionally enables
the default and hardens metadata invalidation and probe recovery. Both source
snapshots and build hashes are retained.

The [local results](benchmarks/h100-deepseek-2026-10-15/gpu-attention/RESULTS.md)
include raw measurements, logits, telemetry, tests and source snapshots.
Archive SHA-256: `{proof['archive_sha256']}`.
All {proof['payload_files_verified']:,} payload files were hash-verified locally.
Local reanalysis reproduced the report byte for byte; numerical JSON fields
agree within FP64 absolute 5e-15 / relative 1e-12 across Windows and Linux.
The modified implementation, tests, CMake file and harness match the measured
final source. The archive and local verification record are also persisted at
`/lambda/nfs/btiny/ntransformer-training/deepseek-attention-20261015`.
{end}

'''
if start in body:
    body = re.sub(re.escape(start) + r'.*?' + re.escape(end) + r'\n\n', lambda _: block, body, flags=re.S)
else:
    anchor = '### 3. Move computation to resident experts as an alternative to copying them'
    assert body.count(anchor) == 1
    body = body.replace(anchor, block + anchor)
doc.write_text(body, encoding='utf-8', newline='\n')
print(json.dumps(proof, indent=2))
