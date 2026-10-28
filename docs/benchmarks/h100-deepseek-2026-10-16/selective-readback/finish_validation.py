import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
FAIR = ROOT.parent / 'ds41-fair-20261016'
sys.path.insert(0, str(FAIR))
import common as b
from admission import admitted
import run as fair_run

def main():
    deadline = time.monotonic() + 1800
    while True:
        status = json.loads((ROOT / 'status.json').read_text())
        if status['state'] == 'complete': break
        if status['state'] == 'failed' or time.monotonic() > deadline: raise RuntimeError(status)
        time.sleep(5)
    # No build or validation overlaps any performance window. Strengthen the
    # existing isolation regression with distinct inputs per layer and owner.
    validation = ROOT / 'post-validation'
    validation.mkdir(exist_ok=False)
    b.ROOT = validation; fair_run.ROOT = validation
    source = ROOT / 'source'
    shutil.copy2(source / 'tests/test_ds41_backbone.cpp', ROOT / 'test_ds41_backbone_initial.cpp')
    shutil.copy2(ROOT / 'test_ds41_backbone_final.cpp', source / 'tests/test_ds41_backbone.cpp')
    compiler = b.PREFIX / 'bin/x86_64-conda-linux-gnu-g++'
    b.command('compile-final-backbone', [compiler, '-std=c++20', '-O3', '-march=native', '-fPIC',
        '-I' + str(source / 'src'), '-I' + str(source / 'include'), '-I' + str(b.PREFIX / 'targets/x86_64-linux/include'),
        source / 'tests/test_ds41_backbone.cpp', ROOT / 'libntransformer-readback.a',
        '-L' + str(b.PREFIX / 'targets/x86_64-linux/lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread',
        '-o', validation / 'test-backbone'])
    with admitted(b), fair_run.monitoring():
        for coop in [0, 1]:
            for batch in [0, 1]:
                b.ENV = b.BASE_ENV | dict(DS41_GPU='1', DS41_VRAM_GB='0.1', DS41_PEER_VRAM_GB='0',
                    DS41_Q8_COOPERATIVE=str(coop), DS41_QKV_READBACK_BATCH=str(batch))
                b.command(f'test-prologue-coop{coop}-batch{batch}', [validation / 'test-backbone', 'prologue'])
        b.ENV = b.BASE_ENV | dict(DS41_GPU='0')
        b.command('test-attention-disabled', [ROOT / 'test-attention', 'disabled'])
        assert b.command('test-backbone-disabled', [validation / 'test-backbone', 'prologue'], required=False) == 77
    shutil.copy2(ROOT / 'build.json', ROOT / 'initial-build.json')
    build = json.loads((ROOT / 'build.json').read_text())
    build['sources']['source/tests/test_ds41_backbone.cpp'] = b.sha(source / 'tests/test_ds41_backbone.cpp')
    build['post_benchmark_test_extension'] = 'Distinct prologue inputs per layer and owner; benchmark executable/library unchanged'
    assert b.sha(ROOT / 'bench-nt') == build['binary'] and b.sha(ROOT / 'libntransformer-readback.a') == build['library']
    (ROOT / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    b.save('validation.json', dict(passed=True, benchmark_binary_unchanged=True, library_unchanged=True,
        source_sha256=b.sha(source / 'tests/test_ds41_backbone.cpp'), assertions_enabled=True,
        timing_scope='All compilation and additional checks started after performance runner completed'))
    b.status('complete')

if __name__ == '__main__': main()
