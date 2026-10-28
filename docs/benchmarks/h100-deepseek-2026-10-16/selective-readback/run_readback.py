import json
from pathlib import Path
import subprocess
import sys

FAIR = Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
AUDIT = FAIR.parent / 'ds41-audit-20261016'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FAIR))
import common as b
from admission import admitted
import run as fair_run

def validate(label, workload, case, selective):
    output = ROOT / label
    assert json.loads((output / 'prompt-ids.json').read_text()) == workload['prompt_ids']
    assert json.loads((output / 'target-ids.json').read_text()) == workload['target_ids']
    expected = b.sha(AUDIT / ('nt-hc-' + case) / 'all-logits.f32')
    assert b.sha(output / 'all-logits.f32') == expected, 'changed logits: ' + label
    rows = [json.loads(line) for line in (output / 'measurements.jsonl').read_text().splitlines()]
    completions = [r for r in rows if r['kind'] == 'completion']
    assert len(completions) == 5 and all(r['repeat_bitwise_equal'] for r in completions)
    phases = [r for r in rows if r['kind'] == 'phase']
    for r in phases:
        assert r['peer_fallback_calls'] == r['routed_cpu_calls'] == r['raw_recovery_calls'] == 0
        assert r['resident_attention_calls'] == r['prologue_calls'] == r['tokens'] * 40
        assert r['prologue_staging_waits'] == 0
        if selective: assert 0 < r['prologue_deferred_calls'] == r['prologue_calls'] - r['prologue_readback_waits']
        else: assert r['prologue_deferred_calls'] == 0 and r['prologue_calls'] == r['prologue_readback_waits']
    b.save(label + '-validation.json', dict(exact_hc_reference=True, logits_sha256=expected,
        prediction_positions=65, logits_per_position=129280, exact_repeats=len(completions),
        gpu_coverage=True, fallback_calls=0, staging_reuse_waits=0))

def main():
    b.verify(); b.ROOT = ROOT; fair_run.ROOT = ROOT
    build = json.loads((ROOT / 'build.json').read_text())
    assert b.sha(ROOT / 'bench-nt') == build['binary']
    for name, digest in build['sources'].items(): assert b.sha(ROOT / name) == digest
    ar = b.PREFIX / 'bin/x86_64-conda-linux-gnu-ar'
    baseline, candidate = AUDIT / 'libntransformer-hc.a', ROOT / 'libntransformer-readback.a'
    members = b.capture([ar, 't', baseline]).splitlines()
    assert members == b.capture([ar, 't', candidate]).splitlines()
    changed = [name for name in members if subprocess.check_output([str(ar), 'p', str(baseline), name]) !=
        subprocess.check_output([str(ar), 'p', str(candidate), name])]
    assert sorted(changed) == ['ds41_forward.cpp.o', 'ds41_gpu.cu.o'], changed
    b.save('object-validation.json', dict(objects=len(members), changed=changed))
    plan = dict(paired_workloads=['p32', 'p512'], independent_launches=2, measured_rounds=[2, 3, 4],
        additional_quality_workloads=['p128', 'heldout-p512'], warmup_rounds=[0, 1], decode_tokens=64,
        baseline='Corrected HC, selective readback disabled', candidate='Same executable, selective readback enabled',
        scope='Paired speed measurement with reversed ordering; full logits checked against fixed-HC reference',
        precision='Unchanged FP32 routed compute and Q8 backbone', model=str(b.MODEL))
    b.save('plan.json', plan)
    with admitted(b), fair_run.monitoring():
        b.ENV = b.BASE_ENV | dict(DS41_GPU='1', DS41_VRAM_GB='0.1', DS41_PEER_VRAM_GB='0')
        for cooperative in [0, 1]:
            b.ENV['DS41_Q8_COOPERATIVE'] = str(cooperative)
            for batched in [0, 1]:
                b.ENV['DS41_QKV_READBACK_BATCH'] = str(batched)
                label = f'test-prologue-coop{cooperative}-batch{batched}'
                b.command(label, [ROOT / 'test-backbone', 'prologue'])
                assert 'PASS selective mirrors:' in (ROOT / (label + '.log')).read_text()
                assert 'FAIL' not in (ROOT / (label + '.log')).read_text()
        for mode in ['tail', 'compressor', 'indexer']:
            b.command('test-' + mode, [ROOT / 'test-backbone', mode])
        b.command('test-attention', [ROOT / 'test-attention'])
        b.command('test-hc', [ROOT / 'test-hc'])
        sanitizer = b.PREFIX / 'bin/compute-sanitizer'
        if sanitizer.exists():
            b.command('test-prologue-memcheck', [sanitizer, '--tool', 'memcheck', '--error-exitcode', '99', ROOT / 'test-backbone', 'prologue'])
        b.save('tests.json', dict(passed=True, sanitizer_available=sanitizer.exists(), assertions_enabled=True))
        if '--checks-only' in sys.argv:
            b.status('checks-complete'); return
        for launch in range(2):
            for index, case in enumerate(['p32', 'p512']):
                modes = [0, 1] if (launch + index) % 2 == 0 else [1, 0]
                for selective in modes: bench(case, selective, launch)
        for case in ['p128', 'heldout-p512']: bench(case, 1, 0)
    assert not b.ABORT.is_set()
    b.status('complete')

def bench(case, selective, launch):
    b.ENV = b.BASE_ENV | dict(DS41_GPU='1', DS41_GPU_ATTN='1', DS41_PROFILE='1',
        DS41_Q8_COOPERATIVE='1', DS41_PEER_MODE='execute', DS41_MXFP4_COMPUTE='fp32',
        DS41_VRAM_GB='62', DS41_PEER_VRAM_GB='62', DS41_BENCH_ALL_LOGITS='1',
        DS41_BENCH_TEACHER_FORCING='1', DS41_CACHE_POLICY='slru', DS41_CACHE_PROTECTED_PERCENT='80',
        DS41_CACHE_TRACE='0', DS41_QKV_SELECTIVE_READBACK=str(selective))
    workload = json.loads((FAIR / ('workload-' + case + '.json')).read_text())
    label = f'nt-selective{selective}-{case}-r{launch}'
    b.command(label, ['/usr/bin/time', '-v', '-o', ROOT / (label + '-resources.txt'), ROOT / 'bench-nt', b.MODEL,
        FAIR / workload['corpus'], len(workload['prompt_ids']), 64, 4, ROOT / label, label])
    validate(label, workload, case, selective)

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.ROOT = ROOT; b.status('failed', error=repr(error)); raise
