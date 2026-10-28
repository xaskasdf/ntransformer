"""Run rev2 regressions, compare all outputs to rev2, and check new admission vectors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from ntrpu_hbm_dma_rev3_vectors import generate

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def mutants(out, bins, env):
    """Require semantic failures from deliberate arithmetic/configuration defects."""
    source = (ROOT / 'fpga/experimental/ntrpu_hbm_dma_bridge.sv').read_text()
    changes = [
        ('lost_request_carry', 'wire [64:0] source_limit=', 'wire [63:0] source_limit=', '+bounds_only'),
        ('wrapped_window_end', 'reg [64:0] window_end,published_end,source_capacity,target_capacity;',
         'reg [63:0] window_end; reg [64:0] published_end,source_capacity,target_capacity;', '+bounds_only'),
        ('exact_page_rejected', "page_limit>13'd4096", "page_limit>=13'd4096", '+bounds_only'),
        ('invalid_config_changes_cache', 'cfg_error<=config_bad;',
         "cfg_error<=config_bad; window_end<={1'b0,cfg.window_offset}+{1'b0,cfg.window_bytes};", '+rev3_only'),
    ]
    evidence = []
    for name, before, after, mode in changes:
        if source.count(before) != 1:
            raise ValueError('Mutation target changed: ' + name)
        stage = out / 'mutants' / name; stage.mkdir(parents=True)
        variant = stage / 'wrapper.sv'; variant.write_text(source.replace(before, after))
        files = [ROOT / 'fpga/experimental' / (n + '.sv') for n in
                 ('ntrpu_peer_read_pkg', 'ntrpu_bridge_async_fifo', 'ntrpu_hbm_dma_bridge_lane',
                  'ntrpu_hbm_dma_bridge_reference')]
        command = [str(bins / 'iverilog.exe'), '-g2012', '-Wall', '-I', str(ROOT / 'fpga/tb'),
                   '-s', 'tb_ntrpu_hbm_dma_bridge', '-o', str(stage / 'mutant.vvp'),
                   *map(str, files), str(variant), str(ROOT / 'fpga/tb/tb_ntrpu_hbm_dma_bridge.sv')]
        p = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, timeout=30)
        compile_log = p.stdout + p.stderr; (stage / 'compile.log').write_text(compile_log)
        if p.returncode or any('is sensitive to all' not in line for line in compile_log.splitlines() if 'warning:' in line):
            raise ValueError('Mutant compilation failed: ' + name)
        p = subprocess.run([str(bins / 'vvp.exe'), str(stage / 'mutant.vvp'), mode,
                            '+vectors=admission-vectors.txt'], cwd=out, env=env,
                           capture_output=True, text=True, timeout=60)
        log = p.stdout + p.stderr; (stage / 'simulation.log').write_text(log)
        failure = re.search(r'REV3 (NUMERIC_ORACLE|ADMISSION_MISMATCH|CYCLE_OUTPUT_MISMATCH|INVALID_CONFIG_CHANGED_BOUNDS)', log)
        if not p.returncode or not failure or 'WATCHDOG' in log:
            raise ValueError('Mutation was not detected by a semantic check: ' + name)
        evidence.append(dict(name=name, semantic_failure=failure[0], returncode=p.returncode))
    return evidence


def run(out, toolchain):
    out.mkdir(parents=True, exist_ok=False)
    sources, tools_before = snapshot(ROOT), snapshot(toolchain)
    started = time.monotonic()
    result = dict(status='FAILED', sources=sources, toolchain=tools_before, directed_runs=[])
    env = dict(os.environ)
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    bins = toolchain / 'mingw64/bin'
    env['PATH'] = os.pathsep.join(map(str, (bins, windows / 'System32', windows)))
    try:
        p = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/run_ntrpu_hbm_dma_rev2.py'),
                            str(out / 'regression'), '--toolchain', str(toolchain)],
                           capture_output=True, text=True, timeout=300)
        (out / 'regression-console.log').write_text(p.stdout + p.stderr)
        if p.returncode:
            raise ValueError('Prior regression failed; see nested result/logs')
        baseline = json.loads((out / 'regression/result.json').read_text())
        if baseline['status'] != 'REV2_BEHAVIOR_PASS':
            raise ValueError('Incomplete prior regression')
        image = out / 'regression/bridge/baseline/bridge.vvp'

        def simulate(name, args, timeout=60):
            p = subprocess.run([str(bins / 'vvp.exe'), str(image), *args], cwd=out,
                               env=env, capture_output=True, text=True, timeout=timeout)
            text = p.stdout + p.stderr
            (out / name).write_text(text)
            if p.returncode or re.search(r'FATAL|WATCHDOG|ERROR', text):
                raise ValueError('Simulation failed: ' + name)
            return text

        vectors = generate(out / 'admission-vectors.txt')
        text = simulate('bounds.log', ['+bounds_only', '+vectors=admission-vectors.txt'])
        if re.findall(r'^REV3 BOUNDS_PASS (\d+)$', text, re.M) != [str(vectors['vectors'])]:
            raise ValueError('Incomplete numeric evidence')
        for seed in range(24):
            text = simulate(f'directed-{seed:02}.log', ['+rev3_only', f'+seed={seed}'])
            if (re.findall(r'^BR PASS (\d+) ', text, re.M) != ['31', '32', '33'] or
                    text.count('REV3 TRANSACTIONS_PASS 3\n') != 1):
                raise ValueError('Incomplete directed evidence')
            result['directed_runs'].append(dict(seed=seed, scenarios=3))
        result['mutation_controls'] = mutants(out, bins, env)
        result.update(status='REV3_BEHAVIOR_PASS', prior_bridge_scenarios=baseline['bridge_scenarios'],
                      prior_fifo_runs=baseline['fifo_runs_count'], new_directed_scenarios=72,
                      vectors=vectors, replayed_transfers=baseline['replayed_transfers'],
                      replayed_bytes=baseline['replayed_bytes'], cycle_matched_reference=True,
                      physical_timing_measured=False, native_hbm_xdma_integrated=False, board_qualified=False)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    result['sources_after'], result['toolchain_after'] = snapshot(ROOT), snapshot(toolchain)
    if result['sources_after'] != sources or result['toolchain_after'] != tools_before:
        result.update(status='FAILED', error='Frozen source or toolchain changed')
    result['seconds'] = round(time.monotonic() - started, 3)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'FILES.json').write_text(json.dumps(snapshot(out), indent=2) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--toolchain', type=Path, required=True)
    args = ap.parse_args()
    result = run(args.output.resolve(), args.toolchain.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('sources', 'sources_after', 'toolchain', 'toolchain_after', 'directed_runs')}))
    raise SystemExit(0 if result['status'] == 'REV3_BEHAVIOR_PASS' else 1)
