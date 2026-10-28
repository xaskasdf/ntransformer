"""Compile and simulate the private bridge RTL with a relocatable Icarus tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from check_ntrpu_hbm_dma_bridge_rtl import audit_log, negative_controls


ROOT = Path(__file__).resolve().parents[1]
RTL = ['fpga/experimental/ntrpu_peer_read_pkg.sv',
       'fpga/experimental/ntrpu_bridge_async_fifo.sv',
       'fpga/experimental/ntrpu_hbm_dma_bridge_lane.sv',
       'fpga/experimental/ntrpu_hbm_dma_bridge.sv', 'fpga/tb/tb_ntrpu_hbm_dma_bridge.sv']
SOURCES = RTL + ['tools/ntrpu_transport_model.py', 'tools/ntrpu_hbm_dma_bridge_model.py',
                 'tools/check_ntrpu_hbm_dma_bridge_rtl.py', 'tools/run_ntrpu_hbm_dma_bridge_rtl.py',
                 'docs/fpga/HBM_DMA_BRIDGE_RTL.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, toolchain):
    output.mkdir(parents=True, exist_ok=False)
    tools = toolchain.resolve() / 'mingw64/bin'
    compiler, simulator = tools / 'iverilog.exe', tools / 'vvp.exe'
    env = dict(os.environ)
    # DLL resolution is private plus Windows, never a mutation of system PATH.
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str, [tools, windows / 'System32', windows]))
    before = {n: sha(ROOT / n) for n in SOURCES}
    tool_before = {p.relative_to(toolchain).as_posix(): sha(p) for p in toolchain.rglob('*') if p.is_file()}
    started = time.monotonic()
    record = dict(status='FAILED', sources=before, toolchain_sources=tool_before,
                  rtl_simulated=True, vendor_ip_simulated=False, physical_cdc_qualified=False,
                  pcie_hbm_integrated=False, board_qualified=False, runs=[])

    def execute(command, name):
        job = subprocess.run([str(v) for v in command], cwd=output, env=env,
                             capture_output=True, text=True, timeout=30)
        text = job.stdout + job.stderr
        (output / name).write_text(text, encoding='utf-8')
        if job.returncode:
            raise ValueError(f'{name}: exit {job.returncode}')
        return text

    try:
        execute([compiler, '-V'], 'compiler-version.log')
        execute([simulator, '-V'], 'simulator-version.log')
        compile_text = execute([compiler, '-g2012', '-Wall', '-s', 'tb_ntrpu_hbm_dma_bridge',
                                '-o', output / 'bridge.vvp', *[ROOT / n for n in RTL]], 'compile.log')
        record['compile_warnings'] = compile_text.count('warning:')
        # Keep benign array sensitivity notices visible; reject other warnings.
        warnings = [line for line in compile_text.splitlines() if 'warning:' in line]
        if any('is sensitive to all' not in line for line in warnings):
            raise ValueError('Unexpected compiler warning requires review')
        for seed in range(24):
            command = [simulator, output / 'bridge.vvp', f'+seed={seed}']
            if seed:
                command.append('+happy_only')
            text = execute(command, f'simulation-{seed:02}.log')
            replay = audit_log(text, seed == 0)
            record['runs'].append(dict(seed=seed, scenarios=25 if seed == 0 else 1,
                                       replay=replay, log_sha256=sha(output / f'simulation-{seed:02}.log')))
            if seed == 0:
                record['negative_controls'] = negative_controls(text)
        record['sources_after'] = {n: sha(ROOT / n) for n in SOURCES}
        record['toolchain_after'] = {p.relative_to(toolchain).as_posix(): sha(p) for p in toolchain.rglob('*') if p.is_file()}
        if record['sources_after'] != before or record['toolchain_after'] != tool_before:
            raise ValueError('Source or toolchain changed during execution')
        record.update(status='BRIDGE_RTL_TRANSACTION_REPLAY_PASS',
                      scenarios=sum(r['scenarios'] for r in record['runs']),
                      replayed_transfers=sum(p['transfers'] for r in record['runs'] for p in r['replay']),
                      replayed_bytes=sum(p['bytes'] for r in record['runs'] for p in r['replay']))
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        record['error'] = str(error)
    record['seconds'] = round(time.monotonic() - started, 6)
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'FILES.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--toolchain', type=Path, required=True, help='Private directory containing mingw64/')
    args = parser.parse_args()
    result = run(args.output.resolve(), args.toolchain.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in
                     ('sources', 'sources_after', 'toolchain_sources', 'toolchain_after', 'runs', 'negative_controls')}))
    raise SystemExit(0 if result['status'] == 'BRIDGE_RTL_TRANSACTION_REPLAY_PASS' else 1)
