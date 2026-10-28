"""Run 144 bridge scenarios and FIFO capacity/ordering tests in three widths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def snapshot(path):
    return {p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--toolchain', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = snapshot(ROOT); toolchain = snapshot(args.toolchain)
    result = dict(status='FAILED', sources=sources, toolchain=toolchain, fifo_runs=[])
    try:
        cmd = [sys.executable, '-B', str(ROOT / 'tools/run_ntrpu_hbm_dma_rev1.py'),
               str(out / 'bridge'), '--toolchain', str(args.toolchain.resolve())]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        (out / 'bridge-console.log').write_text(p.stdout + p.stderr)
        if p.returncode:
            raise ValueError('Bridge regression failed')
        bridge = json.loads((out / 'bridge/result.json').read_text())
        if bridge['status'] != 'REV1_BEHAVIOR_PASS' or bridge['scenarios'] != 144:
            raise ValueError('Incomplete bridge evidence')
        tools = args.toolchain.resolve() / 'mingw64/bin'
        env = dict(os.environ); windows = Path(env.get('SystemRoot', 'C:/Windows'))
        env['PATH'] = os.pathsep.join(map(str, [tools, windows / 'System32', windows]))
        for width in (2, 143, 514):
            cmd = [str(tools / 'iverilog.exe'), '-g2012', '-Wall', '-s', 'tb_ntrpu_bridge_fifo_rev2',
                   f'-Ptb_ntrpu_bridge_fifo_rev2.WIDTH={width}', '-o', str(out / f'fifo-{width}.vvp'),
                   str(ROOT / 'fpga/experimental/ntrpu_bridge_async_fifo.sv'),
                   str(ROOT / 'fpga/tb/tb_ntrpu_bridge_fifo_rev2.sv')]
            p = subprocess.run(cmd, env=env, cwd=out, capture_output=True, text=True, timeout=30)
            (out / f'compile-{width}.log').write_text(p.stdout + p.stderr)
            if p.returncode or re.search('warning:|error:', p.stdout + p.stderr, re.I):
                raise ValueError('FIFO compile requires review')
            for seed in range(24):
                p = subprocess.run([str(tools / 'vvp.exe'), str(out / f'fifo-{width}.vvp'), f'+seed={seed}'],
                                   env=env, cwd=out, capture_output=True, text=True, timeout=30)
                log = p.stdout + p.stderr
                (out / f'fifo-{width}-{seed:02}.log').write_text(log)
                rows = re.findall(r'FIFO_REV2 PASS width=(\d+) seed=(\d+) written=(\d+) read=(\d+) discarded=5', log)
                if p.returncode or re.search(r'FATAL|WATCHDOG|ERROR', log) or len(rows) != 1:
                    raise ValueError(f'FIFO test failed width={width} seed={seed}')
                w, sd, written, read = map(int, rows[0])
                if (w, sd) != (width, seed) or written - read != 5 or read < 60:
                    raise ValueError('FIFO accounting mismatch')
                result['fifo_runs'].append(dict(width=w, seed=sd, written=written, read=read, discarded=5))
        if snapshot(ROOT) != sources or snapshot(args.toolchain) != toolchain:
            raise ValueError('Source or toolchain changed')
        result.update(status='REV2_BEHAVIOR_PASS', bridge_scenarios=144, fifo_runs_count=72,
                      replayed_transfers=bridge['replayed_transfers'], replayed_bytes=bridge['replayed_bytes'],
                      rtl_simulated=True, physical_cdc_qualified=False, board_qualified=False)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'FILES.json').write_text(json.dumps(snapshot(out), indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'toolchain', 'fifo_runs')}))
    raise SystemExit(0 if result['status'] == 'REV2_BEHAVIOR_PASS' else 1)


if __name__ == '__main__':
    main()
