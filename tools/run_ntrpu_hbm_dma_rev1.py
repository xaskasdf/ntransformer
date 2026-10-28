"""Run the donor replay unchanged, then directed revision tests over 24 phases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from run_ntrpu_hbm_dma_bridge_rtl import run

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--toolchain', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = snapshot(ROOT)
    original = run(out / 'baseline', args.toolchain.resolve())
    result = dict(status='FAILED', sources=sources, baseline_status=original['status'],
                  baseline_scenarios=original.get('scenarios', 0), directed_runs=[])
    try:
        if original['status'] != 'BRIDGE_RTL_TRANSACTION_REPLAY_PASS':
            raise ValueError('Donor regression failed: ' + original.get('error', ''))
        env = dict(os.environ)
        windows = Path(env.get('SystemRoot', 'C:/Windows'))
        tools = args.toolchain.resolve() / 'mingw64/bin'
        env['PATH'] = os.pathsep.join(map(str, [tools, windows / 'System32', windows]))
        for seed in range(24):
            proc = subprocess.run([str(tools / 'vvp.exe'), str(out / 'baseline/bridge.vvp'),
                                   '+revision_only', f'+seed={seed}'], cwd=out, env=env,
                                  capture_output=True, text=True, timeout=30)
            log = proc.stdout + proc.stderr
            (out / f'directed-{seed:02}.log').write_text(log)
            if proc.returncode or re.search(r'FATAL|ERROR|WATCHDOG', log):
                raise ValueError(f'Directed seed {seed} failed')
            if re.findall(r'^BR PASS (\d+) ', log, re.M) != ['25', '26', '27', '28'] or log.count('BR REV1_ALL_PASS 4\n') != 1:
                raise ValueError(f'Missing directed evidence for {seed}')
            result['directed_runs'].append(dict(seed=seed, scenarios=4))
        result['sources_after'] = snapshot(ROOT)
        if result['sources_after'] != sources:
            raise ValueError('Frozen sources changed')
        result.update(status='REV1_BEHAVIOR_PASS', scenarios=144, replayed_transfers=original['replayed_transfers'],
                      replayed_bytes=original['replayed_bytes'], rtl_simulated=True,
                      physical_cdc_qualified=False, board_qualified=False)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'FILES.json').write_text(json.dumps(snapshot(out), indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'sources_after', 'directed_runs')}))
    raise SystemExit(0 if result['status'] == 'REV1_BEHAVIOR_PASS' else 1)


if __name__ == '__main__':
    main()
