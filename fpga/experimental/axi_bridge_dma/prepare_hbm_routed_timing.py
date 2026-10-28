"""Prepare read-only diagnostics or one bounded post-route timing experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(donor, checkpoint, output, mode):
    if output.exists():
        raise ValueError('Fresh prepared directory required')
    expected = '885c2ec17d7fd8c4dd939556ce6cd3d3742ca7fdab851cc6695bd769112d7d05'
    if mode != 'reopen' and sha(checkpoint) != expected:
        raise ValueError('Sealed routed checkpoint required')
    output.mkdir(parents=True)
    frozen = {}
    for path in donor.iterdir():
        if path.name == 'pcie_methodology_targets.tsv' and mode != 'reopen':
            continue
        if path.suffix in ('.tcl', '.tsv') and path.name != 'inspect_hbm_native_join.tcl':
            shutil.copy2(path, output / path.name)
            frozen[path.name] = sha(path)
    here = Path(__file__).parent
    script = 'inspect_hbm_routed_timing.tcl' if mode == 'diagnose' else 'optimize_hbm_routed_timing.tcl'
    shutil.copy2(here / script, output / script)
    if mode == 'reopen':
        shutil.copy2(here / 'reconcile_hbm_pcie_timing.tcl', output / 'reconcile_hbm_pcie_timing.tcl')
        shutil.copy2(donor / 'pcie_methodology_targets.tsv', output / 'pcie_methodology_targets.tsv')
    timeout = 2400 if mode == 'explore' else 900
    runner = (here / 'run_hbm_native_join.py').read_text()
    if runner.count('timeout=1200') != 2:
        raise ValueError('Runner timeout contract changed')
    (output / 'run_hbm_native_join.py').write_text(runner.replace('timeout=1200', f'timeout={timeout}'))
    command = 'ht_run {*}$argv' if mode == 'diagnose' else f'ho_run {{*}}$argv {mode}'
    entry = ('set_param general.maxThreads 2\nif {[catch {\n'
             f'    source [file join [file dirname [info script]] {script}]\n'
             f'    {command}\n'
             '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (output / 'inspect_hbm_native_join.tcl').write_text(entry)
    record = dict(mode=mode, checkpoint_sha256=sha(checkpoint), frozen_audits=frozen,
                  script_sha256=sha(output / script), timeout_seconds=timeout,
                  cpu_count=2, nice=10, timing_qualified=False, board_qualified=False)
    (output / 'TIMING_BINDING.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('donor', 'checkpoint', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--mode', choices=('diagnose', 'explore', 'reopen'), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.donor, args.checkpoint, args.output, args.mode)))
