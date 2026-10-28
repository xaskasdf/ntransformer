"""Freeze HBM-domain physical optimization without changing prior preparations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(audits, checkpoint, output, mode, reference=None, remote_reference=None):
    if output.exists():
        raise ValueError('Fresh output required')
    if mode == 'placement' and sha(checkpoint) != 'a990f1837d4dcaef6a271b6b625bc7235d2b0187ec517bde658c913207de1fb5':
        raise ValueError('Exact reset-distribution candidate required')
    if mode=='reopen' and (not reference or not remote_reference or sha(reference)!='a990f1837d4dcaef6a271b6b625bc7235d2b0187ec517bde658c913207de1fb5'):
        raise ValueError('Exact original semantic reference required')
    if remote_reference and any(c in remote_reference for c in '{}\r\n'):
        raise ValueError('Unsupported remote reference encoding')
    here = Path(__file__).resolve().parent
    output.mkdir(parents=True)
    frozen = {}
    for p in audits.iterdir():
        if p.suffix in ('.tcl', '.tsv') and p.name != 'inspect_hbm_native_join.tcl':
            shutil.copy2(p, output / p.name)
            frozen[p.name] = sha(p)
    for name in ('inspect_hbm_response_enable.tcl', 'inspect_hbm_routed_timing.tcl'):
        shutil.copy2(here / name, output / name)
    runner = (here / 'run_hbm_native_join.py').read_text()
    if runner.count('timeout=1200') != 2:
        raise ValueError('Unexpected runner resource boundary')
    (output / 'run_hbm_native_join.py').write_text(runner, newline='\n')
    extra=' {'+remote_reference+'}' if remote_reference else ''
    entry = ('set_param general.maxThreads 2\nif {[catch {\n'
             ' source [file join [file dirname [info script]] inspect_hbm_response_enable.tcl]\n'
             f' he_run {{*}}$argv {mode}{extra}\n'
             '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (output / 'inspect_hbm_native_join.tcl').write_text(entry, newline='\n')
    record = dict(mode=mode, checkpoint_sha256=sha(checkpoint), frozen_audits=frozen,
                  script_sha256=sha(output / 'inspect_hbm_response_enable.tcl'),
                  timeout_seconds=1200, cpu_count=2, nice=10,
                  timing_qualified=False, board_qualified=False)
    if reference:
        record.update(reference_sha256=sha(reference),reference_remote=remote_reference)
    (output / 'RESPONSE_ENABLE_BINDING.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audits', 'checkpoint', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--mode', choices=('placement', 'reopen'), required=True)
    p.add_argument('--reference',type=Path)
    p.add_argument('--remote-reference')
    a = p.parse_args()
    print(json.dumps(prepare(a.audits, a.checkpoint, a.output, a.mode,a.reference,a.remote_reference)))
