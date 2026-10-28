"""Freeze a single-LUT shared BREADY placement experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    if args.output.exists():
        raise ValueError('Fresh output required')
    if bool(args.plan) != (args.mode != 'probe'):
        raise ValueError('An explicit plan is required for eco and reopen')
    args.output.mkdir(parents=True)
    frozen = {}
    for path in sorted(args.audits.iterdir()):
        if path.suffix in ('.tcl', '.tsv') and path.name != 'inspect_hbm_native_join.tcl':
            shutil.copy2(path, args.output / path.name)
            frozen[path.name] = sha(path)
    here = Path(__file__).resolve().parent
    for name in ('prepare_hbm_bready_shared_placement.py', 'inspect_hbm_bready_shared_placement.tcl'):
        if (args.output / name).exists():
            raise ValueError('Helper collision')
        shutil.copy2(here / name, args.output / name)
    shutil.copy2(args.runner, args.output / 'run_hbm_native_join.py')
    if args.plan:
        shutil.copy2(args.plan, args.output / 'placement_plan.tsv')
    entry = ('set_param general.maxThreads 2\nif {[catch {\n'
             'set ::pp_candidate 1\nset ::bp_candidate 1\n'
             'source [file join [file dirname [info script]] inspect_hbm_bready_shared_placement.tcl]\n'
             f'lp_run {{*}}$argv {args.mode}\n'
             '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (args.output / 'inspect_hbm_native_join.tcl').write_text(entry, newline='\n')
    binding = dict(mode='reopen', placement_mode=args.mode,
                   checkpoint_sha256=sha(args.checkpoint), frozen_audits=frozen,
                   incremental_sha256=None, incremental_remote=None, timeout_seconds=1800,
                   registered_ready=True, control_predictors=True, control_boundary=True,
                   registered_bready=True, vendor_feedback_probe=True,
                   shared_predictor_placement=True, clock_repair=False,
                   full_rtl_resynthesis=False, baseline_read_only=args.mode == 'probe',
                   physical_timing_qualified=False)
    binding['prepared_files'] = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'CAPTURE_STAGE_BINDING.json').write_text(json.dumps(binding, indent=2) + '\n')
    print(json.dumps({k: v for k, v in binding.items() if k not in ('frozen_audits', 'prepared_files')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('audits', 'runner', 'checkpoint', 'output'):
        parser.add_argument(key, type=Path)
    parser.add_argument('--mode', choices=('eco', 'reopen'), required=True)
    parser.add_argument('--plan', type=Path)
    prepare(parser.parse_args())
