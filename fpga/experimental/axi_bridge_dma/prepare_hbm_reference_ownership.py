"""Regenerate native HBM with board-owned reference buffers, in a fresh candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


TOP_SHA = 'f73a5c8ff85e6ba70325d0a1c123ee649c030d4fdc50024e77240311bd679462'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Changed source anchor: ' + old[:80])
    return text.replace(old, new, 1)


def prepare(donor, out, synth_audit=False):
    if out.exists():
        raise ValueError('Fresh destination required')
    if sha(donor / 'ntrpu_axi_bridge_hbm_controlled.sv') != TOP_SHA:
        raise ValueError('Expected frozen board frontend r3')
    generation = (donor / 'generate_hbm_native_join.tcl').read_text()
    anchor = '    set_property CONFIG.pipe_sim false [get_ips ntrpu_bridge]'
    probe = '''    set hbm [get_ips native_hbm]
    set f [open $out/hbm_before_config.tsv {WRONLY CREAT EXCL}]
    foreach key [lsort [list_property $hbm]] {
        if {[string match CONFIG.* $key]} {puts $f "$key\\t[get_property $key $hbm]"}
    };close $f
    if {![string equal -nocase [get_property CONFIG.USER_DIS_REF_CLK_BUFG $hbm] false]} {
        error "Donor already disables reference buffers"
    }
    set_property CONFIG.USER_DIS_REF_CLK_BUFG true $hbm
    if {![string equal -nocase [get_property CONFIG.USER_DIS_REF_CLK_BUFG $hbm] true]} {
        error "Reference-buffer setting was not accepted"
    }
'''
    generation = once(generation, anchor, probe + anchor)
    runner = (donor / 'run_hbm_native_join.py').read_text()
    runner = once(runner, 'timeout=1200, stage=args.stage', 'timeout=1800, stage=args.stage')
    runner = once(runner, 'child.wait(timeout=1200)', 'child.wait(timeout=1800)')
    out.mkdir(parents=True)
    for path in donor.iterdir():
        if path.is_file():
            shutil.copy2(path, out / path.name)
    (out / 'generate_hbm_native_join.tcl').write_text(generation)
    (out / 'run_hbm_native_join.py').write_text(runner)
    record = dict(status='REFERENCE_OWNERSHIP_PREPARED_NOT_VALIDATED',
        donor_files={p.name: sha(p) for p in sorted(donor.iterdir()) if p.is_file()},
        unchanged_top_sha256=TOP_SHA, requested_ip_change={'USER_DIS_REF_CLK_BUFG': True},
        generated_rtl_edited=False, board_qualified=False)
    if synth_audit:
        audit = Path(__file__).parent / 'audit_hbm_reference_ownership.tcl'
        shutil.copy2(audit, out / audit.name)
        script = (out / 'elaborate_hbm_native_join.tcl').read_text()
        script = once(script, '    open_project $project', '''    open_project $project
    if {![string equal -nocase [get_property CONFIG.USER_DIS_REF_CLK_BUFG [get_ips native_hbm]] true]} {
        error "This candidate requires board-owned reference buffers"
    }''')
        script = once(script, '        write_xdc -constraints all $out/effective.xdc',
            '''        write_xdc -constraints all $out/effective.xdc
        source $source/audit_hbm_reference_ownership.tcl
        ro_audit $out''')
        (out / 'elaborate_hbm_native_join.tcl').write_text(script)
        record['synth_audit_sha256'] = sha(audit)
    (out / 'REFERENCE_OWNERSHIP_BINDING.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--synth-audit', action='store_true')
    args = parser.parse_args()
    result = prepare(args.donor, args.output, args.synth_audit)
    print(json.dumps({k: v for k, v in result.items() if k != 'donor_files'}))
