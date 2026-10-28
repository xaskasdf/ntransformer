"""Copy the frozen readiness audits into a new board-specific apply/reopen stage."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(donor, output, mode):
    if output.exists():
        raise ValueError('Fresh output required')
    output.mkdir(parents=True)
    names = ['audit_hbm_bar_control.tcl', 'audit_hbm_loader_readiness.tcl',
             'audit_hbm_native_boundary.tcl', 'audit_hbm_native_constraints.tcl',
             'constrain_hbm_native_constraints.tcl']
    for name in names:
        shutil.copy2(donor/name, output/name)
    bc = output/'audit_hbm_bar_control.tcl'
    script = bc.read_text()
    old = 'if {[llength [get_ports]]!=74} {error "Expected only 74 physical/sequencer port bits"}'
    if script.count(old)!=1:
        raise ValueError('Unexpected frozen board-port assertion')
    bc.write_text(script.replace(old, 'if {[llength [get_ports]]!=75} {error "Expected 75 board port bits"}', 1))
    here = Path(__file__).parent
    for name in ['audit_hbm_board_cdc.tcl','audit_hbm_reference_ownership.tcl']:
        shutil.copy2(here/name, output/name)
    shutil.copy2(here/(mode+'_hbm_board_cdc.tcl'), output/'inspect_hbm_native_join.tcl')
    runner = (here/'run_hbm_native_join.py').read_text()
    if runner.count('timeout=1200')!=2:
        raise ValueError('Unexpected runner limit')
    (output/'run_hbm_native_join.py').write_text(runner.replace('timeout=1200','timeout=900'))
    record=dict(mode=mode, frozen_audits={name:sha(donor/name) for name in names},
                changed_legacy_assertion='physical port bits 74 -> 75',
                board_audit_sha256=sha(output/'audit_hbm_board_cdc.tcl'), tool_timeout_seconds=900,
                status='BOARD_CONSTRAINTS_PREPARED_NOT_VALIDATED')
    (output/'BOARD_CONSTRAINTS_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['apply','reopen'],required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.donor,args.output,args.mode)))
