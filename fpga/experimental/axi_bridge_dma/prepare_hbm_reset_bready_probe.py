"""Freeze a read-only reset/BREADY probe on an explicit routed checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(checkpoint,runner,output):
    if output.exists():raise ValueError('Fresh output required')
    output.mkdir(parents=True)
    here=Path(__file__).resolve().parent
    for name in ['prepare_hbm_reset_bready_probe.py','probe_hbm_reset_bready.tcl']:
        shutil.copy2(here/name,output/name)
    shutil.copy2(runner,output/'run_hbm_native_join.py')
    entry='''set_param general.maxThreads 2
if {[catch {
source [file join [file dirname [info script]] probe_hbm_reset_bready.tcl]
rf_run {*}$argv
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
'''
    (output/'inspect_hbm_native_join.tcl').write_text(entry,newline='\n')
    record=dict(status='RESET_BREADY_PROBE_PREPARED',checkpoint_sha256=sha(checkpoint),read_only=True,
                files={p.name:sha(p) for p in output.iterdir()})
    (output/'PREPARED.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','runner','output']:p.add_argument(name,type=Path)
    a=p.parse_args();prepare(a.checkpoint,a.runner,a.output)
