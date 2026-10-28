"""Prepare a fresh board CDC candidate; preserve the frozen reference-clock donor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


OLD = '''    always @(posedge clk or negedge cold_reset_n)begin
        if(!cold_reset_n)begin
            ready_meta<=0;ready_sync<=0;ready_final<=0;'''
NEW = '''    // Status synchronizers share the canonical reader-domain release.
    always @(posedge clk or negedge src_rst_n)begin
        if(!src_rst_n)begin
            ready_meta<=0;ready_sync<=0;ready_final<=0;'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(donor, output):
    if output.exists():
        raise ValueError('Fresh output required')
    if sha(donor/'ntrpu_axi_bridge_hbm_controlled.sv') != 'f73a5c8ff85e6ba70325d0a1c123ee649c030d4fdc50024e77240311bd679462':
        raise ValueError('Wrong board donor')
    rtl = (donor/'ntrpu_hbm_startup.sv').read_text()
    if rtl.count(OLD) != 1:
        raise ValueError('Unexpected reset source')
    output.mkdir(parents=True)
    for path in donor.iterdir():
        if path.is_file():
            shutil.copy2(path, output/path.name)
    (output/'ntrpu_hbm_startup.sv').write_text(rtl.replace(OLD, NEW, 1))
    record = dict(status='BOARD_CDC_PREPARED_PENDING_VALIDATION',
                  donor_files={p.name:sha(p) for p in sorted(donor.iterdir()) if p.is_file()},
                  changed_rtl={'ntrpu_hbm_startup.sv':sha(output/'ntrpu_hbm_startup.sv')},
                  cold_release_chains_unchanged=True, status_sampling_delayed_until_reader_release=True,
                  routed_timing_qualified=False, native_traffic_qualified=False, board_qualified=False)
    (output/'BOARD_CDC_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('donor',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=prepare(args.donor,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='donor_files'}))
