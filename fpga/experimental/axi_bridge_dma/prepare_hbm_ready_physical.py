"""Bind the sealed registered-ready unit candidate into the full board donor."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def board_adapter(donor, unit):
    proof=json.loads((unit/'PREPARED.json').read_text())
    if sha(donor)!=proof['donor_adapter_sha256']:raise ValueError('Untested board donor')
    for name,digest in proof['files'].items():
        if sha(unit/name)!=digest:raise ValueError('Unit source changed: '+name)
    old=donor.read_text()
    matches=list(re.finditer(r'\bmodule\s+ntrpu_hbm_read_tracker\b.*?\bendmodule\b',old,re.S))
    if len(matches)!=1:raise ValueError('Tracker inventory changed')
    candidate=(unit/'candidate_tracker.sv').read_text()
    anchor='module ntrpu_hbm_registered_ready_tracker '
    if candidate.count(anchor)!=1:raise ValueError('Unit tracker module changed')
    candidate=candidate.replace(anchor,'module ntrpu_hbm_read_tracker ',1)
    candidate=candidate.removeprefix('`timescale 1ns/1ps\n').removesuffix('\n')
    m=matches[0]
    return old[:m.start()]+candidate+old[m.end():]+'\n'+(unit/'ntrpu_hbm_ready_capture_queue.sv').read_text()


def prepare(donor,unit,output):
    if output.exists():raise ValueError('Fresh output required')
    here=Path(__file__).resolve().parent
    changed=board_adapter(donor/'ntrpu_hbm_read_adapter.sv',unit)
    output.mkdir(parents=True)
    original={}
    for p in donor.iterdir():
        if p.is_file():shutil.copy2(p,output/p.name);original[p.name]=sha(p)
    (output/'ntrpu_hbm_read_adapter.sv').write_text(changed,newline='\n')
    entry=output/'elaborate_hbm_native_join.tcl'
    text=entry.read_text();anchor='        cp_probe $out/response_capture 0'
    if text.count(anchor)!=1:raise ValueError('Synthesis audit anchor changed')
    entry.write_text(text.replace(anchor,anchor+'\n        source $source/audit_hbm_ready_physical.tcl\n        rp_probe $out/registered_ready 0',1),newline='\n')
    for name in ('audit_hbm_ready_physical.tcl','prepare_hbm_ready_physical.py'):
        shutil.copy2(here/name,output/name)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    delta=sorted(n for n,h in original.items() if files[n]!=h)
    if delta!=['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv']:raise ValueError('Collateral board change')
    record=dict(status='READY_FULL_BOARD_PREPARED',donor_files=original,prepared_files=files,
                tested_unit_binding_sha256=sha(unit/'PREPARED.json'),changed_original_files=delta,
                response_depth=2,full_rtl_resynthesis=True,physical_timing_qualified=False)
    (output/'READY_PHYSICAL_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],files=len(files),changed=delta)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('donor','unit','output'):p.add_argument(key,type=Path)
    a=p.parse_args();prepare(a.donor.resolve(),a.unit.resolve(),a.output.resolve())
