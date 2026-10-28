"""Bind both tested control predictors into the exact complete-board adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def board_adapter(donor,unit):
    proof=json.loads((unit/'PREPARED.json').read_text())
    if proof['status']!='HBM_CONTROL_PREDICTORS_PREPARED' or sha(donor)!=proof['donor_adapter_sha256']:
        raise ValueError('Untested full-board donor')
    for n,h in proof['files'].items():
        if sha(unit/n)!=h:raise ValueError('Changed unit source: '+n)
    text=donor.read_text()
    for old,new,file in [('ntrpu_hbm_read_tracker','ntrpu_hbm_registered_ready_tracker','candidate_tracker.sv'),
                         ('ntrpu_hbm_read_return_cut','ntrpu_hbm_return_valid_queue','candidate_return.sv')]:
        found=list(re.finditer(r'\bmodule\s+'+old+r'\b.*?\bendmodule\b',text,re.S))
        if len(found)!=1:raise ValueError('Unexpected module inventory: '+old)
        candidate=(unit/file).read_text()
        anchor='module '+new+' '
        if candidate.count(anchor)!=1:raise ValueError('Candidate module differs')
        candidate=candidate.replace(anchor,'module '+old+' ',1).removeprefix('`timescale 1ns/1ps\n').removesuffix('\n')
        m=found[0];text=text[:m.start()]+candidate+text[m.end():]
    for file in ('candidate_capture.sv','candidate_validated.sv'):text+='\n'+(unit/file).read_text()
    names=re.findall(r'\bmodule\s+(\w+)',text)
    if len(names)!=len(set(names)):raise ValueError('Duplicate board modules')
    return text


def prepare(donor,unit,output):
    if output.exists():raise ValueError('Fresh output required')
    changed=board_adapter(donor/'ntrpu_hbm_read_adapter.sv',unit)
    output.mkdir(parents=True);original={}
    for p in donor.iterdir():
        if p.is_file():shutil.copy2(p,output/p.name);original[p.name]=sha(p)
    (output/'ntrpu_hbm_read_adapter.sv').write_text(changed,newline='\n')
    entry=output/'elaborate_hbm_native_join.tcl';text=entry.read_text()
    anchor='        rp_probe $out/registered_ready 0'
    if text.count(anchor)!=1:raise ValueError('Synthesis audit anchor changed')
    entry.write_text(text.replace(anchor,anchor+'\n        source $source/audit_hbm_controls_physical.tcl\n        pp_probe $out/control_predictors 0 1',1),newline='\n')
    here=Path(__file__).resolve().parent
    for name in ('prepare_hbm_controls_physical.py','audit_hbm_controls_physical.tcl'):
        if (output/name).exists():raise ValueError('New helper collides with donor')
        shutil.copy2(here/name,output/name)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    delta=sorted(n for n,h in original.items() if files[n]!=h)
    if delta!=['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv']:raise ValueError('Collateral full-board edit')
    record=dict(status='CONTROLS_FULL_BOARD_PREPARED',donor_files=original,prepared_files=files,
                changed_original_files=delta,tested_unit_binding_sha256=sha(unit/'PREPARED.json'),
                response_depth=2,return_depth=2,physical_timing_qualified=False)
    (output/'CONTROLS_PHYSICAL_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],files=len(files),changed=delta)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('donor','unit','output'):p.add_argument(key,type=Path)
    a=p.parse_args();prepare(a.donor.resolve(),a.unit.resolve(),a.output.resolve())
