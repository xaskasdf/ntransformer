"""Compose the tested credit/validated boundary into the complete board."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def board_adapter(donor,unit):
    proof=json.loads((unit/'PREPARED.json').read_text())
    if proof['status']!='HBM_CONTROL_BOUNDARIES_PREPARED' or sha(donor)!=proof['donor_adapter_sha256']:
        raise ValueError('Untested board donor')
    for name,digest in proof['files'].items():
        if sha(unit/name)!=digest:raise ValueError('Changed unit source: '+name)
    text=donor.read_text()
    for old,new,file in [('ntrpu_hbm_read_tracker','ntrpu_hbm_boundary_tracker','candidate_tracker.sv'),
                         ('ntrpu_hbm_predict_validated_queue','ntrpu_hbm_registered_validated_queue','candidate_validated.sv')]:
        found=list(re.finditer(r'\bmodule\s+'+old+r'\b.*?\bendmodule\b',text,re.S))
        if len(found)!=1:raise ValueError('Module inventory: '+old)
        candidate=(unit/file).read_text();anchor='module '+new+' '
        if candidate.count(anchor)!=1:raise ValueError('Candidate module differs')
        candidate=candidate.replace(anchor,'module '+old+' ',1)
        if file=='candidate_tracker.sv':
            anchor='ntrpu_hbm_registered_validated_queue #'
            if candidate.count(anchor)!=1:raise ValueError('Validated queue binding differs')
            candidate=candidate.replace(anchor,'ntrpu_hbm_predict_validated_queue #',1)
        candidate=candidate.removeprefix('`timescale 1ns/1ps\n').removesuffix('\n')
        m=found[0];text=text[:m.start()]+candidate+text[m.end():]
    text+='\n'+(unit/'candidate_credit.sv').read_text()
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
    anchor='        pp_probe $out/control_predictors 0 1'
    if text.count(anchor)!=1:raise ValueError('Synthesis anchor differs')
    entry.write_text(text.replace(anchor,anchor+'\n        source $source/audit_hbm_boundary_physical.tcl\n        bp_probe $out/control_boundary 0 1',1),newline='\n')
    here=Path(__file__).resolve().parent
    for name in ['prepare_hbm_boundary_physical.py','audit_hbm_boundary_physical.tcl']:
        if (output/name).exists():raise ValueError('Helper collision')
        shutil.copy2(here/name,output/name)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    delta=sorted(n for n,h in original.items() if files[n]!=h)
    if delta!=['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv']:raise ValueError('Collateral source change')
    record=dict(status='BOUNDARY_FULL_BOARD_PREPARED',donor_files=original,prepared_files=files,
                changed_original_files=delta,tested_unit_binding_sha256=sha(unit/'PREPARED.json'),
                physical_timing_qualified=False)
    (output/'BOUNDARY_PHYSICAL_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],files=len(files),changed=delta)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['donor','unit','output']:p.add_argument(key,type=Path)
    a=p.parse_args();prepare(a.donor,a.unit,a.output)
