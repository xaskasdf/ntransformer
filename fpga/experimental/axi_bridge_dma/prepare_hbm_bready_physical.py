"""Compose only the tested registered BREADY delta into a full-board donor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def board_loader(donor,unit):
    proof=json.loads((unit/'PREPARED.json').read_text())
    if proof['status']!='RESET_BREADY_UNIT_PREPARED' or sha(donor)!=proof['donor_files']['ntrpu_hbm_weight_loader.sv']:
        raise ValueError('Untested loader donor')
    for name,digest in proof['files'].items():
        if sha(unit/name)!=digest:raise ValueError('Unit source changed: '+name)
    candidate=(unit/'candidate_loader.sv').read_text()
    anchor='module ntrpu_hbm_bready_loader ('
    if candidate.count(anchor)!=1:raise ValueError('Candidate module differs')
    return candidate.replace(anchor,'module ntrpu_hbm_weight_loader (',1)+'\n'+(unit/'ntrpu_hbm_bready_predictor.sv').read_text()


def prepare(donor,unit,output):
    if output.exists():raise ValueError('Fresh output required')
    changed=board_loader(donor/'ntrpu_hbm_weight_loader.sv',unit)
    original={p.name:sha(p) for p in donor.iterdir() if p.is_file()}
    output.mkdir(parents=True)
    for name in original:shutil.copy2(donor/name,output/name)
    (output/'ntrpu_hbm_weight_loader.sv').write_text(changed,newline='\n')
    entry=output/'elaborate_hbm_native_join.tcl';text=entry.read_text()
    anchor='        bp_probe $out/control_boundary 0 1'
    if text.count(anchor)!=1:raise ValueError('Synthesis audit anchor differs')
    entry.write_text(text.replace(anchor,anchor+'\n        source $source/audit_hbm_bready_physical.tcl\n        wb_probe $out/registered_bready 0',1),newline='\n')
    here=Path(__file__).resolve().parent
    for name in ['prepare_hbm_bready_physical.py','audit_hbm_bready_physical.tcl']:
        if (output/name).exists():raise ValueError('Helper collision')
        shutil.copy2(here/name,output/name)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    delta=sorted(n for n,h in original.items() if files[n]!=h)
    if delta!=['elaborate_hbm_native_join.tcl','ntrpu_hbm_weight_loader.sv']:raise ValueError('Collateral source change')
    record=dict(status='BREADY_FULL_BOARD_PREPARED',donor_files=original,prepared_files=files,
                changed_original_files=delta,tested_unit_binding_sha256=sha(unit/'PREPARED.json'),physical_timing_qualified=False)
    (output/'BREADY_PHYSICAL_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status=record['status'],files=len(files),changed=delta)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['donor','unit','output']:p.add_argument(key,type=Path)
    a=p.parse_args();prepare(a.donor,a.unit,a.output)
