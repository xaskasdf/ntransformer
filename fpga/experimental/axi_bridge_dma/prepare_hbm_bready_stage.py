"""Freeze one BREADY board stage without changing inherited timing contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(a):
    if a.output.exists():raise ValueError('Fresh output required')
    if bool(a.incremental)!=bool(a.remote_incremental) or bool(a.incremental)!=(a.mode=='route'):
        raise ValueError('Both incremental paths are required for route only')
    if a.remote_incremental and any(c in a.remote_incremental for c in '{}\n\r'):raise ValueError('Unsupported remote path')
    a.output.mkdir(parents=True);frozen={}
    for p in a.audits.iterdir():
        if p.suffix in ('.tcl','.tsv') and p.name!='inspect_hbm_native_join.tcl':
            shutil.copy2(p,a.output/p.name);frozen[p.name]=sha(p)
    here=Path(__file__).resolve().parent
    for name in ['audit_hbm_bready_physical.tcl','implement_hbm_bready_physical.tcl','prepare_hbm_bready_stage.py']:
        if (a.output/name).exists():raise ValueError('Helper collision')
        shutil.copy2(here/name,a.output/name)
    shutil.copy2(a.runner,a.output/'run_hbm_native_join.py')
    extra=' {'+a.remote_incremental+'}' if a.remote_incremental else ''
    entry=('set_param general.maxThreads 2\nif {[catch {\nset ::pp_candidate 1\nset ::bp_candidate 1\n'
           'source [file join [file dirname [info script]] implement_hbm_bready_physical.tcl]\n'
           f'cp_run {{*}}$argv {a.mode}{extra}\n'
           '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (a.output/'inspect_hbm_native_join.tcl').write_text(entry,newline='\n')
    record=dict(mode=a.mode,checkpoint_sha256=sha(a.checkpoint),frozen_audits=frozen,
                incremental_sha256=sha(a.incremental) if a.incremental else None,incremental_remote=a.remote_incremental,
                timeout_seconds=1800,registered_ready=True,control_predictors=True,control_boundary=True,registered_bready=True,
                full_rtl_resynthesis=True,baseline_read_only=False,physical_timing_qualified=False,vendor_feedback_probe=True)
    record['prepared_files']={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'CAPTURE_STAGE_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('frozen_audits','prepared_files')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['audits','runner','checkpoint','output']:p.add_argument(key,type=Path)
    p.add_argument('--mode',choices=['constrain','route','reopen'],required=True)
    p.add_argument('--incremental',type=Path);p.add_argument('--remote-incremental')
    prepare(p.parse_args())
