"""Freeze a full-board capture implementation stage and its unchanged audits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(a):
    if a.output.exists():raise ValueError('Fresh prepared output required')
    if bool(a.incremental)!=bool(a.remote_incremental) or bool(a.incremental)!=(a.mode=='route'):
        raise ValueError('Both incremental donor paths required for route only')
    if a.remote_incremental and any(x in a.remote_incremental for x in '{}\n\r'):
        raise ValueError('Unsupported remote path')
    a.output.mkdir(parents=True)
    files={}
    for p in a.audits.iterdir():
        if p.suffix in ('.tcl','.tsv') and p.name!='inspect_hbm_native_join.tcl':
            shutil.copy2(p,a.output/p.name);files[p.name]=sha(p)
    here=Path(__file__).resolve().parent
    for name in ('audit_hbm_capture_physical.tcl','implement_hbm_capture_physical.tcl',
                 'inspect_hbm_routed_timing.tcl','prepare_hbm_capture_stage.py'):
        shutil.copy2(here/name,a.output/name)
    # Reuse the archived runner that drains all inherited stdout before sealing.
    shutil.copy2(a.runner,a.output/'run_hbm_native_join.py')
    extra=' {'+a.remote_incremental+'}' if a.remote_incremental else ''
    entry=('set_param general.maxThreads 2\nif {[catch {\n'
           'source [file join [file dirname [info script]] implement_hbm_capture_physical.tcl]\n'
           f'cp_run {{*}}$argv {a.mode}{extra}\n'
           '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (a.output/'inspect_hbm_native_join.tcl').write_text(entry,newline='\n')
    record=dict(mode=a.mode,checkpoint_sha256=sha(a.checkpoint),frozen_audits=files,
                incremental_sha256=sha(a.incremental) if a.incremental else None,
                incremental_remote=a.remote_incremental,timeout_seconds=1800,
                full_rtl_resynthesis=True,physical_timing_qualified=False)
    record['prepared_files']={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}
    (a.output/'CAPTURE_STAGE_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('frozen_audits','prepared_files')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audits','runner','checkpoint','output'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=['constrain','route','reopen'],required=True)
    p.add_argument('--incremental',type=Path);p.add_argument('--remote-incremental')
    prepare(p.parse_args())
