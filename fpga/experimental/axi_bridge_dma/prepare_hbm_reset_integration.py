"""Prepare a private board ECO with one explicit readiness-bound remapping."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(audits,checkpoint,output,mode,edited_reference=None,remote_edited_reference=None):
    if output.exists():raise ValueError('Fresh preparation required')
    if mode=='edit' and sha(checkpoint)!='3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1':
        raise ValueError('Exact routed donor required')
    here=Path(__file__).resolve().parent
    if bool(edited_reference)!=bool(remote_edited_reference) or (edited_reference and mode!='reopen'):
        raise ValueError('Both alias references are required for reopen only')
    if remote_edited_reference and any(c in remote_edited_reference for c in '{}\n\r'):
        raise ValueError('Unsupported reference path encoding')
    output.mkdir(parents=True);frozen={}
    for p in audits.iterdir():
        if p.suffix in ('.tcl','.tsv') and p.name!='inspect_hbm_native_join.tcl':
            shutil.copy2(p,output/p.name);frozen[p.name]=sha(p)
    p=output/'constrain_hbm_native_constraints.tcl'
    text=p.read_text()
    old='{reader/startup/memory_release_reg[2]} {reader/startup/ready_meta_reg[2]/D} mem_clk reader_clk 4.0'
    new='reader/startup/memory_distribution_reg {reader/startup/ready_meta_reg[2]/D} mem_clk reader_clk 4.0'
    if text.count(old)!=1:raise ValueError('Frozen readiness audit differs')
    p.write_text(text.replace(old,new,1),newline='\n')
    script='integrate_hbm_reset_distribution.tcl';shutil.copy2(here/script,output/script)
    timeout=1800 if mode=='route' else 900
    runner=(here/'run_hbm_native_join.py').read_text()
    if runner.count('timeout=1200')!=2:raise ValueError('Runner contract differs')
    (output/'run_hbm_native_join.py').write_text(runner.replace('timeout=1200',f'timeout={timeout}'),newline='\n')
    extra=' {'+remote_edited_reference+'}' if remote_edited_reference else ''
    entry=('set_param general.maxThreads 2\nif {[catch {\n'
           f'    source [file join [file dirname [info script]] {script}]\n'
           f'    ri_run {{*}}$argv {mode}{extra}\n'
           '} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n')
    (output/'inspect_hbm_native_join.tcl').write_text(entry,newline='\n')
    binding=dict(mode=mode,checkpoint_sha256=sha(checkpoint),frozen_audit_origins=frozen,
                 frozen_audits={name:sha(output/name) for name in frozen},
                 changed_audit='constrain_hbm_native_constraints.tcl',old_audit_row=old,new_audit_row=new,
                 script_sha256=sha(output/script),timeout_seconds=timeout,cpu_count=2,nice=10,
                 timing_qualified=False,board_qualified=False)
    if edited_reference:
        binding['alias_reference_sha256']=sha(edited_reference)
        binding['alias_reference_remote']=remote_edited_reference
    (output/'RESET_INTEGRATION_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
    return binding


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audits','checkpoint','output'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=['edit','route','reopen'],required=True)
    p.add_argument('--edited-reference',type=Path)
    p.add_argument('--remote-edited-reference')
    a=p.parse_args();print(json.dumps(prepare(a.audits,a.checkpoint,a.output,a.mode,a.edited_reference,a.remote_edited_reference)))
