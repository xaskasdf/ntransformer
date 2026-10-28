"""Prepare isolated implementation stages from the sealed board CDC audits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(donor,evidence,out,mode):
    if out.exists():raise ValueError('Fresh prepared directory required')
    expected='724bb198d259ec00aaa5c237eccf814df24fcbfd4eb5cadcf96a222bb3b2f10c'
    if sha(evidence/'output/board_cdc_bounded.dcp')!=expected:raise ValueError('Wrong CDC evidence')
    out.mkdir(parents=True)
    frozen={}
    for p in donor.glob('*.tcl'):
        if p.name=='inspect_hbm_native_join.tcl':continue
        shutil.copy2(p,out/p.name);frozen[p.name]=sha(p)
    loader=out/'audit_hbm_loader_readiness.tcl'
    old='''    if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1"} {error "Readiness reset polarity changed"}
    set reset [nc_one [bc_driver [nb_pin $inv/I0]] readiness_reset]
    if {$reset ne [nc_one [bc_driver [nb_pin core/management_cold_rst_n]] canonical_management_reset]} {error "Readiness reset bypasses canonical cold reset"}'''
    text=loader.read_text()
    if text.count(old)!=1:raise ValueError('Readiness reset audit anchor differs')
    loader.write_text(text.replace(old,'    set reset [bi_readiness_clear $out $source $clr]',1))
    here=Path(__file__).parent
    shutil.copy2(here/'implement_hbm_board.tcl',out/'implement_hbm_board.tcl')
    shutil.copy2(evidence/'output/board_synchronous_paths.tsv',out/'expected_synchronous.tsv')
    limit={'optimize':900,'audit':900,'place':2400,'route':2400,'reopen':900}[mode]
    runner=(here/'run_hbm_native_join.py').read_text()
    if runner.count('timeout=1200')!=2:raise ValueError('Unexpected runner')
    (out/'run_hbm_native_join.py').write_text(runner.replace('timeout=1200','timeout='+str(limit)))
    entry='''set_param general.maxThreads 2
if {[catch {
    source [file join [file dirname [info script]] implement_hbm_board.tcl]
    bi_run {*}$argv MODE
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
'''
    (out/'inspect_hbm_native_join.tcl').write_text(entry.replace('MODE',mode))
    record=dict(mode=mode,donor_checkpoint_sha256=expected,frozen_audits=frozen,
                adapted_readiness_audit_sha256=sha(loader),
                synchronous_contract_sha256=sha(out/'expected_synchronous.tsv'),
                implementation_sha256=sha(out/'implement_hbm_board.tcl'),timeout_seconds=limit,
                cpu_count=2,nice=10,pinout_released=False,board_qualified=False)
    (out/'IMPLEMENTATION_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('donor','evidence','output'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=['optimize','audit','place','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(prepare(a.donor,a.evidence,a.output,a.mode)))
