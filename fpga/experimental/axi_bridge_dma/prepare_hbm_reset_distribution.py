"""Freeze an isolated reset candidate against an explicit, sealed RTL donor."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', text))


def prepare(donor, out):
    source=Path(__file__).resolve().parent
    if out.exists():
        raise ValueError('Fresh prepared directory required')
    original=(donor/'ntrpu_hbm_startup.sv').read_text()
    candidate=(source/'ntrpu_hbm_startup_distributed.sv').read_text()
    block='''    always @(posedge mem_clk or negedge memory_release[2])begin
        if(!memory_release[2])memory_distribution<=0;
        else memory_distribution<=1'b1;
    end'''
    rollback=candidate
    for before,after in [
        ('module ntrpu_hbm_startup_distributed (','module ntrpu_hbm_startup ('),
        ('reg memory_distribution;',''),
        ('assign mem_rst_n=memory_distribution;','assign mem_rst_n=memory_release[2];'),
        (block,'')]:
        if rollback.count(before)!=1:
            raise ValueError('Candidate transformation anchor differs')
        rollback=rollback.replace(before,after,1)
    if tokens(rollback)!=tokens(original):
        raise ValueError('Candidate changes more than the explicit distribution stage')
    out.mkdir(parents=True)
    provenance={}
    def copy(path, name=None, text=None):
        target=out/(name or path.name)
        if text is None:
            shutil.copy2(path,target)
        else:
            with target.open('x',newline='\n') as f:
                f.write(text)
        provenance[target.name]=dict(origin=str(path.resolve()),origin_sha256=sha(path),
                                     sha256=sha(target),transformed=text is not None)
    for name in ('ntrpu_hbm_startup_distributed.sv','tb_hbm_reset_distribution.sv',
                 'synthesize_hbm_reset_distribution.tcl','prepare_hbm_reset_distribution.py'):
        copy(source/name)
    copy(source/'run_hbm_reset_distribution.py','run_hbm_cold_release.py')
    copy(donor/'ntrpu_hbm_startup.sv','ntrpu_hbm_startup_reference.sv',
         original.replace('module ntrpu_hbm_startup (','module ntrpu_hbm_startup_reference (',1))
    copy(donor/'ntrpu_hbm_read_adapter.sv')
    path=source/'tb_hbm_cold_release.sv'
    regression=path.read_text()
    if regression.count('ntrpu_hbm_startup dut(')!=1:
        raise ValueError('Original regression binding differs')
    copy(path,text=regression.replace('ntrpu_hbm_startup dut(','ntrpu_hbm_startup_distributed dut(',1))
    manifest=dict(status='ISOLATED_RESET_DISTRIBUTION_PREPARED',files=provenance,
        functional_delta='One ordinary FDCE after three memory ASYNC_REG stages; async assertion; fourth-edge release',
        donor_startup_sha256=sha(donor/'ntrpu_hbm_startup.sv'),
        full_board_integrated=False,physical_timing_qualified=False,
        integration_requirement='Move existing 4 ns ready_meta[2]/D bound to distribution register; time recovery/removal; repeat full board contracts')
    (out/'PREPARED.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],files=len(provenance),output=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    prepare(args.donor.resolve(),args.output.resolve())
