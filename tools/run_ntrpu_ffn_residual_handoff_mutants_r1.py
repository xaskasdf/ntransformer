"""Compile ownership mutations; require explicit failures from controlled completion peers."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import subprocess,time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for n,h in hashes.items():assert sha(a.source/n)==h,n
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-ffn-residual-handoff-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_ffn_residual_handoff_r1.sv').read_text()
    variants={
        'admission_identity_low32':original.replace('command_sequence!=source_sequence','command_sequence[31:0]!=source_sequence[31:0]').replace('command_epoch!=source_epoch','command_epoch[31:0]!=source_epoch[31:0]').replace('command_sequence==source_sequence','command_sequence[31:0]==source_sequence[31:0]').replace('command_epoch==source_epoch','command_epoch[31:0]==source_epoch[31:0]'),
        'live_source_identity_low32':original.replace('source_sequence==completion_sequence && source_epoch==completion_epoch','source_sequence[31:0]==completion_sequence[31:0] && source_epoch[31:0]==completion_epoch[31:0]'),
        'child_identity_low32':original.replace('child_sequence==completion_sequence && child_epoch==completion_epoch','child_sequence[31:0]==completion_sequence[31:0] && child_epoch[31:0]==completion_epoch[31:0]'),
        'epoch_low32':original.replace('expected_epoch!=completion_epoch','expected_epoch[31:0]!=completion_epoch[31:0]').replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'unchecked_bounds':original.replace('else if(bounds_bad)',"else if(1'b0)"),
        'residual_alias_ignored':original.replace("({1'b0,output_address}<residual_end && {1'b0,residual_address}<output_end)","1'b0"),
        'ffn_alias_ignored':original.replace("({1'b0,output_address}<ffn_end && {1'b0,ffn_address}<output_end)","1'b0"),
        'operand_alias_ignored':original.replace("({1'b0,residual_address}<ffn_end && {1'b0,ffn_address}<residual_end)","1'b0"),
        'forgotten_revocation':original.replace('revoked<=1;', 'revoked<=0;'),
        'early_source_ack':original.replace('source_completion_ready=source_owned && release_public','source_completion_ready=source_owned && child_completion_valid'),
        'early_child_ack':original.replace('child_completion_ready=child_owned && release_public','child_completion_ready=child_owned && child_completion_valid'),
        'foreign_source_release':original.replace('can_release_source=!source_owned || (source_completion_valid && source_identity)',"can_release_source=1'b1"),
        'foreign_child_release':original.replace('can_release_child=!child_owned || (child_completion_valid && child_identity)',"can_release_child=1'b1"),
        'withdraw_offer':original.replace('child_command_valid=state==OFFER','child_command_valid=state==OFFER && fault_now==OK'),
        'cancel_not_forwarded':original.replace('child_cancel_valid=child_owned && (cancel_valid || epoch_bad || source_bad || fault!=OK || revoked)',"child_cancel_valid=1'b0"),
        'output_address_low32':original.replace('child_output_address=output_address',"child_output_address={32'd0,output_address[31:0]}"),
        'hls_error_ignored':original.replace('child_completion_status==OK && child_hls_status==0 &&','child_completion_status==OK &&'),
        'short_read_accepted':original.replace('child_read_beats==256 &&',''),
        'short_write_accepted':original.replace('child_write_beats==128 &&',''),
        'error_result_accepted':original.replace(' || (child_completion_status!=OK && child_result_valid)',''),
        'early_completion':original.replace('completion_valid=state==RESPONSE','completion_valid=state==RESPONSE || state==WAIT_CHILD'),
        'early_admission':original.replace('command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&'),
        'leaked_descriptor':original.replace("result_address=result_valid ? output_address:64'd0",'result_address=output_address'),
        'uncaptured_sequence':original.replace('child_command_sequence=completion_sequence','child_command_sequence=command_sequence'),
    }
    tb=a.source/'fpga/tb/tb_ntrpu_ffn_residual_handoff_r1.sv';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    for name,text in variants.items():
        assert text!=original
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_ffn_residual_handoff_r1','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'FRHAND_FAIL' in line),None)
        caught=code==1 and failure is not None and 'timeout' not in failure.lower() and not any(line.startswith('FRHAND_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
