"""Require compiled ownership defects to fail explicit protocol assertions."""
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
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in hashes.items():assert sha(a.source/n)==h,n
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-llama32-ffn-owner-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json'
    original=(a.source/'fpga/rtl/ntrpu_llama32_ffn_owner_r3.sv').read_text()
    variants={}
    def variant(name,old,new,config=0):
        assert original.count(old)==1,(name,original.count(old))
        changed=original.replace(old,new);assert changed!=original;variants[name]=(changed,config)
    variant('source_sequence_truncated','command_source_sequence!=command_sequence','command_source_sequence[31:0]!=command_sequence[31:0]')
    variant('source_epoch_truncated','command_source_epoch!=command_epoch','command_source_epoch[31:0]!=command_epoch[31:0]')
    variant('expected_epoch_truncated','expected_epoch!=command_epoch','expected_epoch[31:0]!=command_epoch[31:0]')
    variant('live_epoch_truncated','expected_epoch!=completion_epoch','expected_epoch[31:0]!=completion_epoch[31:0]')
    variant('public_sequence_truncated','completion_sequence<=command_sequence','completion_sequence<=command_sequence[31:0]')
    variant('public_epoch_truncated','completion_epoch<=command_epoch','completion_epoch<=command_epoch[31:0]')
    variant('untrusted_source_accepted','!command_source_valid ||','')
    variant('residual_span_shortened',"span_valid(residual_address,residual_base,residual_bytes,64'd8192)","span_valid(residual_address,residual_base,residual_bytes,64'd4096)")
    variant('alignment_ignored','addr[5:0]==0 &&','')
    variant('region_overflow_ignored',' && !limit[64]','')
    variant('scratch_bounds_ignored','!span_valid(address[write_index],region_base[write_index],region_bytes[write_index],write_span)',"1'b0")
    variant('residual_overlap_ignored',"({1'b0,address[write_index]}<retained_end && {1'b0,residual_base}<write_end)","1'b0")
    variant('residual_guard_tail_ignored',"retained_end={1'b0,residual_base}+{1'b0,residual_bytes}","retained_end={1'b0,residual_address}+65'd8192")
    variant('private_tag_stride_one',"next_tag+64'd4","next_tag+64'd1")
    variant('private_epoch_not_checked','{child_done_epoch,child_done_sequence}==tag','child_done_sequence==tag[31:0]')
    variant('malformed_success_accepted','else if(child_shape_bad)','else if(1\'b0)')
    variant('early_child_ack','child_done_ready=owned && release_public','child_done_ready=owned && child_done_valid && child_identity')
    variant('premature_command_reuse','command_ready=state==IDLE &&','command_ready=(state==IDLE || state==RESPONSE) &&')
    variant('wrong_ffn_address',"result_valid ? address[7]:64'd0","result_valid ? (address[7]^64'd64):64'd0")
    variant('wrong_residual_address',"result_valid ? residual_address:64'd0","result_valid ? (residual_address^64'd64):64'd0")
    variant('late_cancel_unmasked',' && !cancel_valid && !epoch_bad',' && !epoch_bad')
    variant('late_epoch_unmasked',' && !cancel_valid && !epoch_bad',' && !cancel_valid')
    variant('tag_exhaustion_wrap',"if(next_tag==64'hfffffffffffffffc)exhausted<=1;","if(next_tag==64'hfffffffffffffffc)exhausted<=0;",1)
    variant('misaligned_tag_allowed','exhausted<=INITIAL_TAG[1:0]!=0','exhausted<=0',2)
    variants['revocation_not_sticky']=(original.replace('revoked<=1;','revoked<=0;'),0)
    variants['cancel_fault_ignored']=(original.replace('fault_now=CANCELLED','fault_now=OK').replace('fault<=CANCELLED','fault<=OK'),0)
    tb=a.source/'fpga/tb/tb_ntrpu_llama32_ffn_owner_r7.sv';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    for name,(text,config) in variants.items():
        assert text!=original
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_llama32_ffn_owner_r7','-P','tb_ntrpu_llama32_ffn_owner_r7.CONFIG='+str(config),'-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'FOWNER_FAIL' in line),None)
        caught=code==1 and failure is not None and not any(line.startswith('FOWNER_PASS') for line in lines)
        info['mutants'].append(dict(name=name,config=config,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
