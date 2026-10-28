"""Compile deliberate loader mutations, requiring each to fail the real AXI bank."""
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
    info=dict(stage='V7-post-attention-owner-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_post_attention_owner_r1.sv').read_text()
    variants={
        'identity_low32':original.replace('command_sequence!=source_sequence','command_sequence[31:0]!=source_sequence[31:0]').replace('command_epoch!=source_epoch','command_epoch[31:0]!=source_epoch[31:0]').replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'wrong_input_plane':original.replace('source_q[plane]:32', 'source_q[(plane+1)%3]:32'),
        'wrong_residual_word':original.replace('residual_mem[hls_residual_address]<=hls_residual_d;', "residual_mem[hls_residual_address]<=hls_residual_d^32'd1;"),
        'wrong_abi_word':original.replace('ffn_mem[hls_ffn_address]<=hls_ffn_d;', "ffn_mem[hls_ffn_address]<=hls_ffn_d^64'd1;"),
        'early_pair_publication':original.replace('result_valid=completion_valid &&', 'result_valid=(completion_valid || state==H_WAIT) &&'),
        'missing_residual_count':original.replace('residual_after!=2048 || ffn_after!=512','ffn_after!=512'),
        'missing_abi_count':original.replace('residual_after!=2048 || ffn_after!=512','residual_after!=2048'),
        'missing_ram_response_ignored':original.replace('state==H_WAIT && input_bad', "state==H_WAIT && 1'b0"),
        'ignore_cancel':original.replace('else if(cancel_valid)fault_now=CANCELLED;', "else if(1'b0)fault_now=CANCELLED;"),
        'ignore_timeout':original.replace('else if(timeout_now)fault_now=TIMEOUT;', "else if(1'b0)fault_now=TIMEOUT;"),
        'forgotten_revocation':original.replace('if(busy && (epoch_bad || cancel_valid))revoked<=1;', "if(1'b0)revoked<=1;"),
        'early_source_release':original.replace('source_completion_ready=state==RELEASE_SOURCE &&', 'source_completion_ready=(state==RELEASE_SOURCE || state==H_WAIT) &&'),
        'early_reuse':original.replace('command_ready=state==IDLE &&', 'command_ready=(state==IDLE || state==RESPONSE) &&'),
        'wrong_epsilon':original.replace("hls_epsilon_bits=32'h3727c5ac", "hls_epsilon_bits=32'h3727c5ad"),
        'hls_failure_ignored':original.replace('else if(hls_return!=0)fault_now=COMPUTE_ERROR;', "else if(1'b0)fault_now=COMPUTE_ERROR;"),
        'residual_nonfinite_accepted':original.replace(" || hls_residual_d[30:23]==8'hff", ''),
        'abi_format_ignored':original.replace(' || word_bad)', ')'),
    }
    tb=a.source/'fpga/tb/tb_ntrpu_post_attention_owner_r1.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_post_attention_owner_r1','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'POWNER_FAIL' in line),None)
        caught=code==1 and failure is not None and not any(line.startswith('POWNER_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
