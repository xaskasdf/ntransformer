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
    info=dict(stage='V7-post-attention-loader-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_post_attention_loader_r1.sv').read_text()
    variants={
        'identity_low32':original.replace('command_sequence!=command_source_sequence','command_sequence[31:0]!=command_source_sequence[31:0]').replace('command_epoch!=command_source_epoch','command_epoch[31:0]!=command_source_epoch[31:0]').replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'unchecked_bounds':original.replace('state==CHECK && bounds_bad',"state==CHECK && 1'b0"),
        'wrong_bank':original.replace('1:branch_mem[','1:residual_mem['),
        'wrong_lane':original.replace('pending_data[pending_lane*32+:32]',"pending_data[(pending_lane^4'd1)*32+:32]"),
        'early_publication':original.replace('result_valid=completion_valid &&', 'result_valid=(completion_valid || state==READ) &&'),
        'ignore_cancel':original.replace('else if(cancel_valid && cancel_ready)fault_now=CANCELLED;',"else if(1'b0)fault_now=CANCELLED;"),
        'ignore_timeout':original.replace('else if(timeout_now)fault_now=TIMEOUT;',"else if(1'b0)fault_now=TIMEOUT;"),
        'forgotten_revocation':original.replace('if(busy && (epoch_bad || cancel_valid))revoked<=1;',"if(1'b0)revoked<=1;"),
        'early_reuse':original.replace('command_ready=state==IDLE &&', 'command_ready=(state==IDLE || state==RESPONSE) &&'),
        'nonfinite_accepted':original.replace("if(reader_data[lane*32+23+:8]==8'hff)beat_bad=1;","if(1'b0)beat_bad=1;"),
        'reader_status_ignored':original.replace('reader_status!=0 || reader_bytes!=chunk_bytes || chunk_received!=chunk_beats',"1'b0"),
    }
    reader=a.source/'fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv'
    tb=a.source/'fpga/tb/tb_ntrpu_post_attention_loader_r1.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_post_attention_loader_r1','-o',str(binary),str(path),str(reader),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'FAIL PLOAD' in line),None)
        caught=code==1 and failure is not None and not any(line.startswith('PLOAD_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
