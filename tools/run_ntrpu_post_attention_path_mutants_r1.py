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
    info=dict(stage='V7-post-attention-path-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_post_attention_path_r1.sv').read_text()
    variants={
        'uncaptured_sequence':original.replace('.command_sequence(sequence_reg)', '.command_sequence(command_sequence)'),
        'uncaptured_destination':original.replace('.command_residual_address(residual_address)', '.command_residual_address(command_residual_address)'),
        'wrong_ffn_destination':original.replace('ffn_address<=command_ffn_address;', "ffn_address<=command_ffn_address+64'd64;"),
        'truncated_sequence':original.replace('sequence_reg<=command_sequence;', "sequence_reg<={32'd0,command_sequence[31:0]};"),
        'drop_admission_cancel':original.replace('busy && (cancel_seen || cancel_valid)', 'busy && cancel_valid'),
        'early_reuse':original.replace('state==IDLE && loader_ready && !owner_busy && !store_busy', '(state==IDLE || state==RESPONSE) && loader_ready'),
        'early_store_ack':original.replace('.completion_ready(state==RESPONSE && completion_ready)', '.completion_ready(store_valid)'),
        'early_publication':original.replace('result_valid=completion_valid &&', 'result_valid=busy &&'),
        'missing_read_count':original.replace('completion_read_beats<=loader_beats;', 'completion_read_beats<=0;'),
        'missing_hls_diagnostic':original.replace('completion_hls_status<=owner_hls_status;', 'completion_hls_status<=0;'),
        'forced_success':original.replace('completion_status<=fault_now!=0 ? fault_now:store_status;', 'completion_status<=0;'),
        'truncated_completion_epoch':original.replace('completion_epoch<=store_epoch;', "completion_epoch<={32'd0,store_epoch[31:0]};"),
    }
    tb=a.source/'fpga/tb/tb_ntrpu_post_attention_path_r1.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_post_attention_path_r1','-o',str(binary),str(path),str(tb)]+[str(a.source/n) for n in ['fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv', 'fpga/rtl/ntrpu_post_attention_loader_r1.sv', 'fpga/rtl/ntrpu_post_attention_owner_r1.sv', 'fpga/rtl/ntrpu_post_attention_store_r1.sv']])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1','+FIXTURE='+str(a.source/'fixture')])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'PPATH_FAIL' in line),None)
        caught=code==1 and failure is not None and not any(line.startswith('PPATH_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
