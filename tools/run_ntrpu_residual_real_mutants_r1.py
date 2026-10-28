"""Require explicit bank failures for faults in the composed actual HLS path."""
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in hashes.items():assert sha(a.source/n)==h,n
    compiled=json.loads((a.source/'compiled_sources.json').read_text());assert len(compiled)==17
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-residual-real-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json'
    variants=[
        ('capture_sequence_low32','path','sequence_reg<=command_sequence;',"sequence_reg<={32'd0,command_sequence[31:0]};"),
        ('capture_epoch_low32','path','epoch_reg<=command_epoch;',"epoch_reg<={32'd0,command_epoch[31:0]};"),
        ('source_identity_low32','path',',.command_source_sequence,',",.command_source_sequence({32'd0,command_source_sequence[31:0]}),"),
        ('input_address_low32','path','.command_residual_address(command_input_address)',".command_residual_address({32'd0,command_input_address[31:0]})"),
        ('destination_displaced','path','.command_residual_address(residual_address)',".command_residual_address(residual_address+64'd64)"),
        ('cancel_not_forwarded','path','child_cancel=busy && (cancel_seen || cancel_valid)',"child_cancel=1'b0"),
        ('early_publication','path','result_valid=completion_valid && completion_status==0 && store_result &&',"result_valid=(store_busy || completion_valid) && completion_status==0 &&"),
        ('private_address_leaked','path',"result_residual_address=result_valid ? store_residual_address:64'd0",'result_residual_address=residual_address'),
        ('wrong_output_extent','path',"result_residual_bytes=result_valid ? store_residual_bytes:14'd0", "result_residual_bytes=result_valid ? 14'd8191:14'd0"),
        ('duplicate_operand','hls_ports','.branch_q0(hls_input_q[1])','.branch_q0(hls_input_q[0])'),
        ('operand_bit_flip','hls_ports','.branch_q0(hls_input_q[1])',".branch_q0(hls_input_q[1]^32'd1)"),
        ('short_hls_vector','hls_ports','.elements(hls_elements)',".elements(hls_elements-32'd1)"),
    ]
    errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    for name,kind,before,after in variants:
        relative='fpga/rtl/ntrpu_residual_'+kind+'_r1.sv';original=(a.source/relative).read_text()
        assert original.count(before)==1,(name,original.count(before))
        text=original.replace(before,after);path=out/(name+'.sv');path.write_text(text)
        sources=[str(path if n==relative else a.source/n) for n in compiled];binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_residual_real_r1','-o',str(binary)]+sources)
        if compile_code:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1','+FIXTURE='+str(a.source/'fixture')])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'RESREAL_FAIL' in line),None)
        caught=code==1 and failure is not None and 'timeout' not in failure.lower() and not any(line.startswith('RESREAL_PASS') for line in lines)
        info['mutants'].append(dict(name=name,source_file=relative,parent_sha256=hashes[relative],rtl_sha256=sha(path),
            compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
