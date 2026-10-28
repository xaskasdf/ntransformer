"""Compile integration mutations; require explicit failures from shared AXI protocol peers."""
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
    info=dict(stage='V7-ffn-residual-control-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_llama32_ffn_residual_r3.sv').read_text()
    variants={
        'unchecked_allocations':original.replace('else if(allocation_bad)',"else if(1'b0)"),
        'retained_allocation_skipped':original.replace('region_index==8)state<=FFN_OFFER','region_index==7)state<=FFN_OFFER'),
        'output_bounds_ignored':original.replace('else if(output_bad)',"else if(1'b0)"),
        'source_identity_low32':original.replace('command_sequence!=command_source_sequence','command_sequence[31:0]!=command_source_sequence[31:0]').replace('command_epoch!=command_source_epoch','command_epoch[31:0]!=command_source_epoch[31:0]'),
        'admission_epoch_low32':original.replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'epoch_cause_lost':original.replace('public_status<=fault_now!=0 ? fault_now:handoff_completion_status','public_status<=handoff_completion_status'),
        'held_status_mutable':original.replace('completion_status=state==LOCAL_RESPONSE ? fault:public_status','completion_status=state==LOCAL_RESPONSE ? fault:fault!=0 ? fault:public_status'),
        'uncaptured_identity':original.replace('ffn_command_sequence=sequence_reg','ffn_command_sequence=command_sequence'),
        'address_low32':original.replace('ffn_command_addr[i]=address[i]',"ffn_command_addr[i]={32'd0,address[i][31:0]}"),
        'uncaptured_output':original.replace('handoff_command_output_address=output_address','handoff_command_output_address=command_output_address'),
        'read_address_wrong_owner':original.replace('m_axi_araddr=residual_selected ? res_m_axi_araddr:ffn_m_axi_araddr','m_axi_araddr=ffn_m_axi_araddr'),
        'write_address_wrong_owner':original.replace('m_axi_awaddr=residual_selected ? res_m_axi_awaddr:ffn_m_axi_awaddr','m_axi_awaddr=ffn_m_axi_awaddr'),
        'write_data_wrong_owner':original.replace('m_axi_wdata=residual_selected ? res_m_axi_wdata:ffn_m_axi_wdata','m_axi_wdata=ffn_m_axi_wdata'),
        'read_response_broadcast':original.replace('ffn_m_axi_rvalid=!residual_selected && m_axi_rvalid','ffn_m_axi_rvalid=m_axi_rvalid'),
        'write_response_broadcast':original.replace('ffn_m_axi_bvalid=!residual_selected && m_axi_bvalid','ffn_m_axi_bvalid=m_axi_bvalid'),
        'early_ffn_release':original.replace('ffn_completion_ready=handoff_source_completion_ready','ffn_completion_ready=ffn_completion_valid'),
        'early_residual_release':original.replace('res_completion_ready=handoff_child_completion_ready','res_completion_ready=res_completion_valid'),
        'early_handoff_release':original.replace('handoff_completion_ready=state==FINAL_RESPONSE && completion_ready','handoff_completion_ready=handoff_completion_valid'),
        'early_public_admission':original.replace('command_ready=state==IDLE && ffn_command_ready && !ffn_completion_valid && !ffn_busy && !handoff_busy && res_command_ready && !res_completion_valid','command_ready=state==IDLE || state==FINAL_RESPONSE'),
        'private_output_leak':original.replace("result_address=result_valid ? handoff_result_address:64'd0",'result_address=output_address'),
    }
    tb=a.source/'fpga/tb/tb_ntrpu_llama32_ffn_residual_control_r9.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_llama32_ffn_residual_control_r1','-o',str(binary),str(path),str(a.source/'fpga/rtl/ntrpu_ffn_residual_handoff_r1.sv'),str(a.source/'fpga/tb/ntrpu_ffn_residual_control_peers_r1.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'FFRCTRL_FAIL' in line),None)
        caught=code==1 and failure is not None and 'timeout' not in failure.lower() and not any(line.startswith('FFRCTRL_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
