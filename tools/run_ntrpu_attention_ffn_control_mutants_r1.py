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
    info=dict(stage='V7-attention-ffn-control-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_llama32_attention_ffn_r4.sv').read_text()
    variants={
        'unchecked_spans':original.replace('else if(!span_valid(address[span_index],allocation_base[span_index],allocation_bytes[span_index],required_span(span_index)))',"else if(1'b0)"),
        'unchecked_pairs':original.replace('else if(pair_overlap)',"else if(1'b0)"),
        'skip_last_pairs':original.replace('pair_left==14','pair_left==13'),
        'skip_first_span':original.replace('span_index<=0','span_index<=1'),
        'adjacency_rejected':original.replace("{1'b0,allocation_base[pair_left]}<right_end","{1'b0,allocation_base[pair_left]}<=right_end").replace("{1'b0,allocation_base[pair_right]}<left_end","{1'b0,allocation_base[pair_right]}<=left_end"),
        'admission_sequence_low32':original.replace('command_sequence!=a_cache_sequence','command_sequence[31:0]!=a_cache_sequence[31:0]'),
        'admission_epoch_low32':original.replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]').replace('command_epoch!=a_cache_epoch','command_epoch[31:0]!=a_cache_epoch[31:0]').replace('expected_epoch!=held_command_epoch','expected_epoch[31:0]!=held_command_epoch[31:0]'),
        'source_identity_low32':original.replace('a_completion_sequence==held_command_sequence && a_completion_epoch==held_command_epoch','a_completion_sequence[31:0]==held_command_sequence[31:0] && a_completion_epoch[31:0]==held_command_epoch[31:0]'),
        'epoch_cause_lost':original.replace('public_status<=fault_now!=0 ? fault_now:h_completion_status','public_status<=h_completion_status'),
        'held_status_mutable':original.replace('completion_status=state==LOCAL_RESPONSE ? fault:public_status','completion_status=state==LOCAL_RESPONSE ? fault:fault!=0 ? fault:public_status'),
        'bind_priority_removed':original.replace(' && !bind_valid && !append_valid && a_command_ready',' && a_command_ready'),
        'cache_lock_lost':original.replace('cache_lock=busy || a_cache_lock','cache_lock=a_cache_lock'),
        'q_forwarded_unowned':original.replace('a_q_valid=attention_launched && !ffn_selected && q_valid','a_q_valid=q_valid'),
        'cache_mutation_ignored':original.replace('cache_changed ? 4\'d11:', ''),
        'source_shape_ignored':original.replace('state==WAIT_ATTENTION && a_completion_valid && attention_identity && source_shape_bad ? 4\'d5:', ''),
        'uncaptured_attention_identity':original.replace('a_command_sequence=state==IDLE ? command_sequence:held_command_sequence','a_command_sequence=command_sequence'),
        'uncaptured_ffn_arrays':original.replace('h_command_ffn_addr[i]=address[i+7]','h_command_ffn_addr[i]=command_ffn_addr[i]'),
        'early_attention_release':original.replace('a_completion_ready=h_source_completion_ready','a_completion_ready=a_completion_valid'),
        'early_ffn_release':original.replace('f_completion_ready=h_child_completion_ready','f_completion_ready=f_completion_valid'),
        'early_handoff_release':original.replace('h_completion_ready=state==FINAL_RESPONSE && completion_ready && can_release','h_completion_ready=h_completion_valid'),
        'foreign_public_ack':original.replace('state==FINAL_RESPONSE && h_completion_valid && can_release','state==FINAL_RESPONSE && h_completion_valid'),
        'early_public_valid':original.replace('completion_valid=state==LOCAL_RESPONSE || (state==FINAL_RESPONSE && h_completion_valid && can_release)','completion_valid=state==LOCAL_RESPONSE || h_completion_valid'),
        'read_address_wrong_owner':original.replace('m_axi_araddr=ffn_selected ? f_m_axi_araddr:a_m_axi_araddr','m_axi_araddr=a_m_axi_araddr'),
        'write_address_wrong_owner':original.replace('m_axi_awaddr=ffn_selected ? f_m_axi_awaddr:a_m_axi_awaddr','m_axi_awaddr=a_m_axi_awaddr'),
        'write_data_wrong_owner':original.replace('m_axi_wdata=ffn_selected ? f_m_axi_wdata:a_m_axi_wdata','m_axi_wdata=a_m_axi_wdata'),
        'read_response_broadcast':original.replace('a_m_axi_rvalid=!ffn_selected && m_axi_rvalid','a_m_axi_rvalid=m_axi_rvalid'),
        'write_response_broadcast':original.replace('a_m_axi_bvalid=!ffn_selected && m_axi_bvalid','a_m_axi_bvalid=m_axi_bvalid'),
        'private_output_leak':original.replace("result_address=result_valid ? h_result_address:64'd0",'result_address=address[15]'),
    }
    for signal in ('bind_valid','append_valid','data_valid','commit_valid','append_cancel_valid','append_done_ready'):
        old='a_'+signal+'=state==IDLE && '+signal
        assert old in original
        variants['ungated_'+signal]=original.replace(old,'a_'+signal+'='+signal)
    tb=a.source/'fpga/tb/tb_ntrpu_llama32_attention_ffn_control_r4.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_llama32_attention_ffn_control_r4','-o',str(binary),str(path),str(a.source/'fpga/rtl/ntrpu_post_ffn_handoff_r2.sv'),str(a.source/'fpga/tb/ntrpu_attention_ffn_control_peers_r1.sv'),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'AFC_FAIL' in line),None)
        caught=code==1 and failure is not None and not any(word in failure.lower() for word in ('timeout','missing','premature abort')) and not any(line.startswith('AFC_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
