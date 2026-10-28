"""Semantic corruption tests for token sequencing, retention and context recovery."""
import argparse,json,re,subprocess,time
from pathlib import Path

def mutations(text):
    cases=[]
    def change(name,old,new):
        assert text.count(old)==1,(name,text.count(old))
        cases.append((name,text.replace(old,new)))
    change('ignore_plan_certificate','!command_source_valid || !command_plan_valid','!command_source_valid')
    change('truncate_source_identity','command_source_sequence!=command_sequence','command_source_sequence[31:0]!=command_sequence[31:0]')
    change('ignore_position','command_position!=committed_tokens','1\'b0')
    change('ignore_lease_wrap','command_lease_base==0 || lease_end[64]','command_lease_base==0')
    change('ignore_alignment',"address[5:0]==0 && !limit[64]","1'b1 && !limit[64]")
    change('allow_input_output_alias','overlaps(scan==0 ? input_address:residual_address[scan-1\'b1],8192,residual_address[scan],8192)',"1'b0")
    change('accept_dirty_bind','cache_query_committed_tokens!=0',"1'b0")
    change('ignore_prefix_scan','cache_query_committed_tokens!=completion_position+(state==VERIFY_FINAL ? 1:0)',"1'b0")
    change('skip_last_layer','state<=step==15 ? OFFER_HEAD:ACK_LAYER;','state<=step==14 ? OFFER_HEAD:ACK_LAYER;')
    change('wrong_residual_input','step==0 ? input_address:residual_address[step-1\'b1]','input_address')
    change('wrong_layer_lease',"lease_base+({60'd0,step}<<5)","lease_base+({60'd0,step}<<4)")
    change('ignore_layer_bytes','layer_result_bytes==8192',"1'b1")
    change('ignore_kv_prefix_on_completion','!layer_success || !cache_query_valid || cache_query_committed_tokens!=completion_position+1','!layer_success || !cache_query_valid')
    change('ignore_head_bytes','head_result_bytes==513024',"1'b1")
    change('ignore_head_valid','head_completion_status==OK && head_result_valid && !head_revoked','head_completion_status==OK && !head_revoked')
    change('poison_revives_after_fault','else if(command_valid && command_ready)begin','else if(command_valid && (command_ready || state==UNBOUND))begin')
    change('commit_failed_token','else committed_tokens<=completion_position+1\'b1;',"committed_tokens<=completion_position+1'b1;")
    change('late_revocation_not_sticky','if(fault_now!=OK)begin fault<=fault_now;revoked<=1;',"if(fault_now!=OK)begin fault<=fault_now;revoked<=0;")
    change('early_final_layer_ack','(state==DRAIN_FINAL && !owned_head && !head_busy && !head_completion_valid)','(state==OFFER_HEAD || state==DRAIN_FINAL)')
    change('ignore_bind_identity','bind_completion_sequence!=cache_sequence || bind_completion_epoch!=cache_epoch','bind_completion_epoch!=cache_epoch')
    return cases

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    rtl=a.source/'fpga/rtl/ntrpu_llama32_token_guard_r1.sv';tb=a.source/'fpga/tb/tb_ntrpu_llama32_token_guard_r2.sv'
    rows=[]
    for label,source in mutations(rtl.read_text()):
        folder=a.output/label;folder.mkdir();mutant=folder/'mutant.sv';mutant.write_text(source)
        started=time.monotonic()
        with (folder/'compile.log').open('wb') as f:
            compile_result=subprocess.run(['iverilog','-g2012','-s','tb_ntrpu_llama32_token_guard_r2','-o',str(folder/'bench.vvp'),str(mutant),str(tb)],stdout=f,stderr=subprocess.STDOUT,timeout=10)
        rc=None;timed_out=False
        if compile_result.returncode==0:
            with (folder/'run.log').open('wb') as f:
                try:result=subprocess.run(['vvp',str(folder/'bench.vvp'),'+SEED=1'],stdout=f,stderr=subprocess.STDOUT,timeout=5);rc=result.returncode
                except subprocess.TimeoutExpired:timed_out=True
        text=(folder/'run.log').read_text() if (folder/'run.log').exists() else ''
        fatal=next((line for line in text.splitlines() if 'TOK_FAIL' in line),'')
        caught=compile_result.returncode==0 and rc not in (None,0) and bool(fatal) and 'deadline' not in fatal and 'timeout' not in fatal and not timed_out
        row=dict(name=label,compile_exit=compile_result.returncode,exit_code=rc,timed_out=timed_out,caught=caught,
            assertion=fatal,wall_seconds=time.monotonic()-started)
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='PASS_TOKEN_GUARD_MUTANTS_R1' if all(r['caught'] for r in rows) else 'DIAGNOSTIC_TOKEN_GUARD_MUTANTS_R1',mutants=rows,arithmetic_executed=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if all(r['caught'] for r in rows) else 1

if __name__=='__main__':raise SystemExit(main())
