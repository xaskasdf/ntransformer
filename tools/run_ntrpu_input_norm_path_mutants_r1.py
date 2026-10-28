"""Require compiled ownership mutations to fail explicit protocol assertions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def mutations(original):
    changes={'identity_low32': [('command_sequence!=command_source_sequence', 'command_sequence[31:0]!=command_source_sequence[31:0]'), ('command_epoch!=command_source_epoch', 'command_epoch[31:0]!=command_source_epoch[31:0]')], 'capture_low32': [('completion_sequence<=command_sequence;', "completion_sequence<={32'd0,command_sequence[31:0]};")], 'bounds_preflight_removed': [('state==CHECK_REGION && bounds_bad', "state==CHECK_REGION && 1'b0")], 'alias_preflight_removed': [('state==CHECK_ALIAS && alias_bad', "state==CHECK_ALIAS && 1'b0")], 'live_output_descriptor': [('.command_address(addresses[2])', '.command_address(command_output_address)')], 'output_address_corrupted': [('addresses[2]<=command_output_address;', "addresses[2]<=command_output_address+64'd64;")], 'input_planes_swapped': [('.command_residual_address(addresses[0])', '.command_residual_address(addresses[1])'), ('.command_residual_base(bases[0])', '.command_residual_base(bases[1])')], 'early_publication': [('completion_valid=state==RESPONSE;', 'completion_valid=state==RESPONSE || state==WAIT_STORE;')], 'early_store_ack': [('((state==RESPONSE && completion_ready) || state==DRAIN)', '((state==RESPONSE && completion_ready) || state==DRAIN || state==WAIT_STORE)')], 'early_reuse': [('command_ready=state==IDLE &&', 'command_ready=(state==IDLE || state==RESPONSE) &&')], 'ignore_terminal_status': [('completion_status<=fault_now!=OK ? fault_now:store_status;', 'completion_status<=OK;')], 'wrong_result_extent': [("result_valid ? store_bytes:13'd0", "result_valid ? 13'd2048:13'd0")], 'public_ack_not_consumed': [('RESPONSE:if(completion_ready)state<=DRAIN;', "RESPONSE:if(1'b0)state<=DRAIN;")], 'leak_invalid_address': [("result_valid ? store_address:64'd0", 'store_address')]}
    result={}
    for name,pairs in changes.items():
        text=original
        for old,new in pairs:
            assert old in text,(name,old);text=text.replace(old,new)
        assert text!=original;result[name]=text
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    for n,h in json.loads((a.source/'SNAPSHOT_FILES.json').read_text()).items():assert sha(a.source/n)==h,n
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();start=time.monotonic()
    info=dict(stage='V7-input-norm-path-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_input_norm_path_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_input_norm_path_r1.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_input_norm_path_r1','-o',str(binary),*[str(a.source/n) for n in ['fpga/rtl/ntrpu_axi_read_master.sv','fpga/rtl/ntrpu_residual_loader_r1.sv','fpga/rtl/ntrpu_input_norm_owner_r1.sv','fpga/rtl/ntrpu_activation_writer.sv','fpga/rtl/ntrpu_attention_activation_store_r1.sv']],str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1','+FIXTURE='+str(a.source/'fixture')])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'INPATH_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('INPATH_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
