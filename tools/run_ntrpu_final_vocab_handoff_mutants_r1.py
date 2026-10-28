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
    changes={'identity_low32': [('command_sequence!=command_source_sequence', 'command_sequence[31:0]!=command_source_sequence[31:0]')], 'capture_low32': [('completion_sequence<=command_sequence;', "completion_sequence<={32'd0,command_sequence[31:0]};")], 'source_not_valid': [('else if(!command_source_valid)', "else if(1'b0)")], 'ranges_disabled': [('!span_valid(address[range_index],region_base[range_index],region_size[range_index],span(range_index))', "1'b0")], 'address_alignment': [('addr[5:0]==0 && ', '')], 'base_alignment': [('base[5:0]==0 && ', '')], 'size_alignment': [('bytes[5:0]==0 && ', '')], 'aliases_disabled': [('else if(overlap)', "else if(1'b0)")], 'read_alias_rejected': [('write_index<=2;', 'write_index<=0;')], 'live_norm_address': [('norm_command_addr[n]=address[n]', 'norm_command_addr[n]=command_addr[n]')], 'vocab_shifted_descriptor': [('vocab_command_addr[q]=address[q+2]', 'vocab_command_addr[q]=address[q+1]')], 'early_norm_ack': [('wire release_request=public_ack || state==DRAIN;', 'wire release_request=public_ack || state==DRAIN || state==OFFER_VOCAB;')], 'foreign_norm_ack': [('norm_owned && norm_identity && release_request', 'norm_owned && release_request')], 'foreign_vocab_ack': [('vocab_owned && vocab_identity && release_request', 'vocab_owned && release_request')], 'bus_lost_drain': [('vocab_bus_owned=vocab_owned;', 'vocab_bus_owned=vocab_owned && state!=DRAIN;')], 'norm_byte_count': [('&& norm_result_bytes==4096', '')], 'norm_read_count': [('&& norm_read_beats==256', '')], 'norm_record_count': [('norm_records==64 &&', '')], 'vocab_matrix_count': [('&& vocab_matrices==16', '')], 'vocab_result_shape': [('vocab_result_bytes==513024', "1'b1")], 'diagnostic_norm': [('completion_norm_records<=norm_records;', 'completion_norm_records<=0;')], 'diagnostic_vocab': [('completion_matrix_status<=vocab_child_status;', 'completion_matrix_status<=0;')], 'public_ack_not_consumed': [('RESPONSE:if(completion_ready)state<=DRAIN;', "RESPONSE:if(1'b0)state<=DRAIN;")], 'leak_invalid_address': [("result_valid ? address[4]:64'd0", 'address[4]')], 'sticky_revocation_removed': [('revoked<=1;', 'revoked<=0;')], 'watchdog_removed': [('watchdog>=CHILD_TIMEOUT_CYCLES-1', "1'b0")], 'ignore_norm_failure': [('fault_now!=OK || norm_completion_status!=OK || !norm_success', "1'b0")], 'ignore_vocab_failure': [('completion_status<=fault_now!=OK ? fault_now:vocab_completion_status!=OK ? vocab_completion_status:vocab_success ? OK:BAD_COMPLETION;', 'completion_status<=OK;')]}
    changes.update(vocab_phase=[('vocab_phase==15', "1'b1")],
        vocab_total=[('vocab_total_bytes==513024', "1'b1")],
        vocab_revoked=[('&& !vocab_revoked', '')],
        vocab_result_valid=[('&& vocab_result_valid', '')],
        norm_hls_status=[('&& norm_hls_status==0', '')])
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
    info=dict(stage='V7-final-vocab-handoff-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_final_vocab_handoff_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_final_vocab_handoff_r1.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_final_vocab_handoff_r1','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'FVH_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('FVH_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
