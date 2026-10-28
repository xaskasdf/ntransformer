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
    changes={'skip_final_region': [('else if(region_index==23)state<=CHECK_PAIRS;', 'else if(region_index==22)state<=CHECK_PAIRS;')], 'skip_final_pair': [('if(pair_left==22)state<=RESPONSE;', 'if(pair_left==21)state<=RESPONSE;')], 'final_write_not_protected': [('2,6,7,8,11,12,14,15,19,20,21,22,23:return', '2,6,7,8,11,12,14,15,19,20,21,22:return')], 'missing_kv_exclusion': [(' || cache_overlap)begin', ')begin')], 'allow_write_alias': [('(writable(pair_left) || writable(pair_right)) && pair_overlap', "1'b0 && pair_overlap")], 'reject_read_alias': [('(writable(pair_left) || writable(pair_right)) && pair_overlap', 'pair_overlap')], 'source_epoch_truncated': [('source_epoch!=completion_epoch', 'source_epoch[31:0]!=completion_epoch[31:0]')], 'cache_sequence_truncated': [('saved_cache_sequence!=completion_sequence', 'saved_cache_sequence[31:0]!=completion_sequence[31:0]')], 'ignore_live_cache_size': [('cache_region_bytes!=saved_cache_bytes ||', '')], 'ignore_live_prefix': [('cache_query_committed_tokens!=validated_prefix', "1'b0")], 'revocation_not_sticky': [('if(busy && fault_now!=OK)completion_revoked<=1;', 'if(busy)completion_revoked<=fault_now!=OK;')], 'late_valid_unmasked': [(' && fault_now==OK;', ';')], 'late_status_changed': [('if(busy && fault_now!=OK)completion_revoked<=1;', 'if(busy && fault_now!=OK)begin completion_revoked<=1;completion_status<=fault_now;end')], 'descriptor_from_base': [('address[i]<=command_addr[i];', 'address[i]<=command_base[i];')], 'lease_off_by_one': [("validated_lease_base>64'hffffffffffffffe0", "validated_lease_base>64'hffffffffffffffe1")], 'truncated_layer': [('completion_layer>=16', 'completion_layer[3:0]>=16')], 'context_guard_late': [('wire cache_changed=busy &&', 'wire cache_changed=context_accepted &&')], 'short_vector_span': [("default:return 64'd8192;", "default:return 64'd4096;")]}
    result={}
    for name,pairs in changes.items():
        text=original
        for old,new in pairs:
            assert text.count(old)==1,(name,old);text=text.replace(old,new)
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
    info=dict(stage='V7-layer-preflight-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    original=(a.source/'fpga/rtl/ntrpu_llama32_layer_preflight_r1.sv').read_text()
    tb=a.source/'fpga/tb/tb_ntrpu_llama32_layer_preflight_r2.sv'
    for name,text in mutations(original).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_llama32_layer_preflight_r2','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1','+PLAN='+str(a.source/'fpga/tb/llama32_layer_preflight_plan_r1.mem')])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('LP_FAIL ')),None)
        caught=code==1 and failure is not None and not any(s.startswith('LP_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline','progress','offered','offer reached','final coordinated drain')):caught=False
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
