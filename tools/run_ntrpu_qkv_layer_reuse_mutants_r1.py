"""Require compiled ownership mutations to fail explicit protocol assertions."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SOURCES=['fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv','fpga/rtl/ntrpu_qkv_snapshot_r1.sv',
 'fpga/rtl/ntrpu_qkv_dispatch_r1.sv','fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv',
 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv','fpga/rtl/ntrpu_attention_root_guard_r1.sv',
 'fpga/rtl/ntrpu_attention_head_sequencer_r1.sv']

def mutations(source):
    dispatch=SOURCES[2];engine=SOURCES[4];guard=SOURCES[5]
    specs={
      'idle_prefix_uses_previous_layer':(dispatch,'root_command_layer=state==IDLE ? command_layer:completion_layer;','root_command_layer=completion_layer;'),
      'guard_layer_alias':(guard,'prefixes[held_layer[3:0]]<=engine_done_committed_tokens[17:0];','prefixes[0]<=engine_done_committed_tokens[17:0];'),
      'engine_layer_alias':(engine,"prefixes[layer]<=position+18'd1;","prefixes[0]<=position+18'd1;"),
      'write_position_alias':(engine,"slot<=26'({layer,head})*26'(cache_capacity)+26'(position);","slot<=26'({layer,head})*26'(cache_capacity);"),
      'failed_write_invalidates_other_layers':(guard,'end else if(engine_done_status==0)begin','end else begin'),
      'bind_retains_old_engine_prefixes':(engine,'                        for(int i=0;i<16;i++) prefixes[i]<=0;','                        // mutation: old allocation prefixes retained')}
    result={}
    for name,(path,old,new) in specs.items():
        text=(source/path).read_text();assert text.count(old)==1,(name,old)
        result[name]=(path,text.replace(old,new))
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
    info=dict(stage='V7-qkv-layer-reuse-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';errors=[]
    def execute(name,cmd):
        tick=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-tick))
        inv.write_text(json.dumps(info,indent=2)+'\n');return code
    tb=a.source/'fpga/tb/tb_ntrpu_qkv_layer_reuse_r1.sv'
    for name,(target,text) in mutations(a.source).items():
        path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_qkv_layer_reuse_r1','-o',str(binary)]+[str(path if n==target else a.source/n) for n in SOURCES]+[str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((s for s in lines if s.startswith('FATAL:') and 'DISPATCH_FAIL' in s),None)
        caught=code==1 and failure is not None and not any(s.startswith('LAYERS_PASS') for s in lines)
        if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
        info['mutants'].append(dict(name=name,target=target,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' lacked an explicit non-timeout assertion')
    info.update(exit_code=int(bool(errors)),errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-start,status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
