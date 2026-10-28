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
    info=dict(stage='V7-residual-store-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_residual_store_r1.sv').read_text()
    variants={
        'identity_low32':original.replace('command_sequence!=source_sequence','command_sequence[31:0]!=source_sequence[31:0]').replace('command_epoch!=source_epoch','command_epoch[31:0]!=source_epoch[31:0]').replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'unchecked_bounds':original.replace('state==CHECK_BOUNDS && bounds_bad', "state==CHECK_BOUNDS && 1'b0"),
        'wrong_residual_word':original.replace('record_data[word_index*32+:32]<=source_residual_q;', "record_data[word_index*32+:32]<=source_residual_q^32'd1;"),
        'early_pair_publication':original.replace('result_valid=completion_valid &&', 'result_valid=(completion_valid || state==SEND) &&'),
        'write_error_ignored':original.replace('(m_axi_bresp!=0 || m_axi_bid!=0)', "1'b0"),
        'commit_before_b':original.replace('if(b_fire && m_axi_bresp==0 && m_axi_bid==0)begin','if(w_fire)begin'),
        'leave_before_b':original.replace('SEND:if(b_fire)begin','SEND:if(w_fire)begin'),
        'withdraw_channels_on_fault':original.replace('state==SEND && !aw_seen','state==SEND && !aw_seen && fault==OK').replace('state==SEND && !w_seen','state==SEND && !w_seen && fault==OK'),
        'ignore_cancel':original.replace('else if(cancel_valid && cancel_ready)fault_now=CANCELLED;', "else if(1'b0)fault_now=CANCELLED;"),
        'ignore_timeout':original.replace('else if(timeout_now)fault_now=TIMEOUT;', "else if(1'b0)fault_now=TIMEOUT;"),
        'forgotten_revocation':original.replace('if(busy && (epoch_bad || cancel_valid))revoked<=1;', "if(1'b0)revoked<=1;"),
        'early_reuse':original.replace('command_ready=state==IDLE &&', 'command_ready=(state==IDLE || state==RESPONSE) &&'),
        'nonfinite_residual_accepted':original.replace("word_bad=source_residual_q[30:23]==8'hff;", "word_bad=0;"),
        'missing_ram_response_ignored':original.replace('!source_residual_q_valid', "1'b0"),
        'foreign_source_ack':original.replace('source_sequence==completion_sequence && source_epoch==completion_epoch;', "1'b1;"),
    }
    variants.update({
        'expected_epoch_low32':original.replace('command_epoch!=expected_epoch','command_epoch[31:0]!=expected_epoch[31:0]'),
        'address_low32':original.replace('m_axi_awaddr=residual_address+',"m_axi_awaddr={32'd0,residual_address[31:0]}+"),
        'bounds_wrap64':original.replace("{1'b0,residual_base}+{1'b0,residual_bytes}","{1'b0,(residual_base+residual_bytes)}").replace("{1'b0,residual_address}+65'd8192","{1'b0,(residual_address+64'd8192)}"),
        'extent_one_byte_short':original.replace('residual_end>residual_region_end',"residual_end>residual_region_end+65'd1"),
        'word_order':original.replace('word_index*32+:32','(15-word_index)*32+:32'),
        'short_output':original.replace('else if(record_index==127)','else if(record_index==126)'),
    })
    tb=a.source/'fpga/tb/tb_ntrpu_residual_store_r2.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_residual_store_r2','-o',str(binary),str(path),str(tb)])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1','+FIXTURE='+str(a.source/'fixture')])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'RSTORE_FAIL' in line),None)
        caught=code==1 and failure is not None and 'timeout' not in failure.lower() and not any(line.startswith('RSTORE_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
