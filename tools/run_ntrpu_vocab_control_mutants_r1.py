"""Compiled VOCAB protocol mutations; explicit assertions, never timeouts as passes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def mutations(original):
    variants = {}
    def add(name, replacements, tag_test=0):
        text = original
        for old, new in replacements:
            assert old in text, (name, old)
            text = text.replace(old, new)
        assert text != original
        variants[name] = (text, tag_test)
    add('admission_identity_low32', [(f'{a}{op}{b}', f'{a}[31:0]{op}{b}[31:0]')
        for a,b in [('source_sequence','command_sequence'),('source_epoch','command_epoch')]
        for op in ['==','!=']])
    add('live_source_identity_low32', [('source_sequence==completion_sequence && source_epoch==completion_epoch',
         'source_sequence[31:0]==completion_sequence[31:0] && source_epoch[31:0]==completion_epoch[31:0]')])
    add('epoch_low32', [('expected_epoch!=completion_epoch','expected_epoch[31:0]!=completion_epoch[31:0]'),
                      ('expected_epoch!=command_epoch','expected_epoch[31:0]!=command_epoch[31:0]')])
    add('unchecked_bounds', [('!span_valid(address[range_index],region_base[range_index],region_bytes[range_index],span(range_index))',"1'b0")])
    add('unchecked_aliases', [('else if(overlap)',"else if(1'b0)")])
    add('tag_stride_32', [("tag<=tag+64'd256", "tag<=tag+64'd32")])
    add('reservation_64', [("{1'b0,next_tag}+65'd4096", "{1'b0,next_tag}+65'd64")])
    add('early_source_ack', [('source_completion_ready=source_owned && release_all','source_completion_ready=source_owned && child_done_valid')])
    add('early_child_ack', [('child_done_ready=child_owned && (intermediate_ack || release_all)', 'child_done_ready=child_owned && child_done_valid')])
    add('foreign_child_accepted', [('wire child_identity={child_done_epoch,child_done_sequence}==tag', "wire child_identity=1'b1")])
    add('foreign_source_release', [('(!source_owned || (source_completion_valid && source_identity))', "1'b1")])
    add('withdrawn_offer', [('child_job_valid=state==OFFER','child_job_valid=state==OFFER && fault_now==OK')])
    add('forgotten_revocation', [('revoked<=1;', 'revoked<=0;')])
    add('early_completion', [('completion_valid=state==RESPONSE','completion_valid=state==RESPONSE || state==WAIT_CHILD')])
    add('skip_drain', [('else state<=DRAIN;', 'else state<=IDLE;')])
    add('short_bytes_accepted', [(' && child_done_bytes==phase_bytes','')])
    add('wrong_rows_accepted', [(' && child_done_rows==phase_rows','')])
    add('source_shape_unchecked', [('source_snapshot_address!=address[0] || source_snapshot_bytes!=4096', "1'b0")])
    add('leaked_q_descriptor', [("result_address=result_valid ? address[2]:64'd0",'result_address=address[2]')])
    add('tag_alignment_ignored', [('INITIAL_TAG[7:0]!=0 || INITIAL_TAG>LAST_TAG_BASE','INITIAL_TAG>LAST_TAG_BASE')], 2)
    add('tag_exhaustion_ignored', [('else if(exhausted)',"else if(1'b0)")], 1)
    add('cancel_not_forwarded', [('child_cancel_valid=child_owned && fault!=OK && !cancel_sent',"child_cancel_valid=1'b0")])
    add('uncaptured_public_sequence', [('completion_sequence<=command_sequence;completion_epoch<=command_epoch;',
         'completion_sequence<=command_sequence;completion_epoch<=command_epoch;'),
         ('            case(state)', '            if(busy)completion_sequence<=command_sequence;\n            case(state)')])
    add('output_address_low32', [('child_job_output_addr=address[2]+output_offset',"child_job_output_addr=address[2]+output_offset & 64'h00000000ffffffff")])
    add('live_source_metadata_ignored', [('source_result_address!=source_snapshot_address || source_result_bytes!=source_snapshot_bytes', "1'b0")])
    add('live_source_status_ignored', [('source_completion_status!=completion_source_status || ', '')])
    add('held_child_change_ignored', [('child_done_rows!=completion_child_rows || child_done_bytes!=completion_child_bytes', "1'b0")])
    add('cancel_requires_ready', [('child_cancel_valid=child_owned && fault!=OK && !cancel_sent',
         'child_cancel_valid=child_owned && fault!=OK && !cancel_sent && child_cancel_ready')])
    add('duplicate_cancellation', [('child_cancel_valid=child_owned && fault!=OK && !cancel_sent',
         'child_cancel_valid=child_owned && fault!=OK')])
    add('wrong_tail_rows', [("phase==15 ? 14'd5376:14'd8192", "14'd8192")])
    add('wrong_weight_stride', [("64'(phase)*64'd17825792", "64'(phase)*64'd17825728")])
    add('wrong_output_stride', [("64'(phase)<<15", "64'(phase)<<14")])
    add('fp64_instead_of_fp32', [('child_job_output_fp32=1','child_job_output_fp32=0')])
    add('short_output_allocation', [("default:return 64'd513024", "default:return 64'd512960")])
    add('stop_at_15_matrices', [('if(phase==15)begin completion_status<=OK','if(phase==14)begin completion_status<=OK')])
    return variants


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2]
    os.sched_setaffinity(0,affinity)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for n,h in json.loads((a.source/'SNAPSHOT_FILES.json').read_text()).items():assert sha(a.source/n)==h,n
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-vocab-control-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json'
    def save():inv.write_text(json.dumps(info,indent=2)+'\n')
    save();errors=[]
    def execute(name,cmd):
        start=time.monotonic()
        with (out/name).open('wb') as f:
            try:code=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-start))
        save();return code
    try:
        original=(a.source/'fpga/rtl/ntrpu_llama32_vocab_control_r1.sv').read_text()
        tb=a.source/'fpga/tb/tb_ntrpu_llama32_vocab_control_r1.sv'
        top='tb_ntrpu_llama32_vocab_control_r1'
        for name,(text,mode) in mutations(original).items():
            path=out/(name+'.sv');path.write_text(text);binary=out/(name+'.vvp')
            command=['iverilog','-g2012','-Wall','-s',top]
            if mode:
                tag="64'hfffffffffffff000" if mode==1 else "64'h1"
                command+=['-P'+top+'.INITIAL_TAG='+tag,'-P'+top+'.TAG_TEST='+str(mode)]
            command+=['-o',str(binary),str(path),str(tb)]
            compiled=execute(name+'-compile.log',command)
            if compiled!=0:
                errors.append(name+' compile failed');continue
            code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
            lines=(out/(name+'-test.log')).read_text().splitlines()
            failure=next((l for l in lines if l.startswith('FATAL:') and 'VOCAB_FAIL' in l),None)
            caught=code==1 and failure is not None and not any(x in failure.lower() for x in ('timeout','deadline','not offered')) and not any(l.startswith('VOCAB_PASS') for l in lines)
            info['mutants'].append(dict(name=name,rtl_sha256=sha(path),tag_test=mode,compile_exit_code=compiled,
                test_exit_code=code,caught=caught,diagnostic=failure))
            if not caught:errors.append(name+' not rejected by an explicit protocol assertion')
            save()
    except Exception as exc:
        errors.append(repr(exc))
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
                status='awaiting_independent_review')
    save();print(json.dumps(info),flush=True)


if __name__=='__main__':main()
