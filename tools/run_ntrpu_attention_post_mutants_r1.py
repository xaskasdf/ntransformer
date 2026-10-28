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
    info=dict(stage='V7-attention-post-path-mutants-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,native_only=True,hardware_slot=False,arithmetic_qualified=False,status='running',commands=[],mutants=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    original=(a.source/'fpga/rtl/ntrpu_attention_post_path_r2.sv').read_text()
    variants={
        'uncaptured_input':original.replace('.command_input_address(saved_command_post_input_address)', '.command_input_address(command_post_input_address)'),
        'wrong_O_source':original.replace('.command_branch_address(saved_command_output_addr)', ".command_branch_address(saved_command_output_addr+64'd64)"),
        'uncaptured_norm':original.replace('.command_weight_address(saved_command_post_weight_address)', '.command_weight_address(command_post_weight_address)'),
        'early_root_ack':original.replace('.completion_ready(root_owned && root_identity_match && release_public)', '.completion_ready(root_owned && r_completion_valid)'),
        'early_post_ack':original.replace('.completion_ready(post_owned && post_identity_match && release_public)', '.completion_ready(post_owned && p_completion_valid)'),
        'truncated_root_identity':original.replace('r_completion_sequence==completion_sequence', 'r_completion_sequence[31:0]==completion_sequence[31:0]'),
        'truncated_post_identity':original.replace('p_completion_sequence==completion_sequence', 'p_completion_sequence[31:0]==completion_sequence[31:0]'),
        'drop_late_revoke':original.replace('!revoked && !epoch_bad && !cancel_valid &&', ''),
        'early_bus_switch':original.replace('root_owned<=admission_fault==OK;post_owned<=0;', 'root_owned<=admission_fault==OK;post_owned<=1;'),
        'bypass_bounds':original.replace('!bounds_ok ? BAD_BOUNDS:', ''),
        'drop_admission_cancel':original.replace('cancel_valid ? CANCELLED:', ''),
        'missing_root_diagnostic':original.replace('completion_hls_status<=r_completion_hls_status;', 'completion_hls_status<=0;'),
        'missing_post_diagnostic':original.replace('completion_post_hls_status<=p_completion_hls_status;', 'completion_post_hls_status<=0;'),
        'missing_count':original.replace('completion_post_read_beats<=p_completion_read_beats;', 'completion_post_read_beats<=0;'),
        'truncated_sequence':original.replace('completion_sequence<=command_sequence;', "completion_sequence<={32'd0,command_sequence[31:0]};"),
        'drop_sticky_cancel':original.replace('child_cancel=busy && (cancel_valid || epoch_bad || fault!=OK)', 'child_cancel=busy && cancel_valid'),
    }
    tb=a.source/'fpga/tb/tb_ntrpu_attention_post_native_r1.sv';errors=[]
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
        compile_code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_attention_post_native_r1','-o',str(binary),str(path),str(tb)]+[str(a.source/n) for n in ['fpga/tb/ntrpu_attention_post_protocol_peers_r1.sv']])
        if compile_code!=0:errors.append(name+' compile failed');continue
        code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
        lines=(out/(name+'-test.log')).read_text().splitlines()
        failure=next((line for line in lines if line.startswith('FATAL:') and 'APATH_FAIL' in line),None)
        caught=code==1 and failure is not None and not any(line.startswith('APATH_PASS') for line in lines)
        info['mutants'].append(dict(name=name,rtl_sha256=sha(path),compile_exit_code=compile_code,test_exit_code=code,caught=caught,diagnostic=failure))
        if not caught:errors.append(name+' did not fail with an explicit bank assertion')
    info.update(exit_code=1 if errors else 0,errors=errors,finished_utc=now(),duration_seconds=time.monotonic()-started,
        status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)

if __name__=='__main__':main()
