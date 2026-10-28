"""Bounded xvlog/xelab wiring preflight with arithmetic signatures; no xsim execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[:2];os.sched_setaffinity(0,affinity)
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for name,sha in hashes.items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha,name
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    now=lambda:datetime.now(timezone.utc).isoformat();started=time.monotonic()
    info=dict(stage='V7-llama32-owned-ffn-compile-r2',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,per_command_timeout_seconds=180,native_only=True,hardware_slot=False,compile_only=True,arithmetic_qualified=False,status='running',commands=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (out/'version.log').open('wb') as f:subprocess.run(['xvlog','-version'],stdout=f,stderr=subprocess.STDOUT,check=True)
    commands=[('compile.log',['xvlog','--sv','--work','xil_defaultlib','--define','NTRPU_FFN_FP32','--define','NTRPU_FFN_FIXTURE','--define','NTRPU_FFN_LLAMA_REAL']+[str(a.source/n) for n in ['fpga/rtl/ntrpu_fp64_to_fp32.sv', 'fpga/rtl/ntrpu_axi_read_master.sv', 'fpga/rtl/ntrpu_weight_tile_buffer.sv', 'fpga/rtl/ntrpu_q8_unpacker.sv', 'fpga/rtl/ntrpu_q8_weight_stream.sv', 'fpga/rtl/ntrpu_q8_gemv_parallel.sv', 'fpga/rtl/ntrpu_activation_loader.sv', 'fpga/rtl/ntrpu_q8_gemv_memory.sv', 'fpga/rtl/ntrpu_result_writer.sv', 'fpga/rtl/ntrpu_q8_gemv_store.sv', 'fpga/rtl/ntrpu_matrix_sequencer.sv', 'fpga/rtl/ntrpu_q8_matrix_memory.sv', 'fpga/rtl/ntrpu_activation_writer.sv', 'fpga/rtl/ntrpu_swiglu_memory.sv', 'fpga/rtl/ntrpu_ffn_sequencer.sv', 'fpga/rtl/ntrpu_ffn_memory.sv', 'fpga/rtl/ntrpu_llama32_ffn_owner_r3.sv', 'fpga/rtl/ntrpu_llama32_owned_ffn_r1.sv', 'fpga/tb/ntrpu_q8_block_mac_compile_stub_r1.v', 'fpga/tb/ntrpu_swiglu_q8_block_compile_stub_r1.v', 'fpga/tb/tb_ntrpu_llama32_owned_ffn_r1.sv']]),
        ('elaborate.log',['xelab','-mt','2','-debug','off','-s','owned_ffn_compile','xil_defaultlib.ntrpu_llama32_owned_ffn_r1'])]
    code=0
    for name,cmd in commands:
        t=time.monotonic()
        with (out/name).open('wb') as f:
            try:result=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if code:break
    info.update(exit_code=code,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
