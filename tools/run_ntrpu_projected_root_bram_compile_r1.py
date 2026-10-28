"""Bounded Icarus RTL preflight, separate from Vivado qualification."""
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
    info=dict(stage='V7-projected-root-bram-compile-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,per_command_timeout_seconds=120,native_only=True,hardware_slot=False,compile_only=True,arithmetic_qualified=False,status='running',commands=[])
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (out/'version.log').open('wb') as f:subprocess.run(['iverilog','-V'],stdout=f,stderr=subprocess.STDOUT,check=True)
    commands=[('compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_attention_projected_root_bram_r1','-o',str(out/'compile-only.vvp')]+[str(a.source/n) for n in ['fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_tile_loader_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_cache_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_ram_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_path_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_context_r2.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_memory_r1.sv', 'fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_engine_r1.sv', 'fpga/rtl/ntrpu_attention_kv_read_guard_r1.sv', 'fpga/rtl/ntrpu_attention_state_read_guard_r2.sv', 'fpga/rtl/ntrpu_attention_hls_ports_prefetch_r3.sv', 'fpga/rtl/ntrpu_attention_root_r1.sv', 'fpga/rtl/ntrpu_attention_root_guard_r1.sv', 'fpga/rtl/ntrpu_attention_head_sequencer_r1.sv', 'fpga/rtl/ntrpu_fp64_to_fp32.sv', 'fpga/rtl/ntrpu_weight_tile_buffer.sv', 'fpga/rtl/ntrpu_q8_unpacker.sv', 'fpga/rtl/ntrpu_q8_weight_stream.sv', 'fpga/rtl/ntrpu_q8_gemv_parallel.sv', 'fpga/rtl/ntrpu_activation_loader.sv', 'fpga/rtl/ntrpu_q8_gemv_memory.sv', 'fpga/rtl/ntrpu_result_writer.sv', 'fpga/rtl/ntrpu_q8_gemv_store.sv', 'fpga/rtl/ntrpu_matrix_sequencer.sv', 'fpga/rtl/ntrpu_q8_matrix_memory.sv', 'fpga/rtl/registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv', 'fpga/rtl/vivado_explicit_nets_r2/ntrpu_attention_projection_memory_r1.sv', 'fpga/rtl/ntrpu_attention_activation_store_r1.sv', 'fpga/rtl/ntrpu_attention_quantize_owner_r3.sv', 'fpga/rtl/ntrpu_activation_writer.sv', 'fpga/rtl/ntrpu_attention_quantize_store_r2.sv', 'fpga/rtl/ntrpu_attention_output_path_r2.sv', 'fpga/rtl/ntrpu_attention_projected_root_r2.sv', 'fpga/tb/ntrpu_q8_block_mac_compile_stub_r1.v', 'fpga/tb/ntrpu_quantize_abi1_compile_stub_r1.v', 'fpga/tb/ntrpu_attention_tile_fp32_compile_stub_r1.v', 'fpga/tb/tb_ntrpu_attention_projected_root_bram_r1.sv']])]
    code=0
    for name,cmd in commands:
        t=time.monotonic()
        with (out/name).open('wb') as f:
            try:result=subprocess.run(cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT,timeout=120);code=result.returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if code:break
    info.update(exit_code=code,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
