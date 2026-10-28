"""Compile the attention integration with an interface declaration only.

Never execute the resulting vvp artifact: this preflight has no HLS arithmetic.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


RTL = ['ntrpu_axi_read_master', 'ntrpu_kv_tile_loader_r1',
       'ntrpu_kv_append_r1', 'ntrpu_kv_cache_r1',
       'ntrpu_attention_kv_ram_r1', 'ntrpu_attention_kv_path_r1',
       'ntrpu_attention_context_r2', 'ntrpu_attention_memory_r1',
       'ntrpu_attention_engine_r1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and not args.output.exists()
    hashes = json.loads((args.source / 'SNAPSHOT_FILES.json').read_text())
    for name, sha in hashes.items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == sha, name
    cpus = sorted(os.sched_getaffinity(0))
    affinity = cpus[2:4] if len(cpus) >= 4 else cpus[:2]
    os.sched_setaffinity(0, affinity)
    args.output.mkdir(parents=True)
    output = args.output / 'output'
    output.mkdir()
    started = time.monotonic()
    info = dict(stage='V7-attention-engine-interface-compile-r1',
                source=str(args.source), source_sha256=args.source_sha,
                started_utc=datetime.now(timezone.utc).isoformat(),
                cpu_affinity=affinity, timeout_seconds=120,
                compile_only=True, arithmetic_simulated=False,
                generated_attention_rtl_used=False, hardware_slot=False)
    invocation = args.output / 'invocation.json'
    invocation.write_text(json.dumps(info, indent=2) + '\n')
    with (output / 'version.log').open('wb') as stream:
        subprocess.run(['iverilog', '-V'], stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=10)
    command = ['iverilog', '-g2012', '-Wall', '-s', 'tb_ntrpu_attention_engine_r1',
               '-o', str(output / 'interface-only.vvp')]
    command += [str(args.source / 'fpga/rtl' / (name + '.sv')) for name in RTL]
    command += [str(args.source / 'interface/attention_engine_compile_stub_r1.sv'),
                str(args.source / 'fpga/tb/tb_ntrpu_attention_engine_r1.sv')]
    with (output / 'compile.log').open('wb') as stream:
        process = subprocess.Popen(command, cwd=output, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    info.update(command=command, exit_code=code,
                duration_seconds=time.monotonic() - started,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                status='COMPILE_ONLY_PASS' if code == 0 else 'COMPILE_FAILED')
    invocation.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
