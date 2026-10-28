"""Bounded generated-core integration; the queue verifies numerical evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and not args.output.exists()
    affinity = sorted(os.sched_getaffinity(0))[:2]
    os.sched_setaffinity(0, affinity)
    hashes = json.loads((args.source/'SNAPSHOT_FILES.json').read_text())
    for name, sha in hashes.items():
        assert hashlib.sha256((args.source/name).read_bytes()).hexdigest() == sha, name
    binding = json.loads((args.source/'generated/binding/binding.json').read_text())
    assert binding['status'] == 'INSPECTED_ATTENTION_R1_GENERATED_PORTS'
    assert binding['core_sha256'] == hashes['generated/attention/ntrpu_attention_tile_fp32_r1.v']
    assert binding['binding_sha256'] == hashes['generated/binding/ntrpu_attention_hls_ports_r1.sv']
    assert len(binding['ports']) == 35 and binding['synchronous_active_high_core_reset']
    args.output.mkdir(parents=True)
    command = ['timeout', '--signal=TERM', '--kill-after=30s', '5400s', 'vivado',
               '-mode', 'batch', '-source', str(args.source/'tools/ntrpu_board_attention_engine_r3.tcl'),
               '-log', str(args.output/'vivado.log'), '-journal', str(args.output/'vivado.jou'),
               '-tclargs', str(args.output/'output')]
    now = lambda: datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    info = dict(stage='V7-attention-engine-vivado-r1', started_utc=now(), source=str(args.source),
                source_sha256=args.source_sha, cpu_affinity=affinity, max_threads=2,
                timeout_seconds=5400, part='xcvu5p-flva2104-2-i', clock_ns=5,
                command=command, status='running',
                scope='155 AXI/context/generated-attention queries then OOC synthesis. No physical DDR or routed timing.')
    invocation = args.output/'invocation.json'
    invocation.write_text(json.dumps(info, indent=2)+'\n')
    with (args.output/'environment.log').open('wb') as stream:
        subprocess.run(['vivado', '-version'], stdout=stream, stderr=subprocess.STDOUT, check=True)
    with (args.output/'console.log').open('wb') as stream:
        result = subprocess.run(command, cwd=args.output, stdout=stream, stderr=subprocess.STDOUT)
    info.update(exit_code=result.returncode, duration_seconds=time.monotonic()-started,
                finished_utc=now(), status='awaiting_independent_review')
    invocation.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
