"""One bounded final-normalization/head synthesis and protocol attempt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): result.update(block)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.name == 'posix' and not a.output.exists()
    assert {0,1} <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, {0,1})
    source, output = a.source.resolve(), a.output.resolve()
    assert sha(source.with_suffix('.zip')) == a.source_sha
    hashes = json.loads((source/'SNAPSHOT_FILES.json').read_text())
    assert all(sha(source/n) == h for n,h in hashes.items())
    output.mkdir(parents=True)
    shutil.copy2(source.with_suffix('.zip'), output/'source.zip')
    now = lambda: datetime.now(timezone.utc).isoformat()
    command = ['timeout', '--signal=TERM', '--kill-after=60s', '3600s', 'vivado', '-mode', 'batch',
        '-source', str(source/'tools/ntrpu_board_final_vocab_r1.tcl'),
        '-log', str(output/'vivado.log'), '-journal', str(output/'vivado.jou'),
        '-tclargs', str(output/'output')]
    invocation = dict(started_utc=now(), source=str(source), source_sha256=a.source_sha,
        output=str(output), cpu_affinity=[0,1], max_threads=2, timeout_seconds=3600,
        command=command, protocol_jobs=540,
        automatic_retry=False, board_power_qualified=False, physical_timing_qualified=False,
        whole_model_rtl_qualified=False, scope='Actual final normalization and full vocabulary OOC synthesis. Separate protocol peers; no joint arithmetic execution or physical timing qualification.')
    def write_invocation():
        temp = output/'invocation.pending.json'
        with temp.open('x') as f: json.dump(invocation, f, indent=2); f.write('\n')
        os.replace(temp, output/'invocation.json')
    write_invocation()
    started = time.monotonic()
    code = 125
    try:
        with (output/'environment.log').open('xb') as log:
            subprocess.run(['vivado', '-version'], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output/'console.log').open('xb') as log:
            code = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT).returncode
    except Exception as exc:
        invocation['runner_error'] = repr(exc)
    finally:
        invocation.update(finished_utc=now(), elapsed_seconds=time.monotonic()-started, exit_code=code,
            frozen_source_unchanged=all(sha(source/n) == h for n,h in hashes.items()))
        write_invocation()
    print(json.dumps(invocation), flush=True)
    return code if invocation['frozen_source_unchanged'] else 126


if __name__ == '__main__':
    raise SystemExit(main())
