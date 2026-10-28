"""One immutable DDR activity attempt; bounded, preserved on any failure."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
from prepare_ntrpu_ddr_activity_project_r1 import digest, prepare


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    a = p.parse_args()
    source, output = a.source.resolve(), a.output.resolve()
    assert not output.exists()
    assert digest(source.with_suffix('.zip')) == a.source_sha
    hashes = json.loads((source / 'SNAPSHOT_FILES.json').read_text())
    assert all(digest(source / n) == h for n, h in hashes.items())
    for line in subprocess.check_output(['ps', '-eo', 'pid,comm'], text=True).splitlines()[1:]:
        f = line.split()
        if len(f) != 2 or f[1] not in ('vivado', 'xelab', 'xsimk', 'vitis-run', 'vitis_hls'):
            continue
        try:
            cwd = (Path('/proc') / f[0] / 'cwd').resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        assert not cwd.is_relative_to(output.parent), ('Slot occupied', f, cwd)
    output.mkdir()
    invocation = dict(started_utc=stamp(), source=str(source), source_sha256=a.source_sha,
                      output=str(output), cpu_affinity=[0, 1], timeout_seconds=1200,
                      automatic_retry=False, board_power_qualified=False)
    write(output / 'invocation.started.json', invocation)
    start = time.monotonic()
    code = 125
    try:
        # Include the exact frozen package in the terminal archive.
        shutil.copy2(source.with_suffix('.zip'), output / 'source.zip')
        prepare(source, output / 'output')
        command = ['timeout', '--signal=TERM', '--kill-after=60s', '20m',
                   'taskset', '-c', '0,1', 'vivado', '-mode', 'batch',
                   '-source', str(source / 'tools/ntrpu_board_ddr_activity_r1.tcl'),
                   '-log', str(output / 'vivado.log'), '-journal', str(output / 'vivado.jou'),
                   '-tclargs', str(source), str(output / 'output')]
        invocation['command'] = command
        write(output / 'command.json', invocation)
        with (output / 'console.log').open('xb') as log:
            code = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT).returncode
    except Exception as exc:
        invocation['runner_error'] = repr(exc)
    finally:
        unchanged = all(digest(source / n) == h for n, h in hashes.items())
        invocation.update(finished_utc=stamp(), elapsed_seconds=time.monotonic() - start,
                          exit_code=code, frozen_source_unchanged=unchanged)
        write(output / 'invocation.json', invocation)
    print(json.dumps(invocation), flush=True)
    return code if unchanged else 126


if __name__ == '__main__':
    raise SystemExit(main())
