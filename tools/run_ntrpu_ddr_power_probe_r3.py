"""R3: execute the reviewed TOP-identity successor of the DDR power probe.

This is a batch runner, not a queue coordinator. A failed attempt is retained;
there is no automatic retry. Inputs are copied into the batch for restoration.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


EXPECTED = {
    "ntrpu_board_ddr_power_probe_r2.tcl": "8bbc24239bebee597a84b094333f1960b6e23790ca285e63aa2f2d92c750be57",
    "route_attempt.dcp": "161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c",
}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    assert all(digest(inputs / name) == sha for name, sha in EXPECTED.items())
    assert not output.exists(), "Never overwrite or retry an existing batch"
    # The main thread must already own the slot. Refuse a competing tool in
    # this queue's runs directory without touching the existing process.
    for line in subprocess.check_output(["ps", "-eo", "pid,comm"], text=True).splitlines()[1:]:
        fields = line.split()
        if len(fields) != 2 or fields[1] not in ("vivado", "xelab", "xsimk", "vitis-run", "vitis_hls"):
            continue
        try:
            cwd = (Path("/proc") / fields[0] / "cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        assert not cwd.is_relative_to(output.parent), f"Queue slot occupied: {fields}, {cwd}"
    output.mkdir()
    shutil.copytree(inputs, output / "inputs")
    shutil.copy2(__file__, output / Path(__file__).name)
    before = {name: digest(output / "inputs" / name) for name in EXPECTED}
    write_json(output / "inputs.before.json", before)
    command = [
        "timeout", "--signal=TERM", "--kill-after=60s", "20m",
        "taskset", "-c", "0,1", "vivado", "-mode", "batch",
        "-source", str(output / "inputs/ntrpu_board_ddr_power_probe_r2.tcl"),
        "-log", str(output / "vivado.log"), "-journal", str(output / "vivado.jou"),
        "-tclargs", str(output / "inputs/route_attempt.dcp"), str(output / "reports"),
    ]
    invocation = dict(
        started_utc=stamp(), command=command, inputs=str(inputs), output=str(output),
        expected_input_sha256=EXPECTED, runner_sha256=digest(Path(__file__)),
        cpu_affinity=[0, 1], timeout_seconds=1200, automatic_retry=False,
        board_power_qualified=False, exact_dimm_qualified=False,
        scope="One routed generic DDR4 example; inherited/default activity, no SAIF",
    )
    write_json(output / "invocation.started.json", invocation)
    start = time.monotonic()
    exit_code = 125
    try:
        with (output / "console.log").open("xb") as stream:
            exit_code = subprocess.run(command, cwd=output, stdout=stream, stderr=subprocess.STDOUT).returncode
    except Exception as exc:
        invocation["runner_error"] = repr(exc)
    finally:
        after = {name: digest(output / "inputs" / name) for name in EXPECTED}
        staged_after = {name: digest(inputs / name) for name in EXPECTED}
        write_json(output / "inputs.after.json", after)
        integrity = dict(expected=EXPECTED, before=before, after=after, staged_after=staged_after,
                         unchanged=before == after == staged_after == EXPECTED)
        write_json(output / "input-integrity.json", integrity)
        (output / "exit_code.txt").write_text(str(exit_code) + "\n")
        invocation.update(finished_utc=stamp(), elapsed_seconds=time.monotonic() - start,
                          exit_code=exit_code, input_integrity_verified=integrity["unchanged"])
        write_json(output / "invocation.json", invocation)
    print(json.dumps(invocation), flush=True)
    return exit_code if integrity["unchanged"] else 126


if __name__ == "__main__":
    raise SystemExit(main())
