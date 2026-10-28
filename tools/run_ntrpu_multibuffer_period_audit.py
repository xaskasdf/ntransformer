"""Bounded timing-only audit of selected periods on one routed registered fixture."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

from run_ntrpu_multibuffer_route import digest, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reserve", type=int, choices=(4, 8), required=True)
    parser.add_argument("--periods", choices=("5.715", "6.000", "6.667", "8.000"),
                        nargs="+", default=("6.667", "8.000"))
    args = parser.parse_args()
    if len(set(args.periods)) != len(args.periods):
        raise ValueError("duplicate audit periods")
    checkpoint = args.checkpoint.resolve()
    source_hash = digest(checkpoint)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = ("tools/run_ntrpu_multibuffer_period_audit.py", "tools/ntrpu_multibuffer_period_audit.tcl",
             "tools/run_ntrpu_multibuffer_route.py")
    before = {name: digest(ROOT / name) for name in names}
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ["OMP_NUM_THREADS"] = "2"
    command = ["vivado", "-mode", "batch", "-notrace", "-source",
               str(ROOT / "tools/ntrpu_multibuffer_period_audit.tcl"),
               "-log", str(output / "audit.log"), "-journal", str(output / "audit.jou"),
               "-tclargs", str(checkpoint), str(output / "audit")] + list(args.periods)
    record = dict(status="RUNNING", reserve=args.reserve, sources=before, checkpoint=str(checkpoint),
                  checkpoint_sha256=source_hash, cpus=cpus, nice=os.nice(0), timeout_seconds=300,
                  command=command, periods_ns=list(args.periods), load_before=os.getloadavg())
    save(output / "result.json", record)
    start = time.monotonic()
    try:
        with (output / "console.log").open("w") as log:
            child = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        record.update(returncode=code, timeout=timed_out,
                      status="EXPORTED_PENDING_REVIEW" if code == 0 else "FAILED_OR_TIMEOUT")
    except Exception as error:
        record.update(status="RUNNER_ERROR", error=repr(error))
        raise
    finally:
        record.update(seconds=round(time.monotonic() - start, 3),
                      sources_after={name: digest(ROOT / name) for name in names},
                      checkpoint_sha256_after=digest(checkpoint), load_after=os.getloadavg())
        if before != record["sources_after"] or source_hash != record["checkpoint_sha256_after"]:
            record["status"] = "INPUTS_CHANGED"
        save(output / "result.json", record)
        manifest = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        save(output / "FILES.json", manifest)
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            import hashlib
            assert bundle.testzip() is None
            assert all(hashlib.sha256(bundle.read(n)).hexdigest() == h for n, h in manifest.items())
        save(output / "VERIFIED.json", dict(sha256=digest(archive), crc="PASS", hashes="PASS"))
        print(json.dumps(dict(status=record["status"], seconds=record["seconds"], sha256=digest(archive))), flush=True)
    return 0 if record["status"] == "EXPORTED_PENDING_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
