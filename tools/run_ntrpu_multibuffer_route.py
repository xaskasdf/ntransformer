"""Bounded compact OOC implementation, independent of the principal queue."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ("tools/ntrpu_multibuffer_compact_route.tcl", "tools/run_ntrpu_multibuffer_route.py",
          "fpga/constraints/ntrpu_multibuffer_physical_ooc.xdc")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--synth-input", required=True, type=Path)
    parser.add_argument("--clock-site", default="BUFGCE_X0Y60")
    parser.add_argument("--context", choices=("assumed", "registered"), default="assumed")
    parser.add_argument("--variant", choices=("compact", "control", "decoded", "fillstage", "elastic"), default="compact")
    parser.add_argument("--route-directive", choices=("Default", "RuntimeOptimized", "Quick"), default="Default")
    parser.add_argument("--resume-placed-from", type=Path)
    parser.add_argument("--reserves", type=int, choices=(4, 8), nargs="+", default=(4, 8))
    parser.add_argument("--threads", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--spread-logic", action="store_true", help="re-place registered fixture with AltSpreadLogic_high")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.variant in ("control", "decoded", "fillstage", "elastic") and args.context != "registered":
        raise ValueError("control/decoded/fillstage variants require registered context")
    if not 1 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout must be between 1 and 1800 seconds")
    if len(set(args.reserves)) != len(args.reserves):
        raise ValueError("duplicate reserve profiles")
    checkpoints = {str((args.synth_input / f"reserve_{r}" / "multibuffer_synth.dcp").resolve()):
                   digest(args.synth_input / f"reserve_{r}" / "multibuffer_synth.dcp") for r in args.reserves}
    if args.context != "registered" and (args.resume_placed_from or args.route_directive != "Default" or args.threads != 2 or args.spread_logic):
        raise ValueError("resume/directive/thread options require registered context")
    resumed_files = ({str(p.resolve()): digest(p) for p in args.resume_placed_from.rglob("*") if p.is_file()}
                     if args.resume_placed_from else {})
    names = INPUTS + (("tools/ntrpu_multibuffer_registered_route.tcl",
                      "tools/prepare_ntrpu_multibuffer_physical_fixture.py") if args.context == "registered" else ())
    before = {name: digest(ROOT / name) for name in names}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[-args.threads:]
    if len(cpus) != args.threads:
        raise ValueError("requested CPU budget is unavailable")
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    record = dict(status="RUNNING", variant=args.variant, context=args.context, sources=before, checkpoints=checkpoints,
                  cpus=cpus, nice=os.nice(0), load_before=os.getloadavg(),
                  meminfo_before=Path("/proc/meminfo").read_text(),
                  timeout_seconds=args.timeout_seconds, clock_site=args.clock_site,
                  route_directive=args.route_directive, resumed_files=resumed_files,
                  reserves=list(args.reserves), threads=args.threads,
                  place_directive="AltSpreadLogic_high" if args.spread_logic else "Default", runs=[])
    save(output / "result.json", record)
    try:
        for reserve in args.reserves:
            label = f"reserve_{reserve}"
            checkpoint = (args.synth_input / label / "multibuffer_synth.dcp").resolve()
            script = "ntrpu_multibuffer_registered_route.tcl" if args.context == "registered" else "ntrpu_multibuffer_compact_route.tcl"
            command = ["vivado", "-mode", "batch", "-notrace", "-source",
                       str(ROOT / "tools" / script),
                       "-log", str(output / f"{label}.log"), "-journal", str(output / f"{label}.jou"),
                       "-tclargs", str(reserve), str(checkpoint), args.clock_site, str(output / label)]
            if args.context == "registered":
                command += [args.route_directive, str((args.resume_placed_from / label).resolve())
                            if args.resume_placed_from else "", str(args.threads), record["place_directive"]]
            start = time.monotonic()
            with (output / f"{label}.console.log").open("w") as log:
                child = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                timed_out = False
                try:
                    code = child.wait(timeout=args.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
            entry = dict(reserve=reserve, command=command, returncode=code, timeout=timed_out,
                         seconds=round(time.monotonic() - start, 3))
            record["runs"].append(entry)
            save(output / "result.json", record)
            print(json.dumps(entry), flush=True)
            if code != 0:
                record["status"] = "FAILED_OR_TIMEOUT"
                break
        else:
            record["status"] = "EXPORTED_PENDING_REVIEW"
    except Exception as error:
        record.update(status="RUNNER_ERROR", error=repr(error))
        raise
    finally:
        record.update(sources_after={name: digest(ROOT / name) for name in names},
                      checkpoints_after={p: digest(Path(p)) for p in checkpoints},
                      resumed_files_after={p: digest(Path(p)) for p in resumed_files}, load_after=os.getloadavg())
        if (before != record["sources_after"] or checkpoints != record["checkpoints_after"] or
                resumed_files != record["resumed_files_after"]):
            record["status"] = "INPUTS_CHANGED"
        save(output / "result.json", record)
        manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        save(output / "FILES.json", manifest)
        archive = output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(output)))
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError("CRC failure")
            for name, expected in manifest.items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                    raise RuntimeError("Archive member hash mismatch")
        save(output / "VERIFIED.json", dict(sha256=digest(archive), crc="PASS", hashes="PASS"))
        print(json.dumps(dict(status=record["status"], archive=str(archive), sha256=digest(archive))), flush=True)
    return 0 if record["status"] == "EXPORTED_PENDING_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
