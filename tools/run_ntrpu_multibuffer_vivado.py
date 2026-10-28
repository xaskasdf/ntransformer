"""Bounded isolated Vivado runs from a dedicated source copy, never the main queue."""

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
INPUTS = ("fpga/rtl/ntrpu_buffer_pool_allocator.sv", "fpga/rtl/ntrpu_peer_multibuffer.sv",
          "fpga/constraints/ntrpu_multibuffer_ooc.xdc", "tools/ntrpu_peer_multibuffer_synth.tcl",
          "tools/run_ntrpu_multibuffer_vivado.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--netlist-input", type=Path)
    parser.add_argument("--variant", choices=("reference", "compact", "control", "decoded", "fillstage", "elastic"), default="reference")
    parser.add_argument("--reserves", type=int, choices=(4, 8), nargs="+", default=(4, 8))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 1800:
        raise ValueError("timeout must be between 1 and 1800 seconds")
    if len(set(args.reserves)) != len(args.reserves):
        raise ValueError("duplicate reserve profiles")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ["OMP_NUM_THREADS"] = "2"
    names = INPUTS + (("tools/ntrpu_peer_multibuffer_netlist.tcl", "fpga/tb/tb_ntrpu_peer_multibuffer.sv",
                      "tools/prepare_ntrpu_multibuffer_netlist_tb.py", "tools/check_ntrpu_peer_multibuffer.py")
                     if args.netlist_input else ())
    if args.variant in ("compact", "control", "decoded", "fillstage", "elastic"):
        names += ("fpga/rtl/ntrpu_buffer_pool_allocator_parallel.sv", "fpga/rtl/ntrpu_peer_multibuffer_compact.sv",
                  "tools/ntrpu_multibuffer_compact_synth.tcl")
        if args.netlist_input:
            names += ("tools/ntrpu_multibuffer_compact_netlist.tcl", "tools/prepare_ntrpu_compact_netlist_tb.py")
    if args.variant == "control":
        names += ("fpga/rtl/ntrpu_peer_multibuffer_control.sv", "tools/ntrpu_multibuffer_control_synth.tcl")
    if args.variant in ("decoded", "fillstage", "elastic"):
        names += ("fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv", "fpga/rtl/ntrpu_peer_multibuffer_decoded.sv",
                  "tools/ntrpu_multibuffer_decoded_synth.tcl")
    if args.variant in ("fillstage", "elastic"):
        names += ("fpga/rtl/ntrpu_peer_multibuffer_fillstage.sv", "tools/ntrpu_multibuffer_fillstage_synth.tcl")
    if args.variant == "elastic":
        names += ("fpga/rtl/ntrpu_rv_elastic.sv", "fpga/rtl/ntrpu_peer_multibuffer_elastic.sv",
                  "tools/ntrpu_multibuffer_elastic_synth.tcl")
        if args.netlist_input:
            names += ("tools/ntrpu_multibuffer_elastic_netlist.tcl", "tools/prepare_ntrpu_elastic_tb.py",
                      "tools/check_ntrpu_elastic.py")
    before = {name: digest(ROOT / name) for name in names}
    netlists = {}
    if args.netlist_input:
        netlists = {str((args.netlist_input / f"reserve_{reserve}" / "multibuffer_funcsim.v").resolve()):
                    digest(args.netlist_input / f"reserve_{reserve}" / "multibuffer_funcsim.v") for reserve in args.reserves}
    record = dict(status="RUNNING", variant=args.variant, sources=before, cpus=cpus, nice=os.nice(0),
                  load_before=os.getloadavg(), meminfo_before=Path("/proc/meminfo").read_text(),
                  runs=[], netlists=netlists, timeout_seconds=args.timeout_seconds, reserves=list(args.reserves))
    save(output / "result.json", record)
    for reserve in args.reserves:
        label = f"reserve_{reserve}"
        script = "tools/ntrpu_peer_multibuffer_netlist.tcl" if args.netlist_input else "tools/ntrpu_peer_multibuffer_synth.tcl"
        if args.variant in ("compact", "control", "decoded", "fillstage", "elastic"):
            script = "tools/ntrpu_multibuffer_compact_netlist.tcl" if args.netlist_input else "tools/ntrpu_multibuffer_compact_synth.tcl"
        if args.variant == "control" and not args.netlist_input:
            script = "tools/ntrpu_multibuffer_control_synth.tcl"
        if args.variant == "decoded" and not args.netlist_input:
            script = "tools/ntrpu_multibuffer_decoded_synth.tcl"
        if args.variant == "fillstage" and not args.netlist_input:
            script = "tools/ntrpu_multibuffer_fillstage_synth.tcl"
        if args.variant == "elastic":
            script = "tools/ntrpu_multibuffer_elastic_netlist.tcl" if args.netlist_input else "tools/ntrpu_multibuffer_elastic_synth.tcl"
        tclargs = [str(reserve)]
        if args.netlist_input:
            tclargs += [str((args.netlist_input / label).resolve())]
        tclargs += [str(output / label)]
        command = ["vivado", "-mode", "batch", "-notrace", "-source", str(ROOT / script),
                   "-log", str(output / f"{label}.log"), "-journal", str(output / f"{label}.jou"),
                   "-tclargs"] + tclargs
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
        if code == 0 and args.netlist_input:
            try:
                if args.variant == "elastic":
                    from check_ntrpu_elastic import replay_external
                    entry["pin_replay"] = replay_external((output / label / "simulation.log").read_text(),
                        (2, reserve, 64, 16, 4), 1)
                else:
                    from check_ntrpu_peer_multibuffer import replay
                    entry["pin_replay"] = replay((output / label / "simulation.log").read_text(),
                        (2, reserve, 64, 16, 4), 1, check_bank_leases=False)
            except RuntimeError as error:
                entry["replay_error"] = str(error)
                code = 1
                entry["returncode"] = code
        record["runs"].append(entry)
        save(output / "result.json", record)
        print(json.dumps(entry), flush=True)
        if code != 0:
            record["status"] = "FAILED_OR_TIMEOUT"
            break
    else:
        record["status"] = "EXPORTED_PENDING_REVIEW"
    record.update(sources_after={name: digest(ROOT / name) for name in names}, load_after=os.getloadavg())
    if before != record["sources_after"] or any(digest(Path(p)) != h for p, h in netlists.items()):
        record["status"] = "SOURCES_CHANGED"
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
