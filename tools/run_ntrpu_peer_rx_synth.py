"""Bounded, isolated Linux Vivado synthesis. Run from a dedicated source copy.

The caller sources Vivado settings64.sh. No SSH, global installation, queue
integration, IP generation, placement or routing is performed here.
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import signal
import subprocess
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUTS = ("fpga/rtl/ntrpu_peer_rx_guard.sv", "fpga/constraints/ntrpu_peer_rx_ooc.xdc",
          "tools/ntrpu_peer_rx_synth.tcl", "tools/run_ntrpu_peer_rx_synth.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--netlist-input", type=pathlib.Path,
                        help="Replay the two existing synthesis profiles instead of synthesizing")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Bound all children, including any Vivado helper processes, to two CPUs.
    cpus = sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    os.environ["OMP_NUM_THREADS"] = "2"
    names = INPUTS + (("tools/ntrpu_peer_rx_netlist.tcl", "fpga/tb/tb_ntrpu_peer_rx_guard.sv")
                     if args.netlist_input else ())
    before = {name: digest(ROOT / name) for name in names}
    netlists = {}
    if args.netlist_input:
        netlists = {str((args.netlist_input / f"beats_{b}" / "peer_rx_funcsim.v").resolve()):
                    digest(args.netlist_input / f"beats_{b}" / "peer_rx_funcsim.v") for b in (4, 64)}
    record = dict(status="RUNNING", sources=before, cpus=cpus, nice=os.nice(0),
                  platform=platform.platform(), load_before=os.getloadavg(),
                  meminfo_before=pathlib.Path("/proc/meminfo").read_text(), runs=[], netlists=netlists)
    save(output / "result.json", record)
    for beats in (4, 64):
        label = f"beats_{beats}"
        script = "tools/ntrpu_peer_rx_netlist.tcl" if args.netlist_input else "tools/ntrpu_peer_rx_synth.tcl"
        tclargs = [str(beats)]
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
                code = child.wait(timeout=420)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Only our own new process group; never touch another job.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
        entry = dict(beats=beats, command=command, returncode=code, timeout=timed_out,
                     seconds=round(time.monotonic() - start, 3))
        record["runs"].append(entry)
        save(output / "result.json", record)
        print(json.dumps(entry), flush=True)
        if code != 0:
            record["status"] = "FAILED_OR_TIMEOUT"
            break
    else:
        record["status"] = "EXPORTED_PENDING_REVIEW"
    after = {name: digest(ROOT / name) for name in names}
    record.update(sources_after=after, load_after=os.getloadavg())
    if before != after or any(digest(pathlib.Path(p)) != d for p, d in netlists.items()):
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
                raise RuntimeError("Archive content mismatch")
    save(output / "VERIFIED.json", dict(sha256=digest(archive), crc="PASS", hashes="PASS"))
    print(json.dumps(dict(status=record["status"], archive=str(archive), sha256=digest(archive))), flush=True)
    return 0 if record["status"] == "EXPORTED_PENDING_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
