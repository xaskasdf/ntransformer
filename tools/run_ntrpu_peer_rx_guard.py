"""Standalone Icarus simulation, model trace replay and immutable run evidence.

Does not invoke Vivado, modify the principal's queue or install software.
Run with Python 3.10+; pass --ivl-base when using a relocated Icarus package.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = (
    "fpga/rtl/ntrpu_peer_rx_guard.sv",
    "fpga/tb/tb_ntrpu_peer_rx_guard.sv",
    "tools/run_ntrpu_peer_rx_guard.py",
    "tools/ntrpu_pcie_axi_model.py",
    "tools/ntrpu_transport_model.py",
    "docs/fpga/PEER_RX_GUARD.md",
)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def replay_trace(stdout, max_beats):
    # Import the snapshotted model, inserted in sys.path by main().
    from ntrpu_pcie_axi_model import PcieAxiModel
    from ntrpu_transport_model import TransportError

    jobs = []
    current = None
    active_epoch = 0
    for line in stdout.splitlines():
        if not line.startswith("TRACE "):
            continue
        event = line.split()[1:]
        if event[0] == "CMD":
            current = {"tag": int(event[1]), "epoch": int(event[2]),
                       "address": int(event[3], 16), "beats": int(event[4]),
                       "armed_epoch": active_epoch, "events": []}
            jobs.append(current)
        elif current is not None:
            current["events"].append(event)
        if event[0] == "REARM":
            active_epoch = int(event[1])

    results = []
    for job in jobs:
        tag, address, count = job["tag"], job["address"], job["beats"]
        events = job["events"]
        statuses = [int(e[1]) for e in events if e[0] == "CPL"]
        need(len(statuses) == 1, f"tag {tag}: missing/duplicate completion")
        need(sum(e[0] == "RETIRE" for e in events) == 1, f"tag {tag}: retirement count")
        status = statuses[0]
        bad_desc = (job["epoch"] != job["armed_epoch"] or address % 64 != 0 or
                    count > max_beats or address % 4096 + count * 64 > 4096)
        need((status == 2) == bad_desc, f"tag {tag}: descriptor validation mismatch")
        record = {k: v for k, v in job.items() if k != "events"}
        record["status"] = status
        if status in (2, 3):
            need(not any(e[0] in ("LAUNCH", "AW", "W", "B") for e in events),
                 f"tag {tag}: rejected input reached AXI")
            record["check"] = "local rejection; no AXI exposure"
            results.append(record)
            continue

        # One isolated model per command; RTL epoch checks remain a separate
        # pin-level assertion. Payload framing errors are outside this model.
        model = PcieAxiModel(fragment_bytes=4096, mps=4096, max_axi_beats=max_beats)
        size = count * 64
        source = model.register(0, 0, size)
        target = model.register(1, address, size)
        payload = bytes(((tag + offset) * 29 + 17) & 255 for offset in range(size))
        model.host_write(source, 0, payload)
        work = model.submit(model.session, 1, source, model.bind_window(target), size)
        burst, = model.post(work, 0)
        model.acknowledge_posted(burst)
        seen_w = 0
        visible = 0
        retired = False
        late_writes = 0
        recovered = False
        for event in events:
            kind = event[0]
            if kind == "LAUNCH":
                model.deliver(burst)
                model.launch(burst)
            elif kind == "AW":
                need(int(event[1], 16) == address and int(event[2]) == count,
                     f"tag {tag}: address/length truncation")
                model.accept_aw(burst)
            elif kind == "W":
                observed = int(event[2], 16).to_bytes(64, "little")
                need(observed == payload[seen_w*64:(seen_w+1)*64], f"tag {tag}: wrong bytes")
                model.accept_w(burst, last=bool(int(event[1])))
                seen_w += 1
                if retired:
                    late_writes += 1
            elif kind == "STOP":
                model.fail("TIMEOUT")
            elif kind == "FOREIGN_B":
                model.fail("MEMORY_ERROR")
            elif kind == "B":
                response = int(event[1])
                # Explicit environment assumption: OKAY is final visibility.
                # Error examples leave one beat (SLVERR) or none (DECERR).
                visible = count if response == 0 else (1 if response == 2 else 0)
                for _ in range(visible):
                    model.memory_step(burst)
                model.accept_b(burst, {0: "OKAY", 2: "SLVERR", 3: "DECERR"}[response])
                if model.ready:
                    model.confirm_remote(work)
            elif kind == "CPL":
                expected = {0: "SUCCESS", 1: "TIMEOUT", 4: "MEMORY_ERROR"}[status]
                need(model.completion(work).status == expected, f"tag {tag}: model completion")
            elif kind == "RETIRE":
                model.retire(work)
                retired = True
            elif kind == "REARM":
                expected_raw = payload[:visible*64] + bytes(size-visible*64)
                need(model.debug_read_raw(target, 0, size) == expected_raw,
                     f"tag {tag}: memory before recovery")
                for node in (0, 1):
                    model.source_stop_ack(node)
                    model.isolate_ingress_ack(node)
                model.revoke_inflight()
                model.recover()
                recovered = True
            else:
                raise RuntimeError(f"Unknown trace event: {event}")

        if not recovered:
            expected_raw = payload[:visible*64] + bytes(size-visible*64)
            need(model.debug_read_raw(target, 0, size) == expected_raw, f"tag {tag}: memory bytes")
            if status == 0:
                need(model.read(target, 0, size) == payload, f"tag {tag}: successful read")
            else:
                # The foreign response never drains ID 0, even after error RETIRE.
                for node in (0, 1):
                    model.source_stop_ack(node)
                    model.isolate_ingress_ack(node)
                try:
                    model.recover()
                except TransportError as error:
                    need(error.code == "AXI_DRAIN_PENDING", "wrong quarantine reason")
                else:
                    raise RuntimeError("Foreign BID allowed model recovery")
        record.update(check="model trace replay PASS", w_beats=seen_w,
                      writes_after_error_retire=late_writes, recovered=recovered)
        results.append(record)
    need(len(results) == 37, "missing test cases in trace")
    need(sum(r.get("writes_after_error_retire", 0) for r in results) >= 6,
         "late-write coverage missing")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--ivl-base", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "artifacts/ntrpu-peer-rx-guard" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = {name: sha256(ROOT / name) for name in SOURCES}
    snapshot = output / "sources"
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    sys.path.insert(0, str(snapshot / "tools"))
    compiler = [str(pathlib.Path(args.iverilog).resolve())] if "/" in args.iverilog else [args.iverilog]
    runtime = [str(pathlib.Path(args.vvp).resolve())] if "/" in args.vvp else [args.vvp]
    if args.ivl_base:
        compiler += ["-B", str(args.ivl_base.resolve())]
        runtime += ["-M", str(args.ivl_base.resolve())]
    commands = []

    def run(label, command, *, expect_failure=None):
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=90, cwd=ROOT)
        (output / f"{label}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / f"{label}.stderr.log").write_text(result.stderr, encoding="utf-8")
        commands.append(dict(label=label, command=command, returncode=result.returncode,
                             seconds=round(time.monotonic() - start, 3)))
        if expect_failure is None:
            need(result.returncode == 0, f"{label} failed; see {output}")
        else:
            need(result.returncode != 0 and expect_failure in result.stdout,
                 f"{label}: mutation escaped or failed for unexpected reason")
        return result.stdout

    result = dict(status="FAIL", utc=stamp, platform=platform.platform(), python=sys.version,
                  source_hashes=before, commands=commands)
    try:
        run("iverilog_version", compiler + ["-V"])
        run("vvp_version", runtime + ["-V"])
        rtl = snapshot / SOURCES[0]
        bench = snapshot / SOURCES[1]

        def compile_rtl(label, source, max_beats):
            binary = output / f"{label}.vvp"
            run(label + "_compile", compiler + ["-g2012", "-Wall", "-s", "tb_ntrpu_peer_rx_guard",
                f"-Ptb_ntrpu_peer_rx_guard.MAX_BEATS={max_beats}", "-o", str(binary),
                str(source), str(bench)])
            return binary

        summaries = []
        for max_beats in (4, 64):
            label = f"beats_{max_beats}"
            binary = compile_rtl(label, rtl, max_beats)
            stdout = run(label + "_simulation", runtime + [str(binary)])
            match = re.search(r"^PASS cases=37 max_beats=" + str(max_beats) + r" .*", stdout, re.M)
            need(match is not None, "no RTL PASS summary")
            checks = replay_trace(stdout, max_beats)
            save(output / f"{label}_replay.json", checks)
            summaries.append(dict(max_beats=max_beats, summary=match.group(),
                                  model_replays=sum(r["check"] == "model trace replay PASS" for r in checks)))

        # Deliberately broken copies live ONLY in this evidence directory.
        original = rtl.read_text(encoding="utf-8")
        mutations = (
            ("cancel_aw", "state == ACTIVE && !aw_done;", "state == ACTIVE && !aw_done && !stop;",
             "AW unstable under pause"),
            ("release_lease", "cpl_valid <= 0;\n                if (state == FINISHED",
             "cpl_valid <= 0;\n                lease_held <= 0;\n                if (state == FINISHED",
             "error retirement released window"),
            ("early_rearm", "(state == IDLE || state == FINISHED) &&",
             "(state == IDLE || state == FINISHED || state == ACTIVE) &&", "rearm before AW/W"),
        )
        for name, old, new, diagnostic in mutations:
            need(original.count(old) == 1, f"ambiguous mutation {name}")
            mutant = output / f"mutation_{name}.sv"
            mutant.write_text(original.replace(old, new), encoding="utf-8")
            binary = compile_rtl("mutation_" + name, mutant, 4)
            run("mutation_" + name + "_simulation", runtime + [str(binary)], expect_failure=diagnostic)
        after = {name: sha256(ROOT / name) for name in SOURCES}
        need(before == after, "sources changed during validation")
        result.update(status="PASS", configurations=summaries,
                      rtl_cases=74, expected_mutation_failures=3, source_hashes_after=after)
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        save(output / "result.json", result)

    manifest = {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
                for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "FILES.json", manifest)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as bundle:
        need(bundle.testzip() is None, "archive CRC failure")
        for name, digest in manifest.items():
            need(hashlib.sha256(bundle.read(name)).hexdigest() == digest, "archive content failure")
        need(json.loads(bundle.read("FILES.json")) == manifest, "archive manifest mismatch")
        members = len(bundle.namelist())
    save(output / "VERIFIED.json", dict(archive=str(archive), sha256=sha256(archive), members=members,
                                        crc="PASS", content_hashes="PASS"))
    print(json.dumps(dict(status="PASS", rtl_cases=74, mutation_checks=3,
                          evidence=str(output), archive=str(archive), sha256=sha256(archive)), indent=2))


if __name__ == "__main__":
    main()
