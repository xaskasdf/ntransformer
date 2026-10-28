"""Reproducible local scheduler study; writes only to a NEW output directory."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

from ntrpu_buffer_pool_model import Config, Job, Stalls, simulate


def workload(size, split, count=256, seed=20261004):
    rng = random.Random(seed)
    jobs = []
    for i in range(count):
        beats = {"256B": 4, "4KiB": 64}.get(size)
        if beats is None:
            beats = rng.choice((4, 8, 16, 64))
        channel = {"50_50": i % 2, "90_10": int(i % 10 == 9),
                   "100_0": 0, "blocked_head": int(i >= count // 2)}[split]
        jobs.append(Job(i, channel, beats))
    return jobs


def cases():
    base = Config()
    periodic = Stalls(w_period=256, w_closed=96)
    for size in ("256B", "4KiB", "mixed"):
        for split in ("50_50", "90_10", "100_0"):
            for reserve in (0, 4, 8, 16, 32):
                for outstanding in (1, 4, 8):
                    yield "reserve_sweep", size, split, replace(base, reserve=reserve, outstanding=outstanding), periodic
        for reserve in (0, 4, 32):
            yield "no_pause", size, "50_50", replace(base, reserve=reserve), Stalls()
    for size in ("256B", "4KiB"):
        for split in ("50_50", "100_0"):
            for total in (4, 8, 12, 20, 36):
                yield "equal_capacity_shared", size, split, replace(base, dedicated=0, reserve=total), periodic
        for ingress in ("ordered", "eligible"):
            for reserve in (0, 4, 32):
                yield "blocked_head", size, "blocked_head", replace(base, ingress=ingress, reserve=reserve), Stalls(w=((0, 0, 1024),))
        for ports in (1, 2):
            yield "memory_ports", size, "50_50", replace(base, reserve=32, read_ports=ports), periodic
            yield "reserve_ports", size, "50_50", replace(base, dedicated=0, reserve=36, reserve_read_ports=ports), periodic
        for quota in (2, None):
            yield "quota", size, "100_0", replace(base, reserve=4, reserve_quota=quota), periodic
    for outstanding in (1, 4, 8, 32):
        for release in (False, True):
            yield "b_lifetime", "256B", "100_0", replace(base, reserve=0, outstanding=outstanding, b_delay=128, release_on_b=release), Stalls()
    for slots in (2, 8, 64):
        yield "result_pause", "256B", "50_50", replace(base, result_slots=slots), Stalls(result=((0, 512),))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Never overwrite an earlier evidence batch.
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    inputs = {}
    for relative in ("tools/ntrpu_buffer_pool_model.py", "tools/run_ntrpu_buffer_pool_study.py",
                     "tests/test_ntrpu_buffer_pool_model.py", "docs/fpga/PEER_RX_MULTIBUFFER.md"):
        payload = (root / relative).read_bytes()
        snapshot = args.output / "inputs" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)
        inputs[relative] = hashlib.sha256(payload).hexdigest()
    started = time.perf_counter()
    output = []
    for index, (family, size, split, config, stalls) in enumerate(cases()):
        jobs = workload(size, split)
        result = simulate(config, jobs, stalls, max_cycles=200000)
        metrics = result["metrics"]
        if not metrics["drained"] or metrics["payload_bytes"] != sum(j.beats * config.beat_bytes for j in jobs):
            raise RuntimeError(f"incomplete or corrupt case {index}")
        if metrics["gbps_including_fill_and_drain"] > config.beat_bytes * config.clock_mhz / 1000:
            raise RuntimeError("single ingress bandwidth violated")
        output.append({"id": index, "family": family, "size": size, "split": split,
                       "count": len(jobs), "seed": 20261004, "config": asdict(config),
                       "stalls": asdict(stalls), "metrics": metrics})
        # Representative exact event traces, not a claim of RTL pin traces.
        if index == 0 or family in ("blocked_head", "b_lifetime"):
            (args.output / f"trace_{index:03d}.json").write_text(
                json.dumps(result["trace"], indent=2) + "\n", encoding="utf-8")
    report = {"status": "SCHEDULER_STUDY_PASS_NOT_RTL", "created_utc": datetime.now(timezone.utc).isoformat(),
              "wall_seconds": time.perf_counter() - started, "case_count": len(output),
              "input_sha256": inputs, "cases": output}
    (args.output / "RESULTS.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()
