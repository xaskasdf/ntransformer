"""Run bounded NTransformer smoke benchmarks while recording shared GPU load.

Each GPU runs a separate process. Concurrent wall throughput includes model load;
the CLI's decode throughput is also preserved for each process.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def gpu_snapshot():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=10)
    return result.stdout.strip().splitlines()


def process_memory():
    result = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=10)
    memory = {}
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if len(fields) == 3 and fields[1].strip().isdigit() and fields[2].strip().isdigit():
            memory.setdefault(int(fields[1]), {})[fields[0].strip()] = int(fields[2])
    return memory


def run_group(args, jobs, label):
    processes = []
    before = gpu_snapshot()
    started = time.monotonic()
    try:
        for gpu, model, streaming in jobs:
            log_path = args.output / f"{label}-gpu{gpu}.log"
            log = log_path.open("w", encoding="utf-8")
            command = [str(args.binary), "-m", str(model), "-p", args.prompt,
                       "-n", str(args.tokens), "-c", str(args.context), "-t", "0",
                       "--repeat-penalty", "1.0", "--seed", "42"]
            if streaming:
                command.append("--streaming")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            peer_gpu = getattr(args, "peer_gpu", None)
            if peer_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = f"{gpu},{peer_gpu}"
                command += ["--peer-gpu", "1", "--local-layers", str(args.local_layers)]
                if args.peer_layers is not None:
                    command += ["--peer-layers", str(args.peer_layers)]
                streaming = True
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append({"process": process, "file": log, "log_path": log_path,
                              "gpu": gpu, "model": str(model), "streaming": streaming,
                              "command": command, "peak_memory_mib": None, "peak_memory_by_gpu": {}})
        next_sample = started
        while any(job["process"].poll() is None for job in processes):
            now = time.monotonic()
            if now - started > args.timeout:
                raise TimeoutError(f"Benchmark exceeded {args.timeout} seconds")
            if now >= next_sample:
                memory = process_memory()
                for job in processes:
                    value = memory.get(job["process"].pid)
                    if value is not None:
                        job["peak_memory_mib"] = max(job["peak_memory_mib"] or 0, sum(value.values()))
                        for gpu_uuid, used in value.items():
                            job["peak_memory_by_gpu"][gpu_uuid] = max(job["peak_memory_by_gpu"].get(gpu_uuid, 0), used)
                next_sample = time.monotonic() + 0.2
            time.sleep(0.02)
        elapsed = time.monotonic() - started
    finally:
        for job in processes:
            if job["process"].poll() is None:
                job["process"].terminate()
                try:
                    job["process"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    job["process"].kill()
                    job["process"].wait()
            job["file"].close()
    results = []
    for job in processes:
        text = job["log_path"].read_text(encoding="utf-8", errors="replace")
        if job["process"].returncode:
            raise RuntimeError(f"NTransformer failed: {job['log_path']}\n{text[-3000:]}")
        decode = re.search(r"Decode: (\d+) tokens, ([\d.]+) ms \(([\d.]+) tok/s\)", text)
        prefill = re.search(r"Prompt: (\d+) tokens, ([\d.]+) ms \(([\d.]+) tok/s\)", text)
        if not decode or not prefill:
            raise RuntimeError(f"Missing inference statistics: {job['log_path']}")
        result = {
            "gpu": job["gpu"], "model": job["model"], "streaming": job["streaming"],
            "command": job["command"], "pid": job["process"].pid,
            "prompt_tokens": int(prefill[1]), "prefill_ms": float(prefill[2]),
            "decode_steps": int(decode[1]), "decode_ms": float(decode[2]),
            "decode_tokens_per_second": float(decode[3]),
            "generated_tokens": int(decode[1]) + 1,
            "sampled_peak_process_memory_mib": job["peak_memory_mib"],
            "sampled_peak_process_memory_by_gpu_uuid_mib": job["peak_memory_by_gpu"],
            "log": str(job["log_path"]),
        }
        generation = re.search(r"Prompt tokens: \d+\n(.*?)\n--- Stats ---", text, re.DOTALL)
        if generation:
            result["generation_sha256"] = hashlib.sha256(generation[1].encode()).hexdigest()
        peer = re.search(r"Peer cache: GPU (\d+), (\d+) blocks, ([\d.]+) GiB allocated, (\d+) P2P copies, ([\d.]+) GiB transferred", text)
        if peer:
            result["peer_cache"] = {"visible_device": int(peer[1]), "blocks": int(peer[2]),
                                    "allocated_gib": float(peer[3]), "copies": int(peer[4]),
                                    "transferred_gib": float(peer[5])}
        results.append(result)
        print(label, json.dumps(result), flush=True)
    return {"label": label, "wall_seconds_including_load": elapsed,
            "generated_tokens_per_wall_second_including_load": sum(r["generated_tokens"] for r in results) / elapsed,
            "gpu_before": before, "gpu_after": gpu_snapshot(), "processes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--prompt", default="La inteligencia artificial")
    args = parser.parse_args()
    if args.tokens < 2 or args.tokens > 128 or args.repeats < 1 or args.repeats > 10:
        parser.error("Use 2..128 tokens and 1..10 repeats for bounded smoke tests")
    if args.context < args.tokens + 64:
        parser.error("Context must leave room for the prompt and generation")
    args.binary = args.binary.resolve()
    args.models = [model.resolve() for model in args.models]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"shared_gpu_smoke": True, "prompt": args.prompt, "max_tokens": args.tokens,
              "repeats": args.repeats, "groups": [], "summary": {}}

    def save():
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    for model in args.models:
        for gpu in args.gpus:
            group_results = []
            for repeat in range(args.repeats):
                group = run_group(args, [(gpu, model, False)], f"{model.stem}-resident-r{repeat}")
                report["groups"].append(group)
                group_results.append(group["processes"][0]["decode_tokens_per_second"])
                save()
            report["summary"][f"{model.stem}-gpu{gpu}"] = {
                "median_decode_tokens_per_second": statistics.median(group_results),
                "min_decode_tokens_per_second": min(group_results),
                "max_decode_tokens_per_second": max(group_results),
            }
    tiered = run_group(args, [(args.gpus[0], args.models[0], True)], "tiered")
    report["groups"].append(tiered)
    if len(args.gpus) > 1:
        for repeat in range(args.repeats):
            group = run_group(args, [(gpu, args.models[0], False) for gpu in args.gpus], f"concurrent-r{repeat}")
            report["groups"].append(group)
            save()
    save()
    print("SUMMARY", json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
