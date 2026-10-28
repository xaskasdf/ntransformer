"""Compare resident, host-cached and native CUDA P2P inference on two GPUs."""

import argparse
import json
from pathlib import Path
import statistics
from benchmark_shared_gpus import run_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--peer", type=int, default=1)
    parser.add_argument("--local-layers", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()
    if args.gpu < 0 or args.peer < 0 or args.gpu == args.peer or args.local_layers < 0:
        parser.error("Use two distinct physical GPUs and a nonnegative local layer cap")
    if not 2 <= args.tokens <= 256 or not 1 <= args.repeats <= 5 or args.context < args.tokens + 64:
        parser.error("Use 2..256 tokens, 1..5 repeats, and space in context for the prompt")
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"shared_training_load": True, "model": str(args.model), "prompt": args.prompt,
              "max_tokens": args.tokens, "repeats": args.repeats, "groups": [], "summary": {}}
    configurations = (("resident", None, None, None),
                      ("host", args.peer, args.local_layers, 0),
                      ("peer_half", args.peer, args.local_layers, None),
                      ("peer_all", args.peer, 0, None))
    for name, peer, local, peer_layers in configurations:
        args.peer_gpu = peer
        args.local_layers = local
        args.peer_layers = peer_layers
        processes = []
        for repeat in range(args.repeats):
            group = run_group(args, [(args.gpu, args.model, False)], f"{name}-r{repeat}")
            group["mode"] = name
            report["groups"].append(group)
            processes.append(group["processes"][0])
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        report["summary"][name] = {
            "median_decode_tokens_per_second": statistics.median(p["decode_tokens_per_second"] for p in processes),
            "median_prefill_ms": statistics.median(p["prefill_ms"] for p in processes),
            "generated_tokens": [p["generated_tokens"] for p in processes],
        }
    generations = {p["generation_sha256"] for group in report["groups"] for p in group["processes"] if "generation_sha256" in p}
    report["identical_generated_text"] = len(generations) == 1 and all(
        "generation_sha256" in p for group in report["groups"] for p in group["processes"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY", json.dumps(report["summary"], indent=2))
    print("Identical generated text:", report["identical_generated_text"])
    if not report["identical_generated_text"]:
        raise SystemExit("FAIL: generation differs between cache modes")


if __name__ == "__main__":
    main()
