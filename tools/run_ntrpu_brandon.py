#!/usr/bin/env python3
"""Pinned Brandon FFN weights -> Q8 fixtures -> independent references -> C++ fabric.

Needs NumPy; PyTorch is optional for --device cpu/cuda. Never imports model code,
loads pickle, or silently falls back from CUDA. This is not full-model inference.
"""

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_ID = "xaskasdf/brandon-tiny-10m-instruct"
REVISION = "ca6329de991a1cfc17b2a7a1aaae2cc38c08577a"
WEIGHT_SHA256 = "2ff727cc8921d21e5e7b153babd288efc5524b166e9417e1b439ac8c11fe44cf"
CONFIG_SHA256 = "602cab332932b65003fd386c67816406d14d9dd548740c6dc0395734f45b39ae"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class BrandonWeights:
    """Read only the exact audited BF16 artifact; not a general safetensors parser."""

    def __init__(self, model_dir):
        config_bytes = (model_dir / "config.json").read_bytes()
        if sha256(config_bytes) != CONFIG_SHA256:
            raise ValueError("config.json does not match pinned revision")
        self.config = json.loads(config_bytes)
        data = (model_dir / "model.safetensors").read_bytes()
        if sha256(data) != WEIGHT_SHA256:
            raise ValueError("model.safetensors does not match pinned revision")
        header_size = struct.unpack_from("<Q", data)[0]
        if header_size > 4 * 1024 * 1024 or 8 + header_size > len(data):
            raise ValueError("invalid safetensors header")
        self.header = json.loads(data[8:8 + header_size])
        self.header.pop("__metadata__", None)
        self.payload = memoryview(data)[8 + header_size:]
        for name, item in self.header.items():
            start, end = item["data_offsets"]
            if (item["dtype"] != "BF16" or not 0 <= start <= end <= len(self.payload)
                    or end - start != math.prod(item["shape"]) * 2):
                raise ValueError(f"invalid BF16 tensor: {name}")

    def raw(self, name):
        start, end = self.header[name]["data_offsets"]
        return self.payload[start:end]

    def tensor(self, name):
        # Explicit BF16 -> binary32 bit expansion, no lossy cast through FP16.
        bits = np.frombuffer(self.raw(name), dtype="<u2").astype("<u4") << 16
        return bits.view("<f4").reshape(self.header[name]["shape"]).copy()

    def audit(self):
        for name, item in self.header.items():
            if not name.startswith("_unique_layers."):
                continue
            _, index, suffix = name.split(".", 2)
            for alias_index in (2 * int(index), 2 * int(index) + 1):
                alias = f"layers.{alias_index}.{suffix}"
                if self.header[alias]["shape"] != item["shape"] or self.raw(name) != self.raw(alias):
                    raise ValueError(f"block-sharing mismatch: {name}, {alias}")
        if self.raw("tok_embeddings.weight") != self.raw("output.weight"):
            raise ValueError("tied embedding/output mismatch")
        canonical = [name for name in self.header if not name.startswith("layers.") and name != "output.weight"]
        unique = sum(math.prod(self.header[name]["shape"]) for name in canonical)
        if unique != self.config["total_parameters"]:
            raise ValueError("canonical parameter count differs from config")
        return {"stored_elements": sum(math.prod(v["shape"]) for v in self.header.values()),
                "unique_parameters": unique, "stored_tensors": len(self.header),
                "canonical_tensors": len(canonical), "shared_copies_identical": True}


def round_away(values):
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def quantize_q8(weights):
    """Positive maxabs/127 scale; half storage, round-away values, zero column pad.

    Values are rounded using the unrounded double scale. This is a declared
    experimental encoder; compatibility of packing does not imply GGUF conversion.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or min(weights.shape) == 0 or not np.isfinite(weights).all():
        raise ValueError("Q8 weights must be a nonempty finite matrix")
    padded = np.pad(weights, ((0, 0), (0, (-weights.shape[1]) % 32)))
    blocks = padded.reshape(padded.shape[0], -1, 32)
    scales = np.max(np.abs(blocks), axis=-1) / 127.0
    half = scales.astype("<f2")
    if not np.isfinite(half).all() or np.any((scales != 0) & (half == 0)):
        raise ValueError("Q8 scale cannot be represented in FP16")
    divided = np.divide(blocks, scales[..., None], out=np.zeros_like(blocks), where=scales[..., None] != 0)
    values = np.clip(round_away(divided), -127, 127).astype(np.int8)
    packed = np.empty((*scales.shape, 34), dtype=np.uint8)
    packed[..., :2] = half.view(np.uint8).reshape(*scales.shape, 2)
    packed[..., 2:] = values.view(np.uint8)
    return packed.tobytes(), values, half.astype(np.float64)


def q8_gemv(values, scales, inputs):
    """Independent NumPy integer dots, block scales and double accumulation."""
    blocks = np.asarray(inputs, dtype=np.float64).reshape(-1, 32)
    sx = np.max(np.abs(blocks), axis=-1) / 127.0
    divided = np.divide(blocks, sx[:, None], out=np.zeros_like(blocks), where=sx[:, None] != 0)
    qx = np.clip(round_away(divided), -127, 127).astype(np.int32)
    out = np.zeros(values.shape[0], dtype=np.float64)
    for block in range(blocks.shape[0]):
        dots = values[:, block, :].astype(np.int32) @ qx[block]
        out += dots.astype(np.float64) * scales[:, block] * sx[block]
    return out.astype(np.float32)


def swiglu(gate, up):
    gate = np.asarray(gate, dtype=np.float64)
    e = np.exp(-np.abs(gate))
    sigmoid = np.where(gate >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return ((gate * sigmoid) * np.asarray(up, dtype=np.float64)).astype(np.float32)


def quantized_ffn(quantized, inputs):
    _, g, gs = quantized[0]
    _, u, us = quantized[1]
    _, d, ds = quantized[2]
    outputs = []
    for x in inputs:
        activation = swiglu(q8_gemv(g, gs, x), q8_gemv(u, us, x))
        activation = np.pad(activation, (0, (-activation.size) % 32))
        outputs.append(q8_gemv(d, ds, activation))
    return np.asarray(outputs, dtype=np.float32)


def fp32_ffn(weights, inputs, device):
    gate, up, down = weights
    if device == "numpy":
        g, u = inputs @ gate.T, inputs @ up.T
        # NumPy FP32 sigmoid/products and GEMM, independent of Q8 double model.
        e = np.exp(-np.abs(g))
        sigmoid = np.where(g >= 0, np.float32(1) / (np.float32(1) + e), e / (np.float32(1) + e))
        return ((g * sigmoid) * u) @ down.T
    import torch
    with torch.inference_mode():
        x, g, u, d = [torch.from_numpy(a).to(device=device, dtype=torch.float32)
                       for a in (inputs, gate, up, down)]
        y = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, g)) * torch.nn.functional.linear(x, u), d)
        return y.cpu().numpy()


def write_fixture(path, weights, inputs, reference, quantized_reference, hidden):
    dim = inputs.shape[1]
    with path.open("xb") as file:
        file.write(b"NTRFFN01")
        file.write(struct.pack("<IIII", dim, hidden, (hidden + 31) // 32 * 32, len(inputs)))
        for packed, _, _ in weights:
            file.write(packed)
        for data in (inputs, reference, quantized_reference):
            file.write(np.asarray(data, dtype="<f4").tobytes())


def run(args):
    if not 1 <= args.samples <= 256 or (args.layer is not None and not 0 <= args.layer < 12):
        raise ValueError("samples must be 1..256; layer must be 0..11")
    model = BrandonWeights(args.model_dir)
    audit = model.audit()
    device_info = {"backend": args.device, "numpy": np.__version__}
    if args.device != "numpy":
        import torch
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        device_info.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, tf32=False)
        if args.device == "cuda":
            device_info["gpu"] = torch.cuda.get_device_name(0)
    executable = args.exe.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
              "config_sha256": CONFIG_SHA256, "audit": audit, "reference": device_info,
              "python": platform.python_version(), "platform": platform.platform(),
              "runner_sha256": sha256(Path(__file__).read_bytes()),
              "executable_sha256": sha256(executable.read_bytes()),
              "inputs": "synthetic normal vectors, unit RMS; not model activations",
              "seed": args.seed, "samples_per_layer": args.samples,
              "scope": "isolated FFN with real weights; no norm, residual, attention, logits or quality evaluation",
              "limits": {"relative_l2": 0.05, "q8_reference_atol": 1e-6, "q8_reference_rtol": 1e-5},
              "layers": []}
    report_path = args.output_dir / "report.json"
    try:
        for layer in range(12) if args.layer is None else [args.layer]:
            prefix = f"_unique_layers.{layer}.feed_forward"
            weights = [model.tensor(f"{prefix}.{name}.weight") for name in ("w1", "w3", "w2")]
            if [w.shape for w in weights] != [(720, 256), (720, 256), (256, 720)]:
                raise ValueError("unexpected FFN shapes")
            quantized = [quantize_q8(w) for w in weights]
            rng = np.random.default_rng(args.seed + layer)
            inputs = rng.standard_normal((args.samples, 256)).astype(np.float32)
            inputs /= np.sqrt(np.mean(inputs * inputs, axis=1, keepdims=True))
            reference = fp32_ffn(weights, inputs, args.device)
            q_reference = quantized_ffn(quantized, inputs)
            fixture = args.output_dir / f"ffn_{layer:02d}.bin"
            write_fixture(fixture, quantized, inputs, reference, q_reference, 720)
            process = subprocess.run([str(executable), str(fixture.resolve())], capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(f"layer {layer}: {process.stderr.strip() or process.stdout.strip()}")
            result = json.loads(process.stdout)
            result.update(unique_layer=layer, logical_layers=[2 * layer, 2 * layer + 1],
                          fixture_sha256=sha256(fixture.read_bytes()), shape=[256, 720, 256], padded_hidden=736)
            report["layers"].append(result)
            print(f"layer {layer:02d}: relative L2={result['relative_l2']:.6%}; "
                  f"Q8 reference max abs={result['q8_reference_max_abs_error']:.3g}", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True, help="compiled ntrpu_ffn_fixture")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; refuses overwrite")
    parser.add_argument("--device", choices=("numpy", "cpu", "cuda"), default="numpy")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--layer", type=int, help="unique layer 0..11; default all 12")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
