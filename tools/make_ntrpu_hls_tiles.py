"""Create external, pinned Brandon Q8 tiles for the connected HLS testbench."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from run_ntrpu_brandon import BrandonWeights, MODEL_ID, REVISION, WEIGHT_SHA256, quantize_q8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = args.output.with_suffix(".json")
    if args.output.exists() or metadata.exists():
        parser.error("Output or metadata already exists; choose a new path")
    model = BrandonWeights(args.model_dir)
    model.audit()
    tiles = []
    blob = bytearray(b"NTRQ8T01" + struct.pack("<I", 3))
    for index, (name, start, rows) in enumerate((("w1", 0, 32), ("w3", 704, 16), ("w2", 224, 32))):
        tensor = "_unique_layers.0.feed_forward." + name + ".weight"
        weights = model.tensor(tensor)[start:start + rows]
        packed, values, _ = quantize_q8(weights)
        columns = values.shape[1] * 32
        rng = np.random.default_rng(20261001 + index)
        inputs = rng.standard_normal(weights.shape[1]).astype(np.float32)
        inputs /= np.sqrt(np.mean(inputs * inputs))
        inputs = np.pad(inputs, (0, columns - inputs.size))
        blob.extend(struct.pack("<II", rows, columns))
        blob.extend(packed)
        blob.extend(inputs.astype("<f4").tobytes())
        tiles.append({"tensor": tensor, "first_row": start, "rows": rows, "columns": columns})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        file.write(blob)
    record = {"model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
              "fixture_sha256": hashlib.sha256(blob).hexdigest(), "tiles": tiles,
              "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "encoder_sha256": hashlib.sha256(Path(__file__).with_name("run_ntrpu_brandon.py").read_bytes()).hexdigest(),
              "inputs": "Synthetic unit-RMS FP32, seed 20261001+tile; not model activations",
              "scope": "Three matrix tiles from one FFN; not a complete FFN or model"}
    with metadata.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    print(f"Created {args.output}: {len(blob)} bytes, three pinned tiles")


if __name__ == "__main__":
    main()
