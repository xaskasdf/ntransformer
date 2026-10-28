"""Calculate the proposed Llama 3.2 1B Q8_0 hardware storage/tiling budget.

No model loading or inference. Rates are mathematical ceilings, not measurements.
"""
import argparse
import json
from pathlib import Path


def plan(profile):
    d = profile["hidden_size"]
    h = profile["intermediate_size"]
    layers = profile["num_hidden_layers"]
    kv_width = profile["num_key_value_heads"] * profile["head_dim"]
    vocab = profile["vocab_size"]
    if d != profile["num_attention_heads"] * profile["head_dim"]:
        raise ValueError("inconsistent attention shape")
    if not profile["tie_word_embeddings"]:
        raise ValueError("this budget assumes shared embeddings/output projection")
    shapes = {"q_proj": (d, d), "k_proj": (kv_width, d), "v_proj": (kv_width, d),
              "o_proj": (d, d), "gate_proj": (h, d), "up_proj": (h, d), "down_proj": (d, h)}
    matrices = {}
    for name, (rows, columns) in shapes.items():
        if rows % 32 or columns % 1024:
            raise ValueError("this planner requires full 32 x 1024 tiles")
        matrices[name] = dict(rows=rows, columns=columns, q8_bytes=rows*columns//32*34,
                              row_tiles=rows//32, k_tiles=columns//1024,
                              tiles_per_matrix=(rows//32)*(columns//1024))
    block_bytes = sum(m["q8_bytes"] for m in matrices.values())
    embeddings_bytes = vocab*d//32*34
    norms_bytes = (2*layers + 1)*d*4
    weights_bytes = layers*block_bytes + embeddings_bytes + norms_bytes
    kv_per_token = 2*layers*kv_width*2  # K and V, FP16/BF16 storage, batch=1.
    tile_bytes = 32*1024//32*34
    return {
        "target_model": profile["target_model"], "target_revision": profile["target_revision"],
        "scope": "planning; synthetic bus tests; no model numerical or throughput validation",
        "assumptions": ["all dense weights and tied embeddings Q8_0 (34 bytes per 32 values)",
                        "normalization weights FP32; no bias; no container headers/alignment metadata",
                        "KV storage FP16/BF16, no quantization, batch=1",
                        "tile layout requires a future packer and persistent per-row accumulation"],
        "per_layer_matrices": matrices,
        "per_layer_q8_bytes": block_bytes,
        "tied_embeddings_q8_bytes": embeddings_bytes,
        "norms_fp32_bytes": norms_bytes,
        "unique_weight_bytes": weights_bytes,
        "unique_weight_mib": weights_bytes/2**20,
        "tile": {"rows": 32, "columns": 1024, "bytes": tile_bytes,
                 "axi_512_beats": tile_bytes//64,
                 "bursts_if_4k_aligned": (tile_bytes+4095)//4096,
                 "ping_pong_bytes": 2*tile_bytes},
        "proposed_scratch_bytes": {
            "quantized_max_activation_int8": h,
            "max_activation_scales_fp64": (h//32)*8,
            "gate_and_up_fp32": 2*h*4,
            "one_hidden_fp32": d*4,
            "row_accumulators_fp64": 32*8
        },
        "kv_bytes_per_token_all_layers": kv_per_token,
        "kv_contexts": {str(ctx): {"bytes": ctx*kv_per_token, "mib": ctx*kv_per_token/2**20}
                        for ctx in [2048, 8192, 32768, 131072]},
        "ideal_single_reader_bytes_per_second_at_200mhz": 64*200_000_000,
        "weight_only_decode_tokens_per_second_ceiling": 64*200_000_000/weights_bytes,
        "ideal_single_q8_unpacker_bytes_per_second_at_200mhz": 34*200_000_000,
        "weight_only_single_q8_unpacker_tokens_per_second_ceiling": 34*200_000_000/weights_bytes,
        "ceiling_excludes": "AXI/unpacker bubbles, DDR refresh/arbitration, KV traffic, compute, PCIe and model scheduling"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).resolve().parents[1] /
                        "fpga/profiles/llama32_1b_instruct.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(plan(json.loads(args.profile.read_text(encoding="utf-8"))), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
