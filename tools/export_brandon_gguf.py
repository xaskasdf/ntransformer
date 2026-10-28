"""Export a standard Brandon-Tiny checkpoint to a Llama GGUF for NTransformer.

Requires torch, sentencepiece, protobuf and gguf. Conversion runs entirely on CPU.
Brandon's adjacent RoPE pairs are reordered to GGUF's split-half Q/K layout.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def permute_rope(weight, heads):
    rows, columns = weight.shape
    head_dim = rows // heads
    if rows % heads or head_dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    return weight.reshape(heads, head_dim // 2, 2, columns).transpose(1, 2).reshape(rows, columns)


def export(checkpoint_path, tokenizer_path, output_path, precision):
    torch.set_num_threads(2)
    # Only load trusted, user-owned checkpoints; legacy configs may be dataclasses.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not isinstance(config, dict):
        config = dataclasses.asdict(config)
    for key in ("dense_former", "value_residual", "n_registers", "ternary",
                "relu_squared", "step_embedding", "feedback"):
        if config.get(key, False):
            raise ValueError(f"Unsupported Brandon architecture option: {key}")
    if config.get("n_loops", 1) != 1 or config.get("n_predict", 1) != 1:
        raise ValueError("Looped and multi-token prediction models are unsupported")
    dim = config["dim"]
    heads = config["n_heads"]
    kv_heads = config["n_kv_heads"]
    if dim % heads or heads % kv_heads:
        raise ValueError("Invalid attention dimensions")
    state = checkpoint["model_state_dict"]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if sp.vocab_size() != config["vocab_size"]:
        raise ValueError("Tokenizer and checkpoint vocabulary sizes differ")
    expected_tokenizer_hash = checkpoint.get("tokenizer_sha256")
    tokenizer_hash = sha256(tokenizer_path)
    if expected_tokenizer_hash and tokenizer_hash != expected_tokenizer_hash:
        raise ValueError("Tokenizer hash differs from the checkpoint manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_path), "llama", use_temp_file=True)
    writer.add_name(f"Brandon-Tiny {config['n_layers']}L {dim}D")
    writer.add_context_length(config["max_seq_len"])
    writer.add_embedding_length(dim)
    writer.add_block_count(config["n_layers"])
    writer.add_feed_forward_length(config["hidden_dim"])
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_layer_norm_rms_eps(config["norm_eps"])
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_uint32("llama.vocab_size", config["vocab_size"])
    writer.add_tokenizer_model("llama")
    writer.add_token_list([sp.id_to_piece(i) for i in range(sp.vocab_size())])
    writer.add_token_scores([sp.get_score(i) for i in range(sp.vocab_size())])
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(sp.serialized_model_proto())
    writer.add_token_types([int(piece.type) for piece in proto.pieces])
    writer.add_bos_token_id(sp.bos_id())
    writer.add_eos_token_id(sp.eos_id())
    writer.add_unk_token_id(sp.unk_id())
    if sp.pad_id() >= 0:
        writer.add_pad_token_id(sp.pad_id())
    writer.add_string("brandon.checkpoint_sha256", sha256(checkpoint_path))
    writer.add_string("brandon.tokenizer_sha256", tokenizer_hash)
    writer.add_uint32("brandon.checkpoint_step", checkpoint.get("step", 0))

    mapping = [
        ("tok_embeddings.weight", "token_embd.weight", None),
        ("norm.weight", "output_norm.weight", None),
    ]
    if not config.get("weight_tying", False):
        mapping.append(("output.weight", "output.weight", None))
    elif not torch.equal(state["output.weight"], state["tok_embeddings.weight"]):
        raise ValueError("Checkpoint declares tied weights but output differs")
    layer_tensors = (
        ("attention.wq.weight", "attn_q.weight", heads),
        ("attention.wk.weight", "attn_k.weight", kv_heads),
        ("attention.wv.weight", "attn_v.weight", None),
        ("attention.wo.weight", "attn_output.weight", None),
        ("feed_forward.w1.weight", "ffn_gate.weight", None),
        ("feed_forward.w2.weight", "ffn_down.weight", None),
        ("feed_forward.w3.weight", "ffn_up.weight", None),
        ("attention_norm.weight", "attn_norm.weight", None),
        ("ffn_norm.weight", "ffn_norm.weight", None),
    )
    for layer in range(config["n_layers"]):
        for source, target, rope_heads in layer_tensors:
            mapping.append((f"layers.{layer}.{source}", f"blk.{layer}.{target}", rope_heads))
    expected = {source for source, _, _ in mapping} | {"output.weight"}
    if config.get("block_sharing", False):
        expected |= {key for key in state if key.startswith("_unique_layers.")}
    extra = set(state) - expected
    if extra:
        raise ValueError(f"Unmapped checkpoint tensors: {sorted(extra)}")

    for source, target, rope_heads in mapping:
        weight = state[source].detach().float()
        if not torch.isfinite(weight).all():
            raise ValueError(f"Non-finite weight: {source}")
        if rope_heads is not None:
            weight = permute_rope(weight, rope_heads)
        data = weight.contiguous().numpy()
        if data.ndim == 1 or precision == "f32":
            writer.add_tensor(target, data)
        elif precision == "f16":
            writer.add_tensor(target, data.astype(np.float16))
        else:
            quantized = gguf.quantize(data, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(target, quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "gguf": str(output_path),
        "gguf_sha256": sha256(output_path),
        "gguf_bytes": output_path.stat().st_size,
        "precision": precision,
        "tensor_count": len(mapping),
        "config": config,
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("f32", "f16", "q8_0"), default="f16")
    args = parser.parse_args()
    export(args.checkpoint, args.tokenizer, args.output, args.precision)


if __name__ == "__main__":
    main()
