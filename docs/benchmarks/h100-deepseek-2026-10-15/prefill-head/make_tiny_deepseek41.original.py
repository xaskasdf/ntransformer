#!/usr/bin/env python3
"""Emit a tiny deepseek41 GGUF for exercising the llama.cpp runtime.

Random weights, real structure. The point is to drive the loader and the graph so shape and
plumbing errors surface, not to produce sensible text. Every compress_ratio is 0, so this
covers the hyper-connection lag, the engram lookup, the MoE and the final collapse, and stays
clear of the sparse attention that is not written yet.

  python make_tiny_deepseek41.py out.gguf
"""
import sys
import numpy as np

sys.path.insert(0, "gguf-py")
import gguf  # noqa: E402

ARCH = "deepseek41"

# Tiny, but every relation the loader checks has to hold, and every quantizable tensor's ncols is
# a multiple of 256 so the low-bit formats apply without a fallback.
# Tiny, but every relation the loader checks has to hold. The head dims are not shrunk below 128:
# the K rotation width search in llama_kv_cache starts at 64 and halves back, so a smaller head
# yields a rotation wider than the head itself and llama_mul_mat_hadamard then reshapes to zero
# rows at n_tokens 1. Real models are 512 and 128, so this only bites toy sizes.
N_LAYER      = 6
N_EMBD       = 256
N_HEAD       = 4
N_EMBD_HEAD  = 128          # head_dim, the latent KV width
N_ROT        = 64           # rope_head_dim
Q_LORA       = 128
O_GROUPS     = 2
O_LORA       = 64
HC_MULT      = 4
N_EXPERT     = 4
N_EXPERT_USED= 2
N_FF_EXP     = 128
N_SHARED     = 1
IDX_N_HEAD   = 2
IDX_HEAD_DIM = 128
IDX_TOP_K    = 4
N_VOCAB      = 320   # 3 special + 256 byte + 61 normal

# layer roles, mirroring the real model's shape: a couple of sources publish a compressed stream
# and the layers after each of them read it, at two different ratios
COMPRESS_RATIOS   = [0, 0, 2, 2, 1, 1]
KV_SOURCE_LAYERS  = [2, 4]      # these compress and publish
INDEX_KEY_OWNERS  = [2, 4]      # these turn the latent into index keys
INDEX_SOURCES     = [2, 4, 5]   # these run the indexer; 5 reads layer 4's keys

ENGRAM_LAYERS   = [1]
ENGRAM_KEY_LEN  = 128
ENGRAM_MAX_NG   = 4
ENGRAM_N_HEAD   = 2
ENGRAM_PAD      = 1         # already through the token map, as the runtime expects

HC_DIM     = HC_MULT * N_EMBD
HC_MIX_DIM = (2 + HC_MULT) * HC_MULT
N_COLS     = (ENGRAM_MAX_NG - 1) * ENGRAM_N_HEAD

rng = np.random.default_rng(1234)


def rnd(*shape):
    # small values keep the graph away from inf while still being distinguishable
    return (rng.normal(size=shape) * 0.05).astype(np.float32)


def main(path):
    w = gguf.GGUFWriter(path, ARCH)

    w.add_block_count(N_LAYER)
    w.add_context_length(256)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(1)
    w.add_key_length(N_EMBD_HEAD)
    w.add_value_length(N_EMBD_HEAD)
    w.add_rope_dimension_count(N_ROT)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_file_type(gguf.GGMLQuantizationType.F32)

    w.add_q_lora_rank(Q_LORA)
    w.add_sliding_window(16)

    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_count(N_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_weights_scale(1.0)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SQRTSOFTPLUS)

    # per layer clamp arrays
    w.add_key_value(f"{ARCH}.swiglu_clamp_exp", [7.0] * N_LAYER,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.FLOAT32)
    assert len(COMPRESS_RATIOS) == N_LAYER
    w.add_key_value(f"{ARCH}.attention.compress_ratios", COMPRESS_RATIOS,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_key_value(f"{ARCH}.attention.compress_rope_freq_base", 10000.0,
                    gguf.GGUFValueType.FLOAT32)

    w.add_key_value(f"{ARCH}.attention.indexer.head_count", IDX_N_HEAD, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.attention.indexer.key_length", IDX_HEAD_DIM, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.attention.indexer.top_k", IDX_TOP_K, gguf.GGUFValueType.UINT32)

    w.add_key_value(f"{ARCH}.attention.output_group_count", O_GROUPS, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.attention.output_lora_rank", O_LORA, gguf.GGUFValueType.UINT32)

    w.add_key_value(f"{ARCH}.hyper_connection.count", HC_MULT, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.hyper_connection.sinkhorn_iterations", 3, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.hyper_connection.epsilon", 1e-6, gguf.GGUFValueType.FLOAT32)

    # ---- engram constants ----
    # small distinct primes standing in for the real 16M-ish ones; the runtime only requires that
    # they are distinct, non-zero, and that offsets + primes covers the table
    primes = [101, 103, 107, 109, 113, 127]
    assert len(primes) == N_COLS
    offsets, acc = [], 0
    for p in primes:
        offsets.append(acc)
        acc += p
    n_rows = acc

    mults = [(int(x) * 2 + 1) for x in rng.integers(1, 1 << 20, size=ENGRAM_MAX_NG)]
    token_map = [int(t % 37) for t in range(N_VOCAB)]

    w.add_key_value(f"{ARCH}.engram.layer_ids", ENGRAM_LAYERS,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_key_value(f"{ARCH}.engram.head_count", ENGRAM_N_HEAD, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.engram.key_length", ENGRAM_KEY_LEN, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.engram.max_ngram_size", ENGRAM_MAX_NG, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.engram.pad_id", ENGRAM_PAD, gguf.GGUFValueType.UINT32)
    w.add_key_value(f"{ARCH}.engram.multipliers", [int(m) for m in mults],
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
    w.add_key_value(f"{ARCH}.engram.primes", primes,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
    w.add_key_value(f"{ARCH}.engram.offsets", offsets,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
    w.add_key_value(f"{ARCH}.engram.token_map", token_map,
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)

    # ---- vocab ----
    # SPM falls back to byte tokens for anything it cannot piece together, and throws if they are
    # missing, so include the full 256 even though this model will never say anything sensible.
    tokens = ["<unk>", "<s>", "</s>"]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL]

    for b in range(256):
        tokens.append(f"<0x{b:02X}>")
        types.append(gguf.TokenType.BYTE)

    for i in range(N_VOCAB - len(tokens)):
        tokens.append(f"▁t{i}")
        types.append(gguf.TokenType.NORMAL)

    assert len(tokens) == N_VOCAB, (len(tokens), N_VOCAB)

    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * N_VOCAB)
    w.add_token_types([int(t) for t in types])

    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)

    # ---- tensors. numpy shape is the reverse of the ggml ne ----
    w.add_tensor("token_embd.weight",  rnd(N_VOCAB, N_EMBD))
    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight",      rnd(N_VOCAB, N_EMBD))

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight",     rnd(N_EMBD))
        w.add_tensor(p + "attn_sinks.weight",    rnd(N_HEAD))
        w.add_tensor(p + "attn_q_a.weight",      rnd(Q_LORA, N_EMBD))
        w.add_tensor(p + "attn_q_a_norm.weight", rnd(Q_LORA))
        w.add_tensor(p + "attn_q_b.weight",      rnd(N_HEAD * N_EMBD_HEAD, Q_LORA))
        w.add_tensor(p + "attn_kv.weight",       rnd(N_EMBD_HEAD, N_EMBD))
        w.add_tensor(p + "attn_kv_a_norm.weight",  rnd(N_EMBD_HEAD))
        # file layout (n_head*head_dim/groups, lora*groups); the loader reshapes it
        w.add_tensor(p + "attn_output_a.weight",    rnd(O_LORA * O_GROUPS, N_HEAD * N_EMBD_HEAD // O_GROUPS))
        w.add_tensor(p + "attn_output_b.weight",    rnd(N_EMBD, O_GROUPS * O_LORA))

        w.add_tensor(p + "hc_attn_fn.weight",    rnd(HC_MIX_DIM, HC_DIM))
        w.add_tensor(p + "hc_attn_base.weight",  rnd(HC_MIX_DIM))
        w.add_tensor(p + "hc_attn_scale.weight", rnd(3))
        w.add_tensor(p + "hc_ffn_fn.weight",     rnd(HC_MIX_DIM, HC_DIM))
        w.add_tensor(p + "hc_ffn_base.weight",   rnd(HC_MIX_DIM))
        w.add_tensor(p + "hc_ffn_scale.weight",  rnd(3))

        w.add_tensor(p + "ffn_gate_inp.weight",  rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "exp_probs_b.bias",     rnd(N_EXPERT))
        w.add_tensor(p + "ffn_norm.weight",      rnd(N_EMBD))

        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
        w.add_tensor(p + "ffn_up_exps.weight",   rnd(N_EXPERT, N_FF_EXP, N_EMBD))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_EXP * N_SHARED, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_EXP * N_SHARED))
        w.add_tensor(p + "ffn_up_shexp.weight",   rnd(N_FF_EXP * N_SHARED, N_EMBD))

        if il in KV_SOURCE_LAYERS:
            w.add_tensor(p + "attn_compressor_kv.weight",   rnd(N_EMBD_HEAD, N_EMBD))
            w.add_tensor(p + "attn_compressor_norm.weight", rnd(N_EMBD_HEAD))
            # a layer that pools more than one token per row also carries the gate
            if COMPRESS_RATIOS[il] > 1:
                w.add_tensor(p + "attn_compressor_gate.weight", rnd(N_EMBD_HEAD, N_EMBD))

        if il in INDEX_KEY_OWNERS:
            w.add_tensor(p + "indexer.attn_k.weight", rnd(IDX_HEAD_DIM, N_EMBD_HEAD))
            w.add_tensor(p + "indexer.k_norm.weight", rnd(IDX_HEAD_DIM))

        if il in INDEX_SOURCES:
            w.add_tensor(p + "indexer.attn_q_b.weight", rnd(IDX_N_HEAD * IDX_HEAD_DIM, Q_LORA))
            w.add_tensor(p + "indexer.proj.weight",     rnd(IDX_N_HEAD, N_EMBD))

        if il in ENGRAM_LAYERS:
            w.add_tensor(p + "engram_embd.weight", rnd(n_rows, ENGRAM_KEY_LEN))
            w.add_tensor(p + "engram_wkv.weight",  rnd(N_EMBD * (HC_MULT + 1), N_COLS * ENGRAM_KEY_LEN))
            w.add_tensor(p + "engram_q.weight",    rnd(HC_MULT, N_EMBD))
            w.add_tensor(p + "engram_k.weight",    rnd(HC_MULT, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}: {N_LAYER} layers, ratios {COMPRESS_RATIOS}, kv sources {KV_SOURCE_LAYERS}, "
          f"index sources {INDEX_SOURCES}, engram on {ENGRAM_LAYERS}, engram table {n_rows} rows")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tiny-deepseek41.gguf")
