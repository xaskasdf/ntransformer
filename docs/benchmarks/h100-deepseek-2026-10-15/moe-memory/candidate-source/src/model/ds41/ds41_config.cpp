#include "ds41_config.h"
#include <cstdio>
#include <cstring>

namespace nt {
namespace ds41 {

bool load_config(const GGUFLoader& loader, Config& cfg, std::string& err) {
    auto need_i = [&](const std::string& key, int64_t def, bool required, int64_t* out) -> bool {
        const GGUFValue* v = loader.kv(key);
        if (!v) {
            if (required) {
                err = "missing metadata key " + key;
                return false;
            }
            *out = def;
            return true;
        }
        if (v->is_array() || v->is_string()) {
            err = "metadata key " + key + " has unexpected type";
            return false;
        }
        *out = (v->type == GGUFType::FLOAT32 || v->type == GGUFType::FLOAT64) ? (int64_t)v->f : v->i;
        return true;
    };
    auto need_f = [&](const std::string& key, float def, bool required, float* out) -> bool {
        const GGUFValue* v = loader.kv(key);
        if (!v) {
            if (required) { err = "missing metadata key " + key; return false; }
            *out = def;
            return true;
        }
        if (v->is_array() || v->is_string()) {
            err = "metadata key " + key + " has unexpected type";
            return false;
        }
        *out = (v->type == GGUFType::FLOAT32 || v->type == GGUFType::FLOAT64) ? (float)v->f : (float)v->i;
        return true;
    };

    const std::string p = "deepseek41.";

    if (loader.kv_string("general.architecture", "") != "deepseek41") {
        err = "not a deepseek41 model (general.architecture != deepseek41)";
        return false;
    }

    int64_t tmp = 0;
    if (!need_i(p + "block_count", 40, true, &tmp)) return false;             cfg.n_layers = (int)tmp;
    if (!need_i(p + "embedding_length", 5120, true, &tmp)) return false;      cfg.hidden = (int)tmp;
    if (!need_i(p + "attention.head_count", 64, true, &tmp)) return false;    cfg.n_heads = (int)tmp;
    if (!need_i(p + "attention.key_length", 512, true, &tmp)) return false;   cfg.head_dim = (int)tmp;
    if (!need_i(p + "rope.dimension_count", 64, true, &tmp)) return false;    cfg.rope_dim = (int)tmp;
    if (!need_i(p + "attention.q_lora_rank", 1280, true, &tmp)) return false; cfg.q_lora_rank = (int)tmp;
    if (!need_i(p + "attention.sliding_window", 128, true, &tmp)) return false; cfg.sliding_window = (int)tmp;
    if (!need_i(p + "expert_count", 384, true, &tmp)) return false;           cfg.n_expert = (int)tmp;
    if (!need_i(p + "expert_used_count", 6, true, &tmp)) return false;        cfg.n_expert_used = (int)tmp;
    if (!need_i(p + "expert_shared_count", 1, true, &tmp)) return false;      cfg.n_expert_shared = (int)tmp;
    if (!need_i(p + "expert_feed_forward_length", 2304, true, &tmp)) return false; cfg.n_ff_exp = (int)tmp;
    if (!need_i(p + "attention.indexer.head_count", 32, true, &tmp)) return false; cfg.indexer_n_head = (int)tmp;
    if (!need_i(p + "attention.indexer.key_length", 128, true, &tmp)) return false; cfg.indexer_head_dim = (int)tmp;
    if (!need_i(p + "attention.indexer.top_k", 512, true, &tmp)) return false; cfg.indexer_top_k = (int)tmp;
    if (!need_i(p + "attention.output_group_count", 8, true, &tmp)) return false; cfg.o_groups = (int)tmp;
    if (!need_i(p + "attention.output_lora_rank", 1024, true, &tmp)) return false; cfg.o_lora_rank = (int)tmp;
    if (!need_i(p + "hyper_connection.count", 4, true, &tmp)) return false;   cfg.hc_mult = (int)tmp;
    if (!need_i(p + "hyper_connection.sinkhorn_iterations", 20, true, &tmp)) return false; cfg.hc_sinkhorn_iters = (int)tmp;
    if (!need_i(p + "engram.head_count", 8, true, &tmp)) return false;        cfg.engram_n_head = (int)tmp;
    if (!need_i(p + "engram.key_length", 256, true, &tmp)) return false;      cfg.engram_key_length = (int)tmp;
    if (!need_i(p + "engram.max_ngram_size", 4, true, &tmp)) return false;    cfg.engram_max_ngram = (int)tmp;
    if (!need_i(p + "engram.pad_id", 2, false, &tmp)) return false;           cfg.engram_pad_id = (int)tmp;
    if (!need_i("tokenizer.ggml.bos_token_id", 0, false, &tmp)) return false; cfg.bos_id = (int)tmp;
    if (!need_i("tokenizer.ggml.eos_token_id", 1, false, &tmp)) return false; cfg.eos_id = (int)tmp;

    if (!need_f(p + "attention.layer_norm_rms_epsilon", 1e-6f, true, &cfg.norm_eps)) return false;
    if (!need_f(p + "rope.freq_base", 10000.0f, true, &cfg.rope_theta)) return false;
    if (!need_f(p + "attention.compress_rope_freq_base", 160000.0f, true, &cfg.compress_rope_theta)) return false;
    if (!need_f(p + "expert_weights_scale", 1.5f, true, &cfg.expert_weights_scale)) return false;
    if (!need_f(p + "rope.scaling.factor", 1.0f, false, &cfg.rope_factor)) return false;
    if (!need_f(p + "rope.scaling.yarn_beta_fast", 32.0f, false, &cfg.rope_beta_fast)) return false;
    if (!need_f(p + "rope.scaling.yarn_beta_slow", 1.0f, false, &cfg.rope_beta_slow)) return false;
    if (!need_f(p + "hyper_connection.epsilon", 1e-6f, true, &cfg.hc_eps)) return false;

    {
        const GGUFValue* v = loader.kv(p + "rope.scaling.original_context_length");
        if (v && !v->is_array() && !v->is_string()) cfg.rope_orig_ctx = (int)v->i;
    }
    {
        const GGUFValue* v = loader.kv(p + "expert_weights_norm");
        if (v && !v->is_array() && !v->is_string()) cfg.expert_weights_norm = (v->b || v->i != 0);
    }

    // per layer arrays
    {
        const GGUFValue* v = loader.kv(p + "attention.compress_ratios");
        if (!v || !v->is_array() || (int)v->array.ints.size() < cfg.n_layers) {
            err = "deepseek41.attention.compress_ratios missing or too short";
            return false;
        }
        cfg.compress_ratios.assign(v->array.ints.begin(), v->array.ints.begin() + cfg.n_layers);
    }
    {
        const GGUFValue* v = loader.kv(p + "swiglu_clamp_exp");
        if (!v || !v->is_array()) { err = "deepseek41.swiglu_clamp_exp missing"; return false; }
        cfg.swiglu_clamp_exp.resize(cfg.n_layers);
        for (int i = 0; i < cfg.n_layers; i++) {
            cfg.swiglu_clamp_exp[i] = (i < (int)v->array.reals.size()) ? (float)v->array.reals[i] : 0.0f;
        }
    }
    {
        const GGUFValue* v = loader.kv(p + "swiglu_clamp_shexp");
        cfg.swiglu_clamp_shexp.resize(cfg.n_layers);
        if (v && v->is_array()) {
            for (int i = 0; i < cfg.n_layers && i < (int)v->array.reals.size(); i++) {
                cfg.swiglu_clamp_shexp[i] = (float)v->array.reals[i];
            }
        } else {
            cfg.swiglu_clamp_shexp = cfg.swiglu_clamp_exp;
        }
    }

    // engram
    {
        const GGUFValue* v = loader.kv(p + "engram.layer_ids");
        if (!v || !v->is_array() || v->array.ints.empty()) {
            err = "deepseek41.engram.layer_ids missing";
            return false;
        }
        cfg.engram_layer_ids.assign(v->array.ints.begin(), v->array.ints.end());
    }
    auto load_u64_array = [&](const std::string& key, std::vector<uint64_t>& out) -> bool {
        const GGUFValue* v = loader.kv(key);
        if (!v || !v->is_array()) { err = "missing array " + key; return false; }
        out.assign(v->array.ints.begin(), v->array.ints.end());
        return true;
    };
    if (!load_u64_array(p + "engram.multipliers", cfg.engram_multipliers)) return false;
    if (!load_u64_array(p + "engram.primes",      cfg.engram_primes))      return false;
    if (!load_u64_array(p + "engram.offsets",     cfg.engram_offsets))     return false;
    {
        const GGUFValue* v = loader.kv(p + "engram.token_map");
        if (!v || !v->is_array()) { err = "missing deepseek41.engram.token_map"; return false; }
        cfg.engram_token_map.resize(v->array.ints.size());
        for (size_t i = 0; i < v->array.ints.size(); i++) cfg.engram_token_map[i] = (int32_t)v->array.ints[i];
    }

    // validation mirroring the reference loader
    const size_t n_bucket = (size_t)(cfg.engram_max_ngram - 1) * cfg.engram_n_head;
    if (cfg.engram_multipliers.size() != cfg.engram_layer_ids.size() * (size_t)cfg.engram_max_ngram) {
        err = "engram multiplier count does not match layers * ngram size";
        return false;
    }
    if (cfg.engram_primes.size() != cfg.engram_layer_ids.size() * n_bucket ||
        cfg.engram_offsets.size() != cfg.engram_primes.size()) {
        err = "engram prime or offset count does not match layers * buckets";
        return false;
    }

    // ------------------------------------------------------------
    // Derive layer roles from tensor presence (same rules as the
    // reference llama_model_deepseek41::load_arch_tensors)
    // ------------------------------------------------------------
    cfg.kv_source.assign(cfg.n_layers, -1);
    cfg.index_key_source.assign(cfg.n_layers, -1);
    cfg.topk_source.assign(cfg.n_layers, -1);

    int32_t last_kv_source = -1, last_key_owner = -1, last_index_source = -1;
    for (int i = 0; i < cfg.n_layers; i++) {
        std::string pre = "blk." + std::to_string(i) + ".";
        bool has_comp   = loader.tensor_info(pre + "attn_compressor_kv.weight") != nullptr;
        bool has_idx_k  = loader.tensor_info(pre + "indexer.attn_k.weight") != nullptr;
        bool has_idx_q  = loader.tensor_info(pre + "indexer.attn_q_b.weight") != nullptr;

        if (has_comp)  last_kv_source    = i;
        if (has_idx_k) last_key_owner    = i;
        if (has_idx_q) last_index_source = i;

        if (cfg.compress_ratios[i] == 0) continue;

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            err = "layer " + std::to_string(i) + " reads a compressed stream before any layer publishes one";
            return false;
        }
        if (cfg.compress_ratios[i] != cfg.compress_ratios[last_kv_source]) {
            err = "layer " + std::to_string(i) + " ratio disagrees with its KV source";
            return false;
        }
        cfg.kv_source[i]        = last_kv_source;
        cfg.index_key_source[i] = last_key_owner;
        cfg.topk_source[i]      = last_index_source;
    }

    return true;
}

} // namespace ds41
} // namespace nt
