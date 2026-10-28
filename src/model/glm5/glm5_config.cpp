#include "glm5_config.h"

namespace nt {
namespace glm5 {

bool load_config(const GGUFLoader& loader, Config& cfg, std::string& err) {
    const std::string p = "glm5-next.";

    if (loader.kv_string("general.architecture", "") != "glm5-next") {
        err = "not a glm5-next model (general.architecture != glm5-next)";
        return false;
    }

    cfg.n_layer_all = (int)loader.kv_int(p + "block_count", 46);
    cfg.n_nextn     = (int)loader.kv_int(p + "nextn_predict_layers", 1);
    cfg.n_layers    = cfg.n_layer_all - cfg.n_nextn;
    cfg.hidden      = (int)loader.kv_int(p + "embedding_length", 4096);
    cfg.n_heads     = (int)loader.kv_int(p + "attention.head_count", 64);
    cfg.vocab_size  = (int)loader.kv_int(p + "vocab_size", 154880);
    if (!loader.vocab().tokens.empty()) {
        cfg.vocab_size = (int)loader.vocab().tokens.size();
    }

    cfg.norm_eps = loader.kv_float(p + "attention.layer_norm_rms_epsilon", 1e-5f);
    cfg.ln_eps   = loader.kv_float(p + "attention.layer_norm_epsilon", 1e-7f);

    cfg.dense_lead = (int)loader.kv_int(p + "leading_dense_block_count", 3);
    cfg.n_ff_dense = (int)loader.kv_int(p + "feed_forward_length", 12288);

    cfg.n_expert       = (int)loader.kv_int(p + "expert_count", 288);
    cfg.n_expert_used  = (int)loader.kv_int(p + "expert_used_count", 8);
    cfg.n_expert_shared = (int)loader.kv_int(p + "expert_shared_count", 1);
    cfg.n_ff_exp       = (int)loader.kv_int(p + "expert_feed_forward_length", 2048);
    cfg.n_ff_shexp     = (int)loader.kv_int(p + "expert_shared_feed_forward_length", 2048);
    cfg.expert_weights_scale = loader.kv_float(p + "expert_weights_scale", 2.5f);
    {
        const GGUFValue* v = loader.kv(p + "expert_weights_norm");
        cfg.expert_weights_norm = !v || (v->b || v->i != 0);
    }

    cfg.q_lora_rank = (int)loader.kv_int(p + "attention.q_lora_rank", 1536);
    cfg.kv_lora_rank = (int)loader.kv_int(p + "attention.kv_lora_rank", 512);
    cfg.qk_nope = (int)loader.kv_int(p + "attention.key_length_mla", 256);
    cfg.v_dim   = (int)loader.kv_int(p + "attention.value_length_mla", 256);

    cfg.kda_head_dim = (int)loader.kv_int(p + "kda.head_dim", 128);
    cfg.d_conv       = (int)loader.kv_int(p + "ssm.conv_kernel", 4);
    cfg.kda_gate_lower_bound = loader.kv_float(p + "kda.gate_lower_bound", -5.0f);

    cfg.indexer_n_head   = (int)loader.kv_int(p + "attention.indexer.head_count", 32);
    cfg.indexer_head_dim = (int)loader.kv_int(p + "attention.indexer.key_length", 128);
    cfg.indexer_top_k    = (int)loader.kv_int(p + "attention.indexer.top_k", 2048);
    cfg.indexer_kpool    = (int)loader.kv_int(p + "attention.indexer.kpool", 4);

    cfg.hc_mult            = (int)loader.kv_int(p + "hyper_connection.count", 4);
    cfg.hc_sinkhorn_iters  = (int)loader.kv_int(p + "hyper_connection.sinkhorn_iterations", 20);
    cfg.hc_eps             = loader.kv_float(p + "hyper_connection.epsilon", 1e-6f);

    cfg.bos_id = (int)loader.kv_int("tokenizer.ggml.bos_token_id", 0);
    cfg.eos_id = (int)loader.kv_int("tokenizer.ggml.eos_token_id", 1);

    // swiglu clamps (per layer over n_layer_all)
    cfg.swiglu_clamp_exp.assign(cfg.n_layer_all, 0.0f);
    if (const std::vector<double>* a = loader.kv_float_array(p + "swiglu_clamp_exp")) {
        for (int i = 0; i < cfg.n_layer_all && i < (int)a->size(); i++) {
            cfg.swiglu_clamp_exp[i] = (float)(*a)[i];
        }
    }
    cfg.swiglu_clamp_shexp = cfg.swiglu_clamp_exp;
    if (const std::vector<double>* a = loader.kv_float_array(p + "swiglu_clamp_shexp")) {
        for (int i = 0; i < cfg.n_layer_all && i < (int)a->size(); i++) {
            cfg.swiglu_clamp_shexp[i] = (float)(*a)[i];
        }
    }

    // layer roles: n_head_kv == 0 -> KDA (recurrent), else MLA
    cfg.is_recr.assign(cfg.n_layers, 1);
    if (const std::vector<int64_t>* a = loader.kv_int_array(p + "attention.head_count_kv")) {
        for (int i = 0; i < cfg.n_layers && i < (int)a->size(); i++) {
            cfg.is_recr[i] = ((*a)[i] == 0) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < cfg.n_layers; i++) {
            const std::string pre = "blk." + std::to_string(i) + ".";
            cfg.is_recr[i] = loader.tensor_info(pre + "attn_q.weight") != nullptr;
        }
    }

    if (cfg.hc_mult != 4) {
        err = "mHC with hc_mult != 4 is not supported";
        return false;
    }
    if (cfg.indexer_top_k % cfg.indexer_kpool != 0) {
        err = "indexer top_k must be a multiple of kpool";
        return false;
    }
    return true;
}

} // namespace glm5
} // namespace nt
