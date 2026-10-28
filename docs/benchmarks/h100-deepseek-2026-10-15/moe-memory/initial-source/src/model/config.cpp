#include "config.h"
#include <cstdio>

namespace nt {

template<typename T>
static T get_val(
    const std::unordered_map<std::string, std::variant<int, float, std::string, bool>>& kv,
    const std::string& key,
    T default_val
) {
    auto it = kv.find(key);
    if (it == kv.end()) return default_val;
    if (auto* v = std::get_if<T>(&it->second)) return *v;
    return default_val;
}

void ModelConfig::from_gguf_metadata(
    const std::unordered_map<std::string, std::variant<int, float, std::string, bool>>& kv
) {
    metadata = kv;

    // Architecture name (e.g., "llama")
    architecture = get_val<std::string>(kv, "general.architecture", "llama");
    model_name = get_val<std::string>(kv, "general.name", "unknown");

    std::string prefix = architecture + ".";

    // Dimensions
    vocab_size        = get_val<int>(kv, prefix + "vocab_size", vocab_size);
    hidden_size       = get_val<int>(kv, prefix + "embedding_length", hidden_size);
    intermediate_size = get_val<int>(kv, prefix + "feed_forward_length", intermediate_size);
    n_layers          = get_val<int>(kv, prefix + "block_count", n_layers);
    n_heads           = get_val<int>(kv, prefix + "attention.head_count", n_heads);
    n_kv_heads        = get_val<int>(kv, prefix + "attention.head_count_kv", n_heads);
    head_dim          = hidden_size / n_heads;

    // Context
    max_seq_len       = get_val<int>(kv, prefix + "context_length", max_seq_len);

    // Normalization
    norm_eps          = get_val<float>(kv, prefix + "attention.layer_norm_rms_epsilon", norm_eps);

    // RoPE
    rope_theta        = get_val<float>(kv, prefix + "rope.freq_base", rope_theta);

    // Tokens
    bos_token_id      = get_val<int>(kv, "tokenizer.ggml.bos_token_id", bos_token_id);
    eos_token_id      = get_val<int>(kv, "tokenizer.ggml.eos_token_id", eos_token_id);
}

void ModelConfig::print() const {
    fprintf(stderr, "=== Model Config ===\n");
    fprintf(stderr, "Architecture: %s\n", architecture.c_str());
    fprintf(stderr, "Name: %s\n", model_name.c_str());
    fprintf(stderr, "Vocab: %d, Hidden: %d, Intermediate: %d\n",
        vocab_size, hidden_size, intermediate_size);
    fprintf(stderr, "Layers: %d, Heads: %d, KV Heads: %d, Head dim: %d\n",
        n_layers, n_heads, n_kv_heads, head_dim);
    fprintf(stderr, "Max seq: %d, Norm eps: %e\n", max_seq_len, norm_eps);
    fprintf(stderr, "RoPE theta: %.1f, GQA: %s (group=%d)\n",
        rope_theta, is_gqa() ? "yes" : "no", group_size());
    fprintf(stderr, "BOS: %d, EOS: %d\n", bos_token_id, eos_token_id);
}

} // namespace nt
