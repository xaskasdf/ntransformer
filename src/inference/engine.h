#pragma once

#include "../model/transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include <string>
#include <vector>
#include <functional>

namespace nt {

// ============================================================
// Inference Engine
// High-level API for text generation
// ============================================================

struct GenerateConfig {
    int   max_tokens    = 256;
    float temperature   = 0.7f;
    int   top_k         = 40;
    float top_p         = 0.9f;
    float repeat_penalty = 1.1f;
    int   repeat_window = 64;
    uint64_t seed       = 42;
    bool  verbose       = true;
};

// Callback for streaming tokens
using TokenCallback = std::function<bool(const std::string& token, int token_id)>;

class Engine {
public:
    Engine() = default;
    ~Engine() = default;

    // Load model from GGUF file
    bool load(const std::string& model_path, int max_context = 4096);

    // Generate text from a prompt
    std::string generate(const std::string& prompt, const GenerateConfig& config,
                         TokenCallback callback = nullptr);

    // Interactive chat mode
    void chat(const GenerateConfig& config);

    // Benchmark
    void benchmark(const std::string& prompt, int n_tokens);

    const ModelConfig& config() const { return model_.config(); }

private:
    Transformer model_;
    Tokenizer tokenizer_;

    // Generation stats
    struct Stats {
        int   prompt_tokens = 0;
        int   gen_tokens = 0;
        float prefill_ms = 0;
        float decode_ms = 0;

        float prefill_tok_s() const { return prompt_tokens / (prefill_ms / 1000.0f); }
        float decode_tok_s() const { return gen_tokens / (decode_ms / 1000.0f); }
    };

    void print_stats(const Stats& stats) const;
};

} // namespace nt
