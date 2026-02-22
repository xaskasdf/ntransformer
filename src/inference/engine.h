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
    ~Engine();

    // Load model from GGUF file
    bool load(const std::string& model_path, int max_context = 4096, bool streaming = false);

    // Load draft model for speculative decoding
    bool load_draft(const std::string& draft_path, int max_context = 4096);

    // Generate text from a prompt
    std::string generate(const std::string& prompt, const GenerateConfig& config,
                         TokenCallback callback = nullptr);

    // Generate with speculative decoding (draft + target)
    std::string generate_speculative(const std::string& prompt, const GenerateConfig& config,
                                     TokenCallback callback = nullptr);

    // Generate with self-speculative decoding (VRAM layers as draft, no extra model)
    std::string generate_self_speculative(const std::string& prompt, const GenerateConfig& config,
                                          TokenCallback callback = nullptr);

    // Interactive chat mode
    void chat(const GenerateConfig& config);

    // Benchmark
    void benchmark(const std::string& prompt, int n_tokens);

    const ModelConfig& config() const { return model_.config(); }
    Transformer& model() { return model_; }
    bool has_draft() const { return draft_ != nullptr; }
    void set_draft_k(int k) { draft_k_ = k; }
    void set_self_speculative(bool enable) { self_speculative_ = enable; }

    // Set streaming pipeline buffer count (call before load).
    // n=0 auto-detects from PCIe bandwidth; n>=2 overrides.
    void set_pipeline_depth(int n) { model_.set_pipeline_depth(n); }

private:
    Transformer model_;
    Tokenizer tokenizer_;

    // Speculative decoding
    Transformer* draft_ = nullptr;  // owned, null if not speculative
    int draft_k_ = 5;              // number of draft tokens per iteration
    bool self_speculative_ = false; // use VRAM layers as draft (no extra model)

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
