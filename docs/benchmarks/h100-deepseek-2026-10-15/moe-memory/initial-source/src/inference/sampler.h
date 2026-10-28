#pragma once

#include <vector>
#include <random>

namespace nt {

// ============================================================
// Token sampler: top-k, top-p (nucleus), temperature
// ============================================================

struct SamplerConfig {
    float temperature = 0.7f;
    int   top_k       = 40;
    float top_p       = 0.9f;
    float repeat_penalty = 1.1f;
    int   repeat_window  = 64;
    uint64_t seed     = 42;
};

class Sampler {
public:
    Sampler() = default;

    void init(const SamplerConfig& config);

    // Sample a token from logits [vocab_size] (on CPU)
    int sample(const float* logits, int vocab_size);

    // Apply repeat penalty based on recent tokens
    void apply_repeat_penalty(float* logits, int vocab_size,
                              const std::vector<int>& recent_tokens);

    // Greedy (argmax)
    static int argmax(const float* logits, int vocab_size);

    // Reset RNG
    void set_seed(uint64_t seed);

private:
    SamplerConfig config_;
    std::mt19937 rng_;

    // Working buffers
    std::vector<std::pair<float, int>> candidates_;
};

} // namespace nt
