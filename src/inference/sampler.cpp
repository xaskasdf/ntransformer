#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace nt {

void Sampler::init(const SamplerConfig& config) {
    config_ = config;
    rng_.seed(config_.seed);
}

void Sampler::set_seed(uint64_t seed) {
    config_.seed = seed;
    rng_.seed(seed);
}

int Sampler::argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

void Sampler::apply_repeat_penalty(float* logits, int vocab_size,
                                    const std::vector<int>& recent_tokens) {
    if (config_.repeat_penalty <= 1.0f) return;

    int window = std::min((int)recent_tokens.size(), config_.repeat_window);
    for (int i = (int)recent_tokens.size() - window; i < (int)recent_tokens.size(); i++) {
        int tok = recent_tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            if (logits[tok] > 0) {
                logits[tok] /= config_.repeat_penalty;
            } else {
                logits[tok] *= config_.repeat_penalty;
            }
        }
    }
}

int Sampler::sample(const float* logits, int vocab_size) {
    // Temperature = 0 -> greedy
    if (config_.temperature <= 0.0f) {
        return argmax(logits, vocab_size);
    }

    // Build candidates with temperature-scaled logits
    candidates_.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        candidates_[i] = {logits[i] / config_.temperature, i};
    }

    // Top-K filtering
    int k = config_.top_k;
    if (k > 0 && k < vocab_size) {
        std::partial_sort(candidates_.begin(), candidates_.begin() + k, candidates_.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        candidates_.resize(k);
    } else {
        std::sort(candidates_.begin(), candidates_.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // Softmax
    float max_logit = candidates_[0].first;
    float sum = 0.0f;
    for (auto& [logit, id] : candidates_) {
        logit = expf(logit - max_logit);
        sum += logit;
    }
    for (auto& [logit, id] : candidates_) {
        logit /= sum;
    }

    // Top-P (nucleus) filtering
    if (config_.top_p < 1.0f && config_.top_p > 0.0f) {
        float cumsum = 0.0f;
        int cutoff = (int)candidates_.size();
        for (int i = 0; i < (int)candidates_.size(); i++) {
            cumsum += candidates_[i].first;
            if (cumsum >= config_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates_.resize(cutoff);

        // Renormalize
        sum = 0.0f;
        for (auto& [prob, id] : candidates_) {
            sum += prob;
        }
        for (auto& [prob, id] : candidates_) {
            prob /= sum;
        }
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    for (const auto& [prob, id] : candidates_) {
        cumsum += prob;
        if (r <= cumsum) {
            return id;
        }
    }

    // Fallback (shouldn't reach here)
    return candidates_.back().second;
}

} // namespace nt
