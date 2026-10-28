#include "engine.h"
#include "../core/device.h"
#include <cstdio>
#include <chrono>
#include <iostream>

namespace nt {

using Clock = std::chrono::high_resolution_clock;

Engine::~Engine() {
    delete draft_;
    draft_ = nullptr;
}

bool Engine::load(const std::string& model_path, int max_context, bool streaming) {
    if (!model_.load(model_path, max_context, streaming)) {
        return false;
    }

    tokenizer_.init(model_.vocab(), model_.config().bos_token_id, model_.config().eos_token_id);
    return true;
}

bool Engine::load_draft(const std::string& draft_path, int max_context) {
    fprintf(stderr, "\n=== Loading draft model for speculative decoding ===\n");
    draft_ = new Transformer();
    if (!draft_->load(draft_path, max_context, false)) {  // resident mode
        fprintf(stderr, "Failed to load draft model: %s\n", draft_path.c_str());
        delete draft_;
        draft_ = nullptr;
        return false;
    }
    fprintf(stderr, "Draft model loaded (K=%d draft tokens per iteration)\n", draft_k_);
    fprintf(stderr, "Free VRAM after draft: %.1f GB\n",
        CUDADevice::instance().free_vram() / (1024.0 * 1024 * 1024));
    return true;
}

std::string Engine::generate(const std::string& prompt, const GenerateConfig& config,
                              TokenCallback callback) {
    // Use speculative decoding if draft model is loaded
    if (draft_) {
        return generate_speculative(prompt, config, callback);
    }
    if (self_speculative_) {
        return generate_self_speculative(prompt, config, callback);
    }

    Stats stats;
    Sampler sampler;
    SamplerConfig sconfig;
    sconfig.temperature = config.temperature;
    sconfig.top_k = config.top_k;
    sconfig.top_p = config.top_p;
    sconfig.repeat_penalty = config.repeat_penalty;
    sconfig.repeat_window = config.repeat_window;
    sconfig.seed = config.seed;
    sampler.init(sconfig);

    // Tokenize prompt
    std::vector<int> tokens = tokenizer_.encode(prompt, true);
    stats.prompt_tokens = (int)tokens.size();

    if (config.verbose) {
        fprintf(stderr, "Prompt tokens: %d\n", stats.prompt_tokens);
    }

    // Prefill phase
    auto t0 = Clock::now();
    float* logits = model_.forward(tokens.data(), tokens.size(), 0);
    auto t1 = Clock::now();
    stats.prefill_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // Copy logits to CPU for sampling
    int vocab_size = model_.config().vocab_size;
    std::vector<float> logits_cpu(vocab_size);
    nt_cuda_memcpy_d2h(logits_cpu.data(), logits, vocab_size * sizeof(float));

    // Apply repeat penalty
    sampler.apply_repeat_penalty(logits_cpu.data(), vocab_size, tokens);

    // Sample first token
    int next_token = sampler.sample(logits_cpu.data(), vocab_size);
    tokens.push_back(next_token);

    std::string output;
    std::string token_text = tokenizer_.decode_token(next_token);
    output += token_text;

    if (callback) {
        if (!callback(token_text, next_token)) {
            goto done;
        }
    } else if (config.verbose) {
        fprintf(stdout, "%s", token_text.c_str());
        fflush(stdout);
    }

    // Decode loop
    {
        auto decode_start = Clock::now();
        int pos = stats.prompt_tokens;

        for (int i = 1; i < config.max_tokens; i++) {
            if (next_token == tokenizer_.eos_id()) break;

            // Forward single token
            logits = model_.forward(&next_token, 1, pos);
            pos++;

            // Copy logits to CPU
            nt_cuda_memcpy_d2h(logits_cpu.data(), logits, vocab_size * sizeof(float));

            // Apply repeat penalty
            sampler.apply_repeat_penalty(logits_cpu.data(), vocab_size, tokens);

            // Sample
            next_token = sampler.sample(logits_cpu.data(), vocab_size);
            tokens.push_back(next_token);

            token_text = tokenizer_.decode_token(next_token);
            output += token_text;
            stats.gen_tokens++;

            if (callback) {
                if (!callback(token_text, next_token)) break;
            } else if (config.verbose) {
                fprintf(stdout, "%s", token_text.c_str());
                fflush(stdout);
            }
        }

        auto decode_end = Clock::now();
        stats.decode_ms = std::chrono::duration<float, std::milli>(decode_end - decode_start).count();
    }

done:
    if (config.verbose) {
        fprintf(stdout, "\n");
        print_stats(stats);
    }

    return output;
}

// ============================================================
// Speculative decoding: draft model generates K tokens, target verifies
// ============================================================
std::string Engine::generate_speculative(const std::string& prompt, const GenerateConfig& config,
                                          TokenCallback callback) {
    Stats stats;
    int K = draft_k_;
    int vocab_size = model_.config().vocab_size;
    int eos_id = tokenizer_.eos_id();

    // Tokenize prompt
    std::vector<int> tokens = tokenizer_.encode(prompt, true);
    stats.prompt_tokens = (int)tokens.size();
    int P = stats.prompt_tokens;

    if (config.verbose) {
        fprintf(stderr, "Speculative decoding: K=%d, prompt=%d tokens\n", K, P);
        fprintf(stderr, "Draft: %s (%d layers, resident)\n",
            draft_->config().model_name.c_str(), draft_->config().n_layers);
        fprintf(stderr, "Target: %s (%d layers, %s)\n",
            model_.config().model_name.c_str(), model_.config().n_layers,
            model_.is_streaming() ? "tiered" : "resident");
    }

    // Prefill both models
    auto t0 = Clock::now();
    float* target_logits = model_.forward(tokens.data(), P, 0);
    draft_->forward(tokens.data(), P, 0);
    auto t1 = Clock::now();
    stats.prefill_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (config.verbose) {
        fprintf(stderr, "Prefill: %.1f ms (both models)\n", stats.prefill_ms);
    }

    // Sample first token from target
    std::vector<float> logits_cpu(vocab_size);
    nt_cuda_memcpy_d2h(logits_cpu.data(), target_logits, vocab_size * sizeof(float));
    int first_token = Sampler::argmax(logits_cpu.data(), vocab_size);

    std::string output;
    std::string token_text = tokenizer_.decode_token(first_token);
    output += token_text;
    tokens.push_back(first_token);
    stats.gen_tokens = 1;

    if (callback) {
        if (!callback(token_text, first_token)) goto spec_done;
    } else if (config.verbose) {
        fprintf(stdout, "%s", token_text.c_str());
        fflush(stdout);
    }

    if (first_token == eos_id) goto spec_done;

    // Speculative decode loop
    {
        auto decode_start = Clock::now();
        int pos = P;  // next KV position (first_token goes at pos in the anchor step)
        int last_accepted = first_token;
        bool stop = false;

        // Stats tracking
        int total_draft = 0;
        int total_accepted = 0;
        int spec_iterations = 0;

        // Working buffers
        std::vector<int> draft_input(K + 1);  // [anchor, d0, d1, ..., d_{K-1}]
        std::vector<float> pos_logits(vocab_size);

        while (stats.gen_tokens < config.max_tokens && !stop) {
            spec_iterations++;

            // === Draft phase ===
            // Process anchor (last_accepted) at position pos, then generate K tokens
            draft_input[0] = last_accepted;

            float* dl = draft_->forward(&last_accepted, 1, pos);
            nt_cuda_memcpy_d2h(logits_cpu.data(), dl, vocab_size * sizeof(float));
            draft_input[1] = Sampler::argmax(logits_cpu.data(), vocab_size);

            for (int k = 1; k < K; k++) {
                dl = draft_->forward(&draft_input[k], 1, pos + k);
                nt_cuda_memcpy_d2h(logits_cpu.data(), dl, vocab_size * sizeof(float));
                draft_input[k + 1] = Sampler::argmax(logits_cpu.data(), vocab_size);
            }
            // draft_input = [anchor, d0, d1, ..., d_{K-1}]
            // Draft KV now has entries at pos..pos+K-1

            // === Verify phase ===
            // Feed [anchor, d0, ..., d_{K-1}] = K+1 tokens to target at pos
            float* vl = model_.forward_verify(draft_input.data(), K + 1, pos);
            // vl[k * vocab_size] = logits at position pos+k
            // vl[0] predicts pos+1 → compare with d0 (draft_input[1])
            // vl[k] predicts pos+k+1 → compare with d_k (draft_input[k+1])
            // vl[K] predicts pos+K+1 → bonus token

            // === Accept / Reject ===
            int n_accepted = 0;
            int correction = -1;

            for (int k = 0; k < K; k++) {
                nt_cuda_memcpy_d2h(pos_logits.data(),
                    vl + k * vocab_size, vocab_size * sizeof(float));
                int target_pred = Sampler::argmax(pos_logits.data(), vocab_size);

                if (target_pred != draft_input[k + 1]) {
                    // Draft token d_k rejected
                    correction = target_pred;
                    n_accepted = k;
                    break;
                }
                n_accepted = k + 1;
            }

            bool all_accepted = (correction == -1);
            total_draft += K;

            // Output accepted draft tokens (d0..d_{n_accepted-1})
            for (int k = 0; k < n_accepted; k++) {
                int tok = draft_input[k + 1];
                token_text = tokenizer_.decode_token(tok);
                output += token_text;
                tokens.push_back(tok);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, tok)) { stop = true; break; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (tok == eos_id) { stop = true; break; }
            }

            if (stop) break;

            if (all_accepted) {
                // All K draft tokens accepted — get bonus from last position
                nt_cuda_memcpy_d2h(pos_logits.data(),
                    vl + K * vocab_size, vocab_size * sizeof(float));
                int bonus = Sampler::argmax(pos_logits.data(), vocab_size);

                token_text = tokenizer_.decode_token(bonus);
                output += token_text;
                tokens.push_back(bonus);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, bonus)) { stop = true; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (bonus == eos_id) stop = true;

                last_accepted = bonus;
                total_accepted += K + 1;
                pos += K + 1;
            } else {
                // Mismatch: output correction token
                token_text = tokenizer_.decode_token(correction);
                output += token_text;
                tokens.push_back(correction);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, correction)) { stop = true; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (correction == eos_id) stop = true;

                last_accepted = correction;
                total_accepted += n_accepted + 1;
                pos += n_accepted + 1;
            }
        }

        auto decode_end = Clock::now();
        stats.decode_ms = std::chrono::duration<float, std::milli>(decode_end - decode_start).count();

        if (config.verbose) {
            float accept_rate = total_draft > 0 ? (float)total_accepted / total_draft : 0;
            fprintf(stderr, "\n[speculative: %d iterations, %d/%d accepted (%.0f%%), "
                "avg %.1f tokens/iter]\n",
                spec_iterations, total_accepted, total_draft,
                accept_rate * 100.0f,
                stats.gen_tokens / (float)spec_iterations);
        }
    }

spec_done:
    if (config.verbose) {
        fprintf(stdout, "\n");
        print_stats(stats);
    }

    return output;
}

// ============================================================
// Self-speculative decoding: VRAM-resident layers as draft
// No extra model needed — uses partial forward through tier A
// ============================================================
std::string Engine::generate_self_speculative(const std::string& prompt, const GenerateConfig& config,
                                               TokenCallback callback) {
    Stats stats;
    int K = draft_k_;
    int vocab_size = model_.config().vocab_size;
    int eos_id = tokenizer_.eos_id();

    // Tokenize prompt
    std::vector<int> tokens = tokenizer_.encode(prompt, true);
    stats.prompt_tokens = (int)tokens.size();
    int P = stats.prompt_tokens;

    if (config.verbose) {
        fprintf(stderr, "Self-speculative decoding: K=%d, prompt=%d tokens\n", K, P);
        fprintf(stderr, "Draft: %d VRAM-resident layers (no streaming)\n",
            model_.tier_config().n_vram);
        fprintf(stderr, "Target: %d total layers (tiered)\n", model_.config().n_layers);
    }

    // Prefill with full model (all layers)
    auto t0 = Clock::now();
    float* logits = model_.forward(tokens.data(), P, 0);
    auto t1 = Clock::now();
    stats.prefill_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    if (config.verbose) {
        fprintf(stderr, "Prefill: %.1f ms (%d tokens)\n", stats.prefill_ms, P);
    }

    // Sample first token from target
    std::vector<float> logits_cpu(vocab_size);
    nt_cuda_memcpy_d2h(logits_cpu.data(), logits, vocab_size * sizeof(float));
    int first_token = Sampler::argmax(logits_cpu.data(), vocab_size);

    std::string output;
    std::string token_text = tokenizer_.decode_token(first_token);
    output += token_text;
    tokens.push_back(first_token);
    stats.gen_tokens = 1;

    if (callback) {
        if (!callback(token_text, first_token)) goto self_spec_done;
    } else if (config.verbose) {
        fprintf(stdout, "%s", token_text.c_str());
        fflush(stdout);
    }

    if (first_token == eos_id) goto self_spec_done;

    // Self-speculative decode loop
    {
        auto decode_start = Clock::now();
        int pos = P;  // anchor goes at pos
        int last_accepted = first_token;
        bool stop = false;

        int total_draft = 0;
        int total_accepted = 0;
        int spec_iterations = 0;

        std::vector<int> draft_input(K + 1);
        std::vector<float> pos_logits(vocab_size);

        while (stats.gen_tokens < config.max_tokens && !stop) {
            spec_iterations++;

            // === Draft phase: forward_draft (VRAM layers only) ===
            draft_input[0] = last_accepted;

            float* dl = model_.forward_draft(&last_accepted, 1, pos);
            nt_cuda_memcpy_d2h(logits_cpu.data(), dl, vocab_size * sizeof(float));
            draft_input[1] = Sampler::argmax(logits_cpu.data(), vocab_size);

            for (int k = 1; k < K; k++) {
                dl = model_.forward_draft(&draft_input[k], 1, pos + k);
                nt_cuda_memcpy_d2h(logits_cpu.data(), dl, vocab_size * sizeof(float));
                draft_input[k + 1] = Sampler::argmax(logits_cpu.data(), vocab_size);
            }

            // === Verify phase: full model forward ===
            float* vl = model_.forward_verify(draft_input.data(), K + 1, pos);

            // === Accept / Reject ===
            int n_accepted = 0;
            int correction = -1;

            for (int k = 0; k < K; k++) {
                nt_cuda_memcpy_d2h(pos_logits.data(),
                    vl + k * vocab_size, vocab_size * sizeof(float));
                int target_pred = Sampler::argmax(pos_logits.data(), vocab_size);

                if (target_pred != draft_input[k + 1]) {
                    correction = target_pred;
                    n_accepted = k;
                    break;
                }
                n_accepted = k + 1;
            }

            bool all_accepted = (correction == -1);
            total_draft += K;

            // Output accepted draft tokens
            for (int k = 0; k < n_accepted; k++) {
                int tok = draft_input[k + 1];
                token_text = tokenizer_.decode_token(tok);
                output += token_text;
                tokens.push_back(tok);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, tok)) { stop = true; break; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (tok == eos_id) { stop = true; break; }
            }

            if (stop) break;

            if (all_accepted) {
                // Bonus token from last verify position
                nt_cuda_memcpy_d2h(pos_logits.data(),
                    vl + K * vocab_size, vocab_size * sizeof(float));
                int bonus = Sampler::argmax(pos_logits.data(), vocab_size);

                token_text = tokenizer_.decode_token(bonus);
                output += token_text;
                tokens.push_back(bonus);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, bonus)) { stop = true; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (bonus == eos_id) stop = true;

                last_accepted = bonus;
                total_accepted += K + 1;
                pos += K + 1;
            } else {
                // Output correction token
                token_text = tokenizer_.decode_token(correction);
                output += token_text;
                tokens.push_back(correction);
                stats.gen_tokens++;

                if (callback) {
                    if (!callback(token_text, correction)) { stop = true; }
                } else if (config.verbose) {
                    fprintf(stdout, "%s", token_text.c_str());
                    fflush(stdout);
                }

                if (correction == eos_id) stop = true;

                last_accepted = correction;
                total_accepted += n_accepted + 1;
                pos += n_accepted + 1;
            }
        }

        auto decode_end = Clock::now();
        stats.decode_ms = std::chrono::duration<float, std::milli>(decode_end - decode_start).count();

        if (config.verbose) {
            float accept_rate = total_draft > 0 ? (float)total_accepted / total_draft : 0;
            fprintf(stderr, "\n[self-speculative: %d iters, %d/%d accepted (%.0f%%), "
                "avg %.1f tok/iter, draft=%d VRAM layers]\n",
                spec_iterations, total_accepted, total_draft,
                accept_rate * 100.0f,
                stats.gen_tokens / (float)spec_iterations,
                model_.tier_config().n_vram);
        }
    }

self_spec_done:
    if (config.verbose) {
        fprintf(stdout, "\n");
        print_stats(stats);
    }

    return output;
}

void Engine::chat(const GenerateConfig& config) {
    fprintf(stdout, "NTransformer Chat (type 'quit' to exit)\n");
    fprintf(stdout, "Model: %s (%d params)\n",
        model_.config().model_name.c_str(), model_.config().n_layers);
    if (draft_) {
        fprintf(stdout, "Draft: %s (speculative K=%d)\n",
            draft_->config().model_name.c_str(), draft_k_);
    }
    fprintf(stdout, "---\n");

    // Simple chat loop - no history management (stateless)
    // Each turn is independent for Phase 1
    std::string line;
    while (true) {
        fprintf(stdout, "> ");
        fflush(stdout);

        if (!std::getline(std::cin, line)) break;
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        generate(line, config);
        fprintf(stdout, "\n");
    }
}

void Engine::benchmark(const std::string& prompt, int n_tokens) {
    GenerateConfig config;
    config.max_tokens = n_tokens;
    config.temperature = 0.0f;  // greedy for reproducibility
    config.verbose = false;

    fprintf(stderr, "=== Benchmark ===\n");
    fprintf(stderr, "Prompt: \"%s\"\n", prompt.c_str());
    fprintf(stderr, "Max tokens: %d\n", n_tokens);
    if (draft_) {
        fprintf(stderr, "Mode: speculative (K=%d)\n", draft_k_);
    }

    auto t0 = Clock::now();
    std::string output = generate(prompt, config);
    auto t1 = Clock::now();

    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    fprintf(stderr, "Total time: %.1f ms\n", total_ms);
    fprintf(stderr, "Output length: %zu chars\n", output.size());
}

void Engine::print_stats(const Stats& stats) const {
    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Prompt: %d tokens, %.1f ms (%.1f tok/s)\n",
        stats.prompt_tokens, stats.prefill_ms, stats.prefill_tok_s());
    fprintf(stderr, "Decode: %d tokens, %.1f ms (%.1f tok/s)\n",
        stats.gen_tokens, stats.decode_ms, stats.decode_tok_s());

    size_t free_vram = CUDADevice::instance().free_vram();
    size_t total_vram = CUDADevice::instance().total_vram();
    fprintf(stderr, "VRAM: %.1f / %.1f GB\n",
        (total_vram - free_vram) / (1024.0 * 1024 * 1024),
        total_vram / (1024.0 * 1024 * 1024));
}

} // namespace nt
