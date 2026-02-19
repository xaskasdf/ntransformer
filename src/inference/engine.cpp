#include "engine.h"
#include "../core/device.h"
#include <cstdio>
#include <chrono>
#include <iostream>

namespace nt {

using Clock = std::chrono::high_resolution_clock;

bool Engine::load(const std::string& model_path, int max_context, bool streaming) {
    if (!model_.load(model_path, max_context, streaming)) {
        return false;
    }

    tokenizer_.init(model_.vocab(), model_.config().bos_token_id, model_.config().eos_token_id);
    return true;
}

std::string Engine::generate(const std::string& prompt, const GenerateConfig& config,
                              TokenCallback callback) {
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

void Engine::chat(const GenerateConfig& config) {
    fprintf(stdout, "NTransformer Chat (type 'quit' to exit)\n");
    fprintf(stdout, "Model: %s (%d params)\n",
        model_.config().model_name.c_str(), model_.config().n_layers);
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
