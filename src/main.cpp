#include "inference/engine.h"
#include <cstdio>
#include <cstring>
#include <string>

void print_usage(const char* prog) {
    fprintf(stderr, "NTransformer - High-Efficiency LLM Inference Engine\n\n");
    fprintf(stderr, "Usage: %s [options] -m <model.gguf>\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <path>       Path to GGUF model file (required)\n");
    fprintf(stderr, "  -p, --prompt <text>      Prompt text (default: interactive mode)\n");
    fprintf(stderr, "  -n, --n-tokens <int>     Max tokens to generate (default: 256)\n");
    fprintf(stderr, "  -t, --temperature <float> Temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-k <int>            Top-K sampling (default: 40)\n");
    fprintf(stderr, "  --top-p <float>          Top-P nucleus sampling (default: 0.9)\n");
    fprintf(stderr, "  --repeat-penalty <float> Repeat penalty (default: 1.1)\n");
    fprintf(stderr, "  -c, --ctx-size <int>     Context size (default: 4096)\n");
    fprintf(stderr, "  --seed <int>             Random seed (default: 42)\n");
    fprintf(stderr, "  --streaming              SLEP streaming mode (stream layers from CPU via PCIe)\n");
    fprintf(stderr, "  --early-exit <float>     Early exit threshold (0=off, 0.9999=aggressive)\n");
    fprintf(stderr, "  --skip-threshold <float> Layer skip threshold (0=off, 0.985=moderate)\n");
    fprintf(stderr, "  --benchmark              Run benchmark mode\n");
    fprintf(stderr, "  --chat                   Interactive chat mode\n");
    fprintf(stderr, "  -v, --verbose            Verbose output\n");
    fprintf(stderr, "  -h, --help               Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt;
    int max_context = 4096;
    float early_exit_threshold = 0.0f;
    float skip_threshold = 0.0f;
    bool benchmark_mode = false;
    bool chat_mode = false;
    bool streaming_mode = false;

    nt::GenerateConfig config;
    config.verbose = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i < argc) prompt = argv[i];
        } else if (arg == "-n" || arg == "--n-tokens") {
            if (++i < argc) config.max_tokens = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--temperature") {
            if (++i < argc) config.temperature = std::stof(argv[i]);
        } else if (arg == "--top-k") {
            if (++i < argc) config.top_k = std::stoi(argv[i]);
        } else if (arg == "--top-p") {
            if (++i < argc) config.top_p = std::stof(argv[i]);
        } else if (arg == "--repeat-penalty") {
            if (++i < argc) config.repeat_penalty = std::stof(argv[i]);
        } else if (arg == "--seed") {
            if (++i < argc) config.seed = std::stoull(argv[i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i < argc) max_context = std::stoi(argv[i]);
        } else if (arg == "--streaming") {
            streaming_mode = true;
        } else if (arg == "--early-exit") {
            if (++i < argc) early_exit_threshold = std::stof(argv[i]);
        } else if (arg == "--skip-threshold") {
            if (++i < argc) skip_threshold = std::stof(argv[i]);
        } else if (arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "--chat") {
            chat_mode = true;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: model path required (-m)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    nt::Engine engine;
    if (!engine.load(model_path, max_context, streaming_mode)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    if (early_exit_threshold > 0.0f) {
        engine.model().set_early_exit(early_exit_threshold);
    }
    if (skip_threshold > 0.0f) {
        engine.model().set_layer_skip(skip_threshold);
    }

    // Run
    if (benchmark_mode) {
        std::string bench_prompt = prompt.empty() ? "The meaning of life is" : prompt;
        engine.benchmark(bench_prompt, config.max_tokens);
    } else if (chat_mode || prompt.empty()) {
        engine.chat(config);
    } else {
        engine.generate(prompt, config);
    }

    return 0;
}
