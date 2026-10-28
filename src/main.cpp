#include "inference/engine.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include "model/ds41/ds41_forward.h"
#include "model/glm5/glm5_forward.h"

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
    fprintf(stderr, "  --peer-gpu <int>         Cache overflow layers on another visible GPU via CUDA P2P\n");
    fprintf(stderr, "  --local-layers <int>     Cap resident layers on compute GPU (requires --peer-gpu)\n");
    fprintf(stderr, "  --peer-layers <int>      Cap layers cached on peer GPU (requires --peer-gpu)\n");
    fprintf(stderr, "  --draft-model <path>     Draft model for speculative decoding (e.g. 8B)\n");
    fprintf(stderr, "  --draft-k <int>          Draft tokens per iteration (default: 5)\n");
    fprintf(stderr, "  --self-spec              Self-speculative: use VRAM layers as draft (no extra model)\n");
    fprintf(stderr, "  --early-exit <float>     Early exit threshold (0=off, 0.9999=aggressive)\n");
    fprintf(stderr, "  --skip-threshold <float> Layer skip threshold (0=off, 0.985=moderate)\n");
    fprintf(stderr, "  --requant-q4k            Requantize Q6_K→Q4_K_M for tier B (31%% less H2D)\n");
    fprintf(stderr, "  --delta-model <path>     Delta-encoded .ntd file (33x less H2D bandwidth)\n");
    fprintf(stderr, "  --benchmark              Run benchmark mode\n");
    fprintf(stderr, "  --chat                   Interactive chat mode\n");
    fprintf(stderr, "  -v, --verbose            Verbose output\n");
    fprintf(stderr, "  -h, --help               Show this help\n");
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string draft_model_path;
    std::string delta_model_path;
    std::string prompt;
    int max_context = 4096;
    int draft_k = 5;
    int peer_gpu = -1;
    int max_local_layers = -1;
    int max_peer_layers = -1;
    float early_exit_threshold = 0.0f;
    float skip_threshold = 0.0f;
    bool benchmark_mode = false;
    bool chat_mode = false;
    bool streaming_mode = false;
    bool requant_q4k = false;
    bool self_speculative = false;
    std::vector<int> ds41_token_ids;
    std::string ds41_dump_logits;

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
        } else if (arg == "--peer-gpu" || arg == "--local-layers" || arg == "--peer-layers") {
            if (++i >= argc) {
                fprintf(stderr, "Missing value for %s\n", arg.c_str());
                return 1;
            }
            try {
                size_t consumed = 0;
                int value = std::stoi(argv[i], &consumed);
                if (value < 0 || consumed != std::strlen(argv[i])) throw std::invalid_argument("nonnegative integer required");
                if (arg == "--peer-gpu") peer_gpu = value;
                else if (arg == "--local-layers") max_local_layers = value;
                else max_peer_layers = value;
            } catch (const std::exception&) {
                fprintf(stderr, "Invalid value for %s (requires a nonnegative integer)\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--draft-model") {
            if (++i < argc) draft_model_path = argv[i];
        } else if (arg == "--draft-k") {
            if (++i < argc) draft_k = std::stoi(argv[i]);
        } else if (arg == "--early-exit") {
            if (++i < argc) early_exit_threshold = std::stof(argv[i]);
        } else if (arg == "--skip-threshold") {
            if (++i < argc) skip_threshold = std::stof(argv[i]);
        } else if (arg == "--delta-model") {
            if (++i < argc) delta_model_path = argv[i];
        } else if (arg == "--requant-q4k") {
            requant_q4k = true;
        } else if (arg == "--self-spec") {
            self_speculative = true;
        } else if (arg == "--token-ids") {
            if (++i < argc) {
                std::string s = argv[i];
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t comma = s.find(',', pos);
                    std::string part = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    if (!part.empty()) ds41_token_ids.push_back(std::stoi(part));
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
            }
        } else if (arg == "--dump-logits") {
            if (++i < argc) ds41_dump_logits = argv[i];
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

    // GLM-5.3-Flash ("glm5-next") has its own CPU reference implementation
    {
        nt::glm5::GenOptions gopt;
        gopt.prompt         = prompt;
        gopt.max_tokens     = config.max_tokens;
        gopt.temperature    = config.temperature;
        gopt.top_k          = config.top_k;
        gopt.top_p          = config.top_p;
        gopt.repeat_penalty = config.repeat_penalty;
        gopt.repeat_window  = config.repeat_window;
        gopt.seed           = config.seed;
        gopt.ctx            = max_context;
        gopt.token_ids      = ds41_token_ids;
        gopt.dump_logits    = ds41_dump_logits;
        int rc = nt::glm5::run_cli(model_path, gopt);
        if (rc >= 0) {
            return rc;
        }
    }

    // DeepSeek-V4.1 ("deepseek41") has its own CPU reference implementation
    {
        nt::ds41::GenOptions dopt;
        dopt.prompt         = prompt;
        dopt.max_tokens     = config.max_tokens;
        dopt.temperature    = config.temperature;
        dopt.top_k          = config.top_k;
        dopt.top_p          = config.top_p;
        dopt.repeat_penalty = config.repeat_penalty;
        dopt.repeat_window  = config.repeat_window;
        dopt.seed           = config.seed;
        dopt.ctx            = max_context;
        dopt.token_ids      = ds41_token_ids;
        dopt.dump_logits    = ds41_dump_logits;
        int rc = nt::ds41::run_cli(model_path, dopt);
        if (rc >= 0) {
            return rc;
        }
    }

    if (peer_gpu < 0 && (max_local_layers >= 0 || max_peer_layers >= 0)) {
        fprintf(stderr, "--local-layers and --peer-layers require --peer-gpu\n");
        return 1;
    }
    if (peer_gpu >= 0 && !delta_model_path.empty()) {
        fprintf(stderr, "--peer-gpu does not support --delta-model\n");
        return 1;
    }
    if (peer_gpu >= 0) streaming_mode = true;

    // Auto-enable streaming when using speculative modes
    if (!draft_model_path.empty() && !streaming_mode) {
        fprintf(stderr, "Note: --draft-model implies --streaming for target model\n");
        streaming_mode = true;
    }
    if (self_speculative && !streaming_mode) {
        fprintf(stderr, "Note: --self-spec implies --streaming\n");
        streaming_mode = true;
    }

    // Load draft model first (if specified) so it gets VRAM priority
    // Then load target model — tiered mode auto-adjusts to remaining VRAM
    nt::Engine engine;
    if (peer_gpu >= 0 && !engine.model().set_peer_cache(peer_gpu, max_local_layers, max_peer_layers)) {
        return 1;
    }

    if (!draft_model_path.empty()) {
        engine.set_draft_k(draft_k);
        // Draft model loads first in resident mode (all VRAM)
        if (!engine.load_draft(draft_model_path, max_context)) {
            fprintf(stderr, "Failed to load draft model: %s\n", draft_model_path.c_str());
            return 1;
        }
    }

    if (requant_q4k) {
        engine.model().set_requant_q4k(true);
    }

    if (!delta_model_path.empty()) {
        if (!streaming_mode) {
            fprintf(stderr, "Note: --delta-model implies --streaming\n");
            streaming_mode = true;
        }
        engine.model().set_delta_model(delta_model_path);
    }

    if (!engine.load(model_path, max_context, streaming_mode)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    if (self_speculative) {
        engine.set_self_speculative(true);
        engine.set_draft_k(draft_k);
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

    const auto& peer_cache = engine.model().peer_cache();
    if (peer_cache.block_count() > 0) {
        fprintf(stderr, "Peer cache: GPU %d, %d blocks, %.3f GiB allocated, "
                "%llu P2P copies, %.3f GiB transferred\n", peer_cache.device_id(),
                peer_cache.block_count(), peer_cache.allocated_bytes() / (1024.0 * 1024 * 1024),
                static_cast<unsigned long long>(peer_cache.transfer_count()),
                peer_cache.transferred_bytes() / (1024.0 * 1024 * 1024));
    }
    return 0;
}
