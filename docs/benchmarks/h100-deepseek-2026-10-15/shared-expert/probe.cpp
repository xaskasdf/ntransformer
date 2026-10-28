#include "model/ds41/ds41_forward.h"
#include "model/ds41/ds41_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace nt::ds41;
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("model output.bin");
        Model model; std::string error;
        if (!model.load(argv[1], error)) throw std::runtime_error(error);
        auto head = model.tensor("output.weight");
        if (!gpu_lm_head_init(head.data, head.rows(), head.ne0())) throw std::runtime_error("GPU head unavailable");
        Session s(model); std::vector<float> outputs;
        auto save = [&](int seed, int length, const char* phase, const float* logits) {
            for (int i = 0; i < s.vocab_size(); ++i)
                if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite probe logits");
            int token = int(std::max_element(logits, logits + s.vocab_size()) - logits);
            std::printf("{\"seed\":%d,\"length\":%d,\"phase\":\"%s\",\"argmax\":%d}\n", seed, length, phase, token);
            std::fflush(stdout); outputs.insert(outputs.end(), logits, logits + s.vocab_size());
        };
        for (int seed = 0; seed < 2; ++seed) {
            s.reset(); const float* logits = nullptr;
            for (int i = 0; i < 129; ++i) {
                bool emit = i == 0 || i == 15 || i == 31 || i == 63 || i == 127 || i == 128;
                int token = i == 0 ? model.cfg().bos_id : 3 + (i * 37 + seed * 5003) % (s.vocab_size() - 3);
                logits = s.forward(token, emit);
                if (emit) save(seed, i + 1, "prefill", logits);
            }
            for (int i = 0; i < 4; ++i) {
                int token = int(std::max_element(logits, logits + s.vocab_size()) - logits);
                logits = s.forward(token); save(seed, 130 + i, "decode", logits);
            }
        }
        auto stats = gpu_moe_stats();
        bool shared = !std::getenv("DS41_SHARED_GPU") || std::atoi(std::getenv("DS41_SHARED_GPU")) != 0;
        if (stats.calls != 10640 || stats.shared_calls != (shared ? 10640u : 0u))
            throw std::runtime_error("incomplete GPU coverage");
        FILE* file = std::fopen(argv[2], "wb");
        if (!file) throw std::runtime_error("cannot open output");
        size_t n = std::fwrite(outputs.data(), sizeof(float), outputs.size(), file);
        int closed = std::fclose(file);
        if (n != outputs.size() || closed) throw std::runtime_error("short output");
        std::printf("PASS %zu finite logits, two contexts through window rollover, full GPU coverage\n", outputs.size());
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
