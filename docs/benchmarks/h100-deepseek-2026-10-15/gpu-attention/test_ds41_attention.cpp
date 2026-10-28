#include "model/ds41/ds41_gpu.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nt::ds41;
namespace {
using State = std::unique_ptr<GpuAttentionState, decltype(&gpu_attention_destroy)>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float worst = 0;
uint64_t checked = 0, steps = 0;

std::vector<float> reference(const std::vector<float>& query, const std::vector<float>& raw,
                             int window, int position, int heads, int dim, int rd,
                             const std::vector<float>& comp, const std::vector<int>& ids,
                             const std::vector<float>& sinks, const std::vector<float>& cs,
                             const std::vector<float>& sn) {
    std::vector<const float*> keys;
    for (int j = std::max(0, position - window + 1); j <= position; ++j)
        keys.push_back(raw.data() + (j % window) * dim);
    for (int j : ids) keys.push_back(comp.data() + (size_t)j * dim);
    std::vector<float> out(heads * dim);
    const float scale = 1.0f / std::sqrt((float)dim);
    for (int h = 0; h < heads; ++h) {
        std::vector<float> sc(keys.size()), acc(dim, 0);
        float mx = sinks[h];
        for (size_t j = 0; j < keys.size(); ++j) {
            float dot = 0;
            for (int d = 0; d < dim; ++d) dot += query[h * dim + d] * keys[j][d];
            sc[j] = dot * scale;
            if (sc[j] > mx) mx = sc[j];
        }
        float denom = std::exp(sinks[h] - mx);
        for (size_t j = 0; j < keys.size(); ++j) {
            const float pr = std::exp(sc[j] - mx);
            denom += pr;
            for (int d = 0; d < dim; ++d) acc[d] += pr * keys[j][d];
        }
        float inv = 1.0f / denom;
        for (int d = 0; d < dim; ++d) out[h * dim + d] = acc[d] * inv;
        for (int p = 0; p < rd / 2; ++p) {
            const int i = h * dim + dim - rd + 2 * p;
            const float a = out[i], b = out[i + 1], c = cs[p], s = -sn[p];
            out[i] = a * c - b * s;
            out[i + 1] = a * s + b * c;
        }
    }
    return out;
}

void close(const std::vector<float>& actual, const std::vector<float>& expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
        float error = std::abs(actual[i] - expected[i]);
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite attention output");
        if (error > 2e-6f + 2e-5f * std::abs(expected[i])) {
            std::fprintf(stderr, "index=%zu actual=%.9g expected=%.9g error=%.9g\n", i, actual[i], expected[i], error);
            throw std::runtime_error("CPU/GPU attention mismatch");
        }
        worst = std::max(worst, error); ++checked;
    }
}

void fixture(int dim, int heads, int window, int count, int rd) {
    State states[] = {{gpu_attention_create(2, window, heads, dim), gpu_attention_destroy},
                      {gpu_attention_create(2, window, heads, dim), gpu_attention_destroy}};
    require(states[0] && states[1], "resident allocation failed");
    for (int epoch = 0; epoch < 2; ++epoch) {
        for (auto& s : states) gpu_attention_reset(s.get());
        std::vector<float> raw[2][2], compressed[2];
        for (auto& owner : raw) for (auto& layer : owner) layer.resize(window * dim);
        const int n_steps = epoch ? 9 : count;
        for (int t = 0; t < n_steps; ++t) for (int owner = 0; owner < 2; ++owner) {
            auto& comp = compressed[owner];
            if (t % 2 == 1) {
                const int index = (int)comp.size();
                comp.resize(index + dim);
                for (int d = 0; d < dim; ++d) comp[index + d] = .3f * std::sin((index + d + epoch * 71 + owner * 19) * .017f);
            }
            for (int layer = 0; layer < 2; ++layer) {
                std::vector<float> q(heads * dim), kv(dim), sinks(heads), cs(rd / 2), sn(rd / 2), actual(heads * dim, -123);
                for (int i = 0; i < heads * dim; ++i) q[i] = .4f * std::cos((i + t * 31 + owner * 17 + layer * 7 + epoch * 503) * .023f);
                for (int d = 0; d < dim; ++d) kv[d] = .3f * std::sin((d + t * 13 + owner * 7 + layer * 41 + epoch * 101) * .031f);
                std::copy(kv.begin(), kv.end(), raw[owner][layer].begin() + (t % window) * dim);
                for (int h = 0; h < heads; ++h) sinks[h] = (h % 7 - 3) * .71f;
                sinks[0] = 90; if (heads > 1) sinks[1] = -90;
                for (int p = 0; p < rd / 2; ++p) { cs[p] = std::cos(.003f * (t + 1) * (p + 1)); sn[p] = std::sin(.003f * (t + 1) * (p + 1)); }
                const int visible = std::max(0, (int)comp.size() / dim - (layer && t % 3 == 0 ? 1 : 0));
                std::vector<int> ids;
                for (int j = visible - 1; j >= 0; j -= layer + 1) ids.push_back(j);
                if (!ids.empty() && t % 5 == 0) ids.push_back(ids.front());
                AttentionRows rows{0, comp.data(), visible, ids.data(), (int)ids.size()};
                require(gpu_attention_ready(states[owner].get(), layer, t), "expected sequential state");
                require(gpu_attention_probe(states[owner].get(), layer, t, q.data(), kv.data(), cs.data(), sn.data(), rd, rows, sinks.data(), actual.data()), "attention probe unavailable");
                ++steps;
                for (float f : actual) require(std::isfinite(f), "nonfinite probe output");
                if (count < 32 || t < 2 || t == 15 || t >= window - 1 || epoch)
                    close(actual, reference(q, raw[owner][layer], window, t, heads, dim, rd, comp, ids, sinks, cs, sn));
                std::vector<float> recovered(q.size());
                require(gpu_attention_read_query(states[owner].get(), recovered.data()), "query recovery failed");
                require(std::memcmp(q.data(), recovered.data(), q.size() * 4) == 0, "recovered query changed");
                require(!gpu_attention_ready(states[owner].get(), layer, t), "duplicate position accepted");
            }
        }
    }
    gpu_attention_reset(states[0].get());
    require(!gpu_attention_ready(states[0].get(), -1, 0) && !gpu_attention_ready(states[0].get(), 0, -1), "invalid state index accepted");
    std::vector<float> q(heads * dim, .1f), kv(dim, .2f), cs(rd / 2, 1), sn(rd / 2, 0), sinks(heads, 0), out(heads * dim, -123);
    int bad = 1;
    AttentionRows invalid{0, kv.data(), 1, &bad, 1};
    require(!gpu_attention_probe(states[0].get(), 0, 0, q.data(), kv.data(), cs.data(), sn.data(), rd, invalid, sinks.data(), out.data()), "invalid compressed selection accepted");
    require(std::all_of(out.begin(), out.end(), [](float f) { return f == -123; }), "failure published output");
}
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "disabled") {
            require(!gpu_attention_create(2, 128, 64, 512), "disabled CUDA created state");
            gpu_attention_reset(nullptr); gpu_attention_destroy(nullptr);
            std::puts("PASS disabled resident attention"); return 0;
        }
        if (!gpu_backbone_enabled()) { std::puts("SKIP CUDA unavailable"); return 77; }
        require(!gpu_attention_create(1, 128, 64, 513), "unsupported head dimension accepted");
        fixture(19, 5, 3, 7, 8);
        fixture(32, 3, 5, 21, 16);
        fixture(512, 64, 128, 140, 64);
        require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution error");
        std::printf("PASS resident attention: steps=%llu values=%llu max_abs=%.9g; ring wrap, compressed sharing/growth/order, duplicate keys, sinks, reset, isolated owners, recovery and bounds\n",
            (unsigned long long)steps, (unsigned long long)checked, worst);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
