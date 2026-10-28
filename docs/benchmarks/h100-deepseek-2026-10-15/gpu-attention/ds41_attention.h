#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace nt::ds41::detail {
// Device pointers only. Raw rows form a chronological ring; selected compressed
// rows follow them in the supplied order. K and V share the same latent vector.
void launch_attention(const float* query, const float* raw, int window, int64_t position,
                      const float* compressed, const int* selected, int n_selected,
                      const float* sinks, const float* cos_t, const float* sin_t,
                      int heads, int dim, int rope_dim, float scale,
                      float* scores, float* out, cudaStream_t stream);
}
