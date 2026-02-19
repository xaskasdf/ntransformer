#pragma once

#include "../core/tensor.h"

namespace nt {

// ============================================================
// RMSNorm layer
// ============================================================
class RMSNorm {
public:
    void init(Tensor weight, float eps);

    // Streaming mode: set config without owning weight data
    void init_streaming(int hidden_size, float eps);

    // Point weight at preloaded GPU buffer (non-owning)
    void set_weight(const float* gpu_ptr);

    // In-place normalize and scale (overwrites output buffer)
    void forward(float* output, const float* input, int batch_size, void* stream);

    int hidden_size() const { return hidden_size_; }

private:
    Tensor weight_;            // [hidden_size] on GPU (owning, for resident mode)
    const float* weight_ptr_ = nullptr;  // Non-owning pointer (for streaming mode)
    float eps_ = 1e-5f;
    int hidden_size_ = 0;
};

} // namespace nt
