#pragma once

#include "../core/tensor.h"

namespace nt {

// ============================================================
// RMSNorm layer
// ============================================================
class RMSNorm {
public:
    void init(Tensor weight, float eps);

    // In-place normalize and scale (overwrites output buffer)
    void forward(float* output, const float* input, int batch_size, void* stream);

    int hidden_size() const { return hidden_size_; }

private:
    Tensor weight_;       // [hidden_size] on GPU
    float eps_ = 1e-5f;
    int hidden_size_ = 0;
};

} // namespace nt
