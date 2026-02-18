#pragma once

#include "../core/tensor.h"
#include "config.h"

namespace nt {

// ============================================================
// SwiGLU Feed-Forward Network
// output = Wo * (SiLU(Wgate * x) * Wup * x)
//
// Wgate: [intermediate_size, hidden_size]
// Wup:   [intermediate_size, hidden_size]
// Wdown: [hidden_size, intermediate_size]
// ============================================================

class FFN {
public:
    void init(
        const ModelConfig& config,
        Tensor w_gate, Tensor w_up, Tensor w_down,
        int layer_idx
    );

    void forward(
        float* output,
        const float* input,
        int seq_len,
        void* stream
    );

    size_t workspace_size(int seq_len) const;
    void set_workspace(float* ptr) { workspace_ = ptr; }

private:
    Tensor w_gate_, w_up_, w_down_;
    DType gate_dtype_, up_dtype_, down_dtype_;

    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int layer_idx_ = 0;

    float* workspace_ = nullptr;
};

} // namespace nt
