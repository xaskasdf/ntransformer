#include "ffn.h"
#include "../cuda/kernels.h"
#include <cstdio>

namespace nt {

void FFN::init(
    const ModelConfig& config,
    Tensor w_gate, Tensor w_up, Tensor w_down,
    int layer_idx
) {
    hidden_size_       = config.hidden_size;
    intermediate_size_ = config.intermediate_size;
    layer_idx_         = layer_idx;

    gate_dtype_ = w_gate.dtype();
    up_dtype_   = w_up.dtype();
    down_dtype_ = w_down.dtype();

    if (w_gate.device() == Device::CPU) {
        w_gate_ = w_gate.to(Device::CUDA);
        w_up_   = w_up.to(Device::CUDA);
        w_down_ = w_down.to(Device::CUDA);
    } else {
        w_gate_ = std::move(w_gate);
        w_up_   = std::move(w_up);
        w_down_ = std::move(w_down);
    }
}

size_t FFN::workspace_size(int seq_len) const {
    // gate_buf: [seq_len * intermediate_size]
    // up_buf:   [seq_len * intermediate_size]
    // silu_buf: [seq_len * intermediate_size]  (reuse gate_buf)
    return 2 * seq_len * intermediate_size_ * sizeof(float);
}

void FFN::forward(
    float* output,
    const float* input,
    int seq_len,
    void* stream
) {
    NT_CHECK(workspace_ != nullptr, "FFN workspace not set");

    float* gate_buf = workspace_;
    float* up_buf = gate_buf + seq_len * intermediate_size_;

    for (int t = 0; t < seq_len; t++) {
        const float* inp = input + t * hidden_size_;
        float* gate_out = gate_buf + t * intermediate_size_;
        float* up_out = up_buf + t * intermediate_size_;
        float* out = output + t * hidden_size_;

        // gate = Wgate * x
        cuda::launch_gemv(gate_out, w_gate_.data(), inp,
            intermediate_size_, hidden_size_, gate_dtype_, stream);

        // up = Wup * x
        cuda::launch_gemv(up_out, w_up_.data(), inp,
            intermediate_size_, hidden_size_, up_dtype_, stream);

        // silu_out = SiLU(gate) * up  (in-place into gate_buf)
        cuda::launch_silu_mul(gate_out, gate_out, up_out, intermediate_size_, stream);

        // output = Wdown * silu_out
        cuda::launch_gemv(out, w_down_.data(), gate_out,
            hidden_size_, intermediate_size_, down_dtype_, stream);
    }
}

} // namespace nt
