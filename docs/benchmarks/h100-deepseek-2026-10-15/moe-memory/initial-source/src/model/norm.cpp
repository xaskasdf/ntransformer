#include "norm.h"
#include "../cuda/kernels.h"

namespace nt {

void RMSNorm::init(Tensor weight, float eps) {
    hidden_size_ = weight.size(0);
    eps_ = eps;
    // Transfer weight to GPU if not already there
    if (weight.device() == Device::CPU) {
        weight_ = weight.to(Device::CUDA);
    } else {
        weight_ = std::move(weight);
    }
    weight_ptr_ = weight_.data_as<float>();
}

void RMSNorm::init_streaming(int hidden_size, float eps) {
    hidden_size_ = hidden_size;
    eps_ = eps;
}

void RMSNorm::set_weight(const float* gpu_ptr) {
    weight_ptr_ = gpu_ptr;
}

void RMSNorm::forward(float* output, const float* input, int batch_size, void* stream) {
    const float* w = weight_ptr_ ? weight_ptr_ : weight_.data_as<float>();
    cuda::launch_rmsnorm(
        output, input,
        w,
        batch_size, hidden_size_, eps_,
        stream
    );
}

} // namespace nt
