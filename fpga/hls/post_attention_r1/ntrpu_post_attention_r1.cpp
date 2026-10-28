#include "ntrpu_post_attention_r1.h"
#include "ntrpu_residual_fp32_r1.h"
#include "ntrpu_norm_q8_abi1_r1.h"

using namespace nt::ntrpu::post_attention_r1;

extern "C" uint32_t ntrpu_post_attention_r1(
    const uint32_t residual_input[2048], const uint32_t branch[2048],
    const uint32_t norm_weight[2048], uint32_t residual_output[2048],
    uint64_t ffn_input[512], uint32_t elements, uint32_t epsilon_bits) {
#pragma HLS INTERFACE mode=ap_memory port=residual_input storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=branch storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=norm_weight storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=residual_output storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=ffn_input storage_type=ram_1p depth=512
#pragma HLS INTERFACE mode=ap_none port=elements
#pragma HLS INTERFACE mode=ap_none port=epsilon_bits
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (elements == 0 || elements > MAX_ELEMENTS || elements % 32 != 0)
        return INVALID_SHAPE;
    uint32_t residual[MAX_ELEMENTS];
    uint64_t packed[MAX_WORDS];
#pragma HLS BIND_STORAGE variable=residual type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=packed type=ram_1p impl=bram
    const uint32_t residual_status = ntrpu_residual_fp32_r1(
        residual_input, branch, residual, elements);
    if (residual_status != 0) return residual_status;
    const uint32_t norm_status = ntrpu_norm_q8_abi1_r1(
        residual, norm_weight, packed, elements, epsilon_bits);
    if (norm_status != 0) return norm_status == 1 ? INVALID_SHAPE : norm_status + 2;
    // Both stages have succeeded. The owner still withholds visibility until
    // completion, because these two copy loops are not simultaneous writes.
    for (uint32_t i = 0; i < elements; ++i) {
#pragma HLS PIPELINE II=1
        residual_output[i] = residual[i];
    }
    for (uint32_t i = 0; i < elements / 4; ++i) {
#pragma HLS PIPELINE II=1
        ffn_input[i] = packed[i];
    }
    return OK;
}
