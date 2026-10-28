#include "ntrpu_norm_q8_abi1_r1.h"
#include "ntrpu_rmsnorm_fp32_r1.h"
#include "ntrpu_quantize_q8_0.h"
#include <cstring>

using namespace nt::ntrpu::norm_q8_r1;

extern "C" uint32_t ntrpu_norm_q8_abi1_r1(
    const uint32_t input[2048], const uint32_t weight[2048],
    uint64_t output[512], uint32_t elements, uint32_t epsilon_bits) {
#pragma HLS INTERFACE mode=ap_memory port=input storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=weight storage_type=ram_1p depth=2048
#pragma HLS INTERFACE mode=ap_memory port=output storage_type=ram_1p depth=512
#pragma HLS INTERFACE mode=ap_none port=elements
#pragma HLS INTERFACE mode=ap_none port=epsilon_bits
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
    if (elements == 0 || elements > MAX_ELEMENTS || elements % 32 != 0) return INVALID_SHAPE;
    uint32_t normalized[MAX_ELEMENTS];
    uint64_t packed[MAX_WORDS];
#pragma HLS BIND_STORAGE variable=normalized type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=packed type=ram_1p impl=bram
    const uint32_t norm_status = ntrpu_rmsnorm_fp32_r1(input, weight, normalized, elements, epsilon_bits);
    if (norm_status != OK) return norm_status;
    for (uint32_t block = 0; block < elements / 32; ++block) {
#pragma HLS PIPELINE off
        float values[32];
        int8_t quantized[32];
        double scales[1];
        for (uint32_t lane = 0; lane < 32; ++lane) {
#pragma HLS PIPELINE II=1
            const uint32_t bits = normalized[block * 32 + lane];
            std::memcpy(&values[lane], &bits, sizeof(bits));
        }
        // Reuse the unchanged, independently checked block quantization contract.
        if (ntrpu_quantize_q8_0(values, quantized, scales, 32) != 0) return QUANTIZE_ERROR;
        for (uint32_t word = 0; word < 8; ++word) {
#pragma HLS PIPELINE off
            uint64_t bits = 0;
            if (word < 4) {
                for (uint32_t byte = 0; byte < 8; ++byte) {
#pragma HLS UNROLL
                    bits |= uint64_t(uint8_t(quantized[word * 8 + byte])) << (byte * 8);
                }
            } else if (word == 4) {
                std::memcpy(&bits, &scales[0], sizeof(bits));
            }
            packed[block * 8 + word] = bits;
        }
    }
    // No externally visible write until all stages have succeeded.
    for (uint32_t word = 0; word < elements / 4; ++word) {
#pragma HLS PIPELINE II=1
        output[word] = packed[word];
    }
    return OK;
}
