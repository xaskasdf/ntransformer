#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>


// FP16 type alias - use the device half type when compiling device code.
// hipcc defines __HIPCC__ (not __CUDACC__), so guard on both; the half
// header is routed through the compat shim (cuda_fp16.h -> hip/hip_fp16.h
// on HIP). Host .cpp translation units take the 2-byte storage branch on
// either backend, which is all they need (KV-cache element type).
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_fp16.h>
using float16_t = half;
#else
using float16_t = uint16_t;
#endif

// Full-wavefront mask for __shfl_*_sync. CUDA's mask parameter is a 32-bit
// unsigned; ROCm's __shfl_*_sync requires a 64-bit mask (it rejects an
// implicitly promoted 32-bit literal at compile time) so that one all-ones
// constant covers a full wave64 wavefront. Wave32 lanes ignore the unused
// high bits, so the same constant is correct on wave32 and wave64.
#if defined(__HIPCC__)
#define NT_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#else
#define NT_WARP_MASK 0xFFFFFFFFu
#endif

namespace nt {

// ============================================================
// Data types
// ============================================================
enum class DType : uint8_t {
    F32    = 0,
    F16    = 1,
    Q8_0   = 2,
    Q4_0   = 3,
    Q4_K_M = 4,
    Q6_K   = 5,
    Q5_K   = 6,
    Q2_K   = 7,  // For KV-cache RotateKV
    I32    = 8,
    BF16   = 9,
    MXFP4  = 10, // OCP microscaling FP4 (e2m1 values, e8m0 block scale)
    COUNT
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::I32:  return 4;
        // Quantized types return block size
        case DType::Q8_0:   return sizeof(uint16_t) + 32;         // FP16 scale + 32 bytes = 34
        case DType::Q4_0:   return sizeof(float16_t) + 16;       // half scale + 16 bytes (32 nibbles)
        case DType::Q4_K_M: return 144;                          // K-quant block
        case DType::Q5_K:   return 176;                          // K-quant 5-bit block
        case DType::Q6_K:   return 210;
        case DType::Q2_K:   return 84;
        case DType::BF16:   return 2;
        case DType::MXFP4:  return 17;                           // 1 byte e8m0 + 16 bytes nibbles
        default: return 0;
    }
}

inline size_t dtype_block_size(DType dt) {
    switch (dt) {
        case DType::F32:    return 1;
        case DType::F16:    return 1;
        case DType::I32:    return 1;
        case DType::Q8_0:   return 32;
        case DType::Q4_0:   return 32;
        case DType::Q4_K_M: return 256;
        case DType::Q5_K:   return 256;
        case DType::Q6_K:   return 256;
        case DType::Q2_K:   return 256;
        case DType::MXFP4:  return 32;
        default: return 1;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:    return "F32";
        case DType::F16:    return "F16";
        case DType::Q8_0:   return "Q8_0";
        case DType::Q4_0:   return "Q4_0";
        case DType::Q4_K_M: return "Q4_K_M";
        case DType::Q5_K:   return "Q5_K";
        case DType::Q6_K:   return "Q6_K";
        case DType::Q2_K:   return "Q2_K";
        case DType::I32:    return "I32";
        case DType::BF16:   return "BF16";
        case DType::MXFP4:  return "MXFP4";
        default: return "UNKNOWN";
    }
}

// Bytes needed for `n` elements of a given dtype
inline size_t dtype_row_size(DType dt, size_t n) {
    size_t bs = dtype_block_size(dt);
    assert(n % bs == 0);
    return (n / bs) * dtype_size(dt);
}

// ============================================================
// Quantization block layouts (GGUF compatible)
// ============================================================

// Q4_0: 32 weights per block
// Layout: half scale, 16 bytes of packed nibbles
struct BlockQ4_0 {
    uint16_t d;        // FP16 scale (delta)
    uint8_t  qs[16];   // nibbles: 32 x 4-bit
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 size mismatch");

// Q8_0: 32 weights per block
// Layout: FP16 scale, 32 x int8 (GGML uses fp16 scale, not float)
struct BlockQ8_0 {
    uint16_t d;        // FP16 scale (delta)
    int8_t   qs[32];   // quantized values
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size mismatch");

// Q4_K_M: 256 weights per block (super-block with sub-blocks)
// Simplified layout for GGUF compatibility
struct BlockQ4_K {
    uint16_t d;            // super-block scale (FP16)
    uint16_t dmin;         // super-block min (FP16)
    uint8_t  scales[12];   // sub-block scales and mins
    uint8_t  qs[128];      // nibbles: 256 x 4-bit
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K size mismatch");

// Q5_K: 256 weights per block (super-block with 5-bit quantization)
// Like Q4_K but with additional high bit stored in qh[32]
struct BlockQ5_K {
    uint16_t d;            // super-block scale (FP16)
    uint16_t dmin;         // super-block min (FP16)
    uint8_t  scales[12];   // sub-block scales and mins (same packing as Q4_K)
    uint8_t  qh[32];      // high bits: 256 bits for 5th bit of each weight
    uint8_t  ql[128];     // low 4 bits: 256 x 4-bit nibbles
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K size mismatch");

// Q6_K: 256 weights per block
struct BlockQ6_K {
    uint8_t  ql[128];     // lower 4 bits
    uint8_t  qh[64];      // upper 2 bits
    int8_t   scales[16];  // scales
    uint16_t d;            // super-block scale (FP16)
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K size mismatch");

// MXFP4 (OCP microscaling): 32 weights per block
// Layout: 1 byte e8m0 scale (2^(e-127), e=0 -> 2^-128), 16 bytes = 32 x 4-bit e2m1 values
struct BlockMXFP4 {
    uint8_t e;         // E8M0 scale
    uint8_t qs[16];    // nibbles: 32 x 4-bit (low nibbles first half, high nibbles second half)
};
static_assert(sizeof(BlockMXFP4) == 17, "BlockMXFP4 size mismatch");

// e2m1 values, doubled (the block scale is halved when dequantizing)
inline constexpr int8_t kMXFP4Values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

// 2^(x-128): e8m0 scale for MXFP4 (block scale is 2^(e-127), values doubled -> half scale)
inline float e8m0_to_fp32_mxfp4(uint8_t x) {
    if (x < 2) {
        uint32_t bits = (x == 0) ? 0x00200000u : 0x00400000u;  // 2^-128, 2^-127
        float r;
        memcpy(&r, &bits, sizeof(float));
        return r;
    }
    uint32_t bits = (uint32_t)x << 23;
    float r;
    memcpy(&r, &bits, sizeof(float));
    return r / 2.0f;
}

// FP16 <-> FP32 conversions usable from host code (device code uses CUDA's half)
inline float fp16_to_fp32_host(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    memcpy(&r, &f, sizeof(float));
    return r;
}

inline float bf16_to_fp32_host(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float r;
    memcpy(&r, &f, sizeof(float));
    return r;
}

// ============================================================
// Device enum
// ============================================================
enum class Device : uint8_t {
    CPU  = 0,
    CUDA = 1,
};

// ============================================================
// GGUF constants
// ============================================================
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_3 = 3;

enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGML tensor types -> our DType mapping
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    MXFP4   = 39,
};

inline DType ggml_to_dtype(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return DType::F32;
        case GGMLType::F16:  return DType::F16;
        case GGMLType::Q8_0: return DType::Q8_0;
        case GGMLType::Q4_0: return DType::Q4_0;
        case GGMLType::Q4_K: return DType::Q4_K_M;
        case GGMLType::Q5_K: return DType::Q5_K;
        case GGMLType::Q6_K: return DType::Q6_K;
        case GGMLType::Q2_K: return DType::Q2_K;
        case GGMLType::I32:  return DType::I32;
        case GGMLType::BF16: return DType::BF16;
        case GGMLType::MXFP4: return DType::MXFP4;
        default:             return DType::F32;  // fallback
    }
}

// ============================================================
// Utility macros
// ============================================================
#define NT_CHECK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "NT ERROR: %s at %s:%d\n", msg, __FILE__, __LINE__); abort(); } } while(0)

#if defined(__CUDACC__) || defined(__HIPCC__)
#define NT_CUDA_CHECK(err) \
    do { cudaError_t e = (err); if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); abort(); \
    } } while(0)
#endif

// ============================================================
// Cross-platform aligned memory allocation
// ============================================================

inline void* nt_aligned_alloc(size_t alignment, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return aligned_alloc(alignment, size);
#endif
}

inline void nt_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

} // namespace nt
