#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>


// FP16 type alias - use CUDA's half when compiling with nvcc
#ifdef __CUDACC__
#include <cuda_fp16.h>
using float16_t = half;
#else
using float16_t = uint16_t;
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

// Q2_K: 256 weights per super-block (2 bits per weight)
// Layout: scales[16] + qs[64] + d(FP16) + dmin(FP16)
// scales[i]: lower nibble = scale for sub-block i, upper nibble = min for sub-block i
// qs[i]: packed 2-bit values, 4 per byte
// d: FP16 super-block scale (applied to scale nibbles)
// dmin: FP16 super-block min scale (applied to min nibbles)
// Dequant: w = d * (scale_nibble * q) - dmin * min_nibble
struct BlockQ2_K {
    uint8_t  scales[16];  // 4-bit scales+mins: lower nibble=scale, upper nibble=min
    uint8_t  qs[64];      // 2-bit quantized values, 4 per byte (256 weights total)
    uint16_t d;           // FP16 super-block scale (for scale nibbles)
    uint16_t dmin;        // FP16 super-block min delta (for min nibbles)
};
static_assert(sizeof(BlockQ2_K) == 84, "BlockQ2_K size mismatch");

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
        default:             return DType::F32;  // fallback
    }
}

// ============================================================
// Utility macros
// ============================================================
#define NT_CHECK(cond, msg) \
    do { if (!(cond)) { fprintf(stderr, "NT ERROR: %s at %s:%d\n", msg, __FILE__, __LINE__); abort(); } } while(0)

#ifdef __CUDACC__
#define NT_CUDA_CHECK(err) \
    do { cudaError_t e = (err); if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); abort(); \
    } } while(0)
#endif

// ============================================================
// Cross-platform aligned memory allocation
// ============================================================
inline void* nt_aligned_alloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, size);
}

inline void nt_aligned_free(void* ptr) {
    ::free(ptr);
}

} // namespace nt
