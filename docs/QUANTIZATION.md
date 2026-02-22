# Quantization Formats

Supported GGUF quantization formats, their block layouts, and memory usage.

## Format Summary

| Format | Bits/weight | Block size | Bytes/block | GEMV kernel | Notes |
|--------|------------|-----------|-------------|-------------|-------|
| Q2_K | 2.6 | 256 | 84 | `gemv_q2_k_kernel` | 2-bit weights + 4-bit scales/mins |
| Q4_0 | 4.5 | 32 | 18 | `gemv_q4_0_kernel` | Simple uniform 4-bit |
| Q8_0 | 8.5 | 32 | 34 | `gemv_q8_0_kernel` | Near-lossless, fastest for small models |
| Q4_K_M | 4.5 | 256 | 144 | `gemv_q4_k_kernel` | Super-block with mixed Q4_K + Q5_K |
| Q5_K | 5.5 | 256 | 176 | `gemv_q5_k_kernel` | 5-bit super-block |
| Q6_K | 6.6 | 256 | 210 | `gemv_q6_k_kernel` | 6-bit super-block, high quality |
| F16 | 16 | 1 | 2 | `gemv_f16_kernel` | Half precision |
| F32 | 32 | 1 | 4 | `gemv_f32_kernel` | Single precision |

## Memory Estimates for Llama 3.1

| Format | 8B model | 70B model | Notes |
|--------|---------|---------|-------|
| Q2_K | ~3.1 GB | ~24.8 GB | Fits 70B in 32 GB RAM + 16 GB VRAM |
| Q4_K_M | ~4.9 GB | ~39.0 GB | Best quality/size tradeoff |
| Q6_K | ~6.1 GB | ~52.0 GB | Near-lossless quality |
| Q8_0 | ~8.1 GB | ~66.0 GB | Practical only for 8B on 16 GB VRAM |

---

## Block Layouts

### Q2_K (2-bit super-block)

256 weights per block, 84 bytes total.

```
struct BlockQ2_K {
    uint8_t  scales[16];  // 4-bit scale (low nibble) + 4-bit min (high nibble)
                          //   per sub-block (16 sub-blocks of 16 weights each)
    uint8_t  qs[64];      // 2-bit quantized values, 4 per byte (256 weights)
    uint16_t d;           // FP16 super-block scale (applied to scale nibbles)
    uint16_t dmin;        // FP16 super-block min delta (applied to min nibbles)
};  // total: 16 + 64 + 2 + 2 = 84 bytes
```

16 sub-blocks of 16 weights each. Dequantization:
```
for sub-block s in 0..15:
    scale = (scales[s] & 0xF)          # 0..15
    min   = (scales[s] >> 4)           # 0..15
    ds    = d_f32 * scale
    dm    = dmin_f32 * min

    for weight j in 0..15:
        i        = s*16 + j
        byte_idx = i / 4
        bit_pair = i % 4
        q        = (qs[byte_idx] >> (2 * bit_pair)) & 0x3   # 2-bit value, 0..3
        w[i]     = ds * q - dm
```

CUDA kernel: `gemv_q2_k_kernel` in `src/cuda/gemm.cu`. Uses `template<bool USE_SMEM>` to fall back to L2-cached global reads when `in_features * 4` exceeds 64 KB shared memory (e.g. FFN down-projection on 70B models).

### Q6_K (6-bit super-block)

256 weights per block, 210 bytes total.

```
struct BlockQ6_K {
    uint8_t ql[128];    // lower 4 bits of each weight (interleaved)
    uint8_t qh[64];     // upper 2 bits (4 per byte)
    int8_t  scales[16]; // int8 sub-block scales
    uint16_t d;         // FP16 super-block scale
};
```

Dequantization follows GGML's `dequantize_row_q6_K` exactly (see `gemm.cu` for inline comments).

### Q4_K_M (4-bit mixed super-block)

256 weights per block, 144 bytes total.

```
struct BlockQ4_K {
    uint16_t d;          // FP16 super-block scale
    uint16_t dmin;       // FP16 super-block min
    uint8_t  scales[12]; // packed sub-block scales and mins
    uint8_t  qs[128];    // 4-bit quantized values, 2 per byte
};
```

### Q4_0 (simple 4-bit)

32 weights per block, 18 bytes total.

```
struct BlockQ4_0 {
    uint16_t d;      // FP16 scale
    uint8_t  qs[16]; // 4-bit values, 2 per byte
};
```
Dequant: `w = (q - 8) * d` where `q` ∈ 0..15.

### Q8_0 (simple 8-bit)

32 weights per block, 34 bytes total.

```
struct BlockQ8_0 {
    uint16_t d;       // FP16 scale
    int8_t   qs[32];  // int8 quantized values
};
```
Dequant: `w = qs[i] * d`.

---

## CUDA Kernels

All GEMV kernels follow the same structure:
- `blockDim.x = 32` (one warp), `blockDim.y = GEMV_WARPS` (8 warps per block)
- Each warp handles one output row
- Threads stride across super-blocks: `for b in tid..num_blocks step 32`
- Warp reduction via `__shfl_xor_sync` after accumulation
- Template `<bool USE_SMEM>`: loads input vector `x` to shared memory when it fits (≤64KB), otherwise reads from L2 cache

For 70B models with large FFN projection layers, `in_features` (e.g. 28,672 for Llama 70B FFN) may exceed the 64KB shared memory limit. The `USE_SMEM=false` path reads directly from global memory (L2-cached, ~5-10% slower).
