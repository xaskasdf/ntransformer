#!/usr/bin/env python3
"""
decompose_gguf.py — Delta-encode a GGUF model for ntransformer streaming.

For each weight matrix type (attn_q/k/v/o, ffn_gate/up/down):
  1. Dequantize all N layers to F32
  2. Compute base = mean(W_0..W_{N-1})
  3. Per layer i: R_i = W_i - base
  4. SVD: U, S, V = torch.svd_lowrank(R_i, q=rank)
  5. Store delta_U = U*sqrt(S), delta_V = sqrt(S)*V^T as F16

Output: .ntd file (NTD1 format) containing:
  - 64-byte header
  - Base weights: 7 matrices re-quantized to Q6_K
  - Per-layer deltas: 14 F16 tensors (U/V pairs) x n_layers

Usage:
  python tools/decompose_gguf.py \
    --model llama-3.1-70b-instruct-q6_k.gguf \
    --rank 64 \
    --output llama-70b.ntd
"""

import argparse
import struct
import sys
import os
import numpy as np
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


# ============================================================
# GGUF parsing (minimal, for weight extraction)
# ============================================================

GGUF_MAGIC = 0x46554747
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

# Block sizes per type
GGML_BLOCK_SIZE = {
    GGML_TYPE_F32: 1, GGML_TYPE_F16: 1,
    GGML_TYPE_Q4_0: 32, GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q4_K: 256, GGML_TYPE_Q5_K: 256, GGML_TYPE_Q6_K: 256,
}
GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4, GGML_TYPE_F16: 2,
    GGML_TYPE_Q4_0: 18, GGML_TYPE_Q8_0: 34,
    GGML_TYPE_Q4_K: 144, GGML_TYPE_Q5_K: 176, GGML_TYPE_Q6_K: 210,
}

GGUF_TYPE_READERS = {
    0: ('B', 1),   # UINT8
    1: ('b', 1),   # INT8
    2: ('H', 2),   # UINT16
    3: ('h', 2),   # INT16
    4: ('I', 4),   # UINT32
    5: ('i', 4),   # INT32
    6: ('f', 4),   # FLOAT32
    7: ('?', 1),   # BOOL
    10: ('Q', 8),  # UINT64
    11: ('q', 8),  # INT64
    12: ('d', 8),  # FLOAT64
}


def read_string(data, offset):
    length = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    s = data[offset:offset+length].decode('utf-8', errors='replace')
    return s, offset + length


def read_value(data, offset, vtype):
    if vtype == 8:  # STRING
        return read_string(data, offset)
    elif vtype == 9:  # ARRAY
        arr_type = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        arr_len = struct.unpack_from('<Q', data, offset)[0]
        offset += 8
        result = []
        for _ in range(arr_len):
            val, offset = read_value(data, offset, arr_type)
            result.append(val)
        return result, offset
    else:
        fmt, size = GGUF_TYPE_READERS.get(vtype, ('B', 1))
        val = struct.unpack_from(f'<{fmt}', data, offset)[0]
        return val, offset + size


def skip_value(data, offset, vtype):
    _, offset = read_value(data, offset, vtype)
    return offset


class GGUFFile:
    """Minimal GGUF parser for tensor extraction."""

    def __init__(self, path):
        self.path = path
        self.tensors = {}  # name -> {shape, type, offset, nbytes}
        self.metadata = {}
        self.data_offset = 0
        self._mmap = None
        self._parse()

    def _parse(self):
        file_size = os.path.getsize(self.path)
        f = open(self.path, 'rb')
        self._file = f

        # Map entire file
        import mmap
        self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = self._mmap

        # Header
        magic, version, n_tensors, n_kv = struct.unpack_from('<IIQQ', data, 0)
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic={magic:#x})"
        assert version == 3, f"Unsupported GGUF version {version}"

        offset = 24  # past header

        # Read metadata KV pairs
        for _ in range(n_kv):
            key, offset = read_string(data, offset)
            vtype = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            val, offset = read_value(data, offset, vtype)
            self.metadata[key] = val

        # Read tensor info
        tensor_infos = []
        for _ in range(n_tensors):
            name, offset = read_string(data, offset)
            n_dims = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            shape = []
            for _ in range(n_dims):
                dim = struct.unpack_from('<Q', data, offset)[0]
                offset += 8
                shape.append(dim)
            ttype = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            toffset = struct.unpack_from('<Q', data, offset)[0]
            offset += 8

            n_elements = 1
            for d in shape:
                n_elements *= d
            bs = GGML_BLOCK_SIZE.get(ttype, 1)
            ts = GGML_TYPE_SIZE.get(ttype, 0)
            nbytes = (n_elements // bs) * ts

            tensor_infos.append({
                'name': name, 'shape': shape, 'type': ttype,
                'offset': toffset, 'nbytes': nbytes
            })

        # Data starts at alignment boundary after header
        alignment = self.metadata.get('general.alignment', 32)
        self.data_offset = (offset + alignment - 1) // alignment * alignment

        for ti in tensor_infos:
            self.tensors[ti['name']] = ti

    def get_config(self):
        """Extract model config from metadata."""
        def get(key, default=None):
            # Try with common prefixes
            for pfx in ['llama.', 'general.']:
                if pfx + key in self.metadata:
                    return self.metadata[pfx + key]
            return default

        return {
            'n_layers': get('block_count', 32),
            'hidden_size': get('embedding_length', 4096),
            'intermediate_size': get('feed_forward_length', 11008),
            'n_heads': get('attention.head_count', 32),
            'n_kv_heads': get('attention.head_count_kv', 32),
        }

    def tensor_data_raw(self, name):
        """Return raw bytes for a tensor."""
        ti = self.tensors[name]
        start = self.data_offset + ti['offset']
        return self._mmap[start:start + ti['nbytes']]

    def close(self):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()


# ============================================================
# Dequantization routines (numpy, matching GGML exactly)
# ============================================================

def fp16_to_f32(raw_u16):
    """Convert uint16 FP16 to float32."""
    return np.frombuffer(np.array([raw_u16], dtype=np.uint16).tobytes(), dtype=np.float16)[0].astype(np.float32)


def dequant_q6_k(raw_bytes, out_features, in_features):
    """Dequantize Q6_K data to F32 [out_features, in_features] — vectorized."""
    n_blocks_per_row = in_features // 256
    total_blocks = out_features * n_blocks_per_row
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(total_blocks, 210)

    # Parse block fields
    ql_all = data[:, :128]       # [B, 128]
    qh_all = data[:, 128:192]    # [B, 64]
    sc_all = data[:, 192:208].view(np.int8)  # [B, 16]
    d_all  = data[:, 208:210].copy().view(np.float16).astype(np.float32).ravel()  # [B]

    result = np.zeros((total_blocks, 256), dtype=np.float32)

    # Process two 128-weight halves
    for half_idx in range(2):
        ql = ql_all[:, half_idx*64:(half_idx+1)*64]   # [B, 64]
        qh = qh_all[:, half_idx*32:(half_idx+1)*32]   # [B, 32]
        sc = sc_all[:, half_idx*8:(half_idx+1)*8]      # [B, 8]
        base = half_idx * 128

        for l in range(32):
            is_idx = l // 16
            q1 = ((ql[:, l].astype(np.int32) & 0xF) | (((qh[:, l].astype(np.int32) >> 0) & 3) << 4)) - 32
            q2 = ((ql[:, l+32].astype(np.int32) & 0xF) | (((qh[:, l].astype(np.int32) >> 2) & 3) << 4)) - 32
            q3 = ((ql[:, l].astype(np.int32) >> 4) | (((qh[:, l].astype(np.int32) >> 4) & 3) << 4)) - 32
            q4 = ((ql[:, l+32].astype(np.int32) >> 4) | (((qh[:, l].astype(np.int32) >> 6) & 3) << 4)) - 32

            result[:, base + l]      = d_all * sc[:, is_idx].astype(np.float32) * q1.astype(np.float32)
            result[:, base + l + 32] = d_all * sc[:, is_idx+2].astype(np.float32) * q2.astype(np.float32)
            result[:, base + l + 64] = d_all * sc[:, is_idx+4].astype(np.float32) * q3.astype(np.float32)
            result[:, base + l + 96] = d_all * sc[:, is_idx+6].astype(np.float32) * q4.astype(np.float32)

    return result.reshape(out_features, in_features)


def dequant_f16(raw_bytes, out_features, in_features):
    """Dequantize F16 tensor."""
    arr = np.frombuffer(raw_bytes, dtype=np.float16)
    return arr.reshape(out_features, in_features).astype(np.float32)


def dequant_f32(raw_bytes, out_features, in_features):
    """Load F32 tensor."""
    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    return arr.reshape(out_features, in_features)


def dequant_q8_0(raw_bytes, out_features, in_features):
    """Dequantize Q8_0 data to F32 — vectorized."""
    n_blocks_per_row = in_features // 32
    total_blocks = out_features * n_blocks_per_row
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(total_blocks, 34)

    d = data[:, :2].copy().view(np.float16).astype(np.float32).ravel()  # [B]
    qs = data[:, 2:34].view(np.int8).astype(np.float32)                # [B, 32]

    result = qs * d[:, np.newaxis]  # [B, 32]
    return result.reshape(out_features, in_features)


def dequant_q4_k(raw_bytes, out_features, in_features):
    """Dequantize Q4_K_M data to F32 — vectorized."""
    n_blocks_per_row = in_features // 256
    total_blocks = out_features * n_blocks_per_row
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(total_blocks, 144)

    d_all    = data[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    dmin_all = data[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    scales   = data[:, 4:16]   # [B, 12]
    qs       = data[:, 16:144] # [B, 128]

    result = np.zeros((total_blocks, 256), dtype=np.float32)

    for chunk in range(4):
        is_lo = chunk * 2
        is_hi = chunk * 2 + 1

        def get_sc_m_vec(idx):
            if idx < 4:
                sc = (scales[:, idx] & 0x3F).astype(np.float32)
                m  = (scales[:, idx + 4] & 0x3F).astype(np.float32)
            else:
                sc = ((scales[:, idx + 4] & 0x0F) | ((scales[:, idx - 4] >> 6) << 4)).astype(np.float32)
                m  = ((scales[:, idx + 4] >> 4)    | ((scales[:, idx]     >> 6) << 4)).astype(np.float32)
            return sc, m

        sc_lo, m_lo = get_sc_m_vec(is_lo)
        sc_hi, m_hi = get_sc_m_vec(is_hi)
        d1 = d_all * sc_lo
        m1 = dmin_all * m_lo
        d2 = d_all * sc_hi
        m2 = dmin_all * m_hi

        q = qs[:, chunk*32:(chunk+1)*32]  # [B, 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)

        result[:, chunk*64:chunk*64+32]    = lo * d1[:, None] - m1[:, None]
        result[:, chunk*64+32:chunk*64+64] = hi * d2[:, None] - m2[:, None]

    return result.reshape(out_features, in_features)


def dequant_q5_k(raw_bytes, out_features, in_features):
    """Dequantize Q5_K data to F32 — vectorized."""
    n_blocks_per_row = in_features // 256
    total_blocks = out_features * n_blocks_per_row
    data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(total_blocks, 176)

    d_all    = data[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    dmin_all = data[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    scales   = data[:, 4:16]    # [B, 12]
    qh_all   = data[:, 16:48]   # [B, 32]
    ql_all   = data[:, 48:176]  # [B, 128]

    result = np.zeros((total_blocks, 256), dtype=np.float32)

    u1, u2 = 1, 2
    for chunk in range(4):
        is_lo = chunk * 2
        is_hi = chunk * 2 + 1

        def get_sc_m_vec(idx):
            if idx < 4:
                sc = (scales[:, idx] & 0x3F).astype(np.float32)
                m  = (scales[:, idx + 4] & 0x3F).astype(np.float32)
            else:
                sc = ((scales[:, idx + 4] & 0x0F) | ((scales[:, idx - 4] >> 6) << 4)).astype(np.float32)
                m  = ((scales[:, idx + 4] >> 4)    | ((scales[:, idx]     >> 6) << 4)).astype(np.float32)
            return sc, m

        sc_lo, m_lo = get_sc_m_vec(is_lo)
        sc_hi, m_hi = get_sc_m_vec(is_hi)
        d1 = d_all * sc_lo
        m1 = dmin_all * m_lo
        d2 = d_all * sc_hi
        m2 = dmin_all * m_hi

        ql = ql_all[:, chunk*32:(chunk+1)*32]  # [B, 32]
        qh = qh_all                             # [B, 32]

        lo_base = (ql & 0x0F).astype(np.float32)
        hi_base = (ql >> 4).astype(np.float32)
        lo_extra = np.where((qh & u1) != 0, 16.0, 0.0)
        hi_extra = np.where((qh & u2) != 0, 16.0, 0.0)

        result[:, chunk*64:chunk*64+32]    = (lo_base + lo_extra) * d1[:, None] - m1[:, None]
        result[:, chunk*64+32:chunk*64+64] = (hi_base + hi_extra) * d2[:, None] - m2[:, None]

        u1 <<= 2
        u2 <<= 2

    return result.reshape(out_features, in_features)


DEQUANT_FN = {
    GGML_TYPE_F32: dequant_f32,
    GGML_TYPE_F16: dequant_f16,
    GGML_TYPE_Q8_0: dequant_q8_0,
    GGML_TYPE_Q4_K: dequant_q4_k,
    GGML_TYPE_Q5_K: dequant_q5_k,
    GGML_TYPE_Q6_K: dequant_q6_k,
}


# ============================================================
# Q6_K quantization (numpy → raw bytes)
# ============================================================

def quantize_q6_k(weight_f32):
    """Quantize F32 [out, in] to Q6_K raw bytes.

    GGML Q6_K: 256 weights per block = 210 bytes
    ql[128] | qh[64] | scales[16] | d(FP16)

    Process in two 128-weight halves. Each half has 4 sub-groups of 32.
    Scales are int8, one per 32 weights (8 per half, but interleaved).
    """
    out_features, in_features = weight_f32.shape
    assert in_features % 256 == 0
    n_blocks_per_row = in_features // 256
    total_blocks = out_features * n_blocks_per_row
    result = bytearray(total_blocks * 210)

    for row in range(out_features):
        for b in range(n_blocks_per_row):
            block_offset = (row * n_blocks_per_row + b) * 210
            w = weight_f32[row, b * 256:(b + 1) * 256]

            # Find per-sub-group scales: 8 sub-groups of 32 weights
            # Interleaved as: half0 has sub-groups 0-3 (with scale indices 0,2,4,6 and 1,3,5,7)
            # Actually Q6_K uses 16 scale entries indexed as described in the dequant

            # Compute super-block scale d: max absolute value across all 256 weights
            amax = np.max(np.abs(w))
            if amax < 1e-10:
                # Zero block
                result[block_offset:block_offset + 210] = bytes(210)
                continue

            # Q6_K: each weight is 6-bit signed (-32..31), with per-32-weight int8 scale
            # Total scale: d * scale[i] * q - 32
            # We need to find d and per-group scales

            # Process in two 128-weight halves
            ql = np.zeros(128, dtype=np.uint8)
            qh = np.zeros(64, dtype=np.uint8)
            scales = np.zeros(16, dtype=np.int8)

            # Compute d from overall max
            # max representable = d * 127 * 31 (scale=127, quant=31)
            d = amax / (127.0 * 31.0)
            if d < 1e-15:
                d = 1e-15

            ql_ptr = 0
            qh_ptr = 0
            sc_ptr = 0

            for half_idx in range(2):
                half_w = w[half_idx * 128:(half_idx + 1) * 128]

                # 4 sub-groups of 32 weights each
                for sg in range(4):
                    sg_w = half_w[sg * 32:(sg + 1) * 32]
                    sg_amax = np.max(np.abs(sg_w))

                    if sg_amax < 1e-10:
                        sc = 0
                    else:
                        sc = round(sg_amax / (d * 31.0))
                        sc = max(-128, min(127, sc))

                    # Scale index mapping matches GGML
                    # half 0: sub-groups map to scale indices 0,2,4,6 (even)
                    # half 1: sub-groups map to scale indices 8+0,8+2,8+4,8+6
                    # But wait, the dequant uses sc[is+0], sc[is+2], sc[is+4], sc[is+6]
                    # where is = l/16 (0 or 1) for each sub-group...
                    # Actually this is more complex. Let me match the dequant pattern exactly.
                    pass

                # Simpler approach: quantize each weight and find best scales
                # Following GGML reference implementation for Q6_K quantization

                # 4 scale groups per half: [0..31], [32..63], [64..95], [96..127]
                # Scale indices for half 0: 0, 2, 4, 6 (using l/16 offset: groups 0-1 use is=0, groups 2-3 use is=1)
                # Wait, the dequant has:
                #   for l in 0..31:
                #     is = l/16  (0 or 1)
                #     q1 = ... → weight[l]       → sc[is+0]
                #     q2 = ... → weight[l+32]    → sc[is+2]
                #     q3 = ... → weight[l+64]    → sc[is+4]
                #     q4 = ... → weight[l+96]    → sc[is+6]
                # So: weights 0-15 → sc[0], weights 16-31 → sc[1]
                #     weights 32-47 → sc[2], weights 48-63 → sc[3]
                #     weights 64-79 → sc[4], weights 80-95 → sc[5]
                #     weights 96-111 → sc[6], weights 112-127 → sc[7]

                for sg in range(8):
                    sg_w = half_w[sg * 16:(sg + 1) * 16]
                    sg_amax = np.max(np.abs(sg_w))
                    if sg_amax < 1e-10:
                        scales[sc_ptr + sg] = 0
                    else:
                        sc = round(sg_amax / (d * 31.0))
                        scales[sc_ptr + sg] = np.int8(max(-128, min(127, sc)))

                # Now quantize weights
                for l in range(32):
                    is_idx = l // 16

                    eff_sc = [float(scales[sc_ptr + is_idx + k]) for k in [0, 2, 4, 6]]

                    for qi, offset_w in enumerate([0, 32, 64, 96]):
                        val = half_w[l + offset_w]
                        sc_val = eff_sc[qi]
                        if abs(d * sc_val) < 1e-15:
                            q = 0
                        else:
                            q = round(val / (d * sc_val)) + 32
                            q = max(0, min(63, int(q)))

                        # Pack into ql/qh
                        # q1 → ql[l] lower nibble, qh[l] bits 0-1
                        # q2 → ql[l+32] lower nibble, qh[l] bits 2-3
                        # q3 → ql[l] upper nibble, qh[l] bits 4-5
                        # q4 → ql[l+32] upper nibble, qh[l] bits 6-7
                        lo4 = q & 0xF
                        hi2 = (q >> 4) & 3

                        if qi == 0:    # q1: ql[l] low nibble, qh[l] bits 0-1
                            ql[ql_ptr + l] = (ql[ql_ptr + l] & 0xF0) | lo4
                            qh[qh_ptr + l] = (qh[qh_ptr + l] & 0xFC) | (hi2 << 0)
                        elif qi == 1:  # q2: ql[l+32] low nibble, qh[l] bits 2-3
                            ql[ql_ptr + l + 32] = (ql[ql_ptr + l + 32] & 0xF0) | lo4
                            qh[qh_ptr + l] = (qh[qh_ptr + l] & 0xF3) | (hi2 << 2)
                        elif qi == 2:  # q3: ql[l] high nibble, qh[l] bits 4-5
                            ql[ql_ptr + l] = (ql[ql_ptr + l] & 0x0F) | (lo4 << 4)
                            qh[qh_ptr + l] = (qh[qh_ptr + l] & 0xCF) | (hi2 << 4)
                        elif qi == 3:  # q4: ql[l+32] high nibble, qh[l] bits 6-7
                            ql[ql_ptr + l + 32] = (ql[ql_ptr + l + 32] & 0x0F) | (lo4 << 4)
                            qh[qh_ptr + l] = (qh[qh_ptr + l] & 0x3F) | (hi2 << 6)

                ql_ptr += 64
                qh_ptr += 32
                sc_ptr += 8

            # Pack block: ql[128] | qh[64] | scales[16] | d(FP16)
            result[block_offset:block_offset+128] = bytes(ql.tobytes())
            result[block_offset+128:block_offset+192] = bytes(qh.tobytes())
            result[block_offset+192:block_offset+208] = bytes(scales.tobytes())
            d_f16 = np.float16(d)
            result[block_offset+208:block_offset+210] = d_f16.tobytes()

    return bytes(result)


# ============================================================
# NTD file format
# ============================================================
# Header: 64 bytes
#   magic[4] = "NTD1"
#   rank:     uint32
#   n_layers: uint32
#   hidden_size: uint32
#   intermediate_size: uint32
#   n_heads: uint32
#   n_kv_heads: uint32
#   head_dim: uint32
#   base_dtype: uint32 (5 = Q6_K)
#   delta_dtype: uint32 (1 = F16)
#   base_offset: uint64 (from file start)
#   delta_offset: uint64 (from file start)
#
# Base section: 7 Q6_K weight matrices
#   [attn_q, attn_k, attn_v, attn_o, ffn_gate, ffn_up, ffn_down]
#
# Delta section: n_layers * 14 F16 tensors
#   Per layer: [attn_q_U, attn_q_V, attn_k_U, attn_k_V, ..., ffn_down_U, ffn_down_V]

NTD_HEADER_SIZE = 64
NTD_MAGIC = b'NTD1'

WEIGHT_NAMES = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
GGUF_WEIGHT_SUFFIXES = [
    'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
    'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight'
]


def dequant_tensor(gguf, tensor_name):
    """Dequantize a GGUF tensor to F32 numpy array."""
    ti = gguf.tensors[tensor_name]
    raw = gguf.tensor_data_raw(tensor_name)
    ttype = ti['type']
    shape = ti['shape']

    # GGUF shape is [in_features, out_features] (column-major convention)
    # We want [out_features, in_features] (row-major, as stored in memory)
    if len(shape) == 2:
        out_features, in_features = shape[1], shape[0]
    else:
        raise ValueError(f"Expected 2D tensor, got shape {shape}")

    fn = DEQUANT_FN.get(ttype)
    if fn is None:
        raise ValueError(f"Unsupported tensor type {ttype} for {tensor_name}")

    return fn(raw, out_features, in_features)


def compute_weight_shapes(config):
    """Return {weight_name: (out_features, in_features)} for Llama."""
    h = config['hidden_size']
    i = config['intermediate_size']
    nh = config['n_heads']
    nkv = config['n_kv_heads']
    hd = h // nh

    return {
        'attn_q':      (nh * hd, h),
        'attn_k':      (nkv * hd, h),
        'attn_v':      (nkv * hd, h),
        'attn_output': (h, nh * hd),
        'ffn_gate':    (i, h),
        'ffn_up':      (i, h),
        'ffn_down':    (h, i),
    }


def decompose(args):
    print(f"Loading GGUF: {args.model}")
    gguf = GGUFFile(args.model)
    config = gguf.get_config()
    n_layers = config['n_layers']
    rank = args.rank

    print(f"Model: {n_layers} layers, hidden={config['hidden_size']}, "
          f"intermediate={config['intermediate_size']}")
    print(f"Rank: {rank}")

    shapes = compute_weight_shapes(config)

    # For each weight type: dequant all layers, compute base + SVD deltas
    base_weights = {}   # name -> F32 numpy (out, in)
    delta_U = {}        # name -> list of F16 numpy per layer
    delta_V = {}        # name -> list of F16 numpy per layer

    for wi, wname in enumerate(WEIGHT_NAMES):
        out_f, in_f = shapes[wname]
        suffix = GGUF_WEIGHT_SUFFIXES[wi]
        print(f"\nProcessing {wname} [{out_f} x {in_f}]...")

        # Pass 1: compute base (mean) incrementally — O(1 matrix) RAM
        base_sum = np.zeros((out_f, in_f), dtype=np.float64)
        for layer in range(n_layers):
            tname = f"blk.{layer}.{suffix}"
            print(f"  [pass 1] Dequant layer {layer}/{n_layers}...", end='\r')
            W = dequant_tensor(gguf, tname)
            assert W.shape == (out_f, in_f), f"Shape mismatch: {W.shape} vs ({out_f}, {in_f})"
            base_sum += W
        base = (base_sum / n_layers).astype(np.float32)
        del base_sum
        base_weights[wname] = base
        print(f"  [pass 1] Base computed ({n_layers} layers)                    ")

        # Pass 2: per-layer SVD of residuals — dequant again, one at a time
        delta_U[wname] = []
        delta_V[wname] = []

        errors = []
        for layer in range(n_layers):
            tname = f"blk.{layer}.{suffix}"
            W = dequant_tensor(gguf, tname)
            R = W - base  # [out, in]
            del W

            R_t = torch.from_numpy(R).float()

            # Low-rank SVD
            U, S, V = torch.svd_lowrank(R_t, q=rank)
            # U: [out, rank], S: [rank], V: [in, rank]

            sqrt_S = torch.sqrt(S)
            U_scaled = U * sqrt_S.unsqueeze(0)           # [out, rank]
            V_scaled = (V * sqrt_S.unsqueeze(0)).T        # [rank, in]

            # Reconstruction error (relative to residual AND to weight)
            R_approx = (U_scaled @ V_scaled).numpy()
            res_err = np.linalg.norm(R - R_approx) / (np.linalg.norm(R) + 1e-10)
            errors.append(res_err)

            # Store as F16
            delta_U[wname].append(U_scaled.half().numpy())
            delta_V[wname].append(V_scaled.half().numpy())

            print(f"  [pass 2] Layer {layer}/{n_layers}: residual error = {res_err:.4f}", end='\r')

        avg_err = np.mean(errors)
        max_err = np.max(errors)
        # Also compute ||R||/||W|| for the first layer to show residual fraction
        tname0 = f"blk.0.{suffix}"
        W0 = dequant_tensor(gguf, tname0)
        R0 = W0 - base
        frac = np.linalg.norm(R0) / (np.linalg.norm(W0) + 1e-10)
        del W0, R0
        print(f"  {wname}: avg SVD error = {avg_err:.4f}, max = {max_err:.4f}, "
              f"||R0||/||W0|| = {frac:.4f}   ")

    # Write .ntd file
    print(f"\nWriting {args.output}...")

    # Compute sizes
    base_sizes = {}
    for wname in WEIGHT_NAMES:
        out_f, in_f = shapes[wname]
        # Q6_K: (n_elements / 256) * 210
        n_elements = out_f * in_f
        base_sizes[wname] = (n_elements // 256) * 210

    total_base = sum(base_sizes.values())

    delta_sizes_per_layer = {}
    for wname in WEIGHT_NAMES:
        out_f, in_f = shapes[wname]
        u_size = out_f * rank * 2    # F16
        v_size = rank * in_f * 2     # F16
        delta_sizes_per_layer[wname] = (u_size, v_size)

    per_layer_delta = sum(u + v for u, v in delta_sizes_per_layer.values())

    base_offset = NTD_HEADER_SIZE
    delta_offset = base_offset + total_base

    hd = config['hidden_size'] // config['n_heads']

    with open(args.output, 'wb') as f:
        # Header
        header = bytearray(NTD_HEADER_SIZE)
        header[0:4] = NTD_MAGIC
        struct.pack_into('<I', header, 4, rank)
        struct.pack_into('<I', header, 8, n_layers)
        struct.pack_into('<I', header, 12, config['hidden_size'])
        struct.pack_into('<I', header, 16, config['intermediate_size'])
        struct.pack_into('<I', header, 20, config['n_heads'])
        struct.pack_into('<I', header, 24, config['n_kv_heads'])
        struct.pack_into('<I', header, 28, hd)
        struct.pack_into('<I', header, 32, 5)   # base_dtype = Q6_K
        struct.pack_into('<I', header, 36, 1)   # delta_dtype = F16
        struct.pack_into('<Q', header, 40, base_offset)
        struct.pack_into('<Q', header, 48, delta_offset)
        f.write(header)

        # Base weights (Q6_K)
        for wname in WEIGHT_NAMES:
            print(f"  Quantizing base {wname} to Q6_K...")
            raw = quantize_q6_k(base_weights[wname])
            f.write(raw)

        # Per-layer deltas (F16)
        for layer in range(n_layers):
            for wname in WEIGHT_NAMES:
                u_data = delta_U[wname][layer]
                v_data = delta_V[wname][layer]
                f.write(u_data.tobytes())
                f.write(v_data.tobytes())
            if (layer + 1) % 10 == 0:
                print(f"  Written deltas for {layer + 1}/{n_layers} layers")

    file_size = os.path.getsize(args.output)
    print(f"\nDone! Output: {args.output}")
    print(f"  File size:       {file_size / (1024**3):.2f} GB")
    print(f"  Base (Q6_K):     {total_base / (1024**2):.1f} MB")
    print(f"  Deltas (F16):    {per_layer_delta / (1024**2):.1f} MB/layer x {n_layers} = "
          f"{per_layer_delta * n_layers / (1024**3):.2f} GB")
    print(f"  vs GGUF:         {gguf.metadata.get('general.file_size', 'unknown')}")

    gguf.close()


def main():
    parser = argparse.ArgumentParser(description='Delta-encode GGUF model for ntransformer')
    parser.add_argument('--model', '-m', required=True, help='Input GGUF model file')
    parser.add_argument('--rank', '-r', type=int, default=64, help='SVD rank (default: 64)')
    parser.add_argument('--output', '-o', required=True, help='Output .ntd file')
    args = parser.parse_args()

    decompose(args)


if __name__ == '__main__':
    main()
