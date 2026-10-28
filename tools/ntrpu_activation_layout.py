"""Pack activation ABI1 records consumed by ntrpu_activation_loader."""
MAX_SCALE_BITS = 0x4780204070e1c387


def pack_activations(values: bytes, scales_le: bytes) -> bytes:
    """Return INT8[32], binary64 scale, zero[24] per block (1..256).

    Scales are immutable little-endian IEEE754 bytes. Preserve negative zero
    and subnormals; reject negative nonzero, NaN, infinity and HLS overflow.
    Address allocation, range registration and quantization belong to callers.
    """
    if not isinstance(values, bytes) or not isinstance(scales_le, bytes):
        raise ValueError('Expected immutable activation/scale bytes')
    blocks = len(values)//32
    if len(values) % 32 or not 1 <= blocks <= 256 or len(scales_le) != blocks*8:
        raise ValueError('Expected 1..256 complete INT8[32]/binary64 blocks')
    records = []
    for block in range(blocks):
        scale = scales_le[block*8:(block+1)*8]
        bits = int.from_bytes(scale, 'little')
        magnitude = bits & ((1 << 63)-1)
        if (bits >> 63 and magnitude) or magnitude > MAX_SCALE_BITS:
            raise ValueError(f'Invalid activation scale at block {block}')
        records.append(values[block*32:(block+1)*32]+scale+bytes(24))
    return b''.join(records)
