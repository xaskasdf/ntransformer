# Actual HLS transactions: RMSNorm, RoPE and residual

These results come from Vitis 2026.1 C/RTL co-simulation on
`xcvu5p-flva2104-2-i`, with a 5 ns target clock. Inputs include the existing
Llama 3.2 1B fixtures; the individual block scopes remain separate from a
complete transformer layer or model inference.

## Measured simulation and synthesis estimates

| Block | Transactions | Compared successful values | Maximum latency, cycles | BRAM18K / DSP / FF / LUT |
| --- | ---: | ---: | ---: | --- |
| RMSNorm | 60 | 78,208 | 65,615 | 16 / 11 / 2,121 / 4,894 |
| RoPE table | 48 | 61,760 | 27,511 | 25 / 38 / 3,928 / 7,227 |
| Residual | 43 | 51,556 | 4,110 | 4 / 0 / 1,387 / 4,067 |

RMSNorm passed 44 successful and 16 rejected transactions. For the tested
successful lengths, latency is `32*N + 79` cycles: 65,615 at N=2,048,
or 328.075 microseconds at the simulated clock. The 16 rejected cases leave
all 2,048 output RAM words at the sentinel value in both C and RTL traces.
The other rows report maxima across their own fixtures; they are not
interchangeable workload timings.

Estimated HLS periods are 3.625 / 3.611 / 3.644 ns respectively. Resource and
period estimates are pre-route. Local `ap_memory` simulation excludes physical
DDR transfer, host communication and complete decoder latency.

## Binary trace recovery

RMSNorm C/RTL passed, but its next independent observer attempted to decode
an array transaction file as UTF-8. The observer stopped without invalidating
or repeating the hardware run. The archive's generated C++ writer establishes
the actual format: an eight-byte big-endian word count per memory frame,
fixed-width big-endian words, followed by the terminal marker
`5a5aa5a50f0ff0f0`. Scalar return transactions retain their text format.

`tools/read_ntrpu_hls_tv_r1.py` validates frame counts, widths, depths, exact
termination and scalar transaction order. Observers preserve raw bytes in
base64 and record raw/normalized SHA-256 values. Read-only RTL RAM echoes are
compared against C input frames. Numeric checkers receive normalized output
transactions and the original latency report. Archive sealing verifies the
raw bytes again. The binary format is two-state; it cannot prove absence of
unknown X/Z values in the original four-state simulation.

The RMSNorm recovery rejected 12 damaged trace/latency controls. Later HLS
observers use the frozen decoder and retain their block-specific numerical
and evidence controls. `tests/test_ntrpu_hls_tv_r1.py` also checks 32/64-bit
framing, corrupt lengths/trailers, truncation and scalar rejection.

## Preserved evidence

RMSNorm run `20261013T150621Z-V7-llama32-rmsnorm-hls-r1`:

- Terminal archive: `ee1d2662d2a0bbd552dc46dfd51c1543f7ac9492b64bffecfb8026bee4171e89`.
- Independent numeric review: `74cd4dc205af80a745c92a7bfb81a9b17600a2f9881cf3e2238988cfeb67647b`.
- Recovered transaction review R5: `6da8bb2b78d524a8fdeff9dae6c3b775bc27e56629f59c7c6234532081f40b92`.

RoPE run `20261013T154001Z-V7-llama32-rope-hls-r1`, review:
`cf94bfcf8413f6eab9db66bf335cad7042fb1a04c07663efe21458d24aa29d2f`.
Residual run `20261013T154948Z-V7-llama32-residual-hls-r1`, review:
`09faf097028fa806157b962259d8b6a7b97b6d6fcb9223899e5fb55bae9e9a2f`.
Their coordinators retain terminal archive hashes and verified D:/NFS receipts.

The recovery capsule
`hls-binary-tv-recovery-r1-capsule-20261013T153833Z.zip` preserves 453 members,
SHA-256 `3e3125277f9f4517b992fe45bbddddafac257ce067c2f90246ca4e37b158075e`.
Only observation/decoding changed; hardware sources and queue dependencies
were preserved. The earlier Tcl boolean correction remains separately
recorded in the [queue status](board/VIVADO_STATUS.md).
