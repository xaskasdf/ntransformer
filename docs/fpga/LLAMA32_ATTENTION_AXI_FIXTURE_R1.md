# Attention AXI fixture R1

This fixture prepares the actual HLS/KV integration after the
[checksum composition](LLAMA32_ATTENTION_MEMORY_R1.md). It retains all 143
numerical cases from [attention tile R1](LLAMA32_ATTENTION_TILE_R1.md), including
128 cases derived from real embeddings, layer-0 normalization/Q8, Q/K/V
projections and RoPE for token IDs `[0, 42, 128000, 128255]`.

## Production tile and memory layout

The original unit bench varied tile width from one to 31. The KV controller
uses 64-token tiles plus a final partial tile. A derived native fixture changes
only that header field, preserving Q/K/V, expected outputs and case identity.
It has 147 transactions instead of 291, covering the same 143 final outputs.

The AXI fixture contains 16 cache images, 747 complete token appends and
47,808 beats of 512 bits: 3,059,712 write bytes. The first image holds all eight
KV heads for four real token positions and serves all 128 real queries.
The generator checks that the four Q heads sharing a KV head agree across
every prefix. Loading the full image before shorter queries also supplies
future positions that a causal read must exclude.

Each synthetic case has a separate image. Unselected heads contain distinct
finite padding. Cache offsets alternate between 4,032 and 64 bytes to
exercise page boundaries. The established layout is:

```text
address = base + (((layer*2+plane)*8+kv_head)*capacity+position)*256 + word*64
append_index = plane*32 + kv_head*4 + word
```

`commands.hex` stores five little-endian 32-bit fields per row: case ID,
image ID, query position, Q head and layer. `images.hex` stores image ID,
capacity, first append beat and cache offset. `query.hex` and `expected.hex`
each contain 9,152 FP32 words. The manifest records file/tensor hashes and
command lease identities. All model-derived data remain external in:
`D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/attention-axi-r1/`.

## Independent checks and native execution

`check_ntrpu_attention_axi_fixture_r1.py` independently reconstructs physical
memory from append beats and resolves each query through the plane/head/
position strides. It matches 136,064 causal K/V words against the pinned
parent fixture, plus all query/output words. Twelve semantic corruptions
are rejected independently of the file-hash checks, including plane/head
swaps, wrong addresses, altered late-prefix data and changed partitioning.

Both the existing MSVC executable and a fresh GCC build with
`NTRPU_USE_HLS_MATH` passed the 143 cases. Decimal90 checks found zero ULP
difference in 9,152 final FP32 outputs. All 15,288 state words were checked;
maximum scaled floating-state error was approximately `3.599e-16`, below
`2e-12`. Twelve transcript corruptions and a numerically altered fixture
executed through the AMD binary were rejected.

Native AMD run: `20261013T162456Z-V7-attention-axi-native-amd-r1`.
It used CPU affinity 2/3 while the existing Vitis job retained 0/1.
Compilation took 3.975 s and the positive native run 0.520 s; these are
CPU validation times, not FPGA performance.

- Parent fixture SHA-256: `6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361`.
- Fixed-64 fixture: `8071089d4ce33533cea6357a7ab78d01fc7a55e4b402637ba05a3f4cb0f51dae`.
- AXI manifest: `4e80daca26e268eb84f07b1a7f749ba509454ec1ab3457467540d96432193711`.
- Native AMD D:/NFS archive: `c0e02eae126c68c8649e2c4ebe49b0ddab3a703f0ad00732bd9ddc8b3966e089`.
- Independent combined review: `6cf4dbcfde78538d15f31e0af3aae3646c2beca3ae3b07442313d500477ed6e5`.

## Next integration gate

Consume the generated attention RTL only after its C/RTL review and backup.
Inspect actual port names, directions, widths, reset and synchronous-memory
latency before binding it to context R2 and KV RAM R1. Run these images through
actual append/read transactions, compare every published result and private
state, and exercise cancellation and coordinated reset with the real core.
The native and layout results here do not qualify that RTL connection,
physical DDR, routed timing or complete model inference.
