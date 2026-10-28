# Post-O residual and FFN normalization — R1

**Qualified result, 2026-10-15:** HLS R3 passed C/RTL and its complete archive
is verified on D: and NFS. All 26 transactions passed: 8,480 exact FP32 words,
2,120 exact ABI1 words and unchanged output frames in all 15 failure cases.
The reviewer rejected 11 transcript controls and ten actual-frame controls.
Measured maximum latency is **173,353 cycles**, including each of the four
real 2,048-element prefixes. HLS estimates 37 BRAM18, 11 DSP, 5,349 FF,
13,882 LUT and 3.647 ns; these are not routed timing measurements.

Run `20261015T080756Z` follows the full attention/O
[supplementary review R4](LLAMA32_PROJECTED_ROOT_REVIEW_R4.md).
Archive SHA-256: `ac6e55a53a11e7fd4cb6867483e788fe03abf9e1fb01b4969949ea1cf455b979`.
Review SHA-256: `0c847e444e59baf905a0eee04a2a68b047d99cea09c6e312ca37e0b84b8d492a`.
Source SHA-256:
`468cb215797330de8602aaad596a83b27e7d5ba19c59219136e471cc1ede7752`.
The AXI loader R3 has started. Earlier queue revisions below are historical. The
[transport composition](LLAMA32_POST_ATTENTION_PATH_R1.md) passed native
protocol tests; its binding has been generated from these verified HLS ports,
but actual arithmetic through the transport still requires simulation.

**Update 2026-10-15:** the first queue stopped before launching HLS because
its projected-root dependency failed a testbench tag-alignment check.
The unchanged arithmetic now waits in `llama32-post-attention-hls-r2-coordinator.json`
behind the corrected full-root R3 regression. Source SHA-256 is
`399b989731d829b822ac659f20f0a0e3d8168b0643dbefac55eef10fbb64a592`;
package SHA-256 is
`69099806ec49a8979a19747e1f2a8ee566314271c1a3273a25b1a6767ffb712c`.
The [AXI input loader](LLAMA32_POST_ATTENTION_LOADER_R1.md) has passed native
protocol tests and is queued after this HLS qualification. See the
[current recovery and dependency chain](LLAMA32_PROJECTED_ROOT_TAG_RECOVERY_R3.md).
The queue section below records the original R1 package.

Prepared and queued on 2026-10-14. `ntrpu_post_attention_r1` joins the
already qualified residual and RMSNorm/Q8 kernels into one local-memory
hardware boundary. Native arithmetic passed; the new composite has **not
yet passed RTL simulation or synthesis**.

## Interface and ownership

The top is in `fpga/hls/post_attention_r1/`. Its three inputs are the
preserved layer input, the FP32 O projection, and the layer's FFN RMSNorm
weights. For Llama 3.2 1B each contains 2,048 words. Epsilon is binary32
`0x3727c5ac`; valid lengths are multiples of 32 between 32 and 2,048.

The block returns two distinct buffers:

- `residual_output`: 2,048 FP32 words, retained for the addition after FFN.
- `ffn_input`: 512 64-bit words, comprising 64 ABI1 records. Each record
  contains 32 INT8 values, a binary64 scale and 24 reserved zero bytes.

All five arrays must be separate. Inputs remain immutable until `ap_done`.
Both arithmetic stages finish before either output copy starts. The owner
must withhold publication until completion: the two copy loops are sequential,
not atomic bus writes. Failure leaves both output buffers unchanged.

Status priority is shape (1), nonfinite residual operands (2), residual
overflow (3), epsilon (4), nonfinite normalization operands (5), normalization
overflow (6), quantizer failure (7), then success (0). Status 7 is a defensive
propagation path; the fixture does not claim a reachable example for it.
Reset, cancellation, epoch validation and AXI publication belong to the
future transport wrapper, not to this local arithmetic top.

## Native evidence

Run `20261014T044159Z-V7-post-attention-native-amd-r1` uses the actual C++
kernels and AMD math on CPU2/3. Its 26 transactions include four real
layer-0 attention prefixes for tokens 0, 42, 128000 and 128255, numerical
edge cases, invalid shapes, late failures and recovery.

The 11 successful cases produce **8,480 exact FP32 words and 2,120 exact
ABI1 words**, compared against independent integer and ordered
rational/Decimal references. All 15 failures preserve both output arrays;
the test also checks input immutability, guards and inactive tails.
Eleven corrupted transcripts are rejected. Five compiled implementation
mutants fail, including premature residual publication before normalization.
A corrupted reference also makes the test executable fail.

Archive SHA-256:
`d4b02855ee237420185af35e9c5fe504331bc08dfd47411261834fba58b2366f`.
The complete native result and frozen sources are verified on D: and NFS.
These are CPU arithmetic results, not FPGA cycle or throughput measurements.

## Queued C/RTL qualification

`llama32-post-attention-hls-r1-coordinator.json` waits for reviewed, backed
full projected-root R2 completion and the free primary tool slot. The order is:

1. O projection R3: synthesis passed; full XSim regression running.
2. Actual quantizer → ABI1 memory → O R3.
3. Actual attention root → quantizer → O R2.
4. This post-attention HLS R1 block.

The new package is `20261014T044451Z-post-attention-hls-r1`, targeting
`xcvu5p-flva2104-2-i`, 5 ns, CPU0/1 and a 7,200-second tool limit. It runs
C simulation, synthesis and Verilog co-simulation. Acceptance requires exact
outputs and statuses, all read-only memory echoes, complete unchanged failure
frames, transaction cycles, 11 transcript controls and 10 actual-frame controls.
The frame checker passed a manufactured-frame preflight; that preflight is
checker testing only. Every terminal result, including diagnostics, is archived
and verified before the primary slot is released.

- Source ZIP SHA-256:
  `9c3ef5a2feb6594129ff14f3ebfc45048651089e5d710b46294c6130b9752fa1`.
- Package SHA-256:
  `5f0f36b4f6ea71a405f3fd6016094f99f4c18e8bb2ffab7979d96b2eda8dea85`.
- Fixture SHA-256:
  `a8f7797f71bc37bce6d5ff7a3107337172f63e87d8b5328c629b7fb9d21e1310`.

## Next transport boundary

The next RTL wrapper must load the completed O region, the preserved input
and FFN weights into local RAM, then drive this exact HLS interface. Keep the
64-bit sequence/epoch and region ownership through both output writes and
their AXI B responses. Publish a descriptor only when both regions succeed;
cancel, reset, stale identity or write error must invalidate the pair and
drain outstanding traffic before releasing ownership. Register watchdog faults
before driving cancellation to avoid the previously detected progress loop.

The real integration test must read committed O bytes, not an independent
fixture substituted at the consumer. It must retain the residual while FFN
consumes the published ABI1, then validate the second residual addition.
This transport, complete decoder layers, physical DDR and inference remain
unqualified. See [the projected-root queue](LLAMA32_ATTENTION_PROJECTED_ROOT_REAL_QUEUE_R2.md)
and [current board status](board/VIVADO_STATUS.md).
