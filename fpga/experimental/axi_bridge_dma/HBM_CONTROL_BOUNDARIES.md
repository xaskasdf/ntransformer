# HBM READY credit and validated-response boundary

This isolated VU47P/HBM experiment follows
[HBM_CONTROLS_PHYSICAL.md](HBM_CONTROLS_PHYSICAL.md). Its donor adapter is
SHA-256 `b57d47e3d832f5d348901af602b36d86eafce81224d3443d86a5be2a5f181ec9`.
Evidence belongs under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-control-boundaries/`.

## Motivation from the routed board

The preceding implementation has WNS -0.440 ns. Return-FIFO write enables now
pass, but READY D still fails in channel 0 (-0.174 ns), starting at the capture
credit counter. Return VALID D fails at -0.152/-0.013 ns: reset distribution
passes through `validated_cut/m_valid` and both vendor converters. These are
two distinct control paths; the broad reset distribution remains a separate
physical-design problem.

## Changes and contracts

READY only needs to know whether the next capture credit count is nonzero.
The real eight-bit credit counter and all ownership updates remain unchanged.
For reachable counts 0..128, a capture consumes an existing credit and each
accepted AR adds 1..16 credits. Therefore the predicate is:

```verilog
ar_fire || (|remaining[7:1]) || (remaining[0] && !capture_fire)
```

The bank compares that predicate with integer addition/subtraction for all
8,224 valid input combinations, including every AR length. It explicitly
excludes 32 combinations attempting a capture with zero credit. The integrated
tracker bank separately checks that accepted responses own credit and capacity.

The validated-response queue gains one ordinary asynchronous-clear VALID
register. Its D predicts nonempty occupancy using raw admission from the
tracker. Actual queue writes and ownership still use the existing masked
handshakes. Two payload slots, accepted-data-only writes, full-queue admission
bubbles, ordering and transfer latency are preserved. Cold reset must span
active clock edges to clear synchronous ownership; arbitrary short pulses
are outside this contract.

## Reproduction and acceptance

```text
python -B fpga/experimental/axi_bridge_dma/prepare_hbm_control_boundaries.py DONOR_ADAPTER.sv FROZEN_BENCHES FRESH_OUTPUT
```

`FROZEN_BENCHES` contains the preceding predictor experiment's tracker and
return-queue benches. The preparer exports the adapted benches, exact donor
and candidate modules, runner and reviewer with source hashes.

Native XSIM tests cover tracker depths 1/2/4/8, faults, cold reset and up to
128 capture credits. An independent queue scoreboard tests widths 265/519,
all six-cycle VALID/READY words, random stalls, asynchronous VALID clearing,
payload retention and conservation. Seven independent RTL mutations must fail
for their intended invariants; all other modules are restored before each
mutation. Successful process exit alone is insufficient for XSIM.

Separate reference and candidate synthesis target `xcvu47p-fsvh2892-2-e` with
a 2.222 ns clock. Audits require ordinary FDCE control outputs, proven reset
polarity, reset-free unit D cones, unchanged 1,060-register payload inventory,
VALID-dependent ownership and the response capture enable cut. READY D must
lose its AR-length dependency. No timing exceptions are added.

Each private VM batch uses two CPU affinity, nice 10, bounded XSIM stages and
a 600-second synthesis limit. Every batch is archived, including failures.
The final seal checks source reconstruction, rejection of corrupted evidence,
unchanged predecessors, owned-process closure and a restored ZIP review.

## Native results

Batch `board-control-boundaries-sim-20261014T021116Z` passed all 29 stages in
236.212 seconds, including 117.051 seconds for the two synthesis units. The
review counted 896,718 checks and seven correctly rejected RTL mutations.
Its archive contains 78 verified evidence files plus `FILES.json`, occupies
564,040 bytes, and has SHA-256
`a94ea24afed89ce24b4ec4071399696f522297719723b22cbe183ef7aa44b94e`.

| Tracker unit | Reference | Candidate |
|---|---:|---:|
| LUT primitives | 642 | 634 |
| FDRE / FDCE | 1,139 / 1 | 1,139 / 2 |
| Payload registers | 1,060 | 1,060 |
| Audited ownership input pins | 18 | 18 |
| READY D depends on AR-length registers | Yes | No |
| Validated output | Combinational LUT | Direct FDCE Q |

Both candidate control D cones exclude `rst_n` in this isolated unit. The
candidate checkpoint is
`b2f2c92ec3e5344b044cbf632645130807e2653b83c4529d6514dde0606d2bca`;
the reference is
`8e07bb85132c3661060409a8fa9c9c397defdb5d44acdbae1b119c7f1c68f388`.
These resource counts apply to a single tracker synthesis unit, not the board.

Final sealing requires 18 corrupted-report rejection checks and exact
preparation replay with overwrite rejection. `FINAL_BACKUP.json` is the
authoritative receipt after restored-ZIP verification and the owned-process
audit. The preceding full-board archive and donor remain unchanged.

## Qualification boundary

Unit synthesis omits vendor converters and board routing. An external READY
path can reintroduce reset dependencies after integration. These results must
therefore precede a fresh full-board composition, physical timing comparison
and independent checkpoint reopen. They do not establish 450 MHz closure,
HBM/PCIe traffic correctness or PCB manufacturing readiness.
