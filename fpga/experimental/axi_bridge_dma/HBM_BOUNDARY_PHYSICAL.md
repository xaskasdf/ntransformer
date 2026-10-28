# Full-board HBM credit and validated-response boundaries

This isolated VU47P/HBM experiment integrates the candidates validated in
[HBM_CONTROL_BOUNDARIES.md](HBM_CONTROL_BOUNDARIES.md). Evidence lives under
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-boundary-physical/`. The DDR design and the
selected routed reference are separate.

## Frozen composition

The restored unit bank passed 896,718 checks and rejected seven RTL mutations.
Its READY credit predicate removes the burst-length adder from the control
decision while retaining the real credit counter. Its validated-response queue
registers VALID without adding payload storage or changing accepted writes,
ownership, fault handling, or the existing full-queue admission bubble.
The equivalence assumes coordinated cold reset spans active clock edges, so
synchronous queue state is reset before asynchronous VALID is released.

Both active HBM channels use the exact tested tracker, validated queue and
credit helper under their board module names. The adapter donor SHA-256 is
`b57d47e3d832f5d348901af602b36d86eafce81224d3443d86a5be2a5f181ec9`.
Only the adapter and synthesis audit entry change relative to the frozen donor.
`BOUNDARY_PHYSICAL_BINDING.json` records the authoritative composition; older
manifests in the source snapshot describe historical stages.

The incremental reference is checkpoint
`63a16144ddc9f83258be4aae1d6db0c084c5ac8680b6b15c203afe580b1569f9`:
WNS -0.440 ns, TNS -481.952 ns and 3,047 failing setup endpoints.
The preceding return VALID implementation and its CDC write-enable contracts
remain part of this experiment.

## Native flow and review

Each batch uses its own directory, two-CPU affinity, nice 10 and a 1,800-second
tool limit with Vivado 2026.1, targeting `xcvu47p-fsvh2892-2-e`.
Closed batches are copied, restored and hash-verified before
dependent work. The flow is full synthesis, constraints and optimization,
incremental placement/routing, then independent read-only reopen. It retains
the predecessor's timing constraints and reports existing coverage gaps.

```text
prepare_hbm_boundary_physical.py DONOR_RTL TESTED_UNIT FRESH_OUTPUT
prepare_hbm_boundary_stage.py AUDITS RUNNER INPUT_DCP FRESH_OUTPUT --mode constrain|route|reopen
```

Route preparation requires explicit local and remote incremental donor paths.
`--baseline` permits only read-only reopen. No stage replaces the selected
reference automatically.

The new auditor exports complete READY D fan-in, the validated VALID driver
through all three interfaces, ordinary register attributes, clock, enable and
reset polarity. It inventories all validated payload bits and ownership pins;
only ID storage may disappear, with constant/disconnected evidence at every
affected interface. Payload writes must retain their input-VALID dependency.
Reset dependencies in predictor D cones are measured explicitly because the
integrated vendor converters can reintroduce dependencies absent in unit tests.

Timing checks cover validated VALID output to return VALID D, its own D input
and asynchronous clear, plus inherited READY, return, CDC and reset paths.
Route and reopen must agree on targeted and global measurements, bounded paths,
skew, coverage and routing. Final sealing also reruns corrupted-evidence
rejection checks and exact preparation replay with overwrite rejection.

## Audit correction

The first baseline inspection stopped because a Tcl glob interpreted the
literal brackets in `channels[0]` as a character class. Its complete fan-in
export already contained the four ARLEN registers. The corrected auditor uses
literal prefix comparison; the failed batch and its original sources remain
preserved. This changes inspection only, not RTL, constraints or checkpoints.
The synthesis used the original auditor; its independent Python review checks
ARLEN membership directly against the complete exported fan-in. Subsequent
native inspections use the corrected literal comparison.

## Synthesis results

The native synthesis completed in 831.403 seconds. Its structural audit retains
1,036 validated payload registers across both channels; the 24 omitted storage
bits are ID bits, with 64 constant/disconnected interface observations. Both
READY D cones are reset-free. Both new validated VALID D cones include reset
distribution through the integrated interfaces, despite the isolated predictor
being reset-free. This dependency is retained in the physical measurements.

Full-board synthesis uses 61,109 LUTs and 65,733 flip-flops, compared with
61,119 LUTs and 65,731 flip-flops in the previous synthesis. Each tracker saves
five LUTs and adds one validated VALID register. LUTRAM, SRL and block RAM
counts are unchanged. The prior utilization report is copied with its verified
manifest hash under `reference/`.

## Routed measurements - 2026-10-14 UTC

The candidate checkpoint is
`3463c9ca3420b52a5fd98f53a14e4583784935649c931f8403a71fb146890461`.
Independent read-only reopen reproduced the complete targeted measurements,
global timing, routing counts, preserved bounds, skew groups and coverage.
`COMPARISON.json` records those equality checks and the signed deltas.

| Metric | Previous controls | New boundaries |
|---|---:|---:|
| Global WNS (ns) | -0.440 | -0.409 |
| Global TNS (ns) | -481.952 | -263.161 |
| Failing setup endpoints | 3,047 | 2,136 |
| Global WHS (ns) | +0.005 | +0.005 |
| Hold / pulse-width failures | 0 / 0 | 0 / 0 |
| Channel 0 READY D setup (ns) | -0.174 | -0.209 |
| Channel 1 READY D setup (ns) | +0.063 | +0.344 |
| Channel 0 return VALID D worst setup (ns) | -0.152 | +0.257 |
| Channel 1 return VALID D worst setup (ns) | -0.013 | +0.009 |
| Response-capture enable failures / 1,060 | 17 | 14 |
| Response-capture worst setup (ns) | -0.166 | -0.104 |

The path specifically through the new validated VALID output to return VALID D
passes at +0.339/+0.406 ns; its hold margins are +0.795/+0.753 ns. Other paths
to return VALID D determine the smaller margins in the table above.

The new registers introduce their own input and reset endpoints:

| Validated VALID register margin (ns) | Channel 0 | Channel 1 |
|---|---:|---:|
| D setup | -0.223 | +0.006 |
| D hold | +0.204 | +0.218 |
| CLR recovery | +0.636 | -0.091 |
| CLR removal | +0.695 | +1.047 |

All 1,184 inherited CDC RAM write-enable endpoints still pass setup and hold.
Their worst setup margins are unchanged at +0.397/+0.053 ns and hold at
+0.306/+0.197 ns. Registered READY outputs to HBM pass setup at
+0.767/+0.938 ns and hold at +0.526/+0.394 ns. READY clear recovery is
+0.010/+0.221 ns; the narrow channel 0 margin remains.

The 2,195 bounded paths, 84 synchronous checks and 18 skew groups pass, with
minimum skew slack +0.044 ns. All 100,261 routable nets are routed, without
routing errors. Reset distribution still has 1,090 loads and canonical release
one consumer; release-to-distribution recovery/removal is +1.489/+0.231 ns.
The incremental flow retains 99.93% cell placement and reuses 98.29% of nets
according to the final reuse report.

## Interpretation and next experiment

This candidate improves aggregate setup timing and closes the previous return
VALID D violations. It does not close timing: channel 0 READY D regresses by
0.035 ns, the new validated VALID D fails at -0.223 ns, and channel 1's new CLR
recovery fails at -0.091 ns. Channel 1 VALID D also has only +0.006 ns margin.

The global worst path now runs from `memory_distribution_reg/C` to native HBM
stack 0 AXI 0 `BREADY`, with two logic levels and 2.511 ns datapath delay. The
channel 0 validated VALID D worst path starts at response metadata
`response_cut/slot1_reg[264]/C`; it traverses five logic levels. Its complete
D cone still includes reset, even though reset is not this worst-path source.
The READY D worst path remains sourced by `capture_remaining_reg[0]/C`.

The next experiment should investigate local reset distribution and the HBM
write-response ready boundary, alongside validated metadata prediction. It
must preserve cold-reset release, credit accounting, response ownership and
fault behavior, and repeat functional and physical checks. No timing exception
or removal of a functional reset mask is justified by these measurements.

The older selected checkpoint remains unchanged. Its WNS was -0.413 ns; a
0.004 ns improvement in this candidate does not establish timing closure or
justify automatic promotion.

## Batch provenance

Job suffixes below use the `board-boundary-physical-` prefix. Every batch has
an archive receipt and restored file hashes, including the failed inspection.

| Job suffix | Native seconds | Result |
|---|---:|---|
| `synth-20261014T022302Z` | 831.403 | Full synthesis and structural review passed |
| `inspect-20261014T022406Z` | 304.120 | Initial baseline auditor rejected literal brackets |
| `inspect-20261014T023550Z` | 330.913 | Corrected read-only baseline passed |
| `inspect-20261014T024226Z` | 356.157 | Constraints and optimization passed |
| `inspect-20261014T025142Z` | 1,003.312 | Placement, routing and physical evidence review passed |
| `inspect-20261014T030952Z` | 327.064 | Independent reopen reproduced the measurements |

Passing an evidence review means the measurements and contracts are complete;
it does not mean all timing margins pass. Final sealing requires 54 corrupted-
evidence rejection checks, six exact preparation replays with overwrite
rejection, verified predecessor hashes and an audit of owned processes. The ZIP
is restored and its file hashes, reviews and preparation replays are checked
again. `FINAL_BACKUP.json` is the authoritative completion receipt.

## Evidence limits

Unit equivalence does not establish full-board timing closure. This flow does not
qualify live HBM/PCIe traffic or release a PCB for manufacture. Existing CDC,
DRC and methodology warnings remain in the evidence. Coverage still includes
1,154 unconstrained internal endpoints and two ports without input delays.
Experimental pin assignments require separate qualification.
