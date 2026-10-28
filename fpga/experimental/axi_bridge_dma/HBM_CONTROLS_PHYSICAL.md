# Full-board HBM control predictor integration

This isolated VU47P/HBM experiment integrates both candidates from
[HBM_CONTROL_PREDICTORS.md](HBM_CONTROL_PREDICTORS.md). Evidence is stored in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-controls-physical/`.

## Frozen composition

The full adapter donor is the registered-READY synthesis source, SHA-256
`83acbf7cc1310684eae0f81ed02746092a1c631a5256d91258d6cb07d3711a7c`.
The unit archive was restored and reviewed: 3,752,609 checks and eight rejected
RTL mutations. Both tested modules are substituted under their original board
module names, and their two helper queue modules are appended. All other RTL
is copied unchanged. The same implementation serves both active HBM channels.

The source composition uses the corrected READY auditor already validated in
the preceding physical experiment. `reference/RTL_DONOR_ORIGINS.json` records
that provenance separately from RTL. Current source hashes are authoritative
in `CONTROLS_PHYSICAL_BINDING.json`; inherited older manifests describe their
historical stages. Only the adapter and synthesis audit entry change relative
to this frozen composition.

The historical standalone `ntrpu_hbm_read_return_cut.sv` remains in the source
snapshot for provenance; this synthesis compiles the replacement definition
inside `ntrpu_hbm_read_adapter.sv`.

## Native implementation

The incremental donor is routed checkpoint
`6670d41ebfe8581d1b63cfaa4965441f2267fec327e65f2041c4621eda6bd578`.
Its preceding reopened measurements are WNS -0.458 ns, TNS -599.878 ns and
3,362 setup failures. A new read-only inspection supplies matching return-FIFO
write-enable measurements before comparing the new candidate.

Each native batch runs in its own directory under
`/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join`, with two CPU affinity,
nice 10 and a 1,800-second tool limit. Each closed batch is copied, restored and
reviewed before dependent work proceeds. Placement and routing use the same
`place_design -no_psip` and `route_design -no_psir` commands as the predecessor.
No result replaces the selected reference automatically.

The inherited incremental directive is `RuntimeOptimized`. Its reference WNS
is an estimated implementation target, not the final timing-summary result.
This is a controlled comparison against the existing placement, not an
exhaustive search for timing closure.

Relevant preparers are:

```text
prepare_hbm_controls_physical.py DONOR_RTL TESTED_UNIT FRESH_OUTPUT
prepare_hbm_controls_stage.py AUDITS RUNNER INPUT_DCP FRESH_OUTPUT --mode constrain|route|reopen
prepare_hbm_controls_recovery.py ORIGINAL_SOURCE SYNTH_DCP FRESH_OUTPUT
```

Route preparation requires explicit local and remote incremental donor paths.
`--baseline` permits only read-only reopen. The recovery helper completes
corrected audits on the exact saved synthesis checkpoint without resynthesis.

## Physical proof and comparison

The inherited checks retain 2,195 bounded paths, synchronous checks, 18 skew
groups, reset distribution, clock topology, native HBM READY gate truth tables,
calibration selectors, CDC/DRC findings and experimental pin assignments.
Timing commands and exception multiplicities must match the frozen baseline.

New probes verify direct registered return VALID through the adapter to the
CDC input, ordinary register attributes, constant enable and reset polarity.
They export all D-cone startpoints: external IP may reintroduce a reset path
even when the isolated unit's local predictor is reset-free. Such a dependency
must be reported rather than hidden or assumed absent.

Every data, response and last-bit storage register must remain present. A
pruned ID bit requires proof at the queue, adapter and CDC interface: each
boundary must be disconnected or constant. Return ownership must still depend
on upstream VALID, while wide payload enables depend only on free-slot state.
The initial baseline audit stopped because Vivado left disconnected logical
ID pins; that failed batch is retained and the expanded audit checks all three
boundaries explicitly.

The recovered synthesis audit confirms 1,030 return payload registers and
592 CDC RAM write-enable destinations per channel. Eight constant/disconnected
ID storage bits per channel are pruned, with all three boundaries checked.
Both READY predictor D cones are reset-free. Both return VALID D cones still
include reset distribution through the integrated interfaces; the isolated
unit result does not establish reset-free D cones for the complete board.

Setup and hold are measured for every CDC RAM write-enable endpoint, the
return VALID output, its D input and asynchronous clear, and the existing READY
paths. The final routed checkpoint must reproduce these measurements when
reopened. Global WNS/TNS, hold and pulse-width results, coverage gaps and any
tradeoffs determine the conclusion; a faster control output alone is
insufficient to establish timing closure.

## Routed measurements — 2026-10-14 UTC

The candidate routed checkpoint is
`63a16144ddc9f83258be4aae1d6db0c084c5ac8680b6b15c203afe580b1569f9`.
An independent read-only reopen reproduced the complete targeted measurements,
global timing, preserved bounds, skew groups, coverage and routing counts.
`COMPARISON.json` records the equality checks and signed deltas.

| Metric | Previous READY implementation | Control predictors |
|---|---:|---:|
| Global WNS (ns) | -0.458 | -0.440 |
| Global TNS (ns) | -599.878 | -481.952 |
| Failing setup endpoints | 3,362 | 3,047 |
| Global WHS (ns) | +0.005 | +0.005 |
| Hold / pulse-width failures | 0 / 0 | 0 / 0 |
| Channel 0 CDC WE worst setup (ns) | -0.096 | +0.397 |
| Channel 1 CDC WE worst setup (ns) | -0.458 | +0.053 |
| CDC WE setup failures, both channels | 304 / 1,184 | 0 / 1,184 |
| Channel 0 READY D setup (ns) | -0.372 | -0.174 |
| Channel 1 READY D setup (ns) | -0.193 | +0.063 |

All 1,184 CDC RAM WE endpoints pass hold; worst margins are +0.306/+0.197 ns.
The new return VALID D inputs still fail setup at -0.152/-0.013 ns. Their
hold margins are +0.174/+0.237 ns, clear recovery +0.953/+0.732 ns, and clear
removal +0.509/+0.616 ns. Registered READY outputs to HBM pass setup at
+0.759/+0.896 ns and hold at +0.531/+0.415 ns. READY clear recovery remains
narrow in channel 0 (+0.010 ns); channel 1 is +0.235 ns.

The 2,195 preserved bounded paths, 84 synchronous checks and 18 bus-skew groups
pass; minimum skew slack is +0.044 ns. All 100,272 routable nets are routed,
with no routing errors. Response-capture payload enables retain 17 setup
failures among 1,060 registers, worst -0.166 ns; hold passes at +0.155 ns.
The canonical reset release has one consumer, distribution has 1,090 loads,
and release-to-distribution recovery/removal remains +1.489/+0.231 ns.

The older selected checkpoint remains unchanged. Its WNS was -0.413 ns:
this candidate is 0.027 ns worse on that metric, despite fewer failing
endpoints and a smaller magnitude of total negative slack. It is not promoted
as a replacement.

## Batch provenance

The following job suffixes use the `board-controls-physical-` prefix. Every
batch has an archive receipt and restored file hashes, including failures.

| Job suffix | Native seconds | Result |
|---|---:|---|
| `synth-20261014T010608Z` | 802.797 | Synthesis checkpoint saved; initial ID auditor failed |
| `inspect-20261014T010636Z` | 170.664 | Initial baseline ID auditor failed |
| `inspect-20261014T011245Z` | 431.988 | Corrected baseline audit passed |
| `inspect-20261014T012653Z` | 174.281 | Exact synthesis checkpoint audited read-only |
| `inspect-20261014T013026Z` | 332.728 | Constraints and optimization passed |
| `inspect-20261014T013700Z` | 1,028.474 | Placement, routing and physical audits passed |
| `inspect-20261014T015518Z` | 306.109 | Independent reopen passed |

The initial auditor assumed every logical ID pin had a physical driver.
Vivado pruned disconnected ID bits, so the corrected auditor proves all three
interface boundaries instead. Both failed archives and their original audit
sources remain intact. The saved synthesis checkpoint was reused without
changing RTL or repeating synthesis. Final sealing requires 40 corrupted-proof
rejection checks, seven exact preparation replays with overwrite rejection,
and a restored-ZIP rerun of all reviews.

## Remaining paths and next experiments

The global worst path now runs from `memory_distribution_reg` to channel 0
width-converter command-FIFO output enables (`read_addr_inst/cmd_queue/.../CE`).
The new return VALID input path starts at the same reset distribution,
passes the reset-masked `tracker/validated_cut/m_valid`, then traverses the
protocol and width converters before reaching `return_cut/valid_reg_reg/D`.
The channel 0 READY D path instead starts at `capture_remaining_reg[0]`.

These measurements motivate separate experiments on local reset distribution,
the validated-response control boundary, and the READY occupancy/count
predictor. Each must preserve cold-reset behavior, queue ownership and fault
handling, pass the unit banks, and repeat full-board timing measurements.
Removing a functional reset mask or adding a timing exception is not evidence
that the underlying path was fixed.

## Evidence limits

The final bundle includes all failed and completed stages, source/checkpoint
hashes, preparation replays, corrupted-evidence rejection checks and an audit
of owned processes. `FINAL_BACKUP.json` is the authoritative receipt after ZIP
restoration and verification. This flow measures implementation behavior; it
does not qualify physical HBM/PCIe traffic or release a PCB for manufacture.
The inherited coverage gaps remain: 1,154 unconstrained internal endpoints
and two ports without input-delay constraints. Existing CDC, DRC and
methodology warnings are retained in the reports; passing the targeted checks
does not remove these qualification limits.
