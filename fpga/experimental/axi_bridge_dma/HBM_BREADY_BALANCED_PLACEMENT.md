# HBM BREADY Intermediate Placement

## Scope and hypothesis

This isolated VU47P experiment evaluates a register/predictor location between
the loader control logic and HBM. It starts from the original pre-placement
routed checkpoint, SHA-256
`ada41ba33828cbb83f8f2159ae84c4a1568bf1bde86913665f705ca1337d7d15`.
It does not inherit the rejected clock route from the previous placement trials.

Moving the channel 1 cluster all the way to `SLICE_X122Y1` improved its output
but left input setup at -0.482 ns after the shared-LUT correction. This trial
tests whether an intermediate location can improve the combined input/output
margin. It changes placement only: no RTL, pipeline, INIT or timing exception.

## Explicit movements

| Cell | Original LOC/BEL | Target LOC/BEL |
| --- | --- | --- |
| Channel 1 BREADY FDCE | X41Y11/CFF | X100Y4/CFF |
| Channel 1 predictor LUT4 | X41Y11/C6LUT | X100Y4/C6LUT |
| Channel 1 predictor LUT5 | X41Y11/H6LUT | X100Y4/H6LUT |
| Channel 1 vendor BREADY gate | X113Y11/C6LUT | X122Y0/A6LUT |
| Shared `bready_predictor_i_2` LUT2 | X7Y11/D5LUT | X63Y11/F6LUT |

All LOC names have the `SLICE_` prefix. The original physical inventory shows
X100Y4 is empty. The native script checks each donor and all target BELs,
including paired LUT5 resources, before moving any cell.

## Execution and review

`prepare_hbm_bready_balanced_placement.py` freezes the inherited board audits,
explicit plan, runner and native Tcl. The private VM runner uses two CPUs,
nice 10 and a 1,800-second deadline per batch. It archives each closed batch.
Moving the register requires unrouting and rebuilding only
`clock_generator/inst/clk_out2`, followed by preserved routing and finalization.

Review requires exactly five planned movements among 120,241 primitives,
unchanged types/INIT and connectivity, both BREADY cones, and all 8,321 affected
clock pins. Clock PIP additions/removals are recorded. An independent checkpoint
reopen must reproduce the routed inventory, clock route and timing measurements.
The existing full-board CDC, reset, coverage, bounded-path and bus-skew audits
remain mandatory.

Local acceptance requires both channels' BREADY input/output/native/reset and
vendor timing to pass; no worse global WNS/TNS or setup/hold/pulse failure counts
than the original; passing bounded contracts and bus-skew; unchanged coverage;
and complete routing. This is an experimental acceptance gate, not full-board
qualification or permission to replace the selected reference.

## Evidence

The working evidence directory is
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-bready-balanced-placement/`.
`status.py` inspects its private job; `advance.py` collects a closed batch,
verifies it and starts only that batch's independent reopen. `finalize.py`
verifies negative evidence guards, replays preparations, audits private
processes, and verifies a restored ZIP before writing `FINAL_BACKUP.json`.

## Routed result

The ECO completed in 589.9 seconds. Its reviewed checkpoint is
`6f383d4e9e400c9b5215d920288a8b64385864eca48021a633b247d2a48d73aa`.
Independent reopening reproduced every targeted timing measurement, inherited
board contract, primitive inventory, cone connection and final clock PIP.

| Measurement (slack in ns) | Original | Intermediate placement |
| --- | ---: | ---: |
| Channel 0 BREADY input setup | -0.010 | +0.056 |
| Channel 1 BREADY input setup | -0.231 | +0.279 |
| Channel 1 BREADY output setup | -0.403 | +0.142 |
| Channel 1 vendor feedback setup | -0.159 | +0.542 |
| Global WNS | -0.403 | -0.460 |
| Global TNS | -256.046 | -415.585 |
| Failing setup endpoints | 2,070 | 3,155 |
| Minimum bus-skew slack | +0.014 | -0.079 |
| Failing bus-skew groups | 0 / 18 | 1 / 18 |
| Response enable worst setup | -0.114 | -0.327 |
| Failing response enable endpoints | 14 / 1,060 | 262 / 1,060 |

Both BREADY channels pass input, output, native, vendor feedback and reset
setup/recovery/hold/removal checks. Global hold and pulse-width failures remain
zero, and all 100,262 routable nets are complete with no routing errors.
The exact five-cell scope, 17-cell cones and 1,699 cone-pin records passed review.
Clock connectivity is unchanged; the rebuilt tree has 2,705 PIPs versus 2,725,
with 29 added and 49 removed.

**This candidate fails the local acceptance gate and is not promoted.** The
2,195 bounded paths and 84 synchronous checks still pass. Coverage retains
1,154 unconstrained internal endpoints and two inputs without input delay;
those existing limits also prevent a full-board qualification claim.

## Remaining physical work

Channel 1's worst BREADY input now starts at native HBM WREADY: 1.974 ns data
delay, including 1.235 ns routing, with +0.059 ns clock skew. The output data
delay is 1.468 ns, including 1.300 ns routing, with -0.239 ns clock skew.

The global critical path is `memory_distribution_reg/C` to channel 1
`validated_cut/count_reg[1]/CE`: 2.426 ns data delay and -0.143 ns clock skew.
Other reported failures include the loader state-machine enables and native
HBM reset endpoints. The loader payload bus-skew group measures 1.190 ns against
a 1.111 ns requirement, between `address_reg[25]/D` and `payload_reg[149]/D`.

Further work should preserve the measured BREADY placement while investigating
reset/control distribution and the clock-sensitive loader bus-skew separately.
Any reset replication or pipeline change needs functional validation before a
new full-board physical comparison. The selected reference remains
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.

## Closure and backup

Both native batches are collected and reviewed. Their IDs end in
`20261015T105743Z` (ECO) and `20261015T110855Z` (reopen).
`COMPARISON.json` records the passing BREADY checks and failing global gates.
The closure verifier exercises 90 evidence-rejection checks, replays both
frozen preparations, refuses preparation overwrites, and reruns reviews against
the restored ZIP. It also verifies predecessor hashes and checks this
experiment's private processes for leftovers.

Use `FINAL_BACKUP.json` as the completion receipt: it is created only after
archive restoration and verification, and records the ZIP hash and restored
file count. The main simulation work and DDR variant are outside this change.
