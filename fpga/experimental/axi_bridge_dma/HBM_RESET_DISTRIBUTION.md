# Isolated HBM reset distribution

## Scope and measured baseline

This VU47P/HBM side experiment prepares a reset-distribution candidate without
replacing the integrated board RTL or its routed checkpoint. The DDR board
variant remains separate. Evidence lives in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-reset-distribution/`.

The read-only probe reopens routed donor
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.
`core/reader/startup/memory_release_reg[2]` drives **1,091 leaf input pins**:
999 associated with the memory clock, 91 combinational inputs and one reader
status input. All are in SLR0; the inventory includes 32 HBM AXI reset pins.
This corrects the preliminary 1,090-load count.

Expanded constraints are unchanged. The reopened design remains fully routed,
with WNS -0.413 ns, TNS -803.549 ns and 4,555 failing setup/recovery endpoints.
This reproduces the previous reopened TNS; it does not resolve the earlier
0.067 ns difference from the optimizer's in-memory report.

## Candidate and behavioral contract

`ntrpu_hbm_startup_distributed.sv` retains the canonical three-stage source and
memory synchronizers. An ordinary register after the memory synchronizer
drives `mem_rst_n`, including its existing readiness feedback. Cold assertion
propagates asynchronously through both stages; release moves from the third
to the fourth memory edge. Calibration loss and adapter faults remain sticky
status events and must not start a new reset epoch.

The distribution stage intentionally has no `ASYNC_REG` or `DONT_TOUCH`.
This follows AMD's recommendation to add an ordinary register after a
[high-fanout synchronizer](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Replicate-High-Fanout-Net-Drivers).
Replication and physical improvement still require full-board measurement.

## Validation and reproduction

`prepare_hbm_reset_distribution.py` requires a fresh output directory and an
explicit frozen RTL donor. It checks that removing the distribution stage
recovers the original functional source, then records source hashes.

```powershell
python -B fpga/experimental/axi_bridge_dma/prepare_hbm_reset_distribution.py `
  --donor D:/AI/artifacts/ntrpu-axi-bridge-hbm-board-cdc/prepared-r1 `
  --output D:/AI/artifacts/ntrpu-axi-bridge-hbm-reset-distribution/new-prepared
```

The private `launch.py sim --prepared <directory>` helper invokes the archived
runner on the VM with two CPUs, nice level 10, 120-second simulation steps and
a 180-second synthesis limit. Use `collect.py --collect` and then
`review_jobs.py --write`; archive failed batches as well as successful ones.

The new bench exercises four clock/phase combinations, exact release latency,
independent stopped clocks, asynchronous assertion, restart, calibration,
sticky faults and the actual 265/519-bit consumer queues under backpressure.
Each case drains six words from each queue and discards two only at an explicit
cold epoch. The original four-case cold-release regression is also run.
Three negative RTL controls must fail for early release, synchronous-only
assertion and fault-driven reset.

Native synthesis audits all 21 synchronizer registers, the new FDCE, its clock,
constant D/CE, reset polarity and readiness connection. It accepts either a
verified LUT1 inverter or primitive CLR inversion, and requires a single leaf
load on the canonical memory release. The synthesized checkpoint is saved
before these audits. Isolated synthesis is not routed timing evidence.

## Reviewed result, 2026-10-13

Batch `board-reset-sim-20261013T180152Z` completed in 149 seconds. Independent
review passed 4,570 new functional checks across four cases, four original
regression cases and three intended negative-control failures. The synthesized
candidate retains 21 `ASYNC_REG` registers and one ordinary FDCE. Its canonical
release drives one LUT1 inverter input; that inverter drives only the FDCE CLR.
Its readiness feedback comes from the new distribution output.

The candidate checkpoint SHA-256 is
`01b85c5d34cb93e730121503241c7e8a904e0c72d1285445c14dbccc3b885998`.
The isolated timing report warns that the clock ports have no `HD.CLK_SRC`;
its clock delay/skew estimates cannot qualify the board.

Four failed development batches remain archived: a stale-status wait in the
bench, an earlier assertion catching a negative mutation, a Tcl object lookup,
and an audit that assumed absorbed rather than explicit reset inversion. The
candidate RTL did not change during these bench/audit corrections.

Seven altered-proof controls are rejected independently; five prepared
revisions reproduce their input hashes and reject overwriting. All 172 remote
evidence files and 481 frozen predecessor files were verified. The final
process audit found no remaining processes owned by this experiment.
`FINAL_BACKUP.json` identifies the ZIP and its restored-file verification.

## Next full-board acceptance gate

1. Build a private board candidate from the frozen integration sources. Replace
   only its startup binding; preserve all reset consumers and cold-epoch rules.
2. Move the existing 4 ns bound to `ready_meta_reg[2]/D` from the canonical
   release register to the distribution register. Record this exact mapping;
   preserve the other 2,194 bounded paths without adding clock cuts or waivers.
3. Time the new recovery/removal arcs. Check any physical replicas for matching
   clock, reset, initial state and release cycle; measure their actual fanout.
4. Repeat the board's 84 synchronous checks, 18 bus-skew groups, CDC/DRC,
   routing and native traffic regressions. Preserve checkpoints before audits,
   reopen the selected result and compare its reports.

The present candidate does not qualify full PCIe/HBM traffic, board pinout or
timing closure, and does not replace the previous routed baseline.
