# HBM control predictor unit experiment

This isolated VU47P/HBM experiment follows
[HBM_READY_PHYSICAL.md](HBM_READY_PHYSICAL.md). Its frozen adapter is the
registered-READY full-board synthesis source, SHA-256
`83acbf7cc1310684eae0f81ed02746092a1c631a5256d91258d6cb07d3711a7c`.
Evidence belongs under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-control-predictors/`.

## Measured motivation

The preceding routed and reopened board has WNS -0.458 ns and TNS -599.878 ns.
Its worst path runs from return-queue occupancy to the CDC FIFO write enable.
The two READY register D inputs fail setup by 0.372 and 0.193 ns; both start
at the reset distribution register. READY output timing now passes, so these
experiments target the remaining input and return-control paths.

## Two independent changes

The READY candidate computes next space and next fault from unmasked internal
queue state. Actual interfaces, storage updates, credit accounting and fault
quarantine keep their existing reset handling. While reset is high, each raw
prediction equals the corresponding masked expression. While reset is low,
the ordinary READY register's asynchronous clear controls its output and the
D value is irrelevant. This avoids routing reset through the predictor logic.

The return candidate registers next occupancy's nonempty flag as VALID.
It keeps two payload slots, speculative writes only to a free slot, ordering,
full-queue admission bubbles and the original transfer latency. VALID clears
asynchronously and updates only on active clock edges. This adds one control
register; it does not add a payload slot. Both designs require coordinated cold
reset spanning active clock edges for synchronous ownership state. Arbitrarily
short reset pulses are outside that contract.

## Reproduction and acceptance

Run from this repository with a fresh output directory:

```text
python -B fpga/experimental/axi_bridge_dma/prepare_hbm_control_predictors.py DONOR_ADAPTER.sv FRESH_OUTPUT
```

The preparer exports reference/candidate RTL, transformed existing banks, the
new return bank, native runner and reviewer. The runner uses the private
`ntrpu-side/axi-bridge-hbm-native-join` namespace, two CPU affinity, nice 10,
bounded XSIM stages and a 600-second synthesis limit. Four separate synthesis
units target `xcvu47p-fsvh2892-2-e` with a 2.222 ns clock.

The independent scoreboards test widths 265/519, every five-cycle capture
control word and six-cycle return control word, randomized stalls, capacity,
reset discards and payload retention. Tracker depths 1/2/4/8 exercise up to
128 beat credits, simultaneous address/response transfers, protocol faults
and same-edge request faults. Additional assertions compare active-edge raw
and masked predictions. Eight deliberate RTL mutations must fail for their
specified invariants; XSIM's exit code alone is insufficient.

Native structure checks require direct ordinary FDCE outputs, constant enable,
correct reset polarity and clock, no reset startpoint in either candidate D
cone, full payload inventories and VALID-dependent ownership. Payload enables
must retain their free-slot dependency. No timing exceptions are introduced.

## Evidence scope

Every batch, including failures, is archived with source and file hashes.
`REVIEW.json`, `FINAL_PROCESS_AUDIT.json` and `FINAL_BACKUP.json` record closure.
Preparation replay reconstructs its original bench context from frozen files
in `reference/`; exported benches are already transformed. The final ZIP must
be restored and rechecked before this experiment is considered backed up.

## Native results

The complete batch `board-control-predictors-sim-20261014T005041Z` passed all
32 stages in 341.010 seconds, including 207.268 seconds for four synthesis
units. Its archive contains 103 verified evidence files (plus `FILES.json`),
943,404 bytes, SHA-256
`90530225bc259b76e22c8c16536803c611009e2519bfe7b6ce6d6daed1ea7a46`.

The selected bank performed 3,752,609 checks and rejected all eight RTL
mutations. Both candidate D cones exclude `rst_n`; both outputs are driven
directly by ordinary FDCE Q pins with the expected asynchronous clear.

| Unit | Reference | Candidate |
|---|---:|---:|
| Tracker LUT primitives | 638 | 642 |
| Tracker FDRE / FDCE | 1,139 / 1 | 1,139 / 1 |
| Tracker payload registers | 1,060 | 1,060 |
| Return LUT primitives | 530 | 530 |
| Return FDRE / FDCE | 1,042 / 0 | 1,042 / 1 |
| Return payload registers | 1,038 | 1,038 |

These counts describe separate synthesis units, not full-board utilization.
The candidate checkpoints are:

- Tracker: `e09e228d5aa6a8c388e6d9d8c9a93a9726c8ea81a9082897c76538c4570c40d8`.
- Return: `45b7357e96699b3fb934e76c8cd63a67559f2f6965b69627c78e4797cb27bb58`.

Two earlier failures remain archived: `20261014T004345Z` used the reserved
SystemVerilog identifier `bins` in the new testbench; `20261014T004431Z` passed
simulation but its Tcl auditor treated an empty hierarchy prefix incorrectly
and counted zero return capture registers. The corrections change only that
testbench variable and the auditor condition. `REVISION_DIFFS.json` records
the exact edits; candidate RTL is identical across all three preparations.

The reviewer also rejected 14 deliberately corrupted structure reports.
All three preparations replay exactly and reject overwriting an existing
output directory. Closure audits 216 remote evidence files, 51 staged sources
and three archives; no owned active or zombie process remained. The sealed
predecessor archive and its adapter source passed their hash checks.

## Remaining physical qualification

These are unit simulation and synthesis results. The next physical experiment
must bind the exact tested modules into both board channels, audit READY and
return VALID D/Q/clear paths, measure CDC write-enable timing, and rerun the
existing reset, skew, CDC, routing and checkpoint-reopen checks. Board timing,
HBM/PCIe hardware traffic and manufacturing readiness remain unqualified.
