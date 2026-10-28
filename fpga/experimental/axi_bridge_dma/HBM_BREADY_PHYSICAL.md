# Full-board registered BREADY integration

This isolated VU47P/HBM experiment integrates the complete weight loader tested
in [HBM_RESET_BREADY.md](HBM_RESET_BREADY.md). Its four seeds, ten scenario
groups and exhaustive predictor bank passed 53,568 checks and rejected five
intentional RTL defects. This experiment measures the physical consequences of
that exact candidate. The DDR variant and selected routed reference remain
separate.

## Frozen composition

Evidence lives in `D:/AI/artifacts/ntrpu-axi-bridge-hbm-bready-physical/`.
The board donor is the complete source composition from
[HBM_BOUNDARY_PHYSICAL.md](HBM_BOUNDARY_PHYSICAL.md). Its incremental routed
checkpoint SHA-256 is
`3463c9ca3420b52a5fd98f53a14e4583784935649c931f8403a71fb146890461`.
It has WNS -0.409 ns, TNS -263.161 ns and 2,136 failing setup endpoints.

`prepare_hbm_bready_physical.py` verifies the tested unit manifest and loader
donor hash, restores the board module name and appends the tested predictor.
Only `ntrpu_hbm_weight_loader.sv` and the synthesis audit entry change among
the donor files. `BREADY_PHYSICAL_BINDING.json` is the authoritative composition
record; other source manifests describe historical stages.

No additional response latency is intended: BREADY is predicted for the same
edge that enters or leaves `M_RESPONSE`. Ordinary output registers retain
asynchronous cold clear. Transport reset, transaction ownership, payload,
response attribution, fault handling and sealing keep the tested behavior.
This change adds no reset-distribution stage and does not change the vendor
HBM gate implementing `AXI_BREADY || !AXI_BVALID`.

## Native flow and acceptance evidence

Vivado 2026.1 targets `xcvu47p-fsvh2892-2-e`. Each native batch has a private
directory, two-CPU affinity, nice 10 and a 1,800-second timeout. A batch must be
copied, restored, hash-verified and reviewed before its dependent stage starts.
The sequence is full RTL synthesis, constraint application and optimization,
incremental placement/routing, and independent read-only checkpoint reopen.

```text
prepare_hbm_bready_physical.py BOARD_DONOR TESTED_UNIT FRESH_SYNTH
prepare_hbm_bready_stage.py FROZEN_AUDITS NATIVE_RUNNER INPUT_DCP FRESH_STAGE --mode constrain|route|reopen
```

Route preparation also requires `--incremental LOCAL_DCP` and
`--remote-incremental REMOTE_DCP`. The local hash binds the remote donor.

The new audit proves both FDCE output drivers, clock, constant enable, reset
polarity, full D fan-in and the exact native vendor gate function. It measures
setup/hold at D, register-to-HBM output, worst native BREADY from either source,
and clear recovery/removal. Reset dependencies are reported from the integrated
netlist; absence in the unit alone is insufficient.

Existing response ownership, payload storage, CDC write-enable, synchronous
path, skew, calibration-selector, reference-clock and reset checks remain in
force. Timing constraints must retain their prior semantics; no new waiver or
timing exception is permitted. Trial physical pins remain experimental.

Route and reopen must reproduce targeted and global measurements, routing,
coverage and inherited contracts. The artifact's `compare.py` compares the
result against the frozen donor and checks payload/CDC endpoint inventories.
`verify_guards.py` retains the previous 54 rejection checks and adds 20 BREADY
checks. Exact preparation replay and ZIP restoration are required before
`FINAL_BACKUP.json` can record a sealed result.

## Synthesis result

Full-board synthesis and structural review passed in 1,156.520 seconds. Both
BREADY drivers remain ordinary FDCE/Q registers at the memory clock, with
constant enable and proven asynchronous reset polarity. Their complete D cones
have no reset-distribution dependency after native HBM integration.

The board uses 61,108 LUTs and 65,735 FFs, compared with 61,109 LUTs and 65,733
FFs in the donor. LUTRAM, SRL and block RAM counts are unchanged. The synthesis
checkpoint SHA-256 is
`6cc4a58668cc788ad19f3ed692c6adb99e44945701a6e068cdfbc3eaf7a8a49f`.

## Routed result

The incremental route completed in 1,310.439 seconds. All 100,262 routable nets
are routed, with zero routing errors. The routed checkpoint SHA-256 is
`ada41ba33828cbb83f8f2159ae84c4a1568bf1bde86913665f705ca1337d7d15`.

| Global measurement | Board donor | BREADY candidate |
| --- | ---: | ---: |
| WNS (ns) | -0.409 | -0.403 |
| TNS (ns) | -263.161 | -256.046 |
| Failing setup endpoints | 2,136 | 2,070 |
| Failing hold endpoints | 0 | 0 |
| Minimum bus-skew slack (ns) | +0.044 | +0.014 |

The setup endpoint count changes from 196,247 to 196,251. This is a small global
improvement, not timing closure. All 2,195 bounded paths, 84 synchronous checks
and 18 bus-skew groups pass. Both sets of 592 CDC write-enable endpoints retain
their inventory and pass setup/hold. Payload and pruned-ID inventories are
unchanged. Coverage still reports 1,154 unconstrained internal endpoints and two
inputs without input delays; existing CDC and methodology warnings remain.

### BREADY at the 450 MHz memory clock

All values below are slack in ns; negative setup values fail. The native row
includes both the registered output and the vendor's BVALID feedback path.

| Measurement | Channel 0 | Channel 1 |
| --- | ---: | ---: |
| Donor worst native BREADY setup | -0.409 | -0.203 |
| Candidate worst native BREADY setup | +0.400 | -0.403 |
| Register-to-native BREADY setup | +1.250 | -0.403 |
| Vendor BVALID-to-native BREADY setup | +0.400 | -0.159 |
| BREADY register D setup | -0.010 | -0.231 |
| BREADY register D hold | +0.389 | +0.277 |
| Worst native BREADY hold | +0.245 | +0.963 |
| BREADY clear recovery | +0.110 | +0.563 |
| BREADY clear removal | +0.898 | +0.680 |

Both integrated D cones remain free of reset-distribution dependencies. The
canonical release still has one consumer; its distribution register now has
1,089 loads and recovery/removal slack of +1.489/+0.231 ns. Other failures remain:
READY D setup is -0.209 ns on channel 0, validated VALID D setup is -0.226 ns on
channel 0, and validated VALID clear recovery is -0.083 ns on channel 1.
Response-capture enable has 14 failing setup endpoints out of 1,060, with worst
slack -0.114 ns versus the donor's -0.104 ns.

### Next controlled experiment

Channel 1's new BREADY register is at `SLICE_X41Y11`, while the vendor gate is at
`SLICE_X113Y11`. Its output path has one LUT level and 2.016 ns data delay:
1.903 ns routing and 0.113 ns logic. This supports testing placement/routing of
that register and its input cone. Moving only the register cannot be assumed to
close BREADY: its D path already fails, and the separate vendor BVALID feedback
also fails by 0.159 ns on channel 1. Include the vendor gate and both sources in
the experiment; check clock skew, clear recovery/removal and inherited contracts.
Keep any reset-distribution change as a separate experiment, with no timing
waivers or relaxed constraints.

## Reopen and backup status

Independent checkpoint reopen completed in 520.863 seconds. All mandatory
targeted measurements, global timing, routing, coverage and inherited contracts
match the original route. The reopen adds a read-only report for each vendor
BVALID-to-BREADY path, separating it from the new register's output path without
changing the netlist or constraints. The minimum slack across those two sources
matches the native worst-path report for setup and hold on both channels.

All four native batches were copied locally and hash-verified. All 74 deliberate
evidence corruptions were rejected. `COMPARISON.json` records the comparison;
`REVIEW.json` includes the full proofs and exact preparation replay. Finalization
also checks predecessor hashes and the private process inventory, creates a ZIP,
restores every file and repeats verification from the restored copy.
`FINAL_BACKUP.json` records the sealed archive, SHA-256 and restoration counts.
To recheck a restored bundle, run `python -B finalize.py --verify-only` inside it.

This candidate is not qualified for board timing, native PCIe/HBM traffic,
bitstream release or electrical readiness. The selected reference remains
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.
