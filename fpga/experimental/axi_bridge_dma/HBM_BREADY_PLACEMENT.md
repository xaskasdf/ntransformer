# Channel 1 BREADY placement experiment

This isolated VU47P/HBM experiment tests the registered BREADY path from
[HBM_BREADY_PHYSICAL.md](HBM_BREADY_PHYSICAL.md). Evidence lives in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-bready-placement/`.

## Outcome

**Reject this placement as a replacement.** Moving the channel 1 register,
its two predictor LUTs and the vendor gate next to HBM improves the output but
makes the input control path substantially worse. The design is fully routed;
it does not meet timing. No reference is promoted and no board qualification
is claimed. Independent checkpoint reopening reproduced all targeted timing,
board contracts and physical inventories exactly.

Input checkpoint: `ada41ba33828cbb83f8f2159ae84c4a1568bf1bde86913665f705ca1337d7d15`.
Experimental routed checkpoint:
`1a2eae28f1151ace24f3c6a033c6bd13d6946ec21be3d86b4b85d90b2bfa9102`.
The selected reference remains
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.

## Physical scope

The read-only probe reproduced donor timing and inventoried 12 cone cells,
1,150 nearby sites and 179 completely empty sites. The selected destinations
were empty and adjacent to the native channel 1 HBM tile in the device grid.

| Cell | Original LOC/BEL | Destination LOC/BEL |
| --- | --- | --- |
| Channel 1 BREADY register | X41Y11 / CFF | X122Y1 / CFF |
| Predictor output LUT4 | X41Y11 / C6LUT | X122Y1 / C6LUT |
| Predictor input LUT5 | X41Y11 / H6LUT | X122Y1 / H6LUT |
| Vendor BVALID gate | X113Y11 / C6LUT | X122Y0 / A6LUT |

Across all 120,241 primitives, only these four cells changed LOC/BEL or fixed
flags. Types and INIT values are unchanged. The 865 cone-pin records and all
8,321 affected clock-net pins retain their connectivity. This checks physical
scope and selected connectivity; it is not whole-design formal equivalence.

## Measurements at 450 MHz

Slack values are in ns; positive values pass the corresponding check.

| Measurement | Donor | Candidate |
| --- | ---: | ---: |
| Channel 1 BREADY D setup | -0.231 | -1.981 |
| Channel 1 registered output setup | -0.403 | +1.139 |
| Channel 1 vendor BVALID feedback setup | -0.159 | +0.534 |
| Channel 1 D / output hold | +0.277 / +1.099 | +0.372 / +0.308 |
| Channel 1 CLR recovery / removal | +0.563 / +0.680 | +0.271 / +0.873 |
| Channel 0 BREADY D setup | -0.010 | +0.056 |
| Global WNS / TNS | -0.403 / -256.046 | -1.981 / -416.725 |
| Failing setup endpoints | 2,070 | 3,099 |
| Failing hold / pulse-width endpoints | 0 / 0 | 0 / 0 |
| Minimum bus-skew slack | +0.014 | -0.079 |
| Failing bus-skew groups | 0 / 18 | 1 / 18 |
| Response capture enable setup, worst | -0.114 | -0.323 |
| Failing response enable endpoints | 14 / 1,060 | 240 / 1,060 |

All 100,262 routable nets are fully routed with zero routing errors. The 2,195
bounded paths and 84 synchronous checks still pass. Timing coverage remains
1,154 unconstrained internal endpoints and two inputs without input delay;
there are no missing/multiple clocks or timing loops. These existing coverage
limits remain relevant to any eventual qualification.

The new worst path runs from `FSM_sequential_ms_reg[0]` at `SLICE_X63Y11`,
through shared LUT `bready_predictor_i_2` at `SLICE_X7Y11`, to the relocated
LUT/register at `SLICE_X122Y1`. Its data delay is 3.828 ns, including 3.582 ns
of routing. The shared LUT's output net alone consumes 2.361 ns. The failed
skew group spans the loader payload transfer from `axi_aclk_out` to
`clk_out2_fabric_clocks`; its measured skew is 1.190 ns against a 1.111 ns bound.

## Reproducible flow and diagnostics

`prepare_hbm_bready_placement.py` freezes the native helper, inherited board
audits, checkpoint hash and explicit placement plan. It refuses an existing
output directory. `review_hbm_bready_placement.py` verifies source bindings,
complete primitive inventories, cone/clock connectivity and inherited proofs.
Each private native batch uses two CPUs, nice 10 and a 1,800-second timeout.

The successful ECO applies the declared `place_cell` movements, then runs:

```tcl
set net [get_nets {clock_generator/inst/clk_out2}]
route_design -unroute -nets $net
update_clock_routing -net $net
write_checkpoint placed.dcp
route_design -preserve -no_psir
route_design -finalize
write_checkpoint routed.dcp
```

AMD documents [explicit placement](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/place_cell),
[clock-route cleanup before rebuilding](https://docs.amd.com/r/2024.1-English/ug835-vivado-tcl-commands/update_clock_routing),
and [routing/finalization options](https://docs.amd.com/r/en-US/ug904-vivado-implementation/Using-Other-route_design-Options).
Rebuilding the affected clock can change timing elsewhere in its fanout even
when all cell positions are held constant; the full-board comparisons therefore
remain mandatory.

Six failed/stopped diagnostic batches are retained in `DIAGNOSTICS.json`:
three inspection/export issues before movement, then three incomplete clock
route attempts. Preserved routing alone, a moved-sink clock update, and fixed
placement followed by preserved routing each left 12 HBM clock pins unrouted.
Removing the entire affected clock route before rebuilding resolved that error.
The fixed-placer attempt's `placed.dcp` is preserved as diagnostic evidence.
No timing result from an incomplete route is accepted.

## Next experiment

Evaluate placement of the shared predictor relative to the FSM and both channel
consumers, using fresh donor copies and an expanded explicit movement plan.
Measure both BREADY input/output paths, clock skew, loader bus skew and response
capture timing together. The current evidence supports this investigation, not
a claim that moving the shared LUT will close timing. Replication or an added
pipeline stage would be a separate RTL change requiring functional validation.

All nine native batches are collected: six preserved diagnostics, one read-only
probe, one ECO and one independent reopen. `COMPARISON.json` records the failed
acceptance checks. The local closure verifier runs 87 negative evidence checks,
replays all nine frozen preparations (including refusal to overwrite an existing
output), and verifies every file plus the reviews after restoring the ZIP.
`FINAL_BACKUP.json` records completion, archive SHA-256 and restoration counts.

The DDR design, main simulation jobs, RTL and sealed predecessor artifacts are
outside this experiment. Closure checks the predecessor hashes and selected
reference, and audits only this experiment's private processes for leftovers.
