# Shared BREADY predictor placement

This isolated VU47P/HBM experiment follows [HBM_BREADY_PLACEMENT.md](HBM_BREADY_PLACEMENT.md).
Its direct donor is the rejected four-cell placement candidate
`1a2eae28f1151ace24f3c6a033c6bd13d6946ec21be3d86b4b85d90b2bfa9102`.
Using it for a controlled experiment does not promote it to the selected reference.
Evidence lives in `D:/AI/artifacts/ntrpu-axi-bridge-hbm-bready-shared-placement/`.

## Scope and hypothesis

Move only `weight_loader/bready_predictor_i_2`, a LUT2 with INIT `4'h1`, from
`SLICE_X7Y11/D5LUT` to `SLICE_X63Y11/F6LUT`, next to both FSM state registers.
The previous channel 1 path spends 2.361 ns on the shared LUT output net.
Both channel 0 and channel 1 consume that result, so both must be measured.
The native stage checks that both F5LUT and F6LUT destinations are vacant before
moving anything. No clock sink moves or clock-tree rebuild is requested.

## Evidence and acceptance

Freeze the input hash, plan, inherited audits and runner into a fresh directory.
Use a private native batch with two CPUs, nice 10 and a 1,800-second timeout.
Route with `route_design -preserve -no_psir`, finalize and save a new checkpoint.
Compare all 120,241 primitive records, both input cones, the shared drivers and
two register endpoints, all 8,321 clock-net pins, and the clock's PIP inventory.
Reject collateral cell changes or changes to the clock route/connectivity.
This is selected physical/connectivity evidence, not formal equivalence.

AMD documents [route preservation](https://docs.amd.com/r/en-US/ug904-vivado-implementation/Using-Other-route_design-Options)
and [PIP queries by net](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_pips).

Measure BREADY D, native/vendor output, hold and reset recovery/removal in both
channels, plus full-board timing, bus skew, bounded paths, response capture and
coverage. Compare against both the immediate experimental donor and original
`ada41ba33828cbb83f8f2159ae84c4a1568bf1bde86913665f705ca1337d7d15` design.
The immediate donor already fails bus skew by 0.079 ns; improving BREADY alone
cannot establish full-board closure. Independently reopen the routed checkpoint
before sealing the evidence. Timing waivers and RTL changes are outside scope.

## Measured result

The experiment improves both BREADY inputs relative to the direct donor, but
**does not meet timing and is not promoted**. Independent reopening reproduced all targeted measurements, inherited board
contracts and physical inventories exactly. The routed candidate hash is
`c6e92d6cb5ea3c5a6a966c8ad00dfca7c094d721d474c2fd42d8575f25bb7ea4`.

All slack values below are ns. The original column is the design before the
four-cell placement experiment, not a timing-qualified board.

| Measurement | Original | Direct donor | Shared-LUT candidate |
| --- | ---: | ---: | ---: |
| Channel 0 BREADY D setup | -0.010 | +0.056 | +0.065 |
| Channel 1 BREADY D setup | -0.231 | -1.981 | -0.482 |
| Channel 1 output setup | -0.403 | +1.139 | +1.139 |
| Channel 1 vendor feedback setup | -0.159 | +0.534 | +0.534 |
| Global WNS | -0.403 | -1.981 | -0.488 |
| Global TNS | -256.046 | -416.725 | -415.238 |
| Failing setup endpoints | 2,070 | 3,099 | 3,099 |
| Minimum bus-skew slack | +0.014 | -0.079 | -0.079 |
| Failing bus-skew groups | 0 / 18 | 1 / 18 | 1 / 18 |
| Response enable worst setup | -0.114 | -0.323 | -0.323 |
| Failing response enable endpoints | 14 / 1,060 | 240 / 1,060 | 240 / 1,060 |

Hold and pulse-width failures remain zero. BREADY D hold remains +0.389 ns for
channel 0 and +0.372 ns for channel 1. Native outputs, vendor feedback, and
BREADY reset recovery/removal retain their direct-donor margins. All 100,262
routable nets are complete, with no routing errors.

The full primitive inventory confirms exactly one cell moved. Types/INIT,
1,699 cone-pin records across 17 cells, both shared drivers/endpoints, all 8,321
clock pins and all 2,706 clock PIPs are unchanged. There is no clock repair in
this experiment. The inherited board audits remain part of the review.

## Remaining critical paths

The channel 1 input maximum now starts at `FSM_sequential_ms_reg[1]`, passes
through `aw_pending_i_2`, and reaches the predictor LUT5/LUT4 and BREADY register
at `SLICE_X122Y1`. The `bready_predictor/offer` net from `SLICE_X66Y12` consumes
1.646 ns. The complete data delay is 2.327 ns (1.992 ns routing), with -0.348 ns
clock skew. Improving the shared `response` decode exposed this other path.

The global -0.488 ns path is now `memory_distribution_reg` to synchronous reset
pins in the channel 1 protocol converter. The loader payload bus-skew violation
and response-capture failures from the direct donor remain. Moving one LUT
cannot be presented as fixing those independent limits.

A next physical experiment should evaluate an intermediate location for the
channel 1 register/predictor cluster between the control logic and HBM, keeping
the vendor gate near HBM and checking all input and output paths together.
Moving a register also requires clock-route and bus-skew review; clock behavior
from this combinational-only experiment cannot be assumed for that change.
Start from a fresh copy of the original pre-placement donor and remeasure the
clock tree and loader skew, rather than inheriting the rejected clock result.
Reset distribution remains a separate timing issue.

The selected reference remains
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.
## Closure and backup

Both native batches are collected and reviewed. `COMPARISON.json` distinguishes
improvement over the immediate experimental donor from failure to meet the
acceptance checks against the original design. The 2,195 bounded paths and 84
synchronous checks still pass. Coverage retains 1,154 unconstrained internal
endpoints and two inputs without input delay; these existing limits remain
relevant to full-board qualification.

The closure verifier exercises 90 evidence-rejection checks, replays both
frozen preparations, rejects output overwrites, and reruns the reviews on a
restored ZIP. It also verifies predecessor/selected-reference hashes and checks
only this experiment's private processes for leftovers. `FINAL_BACKUP.json`
records completion, archive SHA-256 and restored-file counts. The main simulation
work, DDR variant and sealed predecessor files remain outside this change.
