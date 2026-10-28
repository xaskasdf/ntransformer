# HBM reset distribution and registered BREADY

This isolated VU47P experiment diagnoses the routed board from
[HBM_BOUNDARY_PHYSICAL.md](HBM_BOUNDARY_PHYSICAL.md) and validates a registered
write-response READY candidate using the complete weight loader. Evidence is
under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-reset-bready/`. The candidate has passed
unit simulation and synthesis; full-board integration and physical timing
qualification remain pending. The DDR design is outside this experiment.

## Physical diagnosis

The read-only probe uses routed checkpoint SHA-256
`3463c9ca3420b52a5fd98f53a14e4583784935649c931f8403a71fb146890461`.
It reproduces WNS -0.409 ns, TNS -263.161 ns, 2,136 failing setup endpoints,
zero hold failures and all 100,261 routable nets routed. These are measurements
of the existing board, not the new BREADY candidate.

`memory_distribution_reg` is an ordinary FDCE at `SLICE_X113Y15`, driven by
canonical `memory_release_reg[2]`. Its 1,090 direct leaf loads are all in SLR0:

| Consumer | Loads |
|---|---:|
| Weight loader | 315 |
| HBM0 adapter | 302 |
| HBM1 adapter | 302 |
| Control snapshot | 133 |
| Native HBM | 32 |
| Memory watch | 3 |
| Two crossings and startup | 3 |

The inventory includes 1,011 memory-clock loads, 78 combinational loads and one
reader-clock load. All 512 worst sampled paths from the distribution register
fail; this bounded sample is not a count of every failing reset path.

| Native boundary margin (ns) | HBM0 | HBM1 |
|---|---:|---:|
| BREADY setup | -0.409 | -0.203 |
| BREADY hold | +0.583 | +0.522 |
| Validated VALID clear recovery | +0.636 | -0.091 |
| Validated VALID clear removal | +0.695 | +1.047 |

HBM0's worst BREADY path starts at the distribution register, crosses two LUTs
and has 2.511 ns data delay. Its loader gate is at `SLICE_X8Y11`. HBM1's worst
path starts inside HBM and crosses the vendor BVALID feedback gate; it has
2.053 ns data delay. The exported LUT truth table and input bindings establish
that the native gate implements `AXI_BREADY || !AXI_BVALID`. Registering the
loader output does not remove that vendor feedback path.

The probe changes neither netlist nor constraints. Before/after XDC comparison
ignores only the generated comment naming `before.xdc` versus `after.xdc`;
all other bytes must match. No timing exceptions are added.

## Candidate behavior

`ntrpu_hbm_bready_predictor.sv` computes the next BREADY from the original
loader FSM, selected lane, pending AW/W flags and live handshakes. READY becomes
active on the same edge that enters `M_RESPONSE`, remains active while waiting
for BVALID, and clears when the response is consumed. The selected lane and
transaction ownership remain governed by the original FSM.

The generated loader replaces only its module name and combinational BREADY
assignment. Two ordinary FDCE registers drive the outputs directly. Cold reset
clears them asynchronously, including with `mem_clk` stopped; transport reset
retains a pending response. Payload, attribution, fault handling, sealing and
the rest of the loader remain unchanged. The existing coordinated cold-reset
contract still applies to the surrounding design.

## Native validation — 2026-10-14 UTC

Vivado/XSIM 2026.1 targeted `xcvu47p-fsvh2892-2-e` in private directories with
two-CPU affinity, nice 10, per-step timeouts and archived success/failure logs.
The successful unit batch took 202.623 seconds, including 101.249 seconds for
reference and candidate synthesis.

- Four loader seeds pass all ten scenario groups: 2,829 protocol assertions
  and 46,643 observations comparing 31 output signals against the frozen loader.
- The predictor passes 4,096 enumerated FSM/input cases, for 53,568 total checks.
- Added scenarios cover earliest B responses, both lanes, all AW/W orderings,
  transport reset while the memory clock is stopped, and asynchronous cold clear.
- Five intentional defects are rejected: stale READY, premature READY,
  ignored response, wrong lane and synchronous-only clear. XSIM can return zero
  after `$fatal`; both runner and reviewer inspect fatal diagnostics.
- Synthesis preserves all 935 port bits. Both output drivers are FDCE/Q with
  INIT=0, `mem_clk`, constant CE and verified active-low asynchronous reset
  through a LUT1 inverter. Their D cones contain no reset port in this unit.
- The complete loader changes from 635 LUTs / 1,335 FFs to 634 LUTs / 1,337 FFs.
  Neither uses RAM or DSP resources. No routed candidate timing is claimed.

The synthesized candidate checkpoint SHA-256 is
`8de2be8a28d2b75849d8c8b2d2aea2c93a44833f51933be2c5cb18e121845014`.
It is a unit checkpoint, not a deployable board checkpoint.

The first unit batch passed simulation but failed the topology audit because
standalone synthesis inserted OBUFs. Revision 2 uses `-mode out_of_context` for
both loaders. The original failed batch, its sources and logs are retained.

## Reproduction and evidence

Run preparation with explicit frozen inputs and a fresh destination:

```text
prepare_hbm_reset_bready_probe.py ROUTED_DCP NATIVE_RUNNER FRESH_PROBE
prepare_hbm_reset_bready.py BOARD_SOURCE ORIGINAL_LOADER_BENCH FRESH_UNIT
run_hbm_reset_bready.py PRIVATE_OUTPUT --prepared FRESH_UNIT --settings VIVADO_SETTINGS
review_hbm_reset_bready.py RESTORED_UNIT BOARD_SOURCE
```

Use the artifact's `review_jobs.py` to verify all batches and `finalize.py
--verify-only` after sealing. `REVIEW.json` records independent structural and
functional review, 22 corrupted-evidence rejection checks, and exact replay of
all three preparations with overwrite rejection. `FINAL_BACKUP.json` is the
authoritative ZIP hash and restoration receipt. `FINAL_PROCESS_AUDIT.json`
records checks of only this experiment's remote processes and frozen inputs.

## Next integration gate

Integrate the exact tested loader delta into a fresh full-board composition.
Repeat synthesis and route, followed by independent checkpoint reopen. Measure
BREADY D setup/hold, both output paths through the native gate, and cold-clear
recovery/removal. Recheck complete D fan-in: vendor integration can reintroduce
reset dependencies absent from an isolated unit. Retain all existing CDC,
ownership, coverage, routing and timing checks, with no new waivers.

Evaluate reset distribution as a separate change, using the measured consumer
groups. Preserve the cold-reset epoch and release cycle; do not add a local
pipeline stage implicitly. Any replicated distribution registers require
updated endpoint inventories and recovery/removal analysis before acceptance.
The canonical synchronizer and vendor BVALID gate remain unchanged in the
current candidate. The selected routed reference remains SHA-256
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.
