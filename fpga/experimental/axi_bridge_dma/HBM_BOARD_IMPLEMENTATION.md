# VU47P board implementation experiment

This private experiment implements the sealed
[board CDC candidate](HBM_BOARD_CDC.md), checkpoint SHA-256
`724bb198d259ec00aaa5c237eccf814df24fcbfd4eb5cadcf96a222bb3b2f10c`.
It keeps the VU47P/HBM node separate from the proposed DDR512GiB card.
The main validation queue and all preceding experimental checkpoints are donors,
not work directories.

## Experimental physical assignments

| Input | Package ball(s) | Trial standard |
| --- | --- | --- |
| PCIe REFCLK P/N | AR40 / AR41 | Dedicated GT reference |
| PERST | BF41 | LVCMOS18 |
| Fabric reference | BK31 | LVCMOS18; BL31 remains reserved |
| Management reference P/N | BJ33 / BJ34 | LVDS |
| HBM reference 0 P/N | BK33 / BL33 | LVDS |
| HBM reference 1 P/N | BH32 / BJ32 | LVDS |
| Cold-reset supervisor | BG33 | LVCMOS18 |

The three differential clock pairs use HP bank 64 in SLR0. BG33 is the
unshared `IO_T2U_N12_64` ordinary input selected for this trial. The script
checks bonding, exact package functions, bank type, differential partners,
SLR and the PCIe reference site `GTYE4_COMMON_X0Y1` against Vivado's device
database. It adds eleven package assignments, nine I/O standards and one
reference-buffer LOC; it does not assign clock-buffer locations manually.

These assignments assume 1.8 V bank supplies and appropriate external sources;
they do not release a PCB pinout, choose an oscillator or resolve PERST level
translation or receiver termination. LVDS use in HP banks follows
[UG571](https://docs.amd.com/api/khub/documents/kFbaUC5HGcXyGNauhgU6Gw/content).
HBM reference GCIO must share the HBM SLR and use the specified buffered clock
path; the source must satisfy the jitter requirement in
[PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking).

## Stage boundaries and checks

`prepare_hbm_board_implementation.py` copies the frozen CDC audits and creates
a fresh stage. `implement_hbm_board.tcl` separately handles pin assignment and
optimization, placement, routing, and final reopening. Each completed physical
operation saves its checkpoint before generating reports or checking contracts.
A failed audit therefore retains useful diagnostics without marking the stage
validated.

The post-stage checks cover 1,503 native transport/control paths, 390 snapshot
paths, 302 loader paths, loader/snapshot synchronizers, the registered-readiness
source, both MMCM LOCKED bounds, and the HBM reference trees. The eighty-eight
board synchronous checks are retained; only complete pruning of the unused
`lock_fault_reg` may remove its four checks. No functional RTL changes, broad
clock cuts, artificial input delays, waivers, severity reductions or dedicated
clock-routing overrides are introduced.

Full CDC, DRC, methodology, timing, bus-skew, exception, utilization, I/O and
route-status reports accompany each stage. Placement/routing diagnostics are
distinct from timing closure, native traffic validation and board qualification.

## Evidence and reproduction

Evidence root: `D:/AI/artifacts/ntrpu-axi-bridge-hbm-board-implementation`.
The private VM runner uses two CPUs and nice 10. Optimization/reopening have
900-second tool limits; placement and routing each have 2,400-second limits.
Each stage is restored locally and hashed before its results are accepted.

```text
python -B prepare_hbm_board_implementation.py FROZEN_CDC_AUDITS RESTORED_CDC_APPLY NEW_PREPARED --mode optimize
python -B NEW_PREPARED/run_hbm_native_join.py NEW_PRIVATE_VM_JOB --prepared NEW_PREPARED --settings VIVADO_SETTINGS --stage inspect --checkpoint DONOR_DCP
python -B review_hbm_board_implementation.py RESTORED_STAGE RESTORED_CDC_APPLY DONOR_DCP --mode optimized
```

## Reviewed optimization and reset encoding

The first optimization batch (`board-impl-inspect-20261013T151853Z`) saved
`optimized.dcp` with SHA-256
`01919f8e55f312873f0620e8ea6c8e980d7fee22bea878a21c7e1032174f2609`.
Pinned and optimized DRC each reported zero findings. Its subsequent audit
failed because the synthesis-only guard expected a separate reset inverter;
the original failed receipt is preserved.

The read-only probe `153051Z` established that Vivado absorbed this inverter
into `loader_readiness/ready_reg` (`FDCE`, `INIT=0`, `IS_CLR_INVERTED=1`).
Its CLR driver remains `core/endpoint/guard/cold_m_reg[2]/Q`. The native FDCE
model computes `CLR_in = CLR ^ IS_CLR_INVERTED`; the active-low canonical
reset is therefore preserved. The adapted checker verifies both equivalent
encodings explicitly. A preceding probe (`152734Z`) failed on a Tcl typed
collection error and is also retained.

Independent re-audit `153427Z` passed on the same optimized DCP without
repeating optimization: all 2,195 bounded paths, 84 remaining board synchronous
checks and the four deliberately pruned unused diagnostic checks were accounted
for. CDC has no Critical findings; all 18 reported bus-skew groups pass before
placement. Estimated WNS/WHS/WPWS are -0.269/-0.842/-0.092 ns, so this stage
does not close timing.

## Placement evidence

Batch `board-impl-inspect-20261013T154043Z` completed in 1,702 seconds. Its
59 evidence files were restored locally from archive SHA-256
`28ae58bc1131d0658489502cafb05192c1713afd777284565b3a3c6f459ded27`.
The placed checkpoint is
`2950361d120bf956db679f723531f7deb8604c9c26aa8fa09f86bded6ee05e35`.
All 2,195 bounded paths and the remaining 84 board synchronous checks pass;
the HBM reference tree and canonical readiness reset remain unchanged.
CDC still has no Critical finding.

Post-placement WNS/WHS/WPWS are -0.533/-0.728/0.000 ns. The worst reported
recovery path runs from `core/reader/startup/memory_release_reg[2]` to the
second HBM stack's AXI15 `ARESET_N`; the worst hold path is PCIe MSI logic.
The loader address/payload transfer has one bus-skew failure: 1.348 ns actual
against 1.111 ns required, slack -0.237 ns. Individual path bounds passing
does not remove this separate bus-skew requirement.

DRC reports the expected unrouted RTSTAT-13 plus fourteen PDCN-1569 warnings.
Routing starts from this exact placed DCP in private batch `161027Z`.

## Routed evidence

Batch `board-impl-inspect-20261013T161027Z` completed in 1,072 seconds and
passed independent review after restoring its 59 evidence files. Archive SHA-256:
`359a2295073a6e0c50df235f91446a56ecdfdc751e5a5e7e2ce12eac9f7b9048`.
Routed DCP SHA-256:
`885c2ec17d7fd8c4dd939556ce6cd3d3742ca7fdab851cc6695bd769112d7d05`.

All 100,260 routable nets are fully routed with zero routing errors. All 2,195
bounded paths and 84 board synchronous checks pass. CDC has zero Critical
findings. All eighteen bus-skew groups pass, but their minimum margin is only
0.002 ns; preserve this check in every subsequent physical revision.

| Timing-summary metric | Placed | Routed |
| --- | ---: | ---: |
| WNS | -0.533 ns | -0.457 ns |
| WHS | -0.728 ns | +0.010 ns |
| Setup/recovery failing endpoints | 3,524 | 5,192 |
| Hold failing endpoints | 6,558 | 0 |
| Pulse-width failing endpoints | 0 | 0 |

The final timing summary identifies these remaining paths:

- HBM AXI0 to channel 0 `response_cut/slot1_reg[230]/CE`: setup -0.457 ns;
  2.360 ns data delay, of which 1.643 ns is routing.
- `memory_release_reg[2]` to HBM stack 1 AXI15 `ARESET_N`: recovery -0.317 ns.
- `source_join/tag_reg[30]` to lane 1 return `output_data_reg[206]/CE`:
  setup -0.172 ns.

These are starting points for the next locality/fanout and pipeline review,
not permission to exclude the paths from timing. Routing alone does not close
setup: more endpoints fail than after placement despite improved worst slack.
The router's extra `tight_setup_hold_pins.txt` is backed up separately with a
hash receipt under this batch's `routing-supplement/` directory.

Post-route DRC retains fourteen PDCN-1569 warnings and one RTSTAT-10 warning
covering nets with no routable loads. Methodology additionally reports eighteen
[TIMING-16 findings](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/TIMING-16-Large-Setup-Violation),
including PCIe paths at -1.058 ns. Their difference from the timing-summary WNS
has not been reconciled; retain both reports for targeted analysis. Neither
report is waived or treated as sign-off. Read-only reopening
`board-impl-inspect-20261013T162951Z` completed in 359 seconds and independently
matched the routed timing, scoped checks, CDC, DRC, methodology, route counts
and effective constraints. Its 58 restored files have archive SHA-256
`cba7c8e16c12a9964f9a32f1aef97699f29a05c1c547c294c0b051e300772072`.

## Outstanding qualification

These runs remain out-of-context; Vivado explicitly warns that not all
connectivity DRCs are exercised. The reviewed design retains 1,154 unconstrained
XSDB endpoints and two missing input-delay checks. Methodology still reports
LUTAR-1 (including a loader reset expression), TIMING-9, TIMING-13 at the MMCM
LOCKED boundary, TIMING-18 for PERST and TIMING-34 for a tight bus-skew bound.
None of these findings has been waived or assigned a lower severity.

The final reviewer requires complete error-free routing separately from
preservation of timing contracts. Timing closure, full top-level sign-off,
electrical pin review, power/thermal design and native hardware traffic remain
separate requirements before releasing a board.

## Local preservation

All seven batches, including both diagnostic failures, are closed and backed
up under the evidence root. The final process audit found no live or zombie
processes in this experiment's fourteen owned groups. It also verified 299
remote evidence files, the 65-file CDC donor and its seven frozen Tcl audits.

`SELECTION.json` identifies the checkpoint chain and accepted reviews.
`seal.py` packages the evidence, frozen donor, prepared revisions and final
sources, restores the archive, repeats six independent reviews and regenerates
the four accepted audit/place/route/reopen preparations. Thirteen altered
evidence cases exercise rejection of incorrect reset polarity, changed pins,
added clock cuts and incomplete routing. The failed first optimization/probe
revisions remain preserved without being relabeled as successful.

Use `FINAL_BACKUP.json` for the sealed archive's digest and restoration receipt.
This milestone is a reproducible physical implementation experiment; it does
not release a bitstream or a PCB design.
