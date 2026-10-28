# Routed HBM timing experiment

## Scope and frozen reference

This isolated VU47P/HBM side experiment starts from the reviewed routed DCP
`885c2ec17d7fd8c4dd939556ce6cd3d3742ca7fdab851cc6695bd769112d7d05`.
It does not change the main validation queue or the separate DDR board design.
The preceding evidence and experimental pin assignments remain frozen.

Local evidence: `D:/AI/artifacts/ntrpu-axi-bridge-hbm-timing-closure/`.
Each VM batch has its own directory, two CPU affinity, nice level 10 and a
bounded runner. Diagnosis/reopening allow 900 seconds; physical optimization
allows 2,400 seconds. Every closed batch, including failures, is downloaded,
hashed, restored and reviewed before selection.

## Procedure and acceptance

`inspect_hbm_routed_timing.tcl` records 128 setup paths, eight hold paths and
six specific HBM, return-transport and PCIe targets. It compares timing before
and after a fresh methodology report, captures ignored-path queries and
requires unchanged effective constraints. A first diagnostic attempt failed
because Vivado has no generic `redirect` command; that failed batch is retained.

`optimize_hbm_routed_timing.tcl` performs one
`phys_opt_design -directive Explore` operation on a private copy. AMD documents
[post-route physical optimization](https://docs.amd.com/r/en-US/ug904-vivado-implementation/phys_opt_design)
and its [rerouting behavior](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Iterative-Flows).
The resulting checkpoint is saved before structural audits. There are no RTL
changes, clock reductions, additional timing cuts, or waivers in this trial.

`review_hbm_routed_timing.py` preserves the previous structural checks and
additionally requires identical timing commands before/after, all 2,195
bounded paths, the 84 retained board synchronous checks, eighteen passing
bus-skew groups and complete error-free routing. CDC Critical findings and
blocking physical DRC findings are rejected. Global timing failure is recorded
explicitly; passing these preservation checks is not timing sign-off.

## Reproduction

Prepare a fresh directory with `prepare_hbm_routed_timing.py`, supplying the
frozen audit directory, exact local donor DCP and `--mode diagnose`, `explore`
or `reopen`. Use the archived private `launch.py` and `collect.py` helpers for
VM execution and recovery. `review_jobs.py --write` independently checks the
restored batches using local frozen references. It never edits native evidence.

The baseline has WNS -0.457 ns, TNS -892.009 ns and 5,192 failing setup/recovery
endpoints. Hold and pulse-width checks pass. Minimum bus-skew margin is only
0.002 ns. Methodology reports eighteen TIMING-16 findings, including PCIe
paths down to -1.276 ns, which require reconciliation with the timing summary.

## Read-only diagnosis

Batch `board-timing-inspect-20261013T164836Z` completed in 232 seconds;
its 62 restored files passed independent review. The before/after timing
summaries and expanded constraints are identical. Three PCIe paths explain
the numerical discrepancy:

| Endpoint | TIMING-16 | Active constrained timing |
| --- | ---: | ---: |
| MSI request 23 reset | -1.058 ns | +0.942 ns |
| MSI request 10 reset | -1.063 ns | +0.937 ns |
| AXI read request FSM 0 enable | -1.276 ns | +0.724 ns |

All three timing paths report `MultiCycle Path Setup -start 2`, a 4 ns
requirement and a 2 ns source clock. The inherited IP constraint file
`ip_pcie4c_uscale_plus_impl_x0y0.xdc` applies paired setup/hold multicycle
constraints through the PCIe CFG output pins. The difference is exactly one
source-clock period. This explains the observed arithmetic; the internal
TIMING-16 algorithm has not been independently verified and the warning
remains unwaived. Reopening also queries all eighteen original endpoints.

The HBM response enable drives 265 register pins; its data path is 69.6%
routing delay. The sampled reset network has 1,091 leaf pins and 97.1% routing
delay. The return-transport enable has 603 leaf pins and seventeen logic
levels in the sampled path. These measurements support a physical locality
experiment before changing pipeline latency or queue behavior.

The diagnostic r2 log includes a deprecated `-nets` warning and ignored
formatting options combined with `-of_objects`. Numerical path tables and
reports were exported successfully. The final helper omits these flags;
archived r2 inputs retain the exact commands that produced the evidence.

## Physical result

Batch `board-timing-inspect-20261013T165312Z` completed in 788 seconds, including
reports and audits; physical optimization itself took 417 seconds. The 63
restored evidence files passed independent review. Candidate DCP SHA-256:
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`.
Archive SHA-256:
`f76a6320849d72bb23166eea835effbc05a17e6b6b5d82f613edb571a3f8382f`.

| Metric | Routed reference | Post-route Explore |
| --- | ---: | ---: |
| WNS | -0.457 ns | -0.413 ns |
| TNS | -892.009 ns | -803.482 ns |
| Failing setup/recovery endpoints | 5,192 | 4,555 |
| WHS | +0.010 ns | +0.005 ns |
| Hold / pulse-width failures | 0 / 0 | 0 / 0 |
| Minimum bus-skew margin | +0.002 ns | +0.002 ns |

The final standalone TNS report differs by 0.001 ns from the optimizer's
progress log; this table uses the restored final report. All 100,260 routable
nets remain fully routed. All 2,195 bounded paths, 84 retained synchronous
checks and eighteen skew groups pass. Expanded timing commands are unchanged;
no XDC commands were removed. CDC, DRC and methodology finding counts match
the reference. Eight deliberately corrupted evidence cases verify rejection
of new clock cuts, removed multicycle constraints, incomplete routing, negative
skew/bounded slack, wrong pins, Critical CDC and unhashed edits.

The worst remaining setup path now starts at `memory_release_reg[2]` and
ends inside channel 1's width-converter command FIFO: -0.413 ns, with 85.2%
of its 2.648 ns data delay in routing. HBM AXI15 reset recovery remains
-0.317 ns. The sampled return-transport group improves to -0.096 ns at
`output_data_reg[535]/CE` but still traverses the source tag/control cone.

The next design experiment should address locality/fanout of the canonical
memory reset release and the wide queue/control enables. Preserve coordinated
cold-reset semantics, accepted transactions, quarantine behavior and CDC
bounds; any added pipeline or local release stage needs functional reset and
backpressure validation. Do not infer timing closure from the smaller WNS.

## Reopening and preservation

Batch `board-timing-inspect-20261013T170928Z` completed in 380 seconds and its
83 files were restored locally. All eighteen original PCIe TIMING-16 paths
have positive constrained slack (minimum +0.724 ns), a 4 ns requirement and
the existing paired multicycle constraint. Every warning differs from the
queried slack by exactly 2 ns. The warnings remain present and unwaived.

Reopening exactly preserves fifteen structural/report fields and eleven
timing metrics, including WNS, WHS and failing/total endpoint counts. Expanded
constraints and all scoped path records also match. **Aggregate TNS does not
match exactly:** it changes from -803.482 to -803.549 ns, a -0.067 ns delta
localized to `clk_out2_fabric_clocks` (-777.550 to -777.617 ns). Both timing
summaries within the reopened session agree. The cause has not been verified;
do not attribute it to rounding without further evidence.

The original strict comparison failed and is retained as `finish_strict_r1.py`
and `REOPEN_DISCREPANCY.json`. `SELECTION.json` records the more conservative
reopened TNS and keeps `timing_reproducibility_qualified=false`. This candidate
is archived for continued experiments, not promoted to a timing-qualified
release. Investigating the aggregate discrepancy remains an explicit follow-up.

The closing audit verified 227 remote evidence files, all eight owned process
groups inactive and 566 prior local files unchanged. `seal.py` creates a local
bundle, restores and hashes every file, repeats the independent reviews and
comparisons, regenerates four exact prepared revisions and repeats eight
negative-evidence guards. See `FINAL_BACKUP.json` for the resulting digest.

## Qualification boundary

All results remain out-of-context experiments. The board is not electrically
qualified, the trial pinout is not released, and native hardware traffic has
not been validated. The inherited unconstrained debug endpoints, missing
PERST input-delay checks and other methodology findings remain visible.
