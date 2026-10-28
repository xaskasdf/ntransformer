# Board integration of HBM reset distribution

## Scope

This private VU47P/HBM experiment integrates the candidate validated in
[HBM_RESET_DISTRIBUTION.md](HBM_RESET_DISTRIBUTION.md). Its local evidence root
is `D:/AI/artifacts/ntrpu-axi-bridge-hbm-reset-integration/`. Main-thread jobs,
sealed predecessors and the separate DDR board variant are outside its scope.

The baseline is the complete routed board checkpoint
`3038b257585c6c64325558c43446613f04c789e507edba178460ca1d25009be1`:
WNS -0.413 ns, TNS -803.549 ns after reopening, 4,555 failing setup/recovery
endpoints, no hold failures and a minimum bus-skew margin of 0.002 ns.

## Exact integration change

`integrate_hbm_reset_distribution.tcl` adds
`core/reader/startup/memory_distribution_reg`, an FDCE with initial zero,
constant D/CE, the existing memory clock and active-low asynchronous clear
from the canonical three-stage memory synchronizer. It drives precisely the
original 1,091 reset consumers, including readiness feedback. The canonical
release then drives only the new CLR pin.

The edit adds one cell and one net. The existing 4 ns datapath-only constraint
to `core/reader/startup/ready_meta_reg[2]/D` moves to the new source register.
No clock period, budget or reset consumer is changed. Release is delayed by
one memory edge, as covered by the isolated functional tests.

`rtl-candidate/` also contains the complete frozen source set with the same
startup change and its original module name. This run uses a netlist ECO;
it does not claim a fresh full-board RTL synthesis or native traffic run.

## Native flow and evidence

Each stage runs in its own VM directory with two CPUs and nice level 10.
Edit/reopen stages allow 900 seconds and incremental implementation 1,800.
Every checkpoint is saved before subsequent audits can fail.

The ECO uses AMD's documented
[netlist cell insertion](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/create_cell),
[incremental checkpoint reuse](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/read_checkpoint)
and subsequent `place_design -no_psip` / `route_design -no_psir`.
The edited donor supplies its retained placement and routing; reuse reports
record the before/placed/routed states. Placer and router physical synthesis
are disabled to keep the netlist delta measurable.

The first physical trial incorrectly used `place_design -eco`. Vivado emitted
`Vivado_Tcl 4-2610` because this option targets Versal, yet continued working.
That private process group was stopped after 810.265 seconds and its complete
22-file failure archive was restored locally. Its estimates are not accepted
as board results. The replacement follows AMD's
[UltraScale+ flow](https://docs.amd.com/r/en-US/ug904-vivado-implementation/Run-Section);
the original edit checkpoint and constraints are retained.

The complete XDC is exported, exactly one source expression is replaced, and
timing constraints are [reset and reloaded](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/reset_timing).
The independent reviewer compares command scopes as well as values. It
normalizes only two observed four-element Tcl list encodings and repeated
identical physical-property assignments; timing and waiver counts are retained.
Duplicate inherited IP waiver warnings and the existing LOCKED segmentation
warning remain visible in the log.

The edit checkpoint passed review:
`f937eab80cbcce8980e2f2195ff5336bd8c03201e47cdec08e551b92cbec6b59`.
Seven negative review controls detect wrong reset polarity, clock, consumers,
budget, cell inventory, property scope and timing-command multiplicity.

## Physical acceptance

The physical review checks all 2,195 bounded paths with exactly one named
source remapping, 84 retained board synchronous checks, 18 bus-skew groups,
reference-clock ownership, loader connections, CDC, DRC and complete routing.
It additionally requires the distribution register to be placed and records
its recovery/removal paths without timing cuts.

The selected result must be reopened and reviewed against its saved reports.
Global timing failure or report differences remain explicit. A completed
route alone does not qualify board pinout, hardware operation or timing closure.

## Measured result: preserve the original reference

The corrected route (`board-reset-integrate-inspect-20261013T185854Z`,
1,129.408 seconds) and independent reopen (`20261013T192438Z`, 481.149 seconds)
passed structural, constraint, CDC, DRC and routing review. Their compared
timing, topology and scoped reports agree exactly at exported precision.

| Metric | Original reopened board | Reset distribution, reopened |
| --- | ---: | ---: |
| WNS | -0.413 ns | -0.509 ns |
| TNS | -803.549 ns | -711.005 ns |
| Failing setup/recovery endpoints | 4,555 | 3,992 |
| Hold failures | 0 | 0 |
| Minimum bus-skew margin, 18 groups | 0.002 ns | 0.044 ns |

All 2,195 bounded paths and 84 retained board synchronous checks pass. The new
FDCE is at `SLICE_X113Y15/SLICEL.AFF2`; its recovery/removal margins are
1.489/0.231 ns. Its canonical source has one load, and the original 1,091
distributed reset consumers are unchanged.

The initial strict review rejected two native representation changes.
Reopening both exact checkpoints established that 12 removed HBM APB status
net aliases had no leaf consumers or top-level ports. All cells remain.
Sixteen PCIe calibration false paths changed from filtered queries to explicit
lists; native queries on both checkpoints prove identical membership of all
624 cells across 32 selectors. The reviewer accepts only these proven changes.
It retains command multiplicity and rejects any additional timing exception.
A failed first probe used an unsupported `get_nets -of_objects` object type;
its 105.834-second batch is also preserved, followed by the corrected query.

The candidate checkpoint is
`a990f1837d4dcaef6a271b6b625bc7235d2b0187ec517bde658c913207de1fb5`.
It is archived **without promotion** because WNS worsens by 0.096 ns. Keep
`3038b257...` as the reference. `SELECTION.json` records this decision;
`ROUTE_DIAGNOSIS.json` identifies the worst path from HBM channel 0 to
`core/reader/channels[0].adapter/tracker/response_cut/slot1_reg[199]/CE`:
2.417 ns datapath and -0.199 ns clock skew. A subsequent physical experiment
should examine this enable path and the remaining distribution fanout before
considering promotion. Full-board RTL resynthesis and hardware traffic remain
unqualified.

Twelve negative review controls passed. The closure audit verified 222 remote
files, 779 unchanged predecessor files, and zero owned active/zombie processes.
Both failed batches remain in the final bundle.

## Reproduction

Prepare a fresh directory with `prepare_hbm_reset_integration.py`, supplying
the frozen audits, exact local donor checkpoint and `--mode edit`, `route` or
`reopen`. Launch with the private archived `launch.py inspect --prepared ...
--checkpoint ...` helper. Collect with `collect.py --collect`, then run
`review_jobs.py --write`. The reopen preparation also requires
`--edited-reference <local-edited.dcp>` and
`--remote-edited-reference <remote-edited.dcp>` for its read-only semantic probe.
Review physical batches only after that probe is collected.

Retain failed batches and review rejections alongside successful evidence.
`seal.py` restores the final archive and repeats reviews, both guard suites,
all five preparation revisions and selection without Vivado. The receipt
`FINAL_BACKUP.json` identifies the bundle and its verification results.
