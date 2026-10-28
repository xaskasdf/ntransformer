# VU47P board startup and resumed placement

This experiment follows [HBM_BOARD_BOUNDARY.md](HBM_BOARD_BOUNDARY.md).
It uses private jobs and immutable donor checkpoints. Placement of the earlier
OOC composition and validation of the new board frontend are separate results;
neither transfers physical timing qualification to the other.

## Board frontend candidate

`ntrpu_hbm_board_inputs.sv` adds three IBUFDS inputs and one supervisor IBUF.
The native composition retains its management and two HBM BUFGCE instances;
the frontend adds no global buffers. The existing native HBM clock cascades
identified below still require correction. The fabric reference remains the
existing single-ended input on BK31. Reserved differential pairs are:

| Input | Positive / negative ball | Destination |
| --- | --- | --- |
| Management, 100 MHz | BJ33 / BJ34 | Existing management BUFGCE |
| HBM stack 0, 100 MHz | BK33 / BL33 | Existing HBM reference 0 BUFGCE |
| HBM stack 1, 100 MHz | BH32 / BJ32 | Existing HBM reference 1 BUFGCE |

The supervisor pad has no selected package ball or voltage standard. Clock
standards, termination, sources and jitter also remain electrical-design tasks.
AMD specifies IBUFDS electrical attributes on the top-level ports
([UG974](https://docs.amd.com/r/2025.1-English/ug974-vivado-ultrascale-libraries/IBUFDS));
HBM reference sources must use a GCIO in the HBM SLR and satisfy the clock-source
requirements in [PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking).
No unspecified standard is treated as an approved board constraint.

`prepare_hbm_board_frontend.py` derives a new native top from the frozen
registered-readiness sources. It replaces the three ideal reference inputs
with differential ports, internalizes `fabric_clock_reset`, `fabric_clock_locked`
and `configuration_rst_n`, and updates primary clocks to the positive pads.
PCIe PERST wiring and existing availability/transaction logic remain unchanged.
`BOARD_FRONTEND_BINDING.json` identifies both source versions. The copied older
binding files remain provenance for their donors, not acceptance of this top.

## Startup and recovery contract

`ntrpu_hbm_board_startup.sv` runs from the independent management clock:

1. Supervisor assertion asynchronously asserts the sequencer reset, including
   when management clocks stop. Deassertion passes through three registers.
2. Hold the fabric clock generator in reset for 16 management cycles by default.
3. Release the clock generator and wait for 32 consecutive high samples of its
   three-stage synchronized `LOCKED` signal before releasing configuration reset.
4. After release, sampled lock loss latches `lock_fault`; it never asserts
   configuration reset or clock-generator reset. Relocking does not clear it.

The cycle counts are prototype parameters, not approved power-sequencing times.
PERST is not an input to this sequencer. The native availability monitor still
owns traffic inhibition; the new diagnostic `lock_fault` has no software exposure.
HBM readiness additionally requires both APB completion indications before AXI
traffic ([PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/Non-AXI-Ports)).

Supervisor assertion denotes a deliberate cold configuration/power boundary.
The test does not approve resetting a running physical HBM controller in place.
Clock-source health, power rails and minimum supervisor pulse width need an
external board contract. A halted management clock cannot detect a short lock
loss that begins and ends while stopped. This sampled sequencer is not a
replacement for an independent hardware clock/power supervisor.

## Validation and scope

`tb_hbm_board_startup.sv` tests cycle-count pairs (1,1), (4,3), (16,32) and
(17,9): missing lock, rejected short lock, startup delay, running lock loss,
sticky fault, cold restart and supervisor action with halted management clocks.
`run_hbm_board_startup.py` also compiles an actual mutation that clears the
configuration epoch on lock loss; the intended assertion must reject it.

The standalone synthesis probe uses the real AMD input and global-clock
primitives. Its unconstrained asynchronous input ports are outside the reported
CDC coverage; an empty CDC report cannot qualify the integrated crossings.
NSTD-1 and UCIO-1 remain explicit findings until electrical standards and pins
are selected. Full native synthesis, traffic regression, CDC/constraint audits,
complete board implementation and routed timing remain required.

The selected standalone run `board-place-sim-20261013T124250Z` passed 554
comparisons across 47 scenario groups. The actual bad RTL was rejected at
`lock loss preserves epoch`; XSIM returned zero even on `$fatal`, so both the
runner and reviewer check diagnostic text rather than exit status alone.
The run completed in 302.72 seconds, including 220.61 seconds for synthesis
and reports. Its checkpoint SHA-256 is
`99c098d88dd3820f325638636942c981db56bae2c53681cc9aba52b5ead05db8`.

The selected sequencer has a direct three-register `LOCKED` synchronizer.
An earlier version synchronously cleared that chain during clock reset,
introducing unnecessary input logic. Its passing functional evidence remains
archived, but the final source removes that logic and was retested. Standalone
synthesis reports 18 flip-flops, 17 LUT primitives, three IBUFDS, one IBUF and
the three existing-style BUFGCEs. This utilization belongs only to the small
frontend probe, not to the complete native design.

The final native candidate is `prepared-native-frontend-r3`. Revision r1
retained obsolete ideal-port clock constraints and was never run; r2 corrects
those ports and moves declarations before use; r3 adopts the tested direct
synchronizer. Any future native constraint audit must account for the new
supervisor-to-sequencer reset source and physical clock ports. The older OOC
reset/clock boundary checks must not be copied over as if still qualified.

Native elaboration `board-place-elaborate-20261013T124928Z` completed in
216.18 seconds with the final sequencer source: 75 external bits, all four
native blocks present, zero errors and zero critical warnings. It retains
146 warnings, including unused IP ports and removal of the HBM IP's redundant
input buffers. This is RTL elaboration, not full native synthesis or timing.

## Completed placement and remaining physical findings

`board-place-inspect-20261013T123309Z` completed in 1,298.12 seconds and
exported `placed.dcp`, SHA-256
`a1c9d7cbb6e6694f6bf5395956007dad24d27443d8d6c2c035cb0542cabf4695`.
It places the previous registered-readiness composition, without the new
board frontend. Placement completion does not establish routed timing.

| Placed timing report | Result |
| --- | --- |
| Setup | WNS -0.537 ns; 1,181 failing endpoints |
| Hold | WHS -0.607 ns; 6,449 failing endpoints |
| Pulse width | WPWS 0.000 ns; zero failing endpoints |
| Bus skew | 18 reported constraints pass; minimum slack +0.147 ns |
| Readiness source register | Minimum setup +9.183 ns; hold +0.209 ns |

Expanded CDC retains 182 CDC-3, nine CDC-6, 36 CDC-9 and 2,425 CDC-15, with
zero critical entries. Readiness source paths preserve 10 ns setup and
0 ns hold requirements. Complete post-optimization constraint equivalence
has not been qualified. There are no waivers.

The DRC report has RTSTAT-13 (insufficient routing) and ten PDCN-1569 LUT
warnings. Methodology additionally reports two TIMING-2 and two TIMING-4
critical warnings: the HBM IP defines primary clocks on internal reference
pins downstream of the external primary clocks. Two CLKC-5 advisories expose
existing cascades from `hbm_reference0/1` to the HBM IP's reference BUFGCEs.
Resolve reference-buffer ownership and primary-clock definitions before
accepting insertion-delay analysis or launching a board routing campaign.

Other retained methodology findings include 16 LUTAR-1, one TIMING-9, the
previous LOCKED path-segmentation TIMING-13, 35 TIMING-16, two TIMING-18 and
one TIMING-34 on the tight skew bound reaching `weight_loader/address_reg[0]`.
A passing bus-skew report does not waive that warning or qualify the bound.

The timing report's worst setup path reaches `weight_loader/FSM_sequential_ms_reg[0]/CE`
from HBM AXI interface 0 at 450 MHz. Worst hold reaches PCIe
`CFGINTERRUPTMSIINT[14]` from its MSI request register. These are diagnostic
targets for the next implementation; no RTL timing fix was applied here.

## Evidence and reproduction

Artifacts belong to `D:/AI/artifacts/ntrpu-axi-bridge-hbm-board-place`.
`resume_hbm_board_placement.tcl` reopens optimized checkpoint SHA-256
`65e2a7ae84397ad674e1691e1029c6410c625a034f685cf48246f5a052eb90a8`,
applies no new assignments and saves placement before generating diagnostic
reports. Its private runner allows 1,800 seconds on two CPUs at reduced priority.
No routing or bitstream generation is requested in this experiment.

The final audit checked five closed jobs and 22 owned process groups: none
remain live or zombie. All 603 remote evidence files, the 236-file generation
donor and five local donor files retain their hashes. `FINAL_BACKUP.json`
records the ZIP hash, full local restore, repeated reviews and regeneration
of the final native candidate from its preserved source donor.

```text
python -B prepare_hbm_board_frontend.py READINESS_PREPARED NEW_CANDIDATE
python -B run_hbm_board_startup.py NEW_PRIVATE_VM_JOB --prepared PREPARED --settings VIVADO_SETTINGS
```
