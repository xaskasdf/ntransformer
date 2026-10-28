# Native HBM board boundary and placement candidate

This stage continues [HBM_LOADER_READINESS.md](HBM_LOADER_READINESS.md) on
`xcvu47p-fsvh2892-2-e`. It preserves the registered-readiness checkpoint,
the earlier experimental artifacts and the main Vivado queue. The 512 GiB
DDR design remains a separate card.

## Device inventory and clock reservation

`probe_hbm_board_boundary.tcl` reopens the frozen checkpoint without assignments.
It exports all 2,892 package balls, 96 bonded GCIO pins, their clock regions and
SLRs, the native serial connections and each external port's primitive boundary.
The independent reviewer requires identical active XDC before and after.

Both HBM reference sites are in **SLR0**, clock region **X4Y0**. GCIO banks
64 through 67 are in SLR0, providing sixteen differential pairs. Banks 68-71
belong to SLR1 and banks 72-75 to SLR2; they are not interchangeable sources
for this HBM reference contract. AMD requires the source GCIO to be in the HBM
SLR, through BUFG or MMCM/BUFG, and specifies source jitter below 3 ps RMS
([PG276 clocking](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking)).

The initial reservation uses four available pairs in bank 64, all at X4Y0:

| Clock, 100 MHz | Positive / negative ball | Current boundary |
| --- | --- | --- |
| Fabric reference | BK31 / BL31 | Single-ended input on BK31; BL31 reserved |
| HBM stack 0 reference | BK33 / BL33 | BUFGCE input, no input pad |
| HBM stack 1 reference | BH32 / BJ32 | BUFGCE input, no input pad |
| Management/APB | BJ33 / BJ34 | BUFGCE input, no input pad |

This is a reservation for design, not a released PCB pinout. The latter three
references need physical input buffers in the board wrapper. Choosing a
differential fabric input would also require changing its existing input
buffer/Clocking Wizard configuration. Oscillator, final signal standards,
jitter, terminations and voltage compatibility remain to be selected.

`build_hbm_board_reservation.py` derives `contract.json`, `clock_reservation.tsv`,
`serial_lanes.tsv` and `trial_pins.tsv` from reviewed evidence. The serial table
contains all 64 PCIe signals, differential partners and lane-to-GT mappings;
connector pin numbers remain unset.

## Existing pads and OOC boundaries

There are 74 top-level port bits: 64 native GT serial pads, four other existing
pad bits, and six ports without pads. The four existing pads are PCIe REFCLK
P/N, PERST and the fabric reference. The six remaining boundaries are management
and both HBM references, `configuration_rst_n`, `fabric_clock_reset`, and
`fabric_clock_locked`. The last three require a defined supervisor/sequencer
connection and physical buffers if they remain external.

Vivado's property getter can return `LVCMOS18` for an unassigned ordinary
port. This default is not an explicit board constraint: the preserved DRC
still reports NSTD-1 on fabric reference and PERST. Reading IOSTANDARD on GT
terminals also emits a misleading "Setting property" warning; the inventory
contains no assignment command and its XDC comparison confirms no change.

## Placement experiment

`place_hbm_board_candidate.tcl` applies only this experimental assignment in
a fresh session: PCIe REFCLK AR40/AR41, PERST BF41, fabric reference BK31.
It requires matching package functions, bonding and the fabric GCIO's SLR.
The reference buffer occupies a BEL within `GTYE4_COMMON_X0Y1`; IBUFDS_GTE4 is
not a standalone site type. PERST and fabric use LVCMOS18 in this trial, assuming
1.8 V banks 65 and 64. PERST level translation and the bank supply design are
not selected or electrically qualified.

The script saves candidate, optimized and placed checkpoints when each stage
completes. It retains DRC, CDC, timing and effective constraints. It introduces
no artificial I/O delay, clock-source location, broad clock cut, waiver,
severity reduction or dedicated-clock-routing override. Placement of this
OOC composition cannot qualify the missing input-pad paths or release a board.

## Evidence and recovery

Artifacts belong to `D:/AI/artifacts/ntrpu-axi-bridge-hbm-board-boundary`.
Each job uses a new private VM directory, two CPUs, nice 10 and a 1,200-second
tool limit. Closed archives are restored locally and every file hash checked.

The read-only probe `board-boundary-inspect-20261013T091428Z` completed in
191.56 seconds. Its expanded CDC report (`-all_checks_per_endpoint -no_waiver`)
retains zero critical entries, 200 CDC-3, nine CDC-6, 36 CDC-9 and 2,425 CDC-15.
Five damaged-inventory controls are rejected by the independent reviewer.

Worst unplaced setup is -0.194 ns in the native read-width converter at
450 MHz; worst hold is -0.842 ns into the PCIe hard block. The reported
WPWS -0.092 ns includes the PCIe PIPECLK/CORECLK maximum-skew check, not a
demonstrated HBM reference-frequency limit. The readiness register's estimated
hold remains -0.066 ns. These findings require implementation evidence.

Two script failures are preserved: `092119Z` checked the negative reference
pin after assigning its positive partner, which Vivado automatically resolves;
`092524Z` queried the buffer reference name as if it were a site type. Neither
attempt reached placement or changed the donor checkpoint. The revised script
checks all donor pins before assigning either half and resolves the GT common
site from the package database.

The selected placement job `board-boundary-inspect-20261013T092934Z` reached
phase 4.1.1.3, post-placement timing optimization, before its 1,200-second
limit. The runner recorded `FAILED`, return code -15 and `timed_out=true`;
there is no exported placed checkpoint or final placed report. The log's
intermediate WNS -0.742 ns / TNS -245.116 ns is diagnostic only.

Both completed stages, pinned and optimized, report zero DRC findings and
zero critical CDC entries. Optimization changes CDC-3 from 200 to 182;
CDC-6/9/15 remain 9/36/2,425. That count change does not independently prove
constraint or crossing equivalence after optimization. The optimized timing
report has WNS -0.269 ns, WHS -0.842 ns and WPWS -0.092 ns. It still lists
1,154 HBM XSDB endpoints unconstrained due to constant clocks and two inputs
without delay. Vivado warns that OOC mode omits some connectivity DRCs and
that bus-skew constraints require a separate `report_bus_skew` review.

`review_hbm_board_placement.py` confirms exactly seven added constraints:
four package assignments, two I/O standards and the reference-buffer site,
with all original constraints retained in the candidate. Three deliberately
damaged assignment controls are rejected. This qualifies the assignment
comparison and report inventory; it does not qualify routed timing, a complete
board wrapper, or the equivalence of constraints after optimization.

The recovered `candidate.dcp` SHA-256 is
`f8cfcb874190c94ff225511159d7206dc756c5fe45fb8a0737899ece0c8bd589`;
`optimized.dcp` is
`65e2a7ae84397ad674e1691e1029c6410c625a034f685cf48246f5a052eb90a8`.
Both are under the selected job's `restored/output/` directory. A follow-up
placement experiment can reopen the optimized checkpoint, avoiding repeated
optimization and its preceding reports, with a separate bounded time budget.
It must retain post-place CDC, bus-skew, timing and constraint checks before
considering routing. No route, bitstream or native traffic rerun occurred here.

The final process audit checked eight owned process groups: none remain active
or zombie. It verified 75 closed remote files, all 236 original generation
files, and the earlier donor checkpoints unchanged. `FINAL_BACKUP.json` in
the artifact root records the consolidated ZIP, its SHA-256 and restore
verification; the bundle includes the donor DCP, both recovered checkpoints,
all four job outcomes, source revisions, reviews and the generated reservation.

Reproduce the inventory review and reservation with:

```text
python -B review_hbm_board_boundary.py PROBE_RESTORED
python -B build_hbm_board_reservation.py PROBE_RESTORED NEW_CONTRACT
python -B review_hbm_board_placement.py PLACEMENT_RESTORED
```
