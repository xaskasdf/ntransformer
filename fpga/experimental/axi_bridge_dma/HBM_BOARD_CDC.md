# Board startup CDC and reset boundary

This isolated VU47P/HBM experiment builds on the board-owned reference-clock
candidate in [HBM_REFERENCE_OWNERSHIP.md](HBM_REFERENCE_OWNERSHIP.md). It corrects
the reader status reset and rebinds the existing native transport, telemetry,
and weight-loader constraints. It does not run placement or routing.

## Candidate and reset change

The frozen native composition reported one critical CDC-11 from
`board_startup/configuration_rst_n_reg/C` to the reader cold-release chain.
The raw configuration reset also drove all fifteen reader status synchronizer
reset controls. Those registers could leave reset before the reader's canonical
three-stage release had completed.

`prepare_hbm_board_cdc.py` copies the frozen reference candidate into a fresh
directory and changes only the status process in `ntrpu_hbm_startup.sv`:
its asynchronous reset becomes `src_rst_n` instead of `cold_reset_n`. The two
cold-release chains, readiness criteria and sticky failure behavior are preserved.
Status sampling waits up to three additional reader clock edges after raw
release. No deployed RTL or earlier checkpoint is modified.

The candidate RTL SHA-256 is
`668da3480e6074e25f46bd50e6d494baa7ed4abb1b90145e06edf148a213e8ba`.
`BOARD_CDC_BINDING.json` records the complete donor inventory. Older binding
files copied alongside it describe their respective historical stages; their
old top/source hashes are provenance, not a new validation claim.

## Simulation evidence

`board-cdc-sim-20261013T141800Z` ran the actual candidate with four clock
frequency/phase combinations: 148 checks in 48 scenario groups. Cases cover
missing calibration, both-stack qualification, faults before and after startup,
sticky calibration loss, cold restart and stopped clocks. The verifier checks
that all status stages stay zero until canonical reader release.

A negative control restores the original raw-reset process in the actual RTL;
it fails precisely the canonical-release invariant. XSIM returns zero even for
this `$fatal`, so acceptance checks its diagnostic text as well as process status.
The six bounded simulation stages completed in 27.20 seconds. Thirteen evidence
files were restored and hashed locally.

## Constraint acceptance

The board audit must retain the previous 1,503 transport/control, 390 snapshot
and 302 loader paths with their original endpoint identities and bounds. It must
also preserve synchronizer interstage setup/hold checks and registered-readiness
source timing. No global clock groups, multicycle exceptions or waivers are added.

New reset exceptions are limited to the supervisor input at the three board
release registers and configuration release at nine reset controls in other
clock domains. Board-local, management and reader status recovery/removal must
remain timed. Both consumers of MMCM `LOCKED` receive a 10 ns datapath bound;
the audit verifies that its incoming model arcs were already disabled by the
timing model, not by user constraints.

The new synchronous inventory contains 88 setup/hold or recovery/removal
checks: eight board synchronizer interstage checks, thirty board recovery
checks, six management cold-release checks, fourteen sequencer feedback/enable
checks and thirty reader status reset checks.

The first apply audit, `board-cdc-inspect-20261013T143816Z`, stopped after
185.41 seconds because it expected fourteen board reset consumers. The native
netlist retains `lock_fault_reg`, producing fifteen. It also exposes seven
local configuration feedback/enable consumers. The revised audit enumerates
these explicitly and requires their same-clock timing; it adds no false paths
for them. The failed run and all 47 evidence files remain archived.

## Native synthesis

`board-cdc-synth-20261013T142043Z` completed in 982.86 seconds, without timeout,
using the same candidate RTL as simulation. Its 283 archived evidence files
were restored and verified. The checkpoint SHA-256 is
`b86fd87a4626aab01555ee0eb12d7a98f6f2a36d3fa386b55ceb2ed089c76a4d`.

The raw synthesis CDC report contains no critical finding: the former CDC-11
is now recognized as another CDC-9 synchronized reset. Counts are 193 CDC-3,
9 CDC-6, 35 CDC-9 and 2,425 CDC-15. The multi-bit and clock-enable warnings
remain subject to their protocol and timing contracts. The synthesis summary
contains zero errors, zero critical warnings and 80,971 warnings; this does
not mean the design has a clean methodology report.

## Applied constraints and limitations

`board-cdc-inspect-20261013T144218Z` applied the revised audit in 320.72 seconds.
Its 65 evidence files are locally restored. It preserves all vendor XDC commands
and adds 54 datapath-only max delays, 10 bus-skew constraints and 30 exact false
paths: eighteen inherited reset exceptions and twelve board-boundary exceptions.
The bounded checkpoint SHA-256 is
`724bb198d259ec00aaa5c237eccf814df24fcbfd4eb5cadcf96a222bb3b2f10c`.

The constrained CDC report has 200 CDC-3, 9 CDC-6, 40 CDC-9 and 2,425 CDC-15;
no critical CDC remains. TIMING-2/4/6/7 and CLKC-5 are absent. Remaining
methodology findings include one critical AVAL-324 for the PCIe reference-buffer
location, fourteen LUTAR-1 warnings, and one each of TIMING-9/13/18/34.
TIMING-13 identifies the explicit `LOCKED` segmentation; both incoming MMCM
model arcs remain disabled with `IS_USER_DISABLED=0`. TIMING-18 concerns PERST.
TIMING-34 flags the existing 1.111 ns loader payload skew budget as aggressive;
this experiment preserves it for routed verification.

The estimated synthesis timing is not closed: WNS -0.194 ns, WHS -0.842 ns and
WPWS -0.092 ns. All eighteen reported skew groups have nonnegative estimated
slack (minimum 0.703 ns), which does not establish routed skew closure.
`check_timing` retains 1,154 unconstrained internal endpoints and two inputs
without input delay, with zero missing clocks, multiple clocks or timing loops.
The internal boundary and ignored vendor exceptions remain visible in the
archived reports; there are no waivers.

`board-cdc-inspect-20261013T144926Z` reopened the bounded checkpoint in 273.12
seconds, with 64 evidence files restored. The independent review passed:
the XDC command inventory, all 2,195 inherited paths, connectivity tables,
88 new synchronous checks, reset exception endpoints and CDC coverage agree
across application and reopening. Reference clocks remain correctly owned by
the board. The selected constraint stages are `prepared-apply-r3` and
`prepared-reopen-r3`; `prepared-r1` is the tested native RTL candidate.

The final process audit verified all 472 closed VM evidence files, 71 local
donor files, both generation donors (236 and 272 files), and the immutable
earlier checkpoints. None of the fifteen owned process groups remain active
or zombie. `FINAL_BACKUP.json` is the completion receipt for the local bundle:
its seal step restores every file, repeats both independent reviews and
reconstructs the three selected prepared directories byte for byte.

## Reproduction and evidence

Run from this directory with fresh output directories:

```text
python -B prepare_hbm_board_cdc.py FROZEN_REFERENCE NEW_CANDIDATE
python -B prepare_hbm_board_constraints.py FROZEN_READINESS_AUDITS NEW_APPLY --mode apply
python -B prepare_hbm_board_constraints.py FROZEN_READINESS_AUDITS NEW_REOPEN --mode reopen
python -B review_hbm_board_cdc.py RESTORED_SIM --synthesis RESTORED_SYNTH --apply RESTORED_APPLY --reopen RESTORED_REOPEN --baseline FROZEN_READINESS
```

Local evidence root: `D:/AI/artifacts/ntrpu-axi-bridge-hbm-board-cdc`.
Each VM stage uses a new private directory, two CPUs, reduced priority, a bounded
tool runtime and hashed sources. The native generation donor is
`reference-owner-generate-20261013T133721Z`; it is copied, never modified.
The main validation queue and earlier experimental milestones remain untouched.

Remaining qualification includes post-placement CDC/timing, routed delay and
bus-skew closure, native HBM traffic, board pin/electrical choices, power
sequencing and physical hardware operation.
