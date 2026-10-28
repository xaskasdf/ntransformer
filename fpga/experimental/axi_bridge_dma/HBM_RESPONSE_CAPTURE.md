# Isolated HBM response free-slot capture

This VU47P/HBM experiment follows [HBM_RESPONSE_ENABLE.md](HBM_RESPONSE_ENABLE.md).
It changes only a private copy of the tracker's 265-bit `response_cut` binding.
The original adapter, validated response queue, board jobs, DDR variant and
previously selected routed checkpoint remain unchanged. Evidence is stored in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-response-capture/`.

## Candidate and contract

`ntrpu_hbm_response_capture_queue.sv` captures incoming data whenever the write
slot is free, including cycles without VALID. Count and write-pointer advancement
still require `VALID && READY`. The read pointer still advances only when the
output is accepted. When full, both payload slots remain unchanged even during
a pop, preserving the existing one-cycle admission bubble. Reset clears ownership
without clearing the payload registers. Invalid output data is unspecified.

The preparation script extracts the queue and tracker from the frozen board
adapter in the reset-integration evidence. It verifies the exact queue delta
and substitutes only `response_cut`; `validated_cut` retains its original module.
This avoids silently testing a different working-tree tracker revision.

## Native tests

The queue bench compares the reference, candidate and independent two-entry
scoreboard. It enumerates all 4,096 six-cycle VALID/READY sequences for widths
265 and 519, then runs 20,000 randomized cycles per width. It checks all twelve
occupancy/input combinations, simultaneous push/pop, long stalls, speculative
writes, accepted-data conservation and explicit reset discards.

The tracker bench uses depths 1, 2, 4 and 8. It checks descriptor credit limits,
offered-address retention, a response offered with the first AR handshake,
ordered bursts of 1/2/4/16 beats, backpressure and payload conservation. Bad RID,
EXOKAY and early RLAST must quarantine the accepted response despite changing
HBM input data. Fault remains sticky until explicit cold reset. An unsolicited
response is checked separately. Reference/candidate interfaces and ownership
state are compared cycle by cycle; this is simulation, not a formal proof.

Three deliberate RTL mutations must fail: advancing ownership without VALID,
writing the occupied slot, and writing while full. Native VU47P out-of-context
synthesis exports both trackers, checkpoints, primitive inventories and complete
enable-cone tables. Acceptance requires all 530 reference response enables to
depend on HBM RVALID and none of the 530 candidate enables to do so. Ownership
must still depend on RVALID; the 530 validated-queue enable cones must agree.

## Reproduction and evidence

```powershell
python -B fpga/experimental/axi_bridge_dma/prepare_hbm_response_capture.py `
  --donor D:/AI/artifacts/ntrpu-axi-bridge-hbm-response-capture/reference `
  --output D:/AI/artifacts/ntrpu-axi-bridge-hbm-response-capture/new-prepared
```

The archived private `launch.py sim --prepared <directory>` runs XSIM and Vivado
2026.1 with two CPUs and nice 10. Each simulation step is limited to 120 seconds;
the two synthesis runs share a 300-second limit. `collect.py --collect` downloads
closed archives and verifies their full SHA-256 and file inventory. Failed
batches are retained. `review_hbm_response_capture.py RESTORED_DIR` independently
checks logs, coverage, negative controls and native topology.

## Integration gate

This stage does not implement or time the complete board. It does not establish
that global WNS improves: the previous worst path moved to HBM RREADY, and reset
distribution still matters. The next physical experiment must bind this queue
in a fresh board build, preserve all existing timing/CDC contracts and compare
the response enable, RREADY, reset, setup, hold and bus-skew results. No timing
waivers or relaxed clock targets are introduced here. Extra switching in an
unowned slot is possible; power has not been characterized. The original routed
`3038b257...` checkpoint remains the selected reference.

## Reviewed result, 2026-10-13

Native batch `board-response-capture-sim-20261013T210540Z` completed in
168.242 seconds, including 99.619 seconds for both synthesis runs. Independent
review accepted 841,814 functional checks, 2,776 clean end-to-end response beats,
three intended RTL mutation failures and nine altered-evidence rejection checks.
XSIM reports these deliberate `$fatal` failures with process exit code zero;
the runner and reviewer therefore inspect the failure text as well as exit status.

| Native synthesis metric | Reference | Candidate |
| --- | ---: | ---: |
| Response payload registers | 530 | 530 |
| Response enables depending on HBM RVALID | 530 | 0 |
| Validated payload registers | 530 | 530 |
| Tracker FDRE primitives | 1,139 | 1,139 |
| Tracker LUT1–LUT6 primitives | 635 | 635 |

All validated-queue enable startpoint sets agree. Candidate response enable
startpoints are only reset, occupancy and write-slot state. Ownership retains
its RVALID dependency. Two LUT4 primitives become LUT3 primitives; there is no
increase in these register/LUT counts and no extra queue latency in simulation.

The candidate synthesis checkpoint SHA-256 is
`a566af78e4e9aa64e16281638587237f9d4c938b4af8dce9155ae6f2f9b03850`.
The successful native archive SHA-256 is
`a8a06ebb64d0a74066831f3b670c470dcccbaaf6bd0f5773a2a906345d26ec17`.
Its 52 evidence files were downloaded and hash-verified. The first, 15-file
failed compilation is also retained: the bench used the reserved word `bins`
as a variable name. Renaming that bench variable was the only revision between
prepared batches; the candidate RTL was unchanged.

The final remote audit caught one append after the synthesis parent exited:
its owned Vivado helper added a 72-byte exit message to `synthesis.log` after
the native archive was made. The archived log remains untouched. A separate
`closure/synthesis-after-helper.log` preserves the final log, and
`LATE_HELPER_OUTPUT.json` verifies its exact archived byte prefix plus the
single expected suffix naming the owned synthesis PID. No other remote
evidence file changed. Checkpoints and reviewed results are unaffected.

`release-r2/FINAL_BACKUP.json` identifies the complete ZIP and its restore
verification. This release corrects the first bundle's documentation-only
68-byte label to the measured 72 bytes; the recorded log sizes, hashes, native
archives and results were already correct and remain unchanged.
`REVIEW.json` includes replay of both preparations, each rejecting overwrite.
`FINAL_PROCESS_AUDIT.json` records remote source/evidence checks, unchanged donor
files and closure of this experiment's processes. None of these isolated
results qualifies full-board timing or native PCIe/HBM traffic.
