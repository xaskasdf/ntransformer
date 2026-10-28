# HBM response-enable physical experiment

This private VU47P/HBM experiment follows
[HBM_RESET_INTEGRATION.md](HBM_RESET_INTEGRATION.md). Its evidence lives in
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-response-enable/`. It does not change the
main-thread jobs, separate DDR design, released pinout or prior sealed evidence.

## Donor and hypothesis

The experimental donor is the reset-distribution candidate
`a990f1837d4dcaef6a271b6b625bc7235d2b0187ec517bde658c913207de1fb5`.
The selected original reference remains `3038b257...`; the donor is not promoted.

All 265 clock enables of channel 0 response `slot1` fail setup before this
experiment. Worst slack is -0.509 ns; minimum hold slack is 0.186 ns. The worst
HBM-to-enable datapath is 2.417 ns, including 1.752 ns routing and two LUTs.
Its final LUT4 at `SLICE_X39Y15` drives all 265 registers. Its logical INIT is
`16'h0800`. This motivates testing physical placement/routing before an RTL edit.

## Native method

Each batch uses a fresh private VM directory, two CPUs, nice 10 and a 1,200-second
limit. Scripts export before/after driver inputs, LUT truth tables, all 265
destinations, tracker connectivity, cell/net inventories and timing paths.
Full board timing, CDC, DRC, bus-skew and frozen contract audits follow.

The first batch tried `phys_opt_design -force_replication_on_nets`. Vivado
2026.1 rejected it with `Vivado_Tcl 4-265`: forced replication is unsupported
in post-route mode. That 173.233-second failure is retained, including its
complete pre-optimization probe. No physical result is accepted from it.

The replacement uses
`phys_opt_design -placement_opt -routing_opt -path_groups clk_out2_fabric_clocks`.
These explicit options request physical placement/routing in the 450 MHz HBM
domain. Other default transformations are disabled, as described in the
[AMD command reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/phys_opt_design).
The full affected domain is measured; this is not claimed as a one-net change.

## Acceptance and limits

The independent reviewer requires unchanged scoped constraints, board contracts,
cell/net inventories, reset consumers, PCIe selectors and tracker logic. It
enumerates all 16 LUT input assignments, checks the 265 enable connections and
rejects collateral tracker changes. All 2,195 bounded paths, 18 bus-skew groups
and distribution recovery/removal checks must pass in the complete physical
batch. Separate Vivado snapshots reopen both exact checkpoints to compare
tracker logic, enable timing, global timing, constraints, reset topology and
routing at the reported precision. Full board contracts are not audited a
second time during this snapshot comparison.

A negative control caught an empty tracker export in the initial script:
Tcl glob matching interpreted `channels[0]` as a character class. The corrected
probe uses a literal prefix and requires at least 530 primitive cells. The
reviewer also requires clock/connectivity rows for all 530 payload registers.
It accepts tracker evidence only from a separate reopening of both exact
input and output checkpoints; the original incomplete exports are retained.

The first corrected reopen was stopped after 360.836 seconds while exporting
tracker connectivity. Repeated shared-clock/constant-net queries were too
expensive. Its partial 29-file archive is preserved. The replacement caches
drivers by net name and clears the cache at every snapshot/design boundary.
`verify_cache.py` compares every complete row retained from the uncached probe
against the corresponding full cached export.

Caching alone was insufficient: that second reopen was also stopped and its
29-file archive retained after 434.832 seconds. The final revision keeps typed
cell/pin objects and uses the documented direct
[`get_pins -leaf` traversal](https://docs.amd.com/r/2024.1-English/ug894-vivado-tcl-scripting/Getting-Objects-by-Relationship).
It avoids repeated name resolution and explicit net-alias expansion.
Reconciliation compares rows by cell/property, retaining every complete row
from both interrupted probes even though native enumeration order differs.

To avoid a redundant third tracker snapshot and repeated global audits, the
private observer stops the final reopen only after both required `he_probe`
calls return. The subsequent native `experiment.tsv` row is the phase marker.
The observer hashes both completed snapshots before stopping its exact owned
process group. The runner remains `FAILED` with return code -15: it is not
reported as a successful full-board reopen. The independent status
`HBM_RESPONSE_ENABLE_SNAPSHOTS_REVIEWED` applies only to the verified completed
snapshots. `SNAPSHOT_COLLECTION_STOP.json` identifies the external observer,
its reason and file hashes; missing or incomplete snapshots are rejected.

This is a physical experiment, not a full-board formal equivalence proof,
fresh RTL synthesis, hardware traffic validation or timing-closure claim.
Existing timing coverage gaps and out-of-context DRC limitations remain; this
run neither hides them nor releases a PCB pinout.

## Native physical result

The physical batch `board-response-enable-inspect-20261013T200453Z` completed
in 719.579 seconds. Its checkpoint is
`312925a03719e3aece53cd197e2c5e05db62920112109d90fd5bdff8da417471`.

| Metric | Reset-distribution donor | Physical candidate |
| --- | ---: | ---: |
| Global WNS | -0.509 ns | -0.427 ns |
| Global TNS | -711.005 ns | -692.348 ns |
| Failing setup/recovery endpoints | 3,992 | 3,991 |
| Hold failures | 0 | 0 |
| Response slot1 enable worst setup | -0.509 ns | -0.416 ns |
| Response slot1 enable minimum hold | 0.186 ns | 0.189 ns |

Seven of the 265 response registers changed physical location/BEL. All 265
enable destinations still fail setup. The global worst path now runs from
`memory_distribution_reg/C` to HBM channel 0 `RREADY`, with 2.556 ns datapath
delay, including 2.385 ns routing. The original reference's -0.413 ns WNS is
still better, so this physical candidate is archived without promotion.

The batch archive was restored locally and all 106 file hashes checked. A
partial SCP transfer was resumed only after verifying its byte prefix against
the closed remote archive; its full SHA-256 is
`3db0e34818b08edb2d8536b21f66648ac262b41807a3a67bbd13a9747931e10d`.
`SELECTION.json` records the subsequent native reopen comparison, while
`FINAL_BACKUP.json` is the final restored-bundle receipt.

## Final review and selection

The completed read-only snapshots (`20261013T203630Z`, controlled stop after
805.000 seconds) preserve 1,774 tracker cells and 6,642 input connections.
Their 13,730 logical-property rows reconcile with all 4,907 and 6,878 complete
rows retained from the earlier probes. The 265 enables have identical logic.
Thirteen negative controls reject corrupted logic or snapshot metadata.

Reopening preserves WNS -0.427 ns, 3,991 failing setup/recovery endpoints,
zero hold failures, target enable timing and complete routing. TNS changes
from -692.348 ns to -692.361 ns, a -0.013 ns difference. Consequently
`reproduced_at_exported_precision` is false. No cause is assumed, and the
original `3038b257...` reference remains selected.

The complete physical batch passes all 2,195 bounded paths, 84 retained board
synchronous checks and 18 bus-skew groups (minimum margin 0.044 ns). These are
physical-batch results, not a claim that the interrupted reopen repeated every
board audit. The final closure audit verified 278 remote files, 565 unchanged
predecessor files and zero owned live/zombie processes. All three failed
attempts and the explicitly controlled snapshot stop are retained.

## Next RTL hypothesis

`fpga/experimental/ntrpu_hbm_read_return_cut.sv` already captures payload into
an unoccupied slot independently of VALID; VALID advances ownership. Applying
that pattern to the 265-bit response queue could remove HBM RVALID from the wide
enable cone. It requires its own backpressure, reset, fault/quarantine and
simultaneous push/pop tests plus full physical integration. This experiment
does not make that RTL change.

## Reproduction

Use `prepare_hbm_response_enable.py AUDITS CHECKPOINT FRESH_OUTPUT --mode placement`
with the frozen audits and exact donor. The archived private launcher runs
the prepared `inspect` stage. Collect the closed physical archive, then prepare
`--mode reopen` from its new checkpoint, supplying `--reference` with the local
original donor and `--remote-reference` with its exact VM path. Collect this
semantic probe before running `review_jobs.py --write`. `finish.py` compares
both reviews. The archived `stop_after_snapshots.py` implements the explicit
phase-boundary stop above. Failed batches remain part of the restored final bundle.
