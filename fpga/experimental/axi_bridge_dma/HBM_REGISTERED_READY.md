# Registered HBM RREADY experiment

This isolated VU47P/HBM candidate targets the reset-distribution-to-HBM-RREADY
path measured at -0.579 ns in [HBM_CAPTURE_PHYSICAL.md](HBM_CAPTURE_PHYSICAL.md).
Evidence belongs under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-rready-register/`.
The frozen input is the exact RTL used by that full-board synthesis, including
the already tested free-slot capture queue. Main-thread jobs, the selected
checkpoint, DDR design and earlier evidence are not modified.

## Admission and capacity invariant

The response queue retains two 265-bit slots. At each active clock edge, the
candidate predicts occupancy after the current push/pop and credits after the
current AR/R handshakes. A single ordinary register advertises READY only when
that next occupancy has room, next credits are nonzero and next fault is clear.
Next fault includes sticky fault, invalid descriptors, malformed responses and
unsolicited offers detected on that same edge.

`READY = queue_space && capture_credit_available && !fault` must hold after
each clock edge. This is an inductive invariant, not permission to accept an
extra beat after capacity or credits run out. Input ownership, credit counters,
validation and quarantine remain cycle-equivalent to the frozen reference.
Payload capture may still update only the unowned slot without RVALID; VALID
controls ownership. The validated queue and its delivery rules are unchanged.

RREADY is driven directly by an ordinary asynchronously cleared FDCE, with no
combinational reset gate after Q. The existing coordinated cold reset must
span active clock edges to clear the synchronous queue/tracker state; the READY
register clears immediately on assertion. It is not a CDC synchronizer and is
not marked `ASYNC_REG` or `DONT_TOUCH`. Recovery/removal of this new register
must remain timed in a later full-board implementation.

AMD documents two-slot buffering with registered READY in its
[fully registered AXI slice guidance](https://docs.amd.com/r/en-US/pg373-axi-register-slice/Fully-Registered).
This custom candidate additionally enforces the tracker's credit/quarantine
contract. It preserves the existing full-queue recovery bubble and does not
claim equivalence to every latency or throughput property of the AMD IP.

## Bounded validation

The queue bank enumerates all `8^5` five-cycle combinations of input VALID,
downstream READY and next-cycle permission, then runs 20,000 randomized cycles,
for widths 265 and 519. An independent occupancy/payload scoreboard checks all
24 occupancy/permission/handshake combinations, accepted-data conservation,
free-slot capture and reset loss accounting.

The tracker bank compares all interfaces, credits, descriptor state, ownership
and valid payload cycle-by-cycle against the frozen reference at `MAX_PENDING`
1, 2, 4 and 8. It also checks an independent end-to-end scoreboard, all 128
outstanding beats, both queues full, simultaneous AR/terminal-R, zero-latency
slave offers, stalled AR, reset with pending data, between-edge reset assertion,
RID/EXOKAY/early/missing-RLAST faults, invalid descriptors and unsolicited data.
SLVERR/DECERR metadata retain the reference behavior.

Five RTL mutants use stale space, stale credit, a delayed fault, omitted request
fault prediction, or synchronous READY reset. Each must fail for the intended reason. XSIM diagnostic text is
checked even when its process exit status is zero.

The final bank adds invalid reservations concurrent with a valid response and
remaining credits at depths 2, 4 and 8. Depth 1 cannot reserve another descriptor
in that state. This distinguishes next-request fault gating from merely waiting
for the sticky fault bit on the following cycle.

Native synthesis compares the same two tracker variants on
`xcvu47p-fsvh2892-2-e`, at a 2.222 ns clock. It exports the direct RREADY driver,
reset polarity/source, register attributes, both payload register inventories,
ownership fan-in, primitive counts, netlists and checkpoints. No false path or
clock exception is added. These are topology checks, not physical timing closure.

## Reproduction and limits

`prepare_hbm_registered_ready.py DONOR_ADAPTER FRESH_OUTPUT` freezes the exact
tracker transformation and reference modules. The prepared native runner uses
two CPUs and nice 10, bounds every simulation/synthesis step, and waits for all
inherited stdout writers before sealing logs. Closed batches, including failed
attempts, are archived and restored locally with file hashes.

Run `review_hbm_registered_ready.py RESTORED DONOR_ADAPTER` for the independent
review. Full-board synthesis, placement/routing, board reset/CDC contracts and
HBM/PCIe hardware traffic remain separate validation steps. Registering RREADY
can move a critical path into the register's input logic; unit synthesis alone
cannot establish a frequency improvement.

## Native results and local evidence

Vivado/XSIM 2026.1 completed two bounded batches on the VM. The second expands
fault coverage without changing either synthesized RTL variant or the synthesis
script. Both batches remain available, with closed logs and native checkpoints.

| Batch (UTC) | Assertions checked | RTL mutants rejected | Elapsed |
| --- | ---: | ---: | ---: |
| `board-registered-ready-sim-20261013T224924Z` | 3,101,603 | 4 | 210.6 s |
| `board-registered-ready-sim-20261013T225411Z` | 3,102,543 | 5 | 228.9 s |

The final bank delivered 3,094 clean-traffic beats across four tracker depths;
3,098 were accepted, with four explicitly accounted for by reset. Fault traffic
is checked separately against the reference state and quarantine rules. All 24
queue coverage bins were reached at both tested widths. These are bounded
simulation checks, not a proof over all possible traffic histories.

Native synthesis confirms `h_rready` is driven directly by
`rready_reg_reg/Q` (FDCE), replacing the reference LUT4 output and its
combinational reset path. Each queue retains 530 payload registers. All three
response ownership registers retain RVALID dependence, while response payload
enables exclude it. Primitive counts change from 635 to 638 LUTs and add one
ordinary FDCE per tracker; the 1,139 FDREs and payload capacity are unchanged.

The selected unit checkpoint SHA-256 is
`bd8566f2d06d082ead67d29afcb1a651d062db9a189fa14909eb4876aea23e41`.
Its batch archive SHA-256 is
`984cf55a2f1a557c5ca7d14eb480de0c6fd5e0045ccdb6fceb6424b4bfa921af`.
The frozen donor SHA-256 is
`b8a8843f0122a89461e1d8c81c3e3f245d404d9f2e6be79aa449e0827dc23d0e`.

The artifact directory contains `finalize.py`, which checks both downloaded
batches, replays both preparations, exercises ten corrupted-evidence rejection
cases, audits owned VM processes and verifies the predecessor archive. It then
seals a ZIP, restores it, repeats the local checks and records the archive hash
in `FINAL_BACKUP.json`. `REVIEW.json` and `FINAL_PROCESS_AUDIT.json` retain the
detailed checks. Each native batch also has its own `BACKUP.json` and file hashes.

Out-of-context synthesis warns that `HD.CLK_SRC` is unset; constant ownership
CE pins have no fan-in startpoints. Neither warning is evidence of board timing
closure. The next physical experiment must integrate this tracker into a fresh
full-board candidate, measure READY input/output paths and reset recovery/removal,
and retain the existing CDC, skew, exception and payload-enable checks. No
full-board checkpoint is promoted by this unit result.
