# Native VU47P/HBM control attachment

This batch connects the [retained control plane](HBM_CONTROL_PLANE.md) to the
frozen native PCIe/HBM composition. `prepare_hbm_bar_control.py` creates the new
`ntrpu_axi_bridge_hbm_controlled` top in a fresh prepared directory. It preserves
the AMD source notices and the native IP configuration. The main thread's running
design and the separate DDR node are not inputs to this mutation.

## Native path and supported host operations

```text
PCIe / ntrpu_bridge.m_axib (512 bits, 64-bit address)
  -> ntrpu_bar32_adapter (range, shape, lane and ID checks)
  -> ntrpu_hbm_control_plane (32-bit AXI4-Lite, PCIe clock)
  -> frozen core cfg/cmd/ack interfaces and retained response mailboxes
```

The frozen profile enables a non-prefetchable, 128 KiB BAR0 with incoming AXI
translation base zero. Only offsets `0x0000–0x0fff` reach the control registers.
The remaining aperture returns DECERR. The full 64-bit AXI address is checked
before selecting the low 12 bits; upper address bits cannot alias a command.
The example BRAM application is removed from the incoming path. Outbound writer
traffic and its mapping-service AXI-Lite interface remain separate.

The supported host contract is an **aligned, single-DW, 32-bit MMIO operation**:
AWLEN/ARLEN=0, AWSIZE/ARSIZE=2, INCR, no lock. Use individual `readl`/`writel`-style
operations through an uncached device mapping, preserving software ordering and
polling the control status after publication. Bulk copies, write combining and
64-bit loads/stores are outside this register ABI. Partial WSTRB within the selected
DW is supported. A zero strobe still reaches the CSR as a no-op write; the CSR may
reject it for a doorbell, whose contract requires all four strobes.

AMD documents single-DW transfers as a special case with AxSIZE=2 on the wider
[bridge master interface](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/AXI-Master-Interface).
The adapter selects WDATA/WSTRB using address bits `[5:2]`, returns RDATA in that
same lane with other lanes zero, and returns the captured BID/RID.

Unsupported metadata returns DECERR without a downstream CSR access. Rejected
write bursts consume exactly AWLEN+1 beats before B; rejected reads return
ARLEN+1 zero/error beats and assert RLAST only on the final beat. A malformed
single-beat WLAST or an out-of-lane write strobe returns SLVERR before forwarding.
An incomplete rejected burst can remain pending until the bus resets; there is no
timeout that silently reinterprets its remaining data as another transaction.
CSR-generated SLVERR responses are propagated. A PCIe posted write does not give
software an AXI B response directly: actual host error reporting still needs a
native TLP/driver test.

AW and W can arrive in either order. Read and write sides are independent, each
with one transaction in flight. The native operational AXI reset clears the
adapter and CSR transport together. Accepted commands, mailboxes and snapshots
remain under canonical cold reset, including when the host loses a B response.
See the control-plane reconciliation rules before retrying a doorbell.

## Telemetry and remaining board boundary

`ntrpu_hbm_telemetry_pack` assigns all 133 observation bits to their previously
audited clock domains. The four 128-bit inputs have a schema/domain tag in bits
31:0 (`0x4e540100 + domain`), followed by packed fields, then zero padding.
`BAR_BINDING.json` records every field's domain, bit offset, width and source clock.

| Domain | Clock | Sample contents |
|---|---|---|
| 0 | PCIe, 250 MHz | Client availability, writer ownership/fault, mapper and source status |
| 1 | Management, 100 MHz | Availability/guard status, HBM initialization, temperatures, CATTRIP, clock supervisors |
| 2 | Reader, 200 MHz | Reader startup/fault/drain/stop and frame-idle state |
| 3 | Memory, 450 MHz | Adapter pending/fault and raw HBM read parity |

Each domain also includes its canonical cold-reset observation. Snapshots retain
the control plane's per-domain coherence and full-round publication semantics.
Raw parity is an observation, not a parity-error detector or proof of a valid read.
The frozen HBM profile has `USER_MC0_RD_DQ_PARITY_EN=false` (and its data-parity
enable is false); this attachment does not enable memory-side parity protection.
Temperatures and CATTRIP are exposed; a complete thermal shutdown policy remains
separate work. Raw MMCM LOCKED remains an external sequencer output; it is not
introduced as an unsynchronized telemetry bit.

The new top removes 1,213 internal transaction/observation port bits, leaving
74 bits for PCIe serial lanes, references and the external sequence interface.
This reduction does not establish a PCB pinout or external timing budget.

## CDC and connectivity evidence

`audit_hbm_bar_control.tcl` re-derives the earlier 1,503 bounded paths and their
90 synchronizer-stage pairs against the new netlist. The previous exact reset
arcs and the 10 ns raw LOCKED bound are retained; the existing LOCKED path
segmentation warning remains visible and is reviewed explicitly.

The audit compares all 112 AXI-Lite connection bits, 1,280 live native bridge
connection bits, canonical reset connections and telemetry source clock domains.
The frozen native synthesis folds AWLOCK and ARLOCK to zero at the adapter and
disconnects the corresponding IP output pins. These two bits have a separate,
exact GND check; they are not counted as live wires. The source must still bind
both inputs directly to the corresponding IP outputs before synthesis. No other
missing connection is accepted. The initial audit that required live LOCK wires
is preserved as a failed attempt; this correction changes only the auditor.
For domains 1–3 it constrains each live held-data capture to 4 ns and bounds
request/acknowledgment paths to their first synchronizer stages (4 ns, except
the request into memory at 2.222 ns). Domain 0 shares the PCIe clock and retains
ordinary synchronous analysis. All 16 new synchronizer-stage pairs retain both
setup and hold analysis. No broad clock groups or CDC severity waivers are added.

The apply stage and a fresh checkpoint reopen must both succeed. These are
post-synthesis bounds and connectivity checks, not placed/routed timing closure,
metastability qualification or board validation. Native PCIe TLP traffic, link
training and host MMIO access are not simulated by this bank.

The post-synthesis DRC retains three critical findings: `AVAL-326` for the PCIe
`refclk_ibuf` LOC; `NSTD-1` for the I/O standards of `fabric_ref_clk_in` and
`sys_rst_n`; and `UCIO-1` for LOCs of those signals plus `sys_clk_p/sys_clk_n`.
These are recorded as physical-design work still required, without severity
changes or waivers. The out-of-context DRC is not a complete board DRC.

## Recorded results (2026-10-12 UTC)

The reviewed selection under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-bar` is:

| Stage | Closed run | Result |
|---|---|---|
| Native elaboration | `bar-elaborate-20261012T001601Z` | Passed |
| Eight RTL simulations | `bar-sim-20261012T002043Z` | 40 named groups, 8,723 checks |
| Native synthesis | `bar-synth-20261012T002033Z` | Passed; physical DRC findings above remain |
| Apply exact bounds | `bar-inspect-20261012T004123Z` | Passed |
| Fresh checkpoint reopen | `bar-inspect-20261012T004456Z` | Passed; connections and bounds preserved |

The snapshot contributes 384 held-data endpoints and six request/acknowledgment
endpoints. Their minimum estimated slack is +1.952 ns. The whole design still
reports **timing not met**: WNS -0.194 ns, WHS -0.842 ns and WPWS -0.092 ns in the
apply report. These are synthesis estimates, not routed timing. There are 1,154
unconstrained internal endpoints, all in the native HBM XSDB hierarchy, plus two
inputs without input delays. CDC reporting retains nine CDC-6 and 2,130 CDC-15
warnings; none are suppressed. The reviewer checks these findings again on reopen.

Native utilization is 60,328 LUTs, 64,372 FFs, 60 RAMB36 and one RAMB18. Of this,
the adapter uses 533 LUTs/1,204 FFs and control uses 1,896 LUTs/3,623 FFs. This is
control/transport integration, not the inference compute design.

The independent reviewer passes 16 deliberately corrupted-evidence controls.
The initial failed LOCK-wire audit and the preliminary two-run smoke bank are
retained separately. `FINAL_PROCESS_AUDIT.json` verifies all seven batch jobs are
closed, 669 remote evidence files and 236 generation-donor files retain their
hashes, and there are no live or zombie processes from this batch. The final ZIP
restores the closed evidence and reruns the reviewer; `FINAL_BACKUP.json` is the
receipt, with its archive path, size and SHA-256.

Bounded checkpoint SHA-256:
`84c727cc7d272db369a281db65e7113c21ce8c02062c163e8b87910312ef6464`.

## Reproduce the isolated batch

1. Run `prepare_hbm_bar_control.py FROZEN_NATIVE_SOURCE BOUNDARY_REVIEW.json NEW_PREPARED`.
2. Use the existing `run_hbm_native_join.py` with that directory and a private copy
   of the closed native generation project; run `--stage elaborate`, then `synth`.
3. Run `prepare_hbm_bar_tests.py FROZEN_CONTROL_SOURCE NEW_SIM_PREPARED`, then
   `run_hbm_bar.py NEW_PRIVATE_OUTPUT --prepared NEW_SIM_PREPARED --settings SETTINGS`.
   It runs four deterministic stall variations of the adapter and real-writer banks.
4. Prepare the audit helpers with `apply_hbm_bar_control.tcl` named
   `inspect_hbm_native_join.tcl`; run native `--stage inspect --checkpoint SYNTH.dcp`.
   Repeat in a fresh stage using `reopen_hbm_bar_control.tcl` and `bar_bounded.dcp`.
5. Run `review_hbm_bar_control.py RESTORED_SIM RESTORED_SYNTH RESTORED_APPLY RESTORED_REOPEN`.

The VM namespace is `/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join`.
Runners inherit the existing two-CPU/nice-10 policy and finite process-group
timeouts. Logs and failed attempts remain under
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-bar`. Local archives are restored, hashed and
reviewed again before the final receipt is issued. HBM weight loading is still
disabled in this native top; this batch does not prove end-to-end inference.
