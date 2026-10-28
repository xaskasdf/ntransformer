# Boot-time weight loading over PCIe

This isolated VU47P/HBM prototype replaces the external preload fixture in
[HBM_TLP.md](HBM_TLP.md) with a synthesizable register-controlled write path.
The frozen reader, writer, BAR adapter, control plane and five AMD IP profiles
remain unchanged. The DDR card and main simulation queue are separate.

## Data path and ownership

```text
Host 32-bit BAR writes -> native PCIe -> existing BAR adapter
    -> loader staging / atomic COMMIT -> held-bus CDC
    -> one native AXI3 write to PC00 or PC16 -> HBM B
    -> retained completion -> host POP -> SEAL -> existing read commands
```

Each COMMIT writes exactly 32 bytes, aligned to 32 bytes, within the 16 GiB
address range. Address bit 33 selects PC00/PC16; the complete 34-bit address
reaches HBM. Upper bits are rejected before truncation. All 48 staging bytes
(address, tag and data) must be written. Accepted work owns an immutable copy;
staging cannot change until its completion is popped. AW and W handshake
independently; neither a posted PCIe write nor an AXI address/data handshake
means the HBM write completed.

SEAL is an irreversible boot-phase fence until canonical cold reset. Before
SEAL, the router rejects the existing command doorbell at `0x010`; afterward,
it rejects weight writes. This prevents this control path from reading while
loading. SEAL requires at least one accepted line, no pending work/completion,
empty staging, available memory, and no retained error. It does **not** verify
a model manifest, all intended addresses, or model completeness.

This initial loader is for functional bring-up. Individual MMIO stores and
per-line acknowledgments are unsuitable for measuring bulk weight-loading
throughput. DMA pull, bulk bursts and dynamic replacement remain future work.

## Register contract

Offsets below are within the existing first 4 KiB of BAR0. Use aligned 32-bit
device MMIO with software ordering and readback; the frozen adapter still rejects
unsupported burst shapes and address aliases. The original CSR space passes
through unchanged except for the command fence.

| Offset | Access | Meaning |
|---|---|---|
| `800`, `804` | R | `0x4e544c44` (NTLD), version 1 |
| `808` | R | Bits 0–5: busy, completion valid, sealed, HBM failure, available, rejected MMIO |
| `80c` | W | COMMIT; value 1, all four strobes |
| `810` | W | POP completion; value 1, all strobes |
| `814` | W | SEAL; value 1, all strobes |
| `818` | W | Clear rejected-MMIO indicator only; value 1, all strobes |
| `81c` | R | Accepted line counter |
| `820–824` | R | Active/completed tag, low word first |
| `828` | R | Completion: bits 1:0 BRESP, bit 2 wrong BID, bit 3 lost memory availability |
| `82c–830` | R | Active/completed address, low word first |
| `834–838` | R | 48-bit staging written-byte mask |
| `840–86c` | RW | Address64, tag64, data256, low word/byte first |

Poll `808`, verify completion tag/address and `828 == 0`, then POP before
staging another line. Successful completion status is `0x12`; sealed and ready
is `0x14`. Writing COMMIT consumes the staging mask. Retrying a COMMIT while busy
or before POP is rejected. Tags identify a retained transaction; they are not a
general deduplication service. A host must reconcile retained state after losing
a PCIe response, rather than blindly repeat its writes.

## Reset, failures and clock crossings

Transport reset clears only bus transactions. Accepted work, staging, completion,
error state and SEAL survive. Both domain cold resets must assert together and
deassert with their own clocks, as in the canonical native reset tree. Independent
memory-side cold reset is unsupported. Cold reset during loading invalidates the
entire software loading session even though HBM may retain some bytes.

The request and response each use three synchronization registers plus a settling
cycle before held-data capture. A stopped memory clock keeps ownership pending.
Once offered, AW/W are retained through backpressure or lost availability. Missing
B never becomes success; recovery may require cold reset. A bad BRESP, wrong BID
or observed availability loss is retained and prevents SEAL until cold reset.

The held-bus protocol follows the full-handshake principle documented by
[AMD](https://docs.amd.com/r/2023.1-English/ug953-vivado-7series-libraries/XPM_CDC_HANDSHAKE),
but this is custom RTL, not that macro. New physical CDC bounds, reset recovery,
placement/routing and timing qualification are still required. The earlier
synthesized native checkpoint does not include this loader.

## Validation and recovery

`tb_hbm_weight_loader.sv` uses the real control plane with an explicit HBM slave
fixture. It checks independent stalls, immutable staging, response retention,
transport/cold resets, clock pause, address limits, unavailable memory, bad BID,
bad BRESP, availability loss and the command/SEAL fence. Run
`run_hbm_weight_loader.py NEW_PRIVATE_RUN --prepared PREPARED --settings SETTINGS`.
The default is native XSIM, with seeds 0, 1 and 3. The installed Icarus parser
rejects the existing initialized output-port syntax; its failed attempt is kept.

`prepare_hbm_weight_native.py FROZEN_CONTROL_SOURCE PIPE_SOURCE NEW_PREPARED`
builds a separate native top with no external preload ports. Its host loads six
lines/192 bytes over PCIe, polls attributed completions, seals, then runs the
existing two HBM-to-host transfers and ACK protocol. The independent reviewer
must match every host operation, native HBM beat, outgoing PCIe byte and VCD
handshake, including continuous ownership and the load/read fence.

The private VM namespace is `ntrpu-side/axi-bridge-hbm-weight-loader`; local
evidence is `D:/AI/artifacts/ntrpu-axi-bridge-hbm-weight-loader`. Runs use two CPUs,
nice 10, bounded subprocess groups and fresh paths. Each attempt is exported,
hashed and restored locally; failed attempts remain distinguishable from passes.
Native functional simulation does not establish physical link, PCB, inference
or routed timing qualification.

### Recorded stages, 2026-10-12 UTC

- `preflight-20261012T042251Z`: failed at Icarus parsing; no simulation claim.
- `preflight-20261012T042332Z`: XSIM passed all 24 groups/1,352 checks in
  25.60 seconds. The independent reviewer rejected 18 altered-log controls.
- `synth-20261012T043102Z`: isolated synthesis completed in 195.94 seconds:
  635 CLB LUTs, 1,335 flip-flops, no latches, BRAM or DSP. All twelve intended
  synchronizer registers retain ASYNC_REG. The report identifies three CDC-3
  crossings and 295 CDC-15 held-data warnings. The external readiness input is
  not covered by that OOC clock-to-clock count.
- `traffic-20261012T042540Z`: full native PASS at 188.110 us simulated. Elaboration
  took 1,419.04 seconds and simulation 2,930.50 seconds; the latter completed
  within its 3,000-second cap. The bank performed **43,468 checks, 201 BAR
  requests (137 writes/64 reads), six confirmed HBM writes and two outbound
  transfers totaling 192 bytes**. These are deterministic test patterns, not
  a complete model or inference run.

The independent native review matches logs to VCD for all transactions. It
checks 28,062 samples of retained loader data, six completions/POPs, one SEAL,
and both read commands after SEAL. The outgoing path retains ownership through
9,063 samples until its matching host ACK and result acceptance. All sixteen
corrupted-evidence controls are rejected, including a missing load fence or
loader ownership in the waveform. These controls test the reviewer; the short
bank provides the actual RTL fault scenarios.

The automatic collector restored and verified 161 native evidence files and
the executable runtime's 19 files. The runtime archive is 331,848,492 bytes,
SHA-256 `3e4decb0a39fb73f6d6170102c93ec260988accedb6c2dedd855a0f74c85cbb1`.
The closed-batch audit verified 196 remote evidence files and fourteen owned
process groups, with no remaining processes or zombies. All frozen donors and
the earlier native bounded checkpoint retain their hashes. `FINAL_BACKUP.json`
records the consolidated archive after restoration and rerunning both reviewers.

The final runner includes a correction to `--resume` validation for identical
HDL. The native run used `--loader-update`, whose accepted changes are unchanged;
its executed runner is preserved separately. `RUNNER_REVISION.diff` records
the helper-only difference. A native `--resume` rerun is not claimed.

The isolated synthesis reports **timing not met**, WNS −0.364 ns, WHS +0.046 ns,
WPWS +0.836 ns, with 138 input and 699 output ports lacking delay budgets.
Unqualified inter-clock paths account for the negative setup estimates; there
are no added false paths, asynchronous clock groups or warning waivers. These
numbers are neither native integrated area nor placed/routed timing closure.

Recover the successful native executable by extracting
`recovery/weight-loader-runtime.tar.gz` from the final bundle into a fresh Linux
directory, sourcing Vivado 2026.1 and running:

```bash
NTRPU_NATIVE_OUT=/absolute/new/output NTRPU_BAR_BASE=80000000 \
  xsim hbm_tlp_behav -tclbatch /absolute/selected/restored/source/run_hbm_tlp_trace.tcl \
  -testplusarg TESTNAME=ntrpu_bridge_native -log simulate.log
```

Use the trace script from this selected run, which includes loader signals.
Archive restoration and hash verification do not claim another native execution.
A source rebuild also needs the earlier frozen PCIe compilation seed and must
regenerate the HBM model inventory at its new path; the exported compile order
contains absolute original paths.
