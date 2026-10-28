# VU47P/HBM internal control plane

This experimental block supplies the internal command producers and response
consumers identified in [the native boundary audit](HBM_NATIVE_BOUNDARY.md).
It is a standalone AXI4-Lite **32-bit** slave, not yet a native PCIe BAR endpoint.
The native incoming AXI interface is 512 bits; its bridge, address decode and
connection to this slave remain separate integration work. The DDR card is unchanged.

## Transaction and reset contract

`ntrpu_hbm_control_plane.sv` buffers AW and W independently and holds B/R until
accepted. It permits one outstanding write response and one read response.
Addresses are byte offsets, aligned to four bytes; unsupported accesses return
SLVERR. Staging registers merge WSTRB bytes, including a successful no-op for zero
strobes. Doorbells require all strobes and data `1`; other values return SLVERR.
AxPROT is accepted but is not an authorization mechanism in this internal prototype.

Every descriptor byte must be freshly written before COMMIT. COMMIT copies the
staging image into an immutable active offer and clears its written-byte mask.
Rewriting staging cannot change that offer. Reserved CMD/ACK bits must be zero.
The writer remains responsible for session, generation, address and transfer checks.

Configuration is blocked while configuration, command or ACK work is outstanding.
Commands are blocked while configuration, command or ACK work is outstanding.
Configuration responses and command events occupy separate one-entry mailboxes;
software must POP them before reusing that operation. POP of an empty mailbox fails.
If both event inputs are presented, rejection takes priority and only one is accepted.

ACK COMMIT requires an accepted command, writer `ack_ready`, no simultaneous event,
and no previous ACK publication for that command. It is posted at most once per
command. Software may only publish an ACK actually received from the remote
consumer; validating that external origin is outside this block. The writer still
requires both successful AXI B and a matching remote ACK before generating a result.
Acceptance into the retained result mailbox can retire the writer; later software
reads and POP do not control that retirement.

`bus_rst_n` clears only the AXI transport. Descriptors, dirty masks, accepted work,
mailboxes and snapshot state survive. A committed operation can lose its B response
during reset: software must reconcile status and **active** descriptors before retrying.
Exactly-once replay across bus reset is not promised. `cold_rst_n` clears all state;
use only the canonical configuration reset, with common assertion and synchronized
release in each domain. Independent telemetry resets during a round are unsupported.

## Register map, version 1

All multiword values use least-significant word first. W means write-only doorbell,
R read-only and RW staging storage. Read payloads are meaningful only when their
mailbox/snapshot valid bit is set; the previous snapshot remains visible while busy.

| Offset | Access | Meaning |
|---|---|---|
| `000`, `004` | R | ID `0x4e545243`, version `1` |
| `008` | R | Bits 0–7: cfg busy, cmd busy, ACK pending, cfg mailbox full, event full, snapshot busy, snapshot valid, ACK already published for last/current command |
| `00c`, `010`, `014` | W | CFG, CMD, ACK COMMIT |
| `018`, `01c` | W | CFG POP, EVENT POP |
| `020`, `030` | W | SNAP REQUEST, SNAP RELEASE |
| `024`, `028`, `02c` | R | Snapshot epoch; cfg error bit; event kind: 1 reject, 2 result |
| `034` | R | Snapshot progress, one bit per domain while busy |
| `038`, `03c`, `040`, `044` | R | Written-byte masks: CFG32, CMD low32, CMD high12, ACK28 |
| `048` | R | Bits 0–4: cfg offer, cfg waiting response, cmd offer, cmd waiting event, ACK offer |
| `100–11c` | RW | CFG: session64, generation64, AXI base64, PCIe base64 |
| `140–168` | RW | CMD: tag128, generation64, source64, offset64, bytes13, reserved19 |
| `180–198` | RW | ACK: tag128, generation64, bytes13, reserved19 |
| `200–210` | R | Event tag128, followed by reject code4 or result bytes13; high bits zero |
| `300–33c` | R | Four domain samples, 128 bits each; domain 0 occupies `300–30c` |
| `400–41c`, `440–468`, `480–498` | R | Active CFG, CMD and ACK images |

## Snapshot semantics and integration requirements

`ntrpu_control_snapshot.sv` transfers request/acknowledgment tokens through three
registers in each direction. Each domain captures its full 128-bit input once,
holds it until the next round, and acknowledges it. The control domain adds a
settling cycle before sampling each held bus, then publishes all four samples
together only after all domains complete. Epoch advances modulo 2^32.

A round is coherent **within each domain**, not simultaneous across all clocks.
No new round is accepted while busy or valid; RELEASE is mandatory after reading.
A stopped source clock keeps the round busy without publishing partial data.
There is no timeout that silently discards a round.

The intended domain ordering is PCIe, management, reader, memory. Native status
packing and top-level connections are pending; the bank uses changing counter
fixtures. The OOC clocks are 250, 100, 200 and 450 MHz, with a separate 250 MHz
control clock; native integration must connect domain 0 to the actual control clock.
Held-data paths still require physical delay constraints, reset review and CDC
qualification in the assembled design. Unconstrained CDC/timing reports are evidence
of open work, not signoff. Existing native constraints must not be copied blindly.

This follows the independent-channel rules of the
[Arm AXI specification](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).
AMD's [full-handshake CDC guidance](https://docs.amd.com/r/2023.1-English/ug953-vivado-7series-libraries/XPM_CDC_HANDSHAKE)
describes the held-bus principle; this block implements its own handshake and does
not instantiate XPM_CDC_HANDSHAKE or inherit that macro's physical qualification.

## Reproduction and evidence

Freeze the six RTL/runner/Tcl files named by `launch.py` plus `PROFILE.json` into a
prepared directory. The writer used here is the earlier frozen implementation,
SHA-256 `885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6`.
Do not substitute the main thread's working copy without treating it as a new revision.

```bash
python3 run_hbm_control_plane.py "$NEW_PRIVATE_OUTPUT" \
  --prepared "$PREPARED" --settings "$VIVADO_SETTINGS" --phase sim
python3 run_hbm_control_plane.py "$ANOTHER_NEW_PRIVATE_OUTPUT" \
  --prepared "$PREPARED" --settings "$VIVADO_SETTINGS" --phase synth
python3 review_hbm_control_plane.py "$RESTORED_SIM" "$RESTORED_SYNTH"
```

Outputs must be fresh subdirectories of
`/home/ubuntu/vivado/ntrpu-side/axi-bridge-hbm-native-join`. The runner uses two
available CPUs, nice 10, two Vivado threads, 120-second simulation-step limits and
a 1,200-second synthesis limit. It can terminate only its own process groups.

The bank exercises eight bus/mailbox/snapshot groups and four real-writer groups,
each at four deterministic phase/stall variations. Mapping, source data, PCIe B and
the remote ACK are explicit fixtures. It does not simulate native PCIe/HBM traffic
or inference. Logs, source hashes, the OOC checkpoint, failed attempts if any, and
backup receipts live under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-control`.

### Recorded batch: 2026-10-12 UTC

- Smoke: `control-sim-20261012T000131Z`, both seed-0 banks passed.
- Final revision: `control-sim-20261012T000344Z`, all eight simulations passed,
  48 named groups and 10,317 assertion checks; 76.36 seconds elapsed in the runner.
- OOC: `control-synth-20261012T000404Z`, 147.29 seconds; 1,933 CLB LUTs
  (2,111 LUT primitives before combining),
  3,620 FDCE registers, 24 ASYNC_REG registers and zero latches.
- CDC evidence: eight CDC-3 token crossings and **512 CDC-15 held-data warnings**.
  No exceptions or severity changes were added. There are 863 input and 839 output
  ports without delay constraints. Estimated overall WNS is −0.278 ns, on the
  unconstrained inter-clock crossings; this is not timing closure.
- The local reviewer checks identities, exact event order, ACK/B ownership,
  stopped-clock progress, source hashes and OOC structure. Its 13 negative controls
  reject altered or incomplete evidence. The bundle is restored and reviewed again
  before the final SHA-256 receipt is written.

Next integration: bind the native status signals to these four sample inputs,
connect the control producers/consumers in the frozen native top, then implement
and validate the native 512-bit AXI to 32-bit register path and its BAR decode.
Re-run native boundary/CDC review for those actual clocks and paths.
