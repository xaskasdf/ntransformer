# Native PCIe traffic to the HBM control registers

This isolated bank attaches the frozen [BAR adapter and CSR](HBM_BAR_CONTROL.md)
to the actual AMD XDMA/Bridge simulation model. It replaces the example BRAM
target in a private copy. Requests originate at the AMD Root Port, traverse the
PCIe PIPE models and native `m_axib` interface, and receive completions at the
Root Port. No AXI traffic is injected behind the bridge and no internal IP
registers or responses are forced.

## Scope and fixtures

The part is `xcvu47p-fsvh2892-2-e`, with Vivado 2026.1 and XDMA/Bridge 4.2 rev. 4.
The source IP settings match the synthesized control integration except for
`pipe_sim=true` versus `false`. PIPE models the digital link without the serial
transceiver path. This is not a physical-link, board, OS-driver or full-HBM-top
simulation. AMD describes the Root Port as a traffic source and receiver, not a
complete verification BFM: [PG194 Root Port model](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Root-Port-Model-Test-Bench-for-Endpoint).

The CSR's downstream readiness and response inputs are tied low. A valid CFG
offer therefore remains held and can be inspected through its active readback.
This tests publication and immutability; it does not test writer acceptance,
mapping, DMA, remote ACK or HBM data. Four explicit telemetry clock fixtures
carry known words `0x5a010000 + word_index`; actual sensor values are not modeled.
The earlier standalone and native structural evidence remains separate.

## Traffic and acceptance criteria

Five groups contain 65 host requests (16 posted writes and 49 reads):

- Read the ID, version and initially clear descriptor mask.
- Write/read all eight CFG words, apply a partial-byte write and verify the mask.
- Reject an upper-aperture alias and two-DW transfers before CSR access; verify
  staging remains unchanged and an unimplemented CSR returns an explicit error.
- Request a snapshot, read all 16 DW lanes of the 512-bit native bus and release it.
- Publish CFG, inspect all active words, rewrite staging and reject a repeated
  commit while the original offer remains held.

Every read must receive one bounded, attributed completion. The test captures
the Root Port's public allocated tag, not merely the tag proposed in RQ.
Completion status, error code, payload length, poison and Request Completed are
checked according to [PG213 descriptor format](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Descriptor-Format)
and [tag management](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Tag-Management-for-Non-Posted-Transactions).
A timeout never counts as successful rejection. Posted-write disposition is
observed on native AXI B; no PCIe completion is expected for a posted write.

The independent reviewer reconstructs expected addresses, byte enables, payloads,
AXI responses, CSR admissions and host readbacks. It compares logged handshakes
against values before clock edges in the VCD, including native tag allocations.
The bank is serialized; concurrent outstanding requests, PCIe reset/retraining
and exhaustive malformed-packet coverage remain separate tests.

## Reproduce and preserve

`prepare_bar_tlp.py STOCK_EXAMPLE FROZEN_BAR_SOURCE NEW_PREPARED` preserves AMD
notices and creates the private endpoint binding and bench sources. The stock
example and serial XCI used for the profile comparison are the closed local
artifacts named in that script. `run_bar_tlp.py NEW_OUTPUT --seed COMPILED_SEED
--prepared NEW_PREPARED --settings VIVADO_SETTINGS` copies the backed-up native
seed, recompiles the new endpoint/bench and runs the native model.

For a second address allocation, use `--replay --seed CLOSED_RUN/work
--bar-base 0x80000000`. Replay requires a successful donor, its compiled traffic
snapshot and identical endpoint, board, include and probe RTL hashes. It copies
the snapshot into a fresh run and skips compilation and elaboration. At 1 us,
the trace script uses Vivado `set_value` only on the Root Port testbench's
`BAR_INIT_P_MEM32_START`. The normal AMD `BAR_INIT` task subsequently programs
the endpoint through PCIe configuration transactions. The reviewer requires the
requested base, actual RQ addresses and translated AXI offsets to agree.

The runner accepts only fresh paths under `/home/ubuntu/vivado/ntrpu-side`, uses
two available CPUs and nice 10, and bounds compilation/elaboration/simulation to
90/900/900 seconds. It records every owned process group and hashes the compiled
seed before and after execution. Each run exports source, logs, VCD and a closed
file manifest, including failed attempts. Compiled tool databases are recoverable
from the original native seed archive and are not duplicated in each evidence tar.

Local evidence lives in `D:/AI/artifacts/ntrpu-axi-bridge-bar-tlp`. Review restored
evidence with `python -B review_bar_tlp.py RESTORED_RUN --output REVIEW.json`.
The final bundle must be restored and reviewed again before its backup receipt
is issued. Existing main-thread work, native checkpoints and queues are untouched.

## Recorded results (2026-10-12 UTC)

| Closed run | BAR0 base | Groups / checks | Requests / completions |
|---|---|---|---|
| `bar-tlp-20261012T011539Z` | `0x00000000` | 5 / 5,872 | 65 / 49 |
| `bar-tlp-20261012T013412Z` | `0x80000000` | 5 / 5,872 | 65 / 49 |

Both banks passed independent log/VCD review: 130 requests, 98 completions and
11,744 checks in total. Each matched 16 native AW/W/B transactions, 49 AR/R
transactions and 49 allocated completion tags. Only 14 writes and 47 reads per
bank reached the CSR address channels, as planned. The relocated bank verified
host address `base + offset` becomes the same native AXI/CSR offset with identical
readbacks. All three frozen control RTL files and the bench remained unchanged.

The PIPE link reported 8 GT/s x16; the measured native AXI clock was 250 MHz.
Initial compilation, elaboration and simulation took 2.8, 358.2 and 278.7 seconds.
The relocated bank reused that compiled snapshot and simulated in 270.8 seconds,
with no second compilation or elaboration.

For this IP configuration, the observed error mapping is AXI DECERR to PCIe
Unsupported Request (status 1), and AXI SLVERR to Completer Abort (status 4).
Both unsuccessful read completions carry RC error code 2 and no payload. The
reviewer enforces this exact mapping and rejects ten deliberately damaged
evidence variants, including wrong status, base, data, strobe, missing response
and disagreement between the text log and waveform.

Warnings remain visible in the archived logs: the example wrapper uses a signal
before its explicit `bar_bid` declaration; AMD-generated sources also report
width/index warnings, and XDMA reports `unique case` initialization warnings at
time zero. This result qualifies the exercised transactions, not a warning-free
model or exhaustive protocol compliance.

The original compiled libraries and kernel are backed up in native-r4's archive
(SHA-256 `eed001b2ea062f7cfa7e891a505e34455c5ef2a64ffd56cc9532bade4ec7d040`).
An independent comparison matched 366 library/kernel files against the donor.
The final bundle includes this seed archive, both closed runs and the verifier.
`seal.py` restores the ZIP, verifies its complete hash inventory and repeats both
evidence reviews before issuing `FINAL_BACKUP.json`. This does not claim that
Vivado was reinstalled or that XSIM was rerun from the restored ZIP.

The final read-only audit checked six owned process groups: no live or zombie
processes remained. All 52 closed evidence files, the two compiled donors
(399 and 621 files) and the previous bounded native checkpoint retained their
hashes. The source and compiled donors were copied, never simulated in place.
Writer acceptance and the actual HBM path, physical PCIe, board constraints and
timing closure remain pending, as detailed in the earlier control integration.
