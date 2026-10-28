# Full-board registered HBM READY experiment

This isolated VU47P/HBM experiment integrates the two-slot candidate validated in
[HBM_REGISTERED_READY.md](HBM_REGISTERED_READY.md). Evidence belongs under
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-ready-physical/`. It retains the previous
selected checkpoint, sealed evidence and separate DDR design.

## Frozen RTL and comparison

The donor is the complete source composition from the capture experiment's
`board-capture-physical-synth-20261013T212305Z`. Its adapter hash matches the
sealed unit donor. The only RTL change replaces that adapter's tracker module
with the tested predictive-READY tracker, restores its original module name
for existing instantiations, and appends the exact tested queue definition.
Both active HBM channels therefore use the same validated implementation.
All other original RTL and the native IP generation donor remain unchanged.

The incremental placement donor is the capture candidate's routed checkpoint:
`0b38a6ab714dca42c88f3f98b69bc1ff00d57856236daed17d4ba93a5f65afc8`.
Its reopened WNS is -0.579 ns, TNS -572.291 ns, with 3,435 failing setup
endpoints. The original selected reference remains at WNS -0.413 ns. A fresh
full synthesis means timing differences also include changes in implementation;
they cannot be attributed exclusively to the new register.

## Native stages

`prepare_hbm_ready_physical.py DONOR_RTL TESTED_UNIT FRESH_OUTPUT` prepares the
full synthesis. The archived `pipeline.py` collects and reviews this batch,
then prepares separate constraint/optimization, placement/routing and saved
checkpoint reopen stages. Each uses a fresh private directory, two CPUs,
nice 10 and a 1,800-second tool limit. An unsuccessful native batch is backed
up before further stages stop. Process control is restricted to owned groups.

`prepare_hbm_ready_stage.py AUDITS RUNNER INPUT_DCP FRESH_OUTPUT --mode MODE`
copies the frozen full-board implementation and audits. Its small wrapper adds
READY checks to the existing response queue probe. Route mode requires local
and remote paths to the exact incremental donor; the inherited commands remain
`place_design -no_psip` and `route_design -no_psir`.

## Review criteria

Both external AXI RREADY inputs of the HBM IP must come directly from ordinary
FDCE Q pins, with constant enable, the existing 450 MHz clock and reset from
the distribution register. Inside the unchanged AMD IP, `hbm_top.sv` drives
primitive READY as `AXI_READY || !AXI_VALID`. The probe separately verifies that
LUT2 function, both input drivers and its complete fan-in. The direct-register
requirement applies at the AXI interface, not beyond this vendor-owned gate.
The native evidence proves either primitive clear inversion or an explicit
LUT1 inverter. Each response ownership register must still depend on RVALID.
All 1,060 response payload enables retain the free-slot capture cut.

For each channel, max/min timing is exported for the READY D input, Q-to-HBM
output and distribution-to-CLR path. Reset recovery/removal must remain timed,
without new exceptions. A registered output alone does not establish closure:
the predictor's input path or asynchronous clear can become critical.

The unchanged board review retains 2,195 bounded paths, the synchronous checks,
18 skew groups, clock topology, reference ownership, calibration selectors,
CDC/DRC findings and experimental pin assignments. Effective timing commands
must match the previous donor; only regenerated internal waiver timestamps are
excluded from the semantic comparison. Existing coverage gaps remain reported.

## Evidence and limits

Each closed batch has a local archive, file hashes and `BACKUP.json`. The final
review compares routed and reopened measurements and exercises rejection of
corrupted READY and inherited-contract evidence. Sources and preparations are
retained for replay; the final bundle is restored and checked before closure.

This experiment measures implementation timing. It does not validate physical
HBM/PCIe traffic, PCB pin release, power, thermal behavior or manufacturability.
No candidate is promoted merely because the former combinational RREADY path
disappears.

## Integration audit correction

The first full synthesis saved checkpoint
`7f2f6cb27a282222787ebf76511e2db9651178f33125c4cddb668006d39de2cf`
but stopped after 704.529 seconds: the new auditor incorrectly required a direct
register connection at the internal HBM primitive. Its 91,912,448-byte archive
was retained as failed and restored locally (277 verified files). A separate
95.877-second read-only binding probe identified the native gate on both stacks;
the generated AMD HDL confirms the exact `READY || !VALID` function.

Recovery reopens the same checkpoint, verifies the vendor gate explicitly and
exports the remaining synthesis audits. No RTL, IP function, clock or timing
exception is changed to satisfy this correction. The original failed batch and
its prepared auditor remain immutable. The initial SCP transfer was incomplete;
collection resumed through a binary SSH stream after checking the downloaded
prefix, and the complete archive hash matched the remote receipt.

An initial recovery also rejected the assumed internal `RVALID` pin name. A
second binding probe exported `RVALID_PIPE` and the actual LUT2 `INIT=4'hB`,
with I0 from registered READY and I1 from that VALID output, on both stacks.
The corrected read-only recovery completed in 173.530 seconds and passed local
review: both ordinary READY registers, native gate truth tables, six ownership
registers and all 1,060 response payload enables matched their contracts.
Both diagnostic batches and both rejected audits remain in the evidence chain.

The constraint/optimization batch completed in 350.601 seconds. Its reviewed
checkpoint is
`69e385861ffe1b927a97bfde15945ce408edf2edce7e1dfc1078bb82f167d568`.
All 2,195 bounded paths and 84 retained synchronous checks keep their original
contracts, and CDC has no critical finding. Pre-route timing estimates are not
used as the physical comparison; placement, routing and reopen are required.

## Routed measurements

Placement/routing and audits completed in 1,094.763 seconds. The routed DCP is
`6670d41ebfe8581d1b63cfaa4965441f2267fec327e65f2041c4621eda6bd578`.
All 100,268 routable nets are routed, with zero routing errors. Final reuse is
99.85% of cells and 99.36% of nets. The 85,014,428-byte native archive was restored
locally with 135 files verified; its SHA-256 is
`92b331b3bd3adfe17b7e2094a11758a9b69ade0978707f0df302f79a9c396d84`.

| Checkpoint | WNS (ns) | TNS (ns) | Failing setup endpoints |
| --- | ---: | ---: | ---: |
| Original selected reference | -0.413 | -803.549 | 4,555 |
| Capture donor, reopened | -0.579 | -572.291 | 3,435 |
| Registered READY, routed | -0.458 | -599.878 | 3,362 |

The worst setup margin improves versus the capture donor, while aggregate setup
deficit worsens by 27.587 ns. Hold and pulse-width failures remain zero (WHS
+0.005 ns). This is a mixed result, not timing closure or a promoted reference.

| Targeted path margin (ns) | Channel 0 | Channel 1 |
| --- | ---: | ---: |
| READY D setup | -0.372 | -0.193 |
| READY D hold | +0.335 | +0.275 |
| Registered Q to native HBM READY setup | +0.773 | +0.882 |
| Registered Q to native HBM READY hold | +0.516 | +0.420 |
| READY CLR recovery | +0.011 | +0.241 |
| READY CLR removal | +0.949 | +0.865 |

Including the native VALID-to-READY branch, the worst HBM READY setup margin is
+0.615 ns, versus -0.579 ns before. The predictor input now exposes a timed
distribution-to-D path through five logic levels on channel 0. Reset recovery
passes but its +0.011 ns minimum provides little margin. The global worst path
is channel 1 `return_cut/count_reg[1]/C` to the read CDC FIFO RAM write enable,
with two logic levels and 2.306 ns datapath delay.

All 2,195 bounded paths, 84 synchronous checks and 18 skew groups pass; minimum
skew slack is +0.044 ns. The canonical release retains one distribution consumer
and recovery/removal +1.489/+0.231 ns. Its distribution has 1,092 leaf loads.
The 1,060 payload enables still exclude RVALID, but 20 fail setup (worst
-0.169 ns), versus 15 at -0.157 ns in the donor. Their hold margin is +0.155 ns.
Existing coverage gaps remain: 1,154 unconstrained internal endpoints and two
ports without input delay. CDC has no critical finding; inherited warnings are
retained in the reports.

Follow-up work should investigate the predictor's reset-to-D logic and the
return FIFO write-enable path, retaining the bounded functional tests. No new
RTL optimization is included merely to improve this experiment's reported result.

## Reopen and closure

Reopening the exact routed checkpoint took 295.200 seconds. Global setup, hold
and pulse-width totals, all targeted READY/queue/reset margins, bounded paths,
skew and routing counts match the routed reports exactly. The 424,214-byte
reopen archive was restored with 129 files verified. `COMPARISON.json` records
the measured differences and retains `selected_reference_replaced=false`.

The eight native batches use the prefix `board-ready-physical-`:

| Suffix (UTC) | Purpose | Outcome |
| --- | --- | --- |
| `synth-20261013T230758Z` | Full synthesis and original probe | Checkpoint saved; probe rejected |
| `inspect-20261013T233829Z` | Native READY binding diagnosis | Reviewed |
| `inspect-20261013T234330Z` | First synthesis audit recovery | Pin-name assumption rejected |
| `inspect-20261013T234702Z` | Gate truth table and VALID pin diagnosis | Reviewed |
| `inspect-20261013T234943Z` | Corrected read-only recovery | Reviewed |
| `inspect-20261013T235339Z` | Constraints and optimization | Reviewed |
| `inspect-20261014T000038Z` | Placement, routing and audits | Reviewed; setup fails |
| `inspect-20261014T002402Z` | Reopen saved routed checkpoint | Same measured margins |

Twenty-six corrupted-evidence controls are rejected, including altered native
gate functions, reset polarity, missing timing and added exceptions. The final
closure script replays four preparations, checks source hashes and owned VM
processes, then creates and restores a ZIP and repeats the local reviews.
`FINAL_BACKUP.json` records the bundle hash and restoration receipt;
`FINAL_PROCESS_AUDIT.json` records process and predecessor integrity checks.
`review_jobs.py` reproduces the complete review, including read-only recovery
and the preserved failures. Transfer retries are recorded separately and never
change a native result or checkpoint.
