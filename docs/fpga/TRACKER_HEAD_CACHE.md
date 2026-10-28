# Registered tracker head length

**2026-10-08: internal OOC timing and public post-route functional regression
PASS.** Select `refine1` with `drc2` and `routed3`. Native HBM controller replay,
external I/O timing, PCIe integration and board qualification remain pending.

## Purpose and scope

The mapped peer reader's selected routed baseline misses the 2.222 ns memory
clock by 0.019 ns at 57 endpoints. Its critical path reads
`lengths[validate_ptr]` and checks RLAST before enabling a validated response slot.
Transport timing passes at 5.000 ns. See `PEER_REGIONS_PHYSICAL.md`.

`fpga/experimental/ntrpu_hbm_read_head_tracker.sv` isolates a candidate that caches
the current descriptor length in four registers. It leaves the reference tracker,
queue depth, credit accounting, response quarantine, clocks and CDC constraints
unchanged. A private source tree replaces both the standalone tracker and the
identical module embedded inside the frozen adapter with the candidate;
the repository's integration files are not replaced.

## Descriptor transitions

The intended invariant is `head_length == lengths[validate_ptr]` whenever any
descriptor is reserved:

| Event | Cached head update |
| --- | --- |
| Reserve into an empty queue | Bypass the incoming length |
| Release the only descriptor while reserving another | Bypass the incoming length |
| Release with more than one descriptor reserved | Load the next descriptor's stored length |
| Otherwise | Hold |

The bypass avoids depending on same-cycle RAM read/write behavior. With more
than one reserved descriptor, the next descriptor already exists; a concurrent
new reservation writes a different entry. Depth one cannot reserve while full.
This argument is supplemented by simulation, not a formal equivalence proof.

## Validation and provenance

`tools/run_ntrpu_tracker_head_unit.py` exercises depths 1/2/4/8 with deterministic
mixed-length traffic, ring wrap, stalls and zero-latency response offers. It
compares public signals and accounting state against the frozen reference every
cycle and checks payloads independently. The existing six quarantine/credit tests
run at all four depths. Five mutations target stale head reuse, wrong next head,
premature credit release, masked RID and combinatorial READY.

The first differential batch exposed a testbench fill-barrier deadlock. It remains
archived as failed; the correction gates native descriptor progress, without
changing the candidate RTL or the cycle limit.

The next gates are the six-case mapped public ledger, then bounded synthesis and
routing at the original clocks. A changed tracker does not inherit its donor's
native HBM simulation qualification. Post-route behavior, native HBM replay,
external I/O timing and board qualification remain separate gates.

Evidence root: `D:/AI/artifacts/ntrpu-tracker-head/20261008T152306773324Z`.
Each completed batch is hashed, stored incrementally in `ntrpu-content-store`,
restored and independently re-audited. Final measurements are recorded below.

## Completed behavioral gates

- `20261008T153144470428Z-unit2`: eight differential runs, 3,072 descriptors,
  25,624 beats; all pass. Twenty-four quarantine/credit cases pass and all five
  deliberate faults are rejected. Runtime: 202.940 s.
- `20261008T154734955910Z-public2`: all six integrated public-port cases pass;
  37 requests, 31 admissions, six rejections, 27 successful results, four retained
  failed results, 118 native-interface AR handshakes. All 12 ledger mutations
  are rejected. Runtime: 62.879 s.
- `PUBLIC_COMPARISON.json`: all 6,808 `SP`/`SF` events, including timestamps,
  exactly match the selected baseline. Canonical trace SHA-256:
  `a3ae2f0844ea1f1e82024e43eaa3eab53741c6821659db5b2a2fc35a85d029b1`.

The unit and public batch manifests were restored and re-audited (343 and 206
files respectively). The candidate RTL SHA-256 is
`90f8bb067e1b52e718be0c826912bc57efc3129b3322713bebcb0ec2f262ac05`.

## Withdrawn integration evidence

`DISQUALIFICATION.json` explicitly withdraws `public1` and `synth1` as evidence
for the candidate: they changed an unused standalone file while compiling the
reference module embedded in the adapter. Their sealed files are retained;
neither is a donor for the corrected physical flow. `unit2` directly instantiates
the distinctly named candidate and remains valid.

The corrected preparation verifies both tracker copies against one candidate.
Synthesis must additionally find eight `tracker/head_length_reg` registers and
export `head_registers.tsv`; missing or malformed evidence fails the audit.

## Corrected synthesis

`20261008T155050969320Z-synth2` confirms all eight head registers. It uses 25,185
LUTs and 24,367 registers: +32 LUTs and +9 registers relative to the baseline
synthesis, with zero BRAM tiles. Internal timing coverage has no missing clocks,
unconstrained internal endpoints, loops, latches or black boxes. The CDC
inventory remains 13 CDC-3 and eight CDC-9 findings.

Pre-route setup failures decrease from 1,102 to 32 endpoints and total negative
slack from -189.910 to -4.051 ns; these estimates do not qualify routed timing.
The synthesized checkpoint SHA-256 is
`5b5ca23797bad5c20bfa2e7259bf8004fc3adca331a3da492771ca6cb05d65c5`.
Its 219-file backup was restored and re-audited. Routing has a 2,400-second limit;
the clock periods and acceptance thresholds remain unchanged.

## Routed candidate

`20261008T155759300496Z-route1` finishes in 1,313.262 s. All 38,691 routable nets
are routed without routing errors. The internal timing gate remains **OPEN**:

| Metric | Baseline | Cached head |
| --- | ---: | ---: |
| Transport setup slack (ns) | +0.035 | +0.029 |
| Memory setup slack (ns) | -0.019 | -0.002 |
| Failing setup endpoints | 57 | 1 |
| Total negative setup slack (ns) | -0.622 | -0.002 |
| Transport hold slack (ns) | +0.018 | +0.013 |
| Memory hold slack (ns) | +0.007 | +0.006 |
| LUTs | 24,771 | 25,039 |
| Registers | 24,344 | 24,357 |

All eight bus-skew checks pass; minimum margin is 1.491 ns. CDC, internal coverage
and registered native READY checks pass unchanged. Six PDCN-1569 warnings and
one RTSTAT-10 warning require the separate object-level audit.

The sole failing path now starts at channel 1's width-converter `first_word_reg`
and ends at `tracker/validated_cut/read_slot_reg/D`. It traverses the width and
protocol converters' READY logic: seven LUT levels, 2.123 ns data delay, of which
1.566 ns is routing. The original length-lookup-to-payload-enable path is no
longer a failing endpoint. A separate 420-second post-route `Explore` attempt
uses an untouched input copy and the same timing criteria.

## Selected physical result

`20261008T162246647239Z-refine1` closes internal timing in 155.570 s:
transport setup **+0.029 ns**, memory setup **+0.020 ns**, hold **+0.013/+0.006 ns**.
Setup, hold and pulse-width reports have zero failing endpoints and zero total
negative slack. All eight bus-skew checks still pass. Clock periods, uncertainties,
RTL and resource counts are unchanged from `route1`.

Vivado reports a critical-path logic transformation and READY-net improvement
worth 0.022 ns. The selected checkpoint SHA-256 is
`3b37b0ec666f6cae78679510b2880f370fcf4cc9a77f02a75b8745a8db915646`;
the untouched functional netlist SHA-256 is
`187c267d0ffffe780a8c56477669c5be93f85d4ca6e51b67116dcf7bdf26c514`.

`20261008T162813789624Z-drc2` audits that exact checkpoint: all six PDCN-1569
instances and the RTSTAT-10 finding are classified, with no unresolved objects
and no waivers. The 3,403 RTSTAT nets comprise 3,301 external OOC ports without
internal loads and 102 nets without logical leaf loads. This is a reviewed warning
set, not a clean DRC report or an external-I/O qualification.

## Netlist bench connection correction

The exported netlist splits `cfg_map` into eight escaped public port names.
`routed1` failed elaboration with implicit connections; `routed2` stopped during
preparation on an overly strict DUT-text guard, before Vivado was invoked. Both
are archived as failed, with `routed2` explicitly identified as collector-recorded
preparation failure. The corrected generator verifies all eight names and widths,
connects their struct fields explicitly, and preserves the other existing public
connections. The netlist itself is byte-for-byte unchanged.

`20261008T163445722283Z-routed3` passes all six cases in 146.342 s, with all 12
ledger mutations rejected. Its counters match the RTL preflight: 37 requests,
27 successful results/ACKs/DMA completions, six rejections and four retained
failed results. Both mapped stacks and generation remapping are exercised.
This is functional netlist simulation without SDF; it is not a formal or
exhaustive cycle-equivalence claim. Static timing is qualified separately by the
selected checkpoint's reports.

## Handoff and next gate

`OUTCOME.json`, `COMPARISON.json` and the restored CAS handoff manifest record the
selected checkpoint, withdrawn/failed attempts, source hashes and per-batch
receipts. All 83 previously protected files remain unchanged. The original
tracker and principal workflow are not replaced by this isolated candidate.

Next, rebuild the native HBM simulation snapshot with the exact embedded-adapter
delta and repeat the mapped PC0/PC16 traffic before integrating this candidate
with the PCIe shell. Preserve the selected IP configurations, clock constraints,
ownership/error checks and evidence binding. The candidate does not inherit the
baseline controller's qualification merely because its public traces match.
