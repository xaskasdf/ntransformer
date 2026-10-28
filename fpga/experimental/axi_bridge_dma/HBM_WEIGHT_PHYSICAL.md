# Native weight-loader synthesis and clock crossings

This isolated VU47P/HBM stage composes the boot-time BAR loader with the native
PCIe bridge, HBM controller, clock generator and read converters. It does not
change the DDR512GiB variant or the main Vivado queue.

## Source contract

`prepare_hbm_weight_physical.py` removes only the known AMD PIPE simulation
scaffolding from the successful loader bank. It independently reconstructs the
same serial top from the previous native synthesis donor, then requires exact
text equality. All 22 frozen RTL files and `ntrpu_hbm_weight_loader.sv` retain
their validated hashes. This is a checked source composition, not formal
sequential equivalence or a new simulation run.

The top remains `ntrpu_axi_bridge_hbm_controlled`, targeting
`xcvu47p-fsvh2892-2-e`. Its 74 external bits are physical/sequencer signals;
the HBM write interface is internal. The native functional bank remains the
evidence for six PCIe-loaded lines, 192 bytes and 43,468 checks; see
[HBM_WEIGHT_LOADER.md](HBM_WEIGHT_LOADER.md).

## Audit and constraint contract

`audit_hbm_weight_physical.tcl` retains the earlier native/control audits and
checks both 112-bit AXI-Lite segments through the loader. It traces the write
signals to native HBM PC00/PC16, including folded constant fields, and verifies
the canonical clocks and separate cold/transport resets.

Loader constraints use exact source cells and destination pins:

- Payload/address capture: 2.222 ns maximum datapath delay and 1.111 ns skew.
- Completion capture: 4 ns maximum delay and 2 ns skew; all four paths into the
  sticky failure reduction receive the same 4 ns delay bound.
- Request and readiness into memory: 2.222 ns; acknowledge and synchronized
  readiness into PCIe: 4 ns.
- Twelve synchronizer registers must retain ASYNC_REG. All eight adjacent
  stage pairs retain ordinary setup and hold analysis.

These are conservative implementation budgets alongside the retained-data
handshake and settling cycle. They are not a timing pass. No broad clock cuts,
new false paths, CDC waivers or invented board pin assignments are introduced.
The existing eighteen individually audited asynchronous control exceptions
remain scoped to their original pins.

AMD distinguishes a recognized CE-controlled crossing from proof of its
functional correctness ([UG906](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/CE-Controlled-CDC)).
The maximum-delay and bus-skew approach follows the handshake guidance in
[UG903](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/set_bus_skew-Example-One);
the actual budgets above are this project's choices.

## Execution and recovery

Artifacts belong to `D:/AI/artifacts/ntrpu-axi-bridge-hbm-weight-physical`.
Remote runs use fresh `weight-physical-*` directories under the private native
integration namespace, two CPUs, nice 10 and a 1,200-second per-tool limit.
Synthesis, constraint application and fresh checkpoint reopening are separate
batches. Each closed archive is restored locally and every file hash checked.

Run the independent review against those three restored roots:

```text
python -B review_hbm_weight_physical.py SYNTH_ROOT APPLY_ROOT REOPEN_ROOT
```

The reviewer checks checkpoint provenance, connections, constraint persistence
and deliberately damaged evidence. CDC/DRC warnings and timing coverage remain
visible. A bounded synthesized checkpoint does not qualify routing, a physical
PCIe link, the PCB or model inference.

## Recorded findings, 2026-10-13 UTC

Full native synthesis `weight-physical-synth-20261013T052939Z` completed in
776.54 seconds with unchanged inputs, no black boxes and no latches. Integrated
utilization is 61,093 LUTs and 65,707 registers, with 60 RAMB36 and one RAMB18.
The loader instance accounts for 636 LUTs and 1,335 registers. Its synthesized
top has the same functional attachment as the successful native simulation.

The raw synthesis timing estimate does **not** meet constraints: WNS -0.413 ns,
WHS -0.842 ns and WPWS -0.092 ns. There remain 1,154 unconstrained internal
endpoints and two inputs without delay budgets, inherited from the native
integration. The endpoints are HBM XSDB logic with constant clocks; the inputs
are `configuration_rst_n` and `sys_rst_n`. These counts are retained in the
reports, not treated as coverage.

One new **critical CDC-10** is open at
`weight_loader/mem_ready_sync_reg[0]/D`. Its combinational input combines
`availability/ready_mgmt_reg[0:1]` with `availability/fault_reg`. Bounding the
three source paths does not remove the combinational-before-synchronizer
hazard or qualify this crossing. The next RTL candidate should register the
combined availability in the management domain, then retest startup, loss,
clock halt and cold-reset behavior, accounting for the added management-cycle
latency. It must receive new integration/CDC evidence before replacing this
frozen baseline.

Two initial audit attempts remain archived as failures: `054429Z` compared a
one-element Tcl list with an unwrapped pin name; `054755Z` expected Q pins where
`all_fanin -startpoints_only` reports register C timing startpoints. These were
auditor corrections; neither attempt changed the RTL or synthesis checkpoint.

The selected apply run `weight-physical-inspect-20261013T055100Z` completed in
203.20 seconds. Reopening it in `weight-physical-inspect-20261013T055532Z` took
158.48 seconds and reproduced the connection, clock, constraint and timing
evidence. The independent reviewer passed twelve damaged-evidence controls.
It confirmed 704 HBM write connection bits, 224 AXI-Lite connection bits,
304 loader source-to-destination paths covering 299 distinct endpoints,
twelve ASYNC_REG registers and all eight setup/hold stage pairs. The earlier
1,503 bounded endpoints and 390 snapshot endpoints remain covered.

The loader's minimum estimated constrained-path slack is +1.925 ns. Overall
timing still fails: WNS -0.194 ns, WHS -0.842 ns, WPWS -0.092 ns. Reports retain
one critical CDC-10, nine CDC-6 warnings and 2,425 CDC-15 warnings. The existing
AVAL-326, NSTD-1 and UCIO-1 physical DRC findings remain open. No placement or
routing was run.

The selected bounded checkpoint SHA-256 is
`7064c9176706738149bae36d00062b56735e37b50f0366a06728ff16515065aa`.
The closed-process audit checked all five attempts, ten owned process groups
and 446 remote evidence files: no remaining processes or zombies. The 236-file
generation donor and earlier native bounded checkpoint retain their hashes.
`FINAL_BACKUP.json` records the final ZIP, full restoration and repeated local
review. Restoring that archive does not claim another Vivado run.
