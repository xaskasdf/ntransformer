# Full-board HBM response capture experiment

This isolated VU47P/HBM experiment integrates the candidate validated in
[HBM_RESPONSE_CAPTURE.md](HBM_RESPONSE_CAPTURE.md). Evidence belongs under
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-response-physical/`. It does not modify the
main-thread jobs, separate DDR design, selected checkpoint or earlier bundles.

## Frozen inputs and intended comparison

The full RTL donor is the 60-file reset-distribution composition archived in
`ntrpu-axi-bridge-hbm-reset-integration/rtl-candidate`. The only original RTL
file changed is `ntrpu_hbm_read_adapter.sv`: its `response_cut` instance uses
the tested free-slot capture module, whose exact definition is appended.
The validated queue, return queue and tracker ownership logic are unchanged.
Both active reader channels instantiate this same adapter, so the native probe
must cover 1,060 payload clock enables, not only the 530 checked in the unit test.

The distributed-reset routed donor is
`a990f1837d4dcaef6a271b6b625bc7235d2b0187ec517bde658c913207de1fb5`:
WNS -0.509 ns and TNS -711.005 ns in its archived reports. The original selected
`3038b257...` reference remains at reopened WNS -0.413 ns. Neither is overwritten.
This is a fresh full RTL synthesis and subsequent implementation, not a manual
LUT edit. Consequently any measured timing difference includes implementation
changes induced by resynthesis; it is not attributed solely to one LUT.

## Stages and resource limits

Each stage runs in a new private VM directory with two CPUs, nice 10 and an
1,800-second tool limit. Synthesis copies the previously generated IP project
and verifies that its IP files resolve inside that copy. It uses the same
`xcvu47p-fsvh2892-2-e` part, Vivado 2026.1 and frozen elaboration flow.

The implementation applies the existing scoped constraints to the new synthesis
checkpoint, assigns the same experimental pins, optimizes, and saves a checkpoint.
Placement/routing may reuse the exact distributed-reset donor with
`read_checkpoint -incremental`, `place_design -no_psip` and `route_design -no_psir`.
These are UltraScale+ commands already exercised in the preceding integration.
No Versal-only ECO switch, new waiver, clock-group cut or relaxed period is used.

The runner waits for EOF from the tool and inherited stdout writers before
sealing its logs. Progress remains visible in `vivado.log`. This addresses the
late helper exit message observed in the previous unit batch. Closed native
archives are copied locally, restored and verified against their file hashes;
failed attempts are retained alongside successful stages.

## Synthesis and reset representation

The fresh full-board synthesis completed in 1,139.860 seconds. Independent review
confirmed all 1,060 response enables exclude RVALID while ownership retains it.
Its checkpoint SHA-256 is
`ea5ae92ef50959e19f227e124826f7cd77e2789fbe2aa0098ce3b5dc136dd919`.
The 92,058,906-byte native archive was restored locally and all 289 files verified.

The first constraint attempt stopped before optimization because the inherited
post-route readiness auditor required an explicit cell inversion property.
A separate read-only native probe showed the fresh synthesis has an unset
`IS_CLR_INVERTED`, actual CLR pin `IS_INVERTED=0`, and the explicit
`loader_readiness/ready_i_2` LUT1 (`INIT=2'h1`) driven by the canonical management
release. Its 108.663-second probe and the 126.789-second failed attempt are kept.

The implementation wrapper recognizes only this proven unset-property case.
It records the raw empty property and the pin-based effective value in separate
columns, verifies the inverter function/source, and delegates all other cases
to the unchanged inherited auditor. It does not set properties, alter the
netlist, or relax reset timing. The reviewer independently checks this encoding.

The corrected constraint/optimization stage completed in 488.904 seconds and
was restored from its 46,958,199-byte native archive (105 verified files).
Its optimized checkpoint is
`0e785870ad3f84612ec4e364dc5efbdcda8f6367ddfc6941f7e5e81342215476`.
Resynthesis also regenerated timestamps on identical internal IP waiver commands.
The comparison excludes only their trailing timestamp metadata; complete waiver
commands and multiplicities, clocks and timing exceptions must still match.
No waiver or constraint is removed from the checkpoint or native evidence.

## Acceptance checks

The full-board probe requires 530 FDRE payload registers in each response queue
and the existing 450 MHz memory clock. Their enables may depend only on queue
occupancy, write-slot state and the distributed reset. HBM RVALID must reach
ownership logic while reaching none of these payload enables. Both slots and
both channels are enumerated explicitly.

The reset probe checks the canonical three-stage synchronizer followed by the
ordinary distribution register, its constant data/enable, reset polarity,
single canonical-release consumer and readiness feedback. It accepts a proven
explicit inverter or primitive CLR inversion. Recovery/removal stays timed.

The independent reviewer compares all 2,195 bounded paths, retained board
synchronous checks, 18 bus-skew groups, clock topology, reference ownership,
PCIe calibration selector membership, CDC, DRC and experimental pin assignments.
Timing clocks and exceptions must match the frozen reset donor semantically.
Global setup/hold, all 1,060 payload enables, HBM RREADY and reset release are
reported separately. A routed candidate is reopened to verify the saved result.

## Measured physical result

The placement/routing/audit batch completed in 1,333.001 seconds and its
85,590,076-byte native archive was restored with all 94 files verified.
The routed checkpoint SHA-256 is
`0b38a6ab714dca42c88f3f98b69bc1ff00d57856236daed17d4ba93a5f65afc8`.
All 100,261 routable nets are routed, with zero routing errors. Final incremental
reuse is 99.99% of cells and 98.60% of nets.

| Checkpoint | WNS (ns) | TNS (ns) | Failing setup endpoints |
| --- | ---: | ---: | ---: |
| Original selected reference, reopened | -0.413 | -803.549 | 4,555 |
| Distributed-reset donor | -0.509 | -711.005 | 3,992 |
| Capture candidate, native routed report | -0.579 | -572.291 | 3,435 |

Reopening the exact routed checkpoint took 394.462 seconds. Its complete global
setup/hold/pulse summary, queue margins, RREADY margins and reset margins match
the native routed reports exactly. The 405,124-byte reopen archive contains
88 verified files. Per-slot worst setup margins are +0.154/-0.030 ns for channel
0 and -0.144/-0.157 ns for channel 1; failing endpoint counts are 0/8 and 4/3.

The aggregate setup deficit decreases, but the worst setup margin deteriorates.
The candidate is therefore evidence for the next experiment, not a replacement
for the selected reference. Hold and pulse-width failures are zero (WHS +0.005 ns).
All 2,195 bounded paths, 84 retained synchronous checks, and 18 bus-skew groups
pass; the minimum bus-skew slack remains +0.044 ns. CDC has no critical finding.
Existing OOC coverage gaps remain: 1,154 unconstrained internal endpoints and two
ports without input delay. Existing DRC/methodology warnings remain recorded.

The 1,060 payload enables still exclude RVALID after physical implementation;
ownership still depends on it. Their worst setup slack is -0.157 ns, with 15
failing endpoints; their worst hold slack is +0.155 ns. The global worst path
is `memory_distribution_reg/C` to stack 0 HBM `RREADY`, through three logic
levels, with a 2.708 ns datapath and 2.222 ns requirement. Canonical release to
the distribution register remains timed, with recovery/removal slack
+1.489/+0.231 ns and exactly one canonical-release consumer.

Fourteen negative evidence controls reject missing/duplicate registers, RVALID
dependencies, clock changes, lost ownership, incomplete timing, reset polarity
or readiness changes, relaxed bounds, calibration changes, added clock cuts,
and changed/duplicated internal IP waivers.

The next bounded experiment should target the distribution-to-RREADY path.
A registered ready/skid-buffer design needs explicit capacity accounting and
tests for backpressure, reset, fault quarantine and in-flight responses before
another full-board run. That RTL change is not part of this candidate.

## Evidence and closure

The native batch names share the prefix `board-capture-physical-`:

| Batch suffix | Purpose | Outcome |
| --- | --- | --- |
| `synth-20261013T212305Z` | Fresh full RTL synthesis | Reviewed |
| `inspect-20261013T214416Z` | Initial constraint audit | Failed; preserved |
| `inspect-20261013T214825Z` | Read-only reset encoding probe | Reviewed |
| `inspect-20261013T215143Z` | Corrected constraints and optimization | Reviewed |
| `inspect-20261013T220223Z` | Incremental placement/routing | Reviewed; setup fails |
| `inspect-20261013T222621Z` | Reopen saved routed checkpoint | Same measured margins |

`COMPARISON.json` holds the measured comparison. `finalize.py` verifies all native
archives and staged sources, checks only owned process IDs/groups for closure,
replays five source preparations, and seals a new ZIP. It then restores that ZIP,
verifies every file, and reruns all six batch reviews and fourteen negative
evidence controls. `FINAL_BACKUP.json` records the resulting archive hash and
restoration receipt; `FINAL_PROCESS_AUDIT.json` records process/source closure.
The native IP generation donor and selected original checkpoint remain external
predecessor evidence; the exact incremental routed donor is included locally.

## Reproduction

`prepare_hbm_capture_physical.py DONOR_RTL TESTED_UNIT FRESH_OUTPUT` prepares
full synthesis. The archived launcher uses its `synth` mode and the recorded
IP-generation donor. `prepare_hbm_capture_stage.py AUDITS RUNNER CHECKPOINT
FRESH_OUTPUT --mode constrain|route|reopen` freezes subsequent stages; route
also requires both local and remote paths of the exact incremental donor.
Use `collect.py --collect` for closed batches, then run
`review_hbm_capture_physical.py RESTORED REFERENCE INPUT_DCP --mode MODE`.

## Limits

Routing and native tool reports do not validate traffic on physical hardware,
PCB manufacturability, released pinout, thermal behavior or power consumption.
Capturing an unowned slot can increase internal switching. Existing timing
coverage gaps remain visible. The candidate is not promoted merely because
the RVALID-to-payload enable dependency disappears.
