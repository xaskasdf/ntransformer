# Registered HBM availability for the native loader

This isolated VU47P/HBM candidate addresses the critical CDC-10 recorded in
[HBM_WEIGHT_PHYSICAL.md](HBM_WEIGHT_PHYSICAL.md). The previous synthesized and
simulated artifacts remain frozen. The DDR512GiB design and main Vivado queue
are outside this change.

## RTL contract

`ntrpu_hbm_loader_readiness.sv` registers `(&ready_mgmt) && !fault_mgmt` on the
100 MHz management clock. Its output directly drives the loader's first
memory-domain synchronizer stage. The source register is an ordinary register;
the existing twelve loader synchronizer registers retain `ASYNC_REG`.

This adds **one management cycle, 10 ns**, after the existing management
availability logic, plus the unchanged memory and PCIe synchronization latency.
It is not an asynchronous availability kill. A stopped management clock holds
the sampled state; a stopped memory clock delays its downstream observation.
Canonical cold reset clears the source asynchronously, including with stopped
clocks. Transport reset does not clear retained loader ownership or this source.
Availability loss remains sticky until cold reset; restoring lock does not
rearm a failed load or permit sealing failed weights.

`prepare_hbm_loader_readiness.py` accepts the frozen physical composition,
checks its hash, adds the source instance, and replaces only the loader's
availability expression. It preserves the loader and all 22 frozen RTL files.
The new composition no longer claims identical functional attachment to the
old native simulation: this stage changes readiness latency.

## Functional evidence

The short XSIM bank uses the actual availability, source register, loader and
control RTL with an HBM response fixture. Seeds 0, 1 and 3 cover fourteen named
groups each: prior loader protocol/error cases, staggered stack startup,
management-clock halt, memory-clock halt with late B completion, failed relock,
cold reset with both clocks stopped, and successful recovery/sealing.

The independent review checks 6,175 assertions, 4,552 source samples, exact
addresses/payloads and 11 AW / 11 W / 9 B handshakes per seed. The unequal
response count includes intentionally abandoned cold-reset transactions.
Twenty-four deliberately damaged-log controls must be rejected.

An actual negative RTL run replaces only the source's clocked `always` block
with `always @*`. It compiles and elaborates, then fails the registered-latency
assertion at 75.001 ns before any PASS. Its `FAILED` archive is expected evidence,
not a passing design run. The source binding file in that negative copy retains
the original candidate identity; its recorded source hash exposes the mutation.

## Native audit contract

`audit_hbm_loader_readiness.tcl` checks the synthesized FDCE, its always-enabled
CE, canonical cold-reset inverter, management clock and sole direct fanout to
`weight_loader/mem_ready_sync_reg[0]/D`. It retains ordinary setup/hold timing
from all three management inputs to the new register and rejects combinational
logic after that register.

The loader retains its payload/completion bounds and eight synchronizer stage
pairs. The three former readiness source paths become one registered crossing:
302 source-to-destination paths, 299 distinct endpoints and ten loader maximum
delay groups. No broad clock exclusions or CDC waivers are added. Earlier
1,503 bounded endpoints and 390 snapshot endpoints must remain covered.

The audit applies constraints in one fresh Vivado process and reopens the saved
checkpoint in another. `review_hbm_loader_readiness.py` checks both closed
inventories, checkpoint hashes, connectivity, constraint persistence and the
registered crossing's CDC classification. Physical findings remain visible.

## Execution and recovery

Artifacts are under `D:/AI/artifacts/ntrpu-axi-bridge-hbm-loader-readiness`.
Each VM job uses a fresh private directory, two CPUs, nice 10 and a bounded
tool timeout. Closed archives are restored locally and every file hash checked.
Run the independent reviewers on restored evidence:

```text
python -B review_hbm_readiness_preflight.py PREFLIGHT_ROOT BYPASS_ROOT
python -B review_hbm_loader_readiness.py SYNTH_ROOT APPLY_ROOT REOPEN_ROOT
```

This stage does **not** rerun the complete native PCIe/HBM traffic bank. Its
functional evidence is the short RTL bank; native synthesis and checkpoint
audits provide integration/CDC evidence. The previous 43,468-check native bank
remains evidence for the previous composition. Neither bank qualifies a physical
PCIe link, board power/pinout, routed timing or model inference.

## Recorded findings, 2026-10-13 UTC

Preflight `loader-readiness-preflight-20261013T071302Z` passed all three seeds.
The deliberate RTL-bypass run `loader-readiness-preflight-20261013T072104Z`
failed at the intended assertion. Both archives were restored and independently
reviewed before the native audit.

Native synthesis `loader-readiness-synth-20261013T071849Z` completed in
1,019.48 seconds with unchanged inputs, no black boxes and no latches. Integrated
utilization is 61,094 LUTs and 65,708 registers: one LUT and one register more
than the frozen baseline, with the same 60 RAMB36 and one RAMB18. The loader
itself remains at 636 LUTs and 1,335 registers.

The raw synthesized report contains no critical CDC findings. The former
CDC-10 is now CDC-3, from `loader_readiness/ready_reg/C` to
`weight_loader/mem_ready_sync_reg[0]/D`, with three synchronization stages.
The raw report retains nine CDC-6 and 2,425 CDC-15 warnings. Its timing estimate
still fails: WNS -0.413 ns, WHS -0.842 ns and WPWS -0.092 ns.

Constraint application `loader-readiness-inspect-20261013T073814Z` completed
in 255.99 seconds. The new source is an ordinary FDCE, always enabled, with
canonical `cold_m_reg[2]/Q` cold reset and exactly one leaf consumer: the first
memory readiness synchronizer. All three source input paths retain 10 ns setup
and 0 ns hold requirements. Their worst **unplaced estimates** are +9.523 ns
setup and **-0.066 ns hold**. Hold closure remains an implementation task; these
requirements were not cut or relaxed to remove the finding.

Fresh reopen `loader-readiness-inspect-20261013T074400Z` completed in 212.75
seconds and reproduced the audited tables and constraint semantics. The
independent reviewer rejected fifteen damaged-evidence controls. It verified
704 HBM write bits, 224 AXI-Lite connection bits, 302 loader paths covering
299 endpoints, twelve synchronizer registers and eight setup/hold stage pairs.
The earlier 1,503 bounded and 390 snapshot endpoints remain covered. The loader's
minimum estimated constrained-path slack is +1.963 ns.

The applied and reopened reports both contain zero critical CDC findings,
200 CDC-3 informational entries, nine CDC-6 warnings, 36 CDC-9 informational
entries and 2,425 CDC-15 warnings. Overall estimated timing still fails:
WNS -0.194 ns, WHS -0.842 ns and WPWS -0.092 ns. The 1,154 unconstrained HBM
XSDB endpoints, two inputs without delay budgets, and AVAL-326/NSTD-1/UCIO-1
physical DRC findings remain open. The prior `LOCKED` path-segmentation warning
remains explicitly audited; no additional timing exception or waiver hides it.

Bounded checkpoint SHA-256:
`73570c8e393c01ead212b5a11c283d346ea54213749b1941265bcdb82b7a551d`.
The closed-process audit checked five jobs, sixteen owned process groups and
414 remote evidence files: no live processes or zombies. The 236-file generation
donor and both earlier native checkpoints retain their hashes. `FINAL_BACKUP.json`
records the final ZIP, verified restoration and repetition of both independent
reviewers. Restoring the ZIP does not constitute another Vivado execution.
