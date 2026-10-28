# Queued generated-attention integration R1

The local coordinator started on 2026-10-13 at 17:14:10 UTC, PID 173272.
It waits for both `llama32-attention-memory-vivado-r1-coordinator.json`
and `llama32-attention-hls-r5-coordinator.json`. Existing owners, sources
and dependency order are unchanged. No integration tool has started yet.

## Frozen package and gates

- Package: `llama32-attention-engine-vivado-r1-package.json`, SHA-256
  `064ad71fd858139a26804d30e8ad79aecbdc6e6cc149a59ee6b6674769e82b25`.
- Source ZIP, 81 members, verified D:/NFS:
  `69789acb3c7ce63799386780662a8671e29d2a2d3c71f085b8bb9cd8b0ce4bfe`.
- Driver: `queue_llama32_attention_engine_vivado_r1.py`, SHA-256
  `7abe9b4ad4376d87e883ba43c5629391f9cb4005385a5d5588267241920eca49`.
- Queue preflight:
  `a24883c74c27915621cee9425975953643e2bfb7197bff6bf77b8de8144e063c`.

The package contains the shell, bench, frozen children, numeric/physical
reviewer, fixture bytes, actual generated RTL/IP scripts and adapter.
Before launch, both parents must pass independent reviews with verified
terminal archives. Every generated file is compared against the actual
HLS archive. A diagnostic parent stops launch. The shared tool slot must
also be empty, and all worktree/source hashes must match the package.

## Execution and acceptance

The runner uses two CPU cores and a 5,400-second total timeout. XSIM has
automatic waveform capture disabled. It runs all 143 numerical cases and
12 additional lifecycle queries, then performs OOC synthesis at a 5 ns
clock. Assertions and the independent reviewer check physical AXI data,
query/head identity, K/V consumption, continuation state, cancellation,
lease drainage, reset, late revocation and published FP32 outputs.

The expected bank has 155 queries, 152 completions and 9,600 published
words. Synthesis must resolve all black boxes and inferred latches;
resource reports, clock checks and a checkpoint are preserved. Reported
setup/hold estimates do not establish routed timing closure.

Terminal evidence must pass the independent numeric/physical checker,
its 27 corruptions, compiled-source verification and archive verification
before the coordinator releases the slot. Failures are also archived.
An SSH observation failure does not restart a run.

The [preflight document](LLAMA32_ATTENTION_ENGINE_R1_PREFLIGHT.md)
distinguishes compile checks, native numeric checks and synthetic reviewer
self-tests from actual generated-core integration, which remains pending.
Use `audit_hls_binary_tv_active_r2.py` to include this new owner when
auditing live processes. Do not rerun a launcher while its owner is alive.
