# Attention engine R1: integration preflight

Status on 2026-10-13: the integration shell and bench compile; the actual
generated HLS ports have been inspected and bound. The complete engine
has **not** been simulated or synthesized. Arithmetic qualification of
the live HLS parent and physical/numerical review of the integrated trace
remain required before claiming an attention engine result.

## Composition and intended bank

`fpga/rtl/ntrpu_attention_engine_r1.sv` connects the frozen memory R1
composition to `ntrpu_attention_hls_ports_r1`. Its 102 public ports expose
cache binding, full-token appends, AXI512, query admission, cancellation,
completion and gated result reads. Q, K/V, ping-pong continuation state,
output RAM and the HLS handshake remain internal. Existing memory and
context sources are unchanged.

`fpga/tb/tb_ntrpu_attention_engine_r1.sv` uses the
[AXI fixtures](LLAMA32_ATTENTION_AXI_FIXTURE_R1.md): 143 numerical cases,
including 128 real layer-0 projection cases. Appends populate physical
AXI memory before queries. Twelve additional queries exercise cancellation,
epoch changes, recovery and reset during AXI reading, active HLS work and
held completion. Planned coverage is 155 queries, 152 completions,
9,600 published output words, 756 appends and four resets.

The reset test now waits for an observed arithmetic start and logs reset
when it is actually asserted. Late epoch revocation logs both immediate
result gating and persistent gating after restoring the original epoch.

## Completed checks

The latest compile-only attempt is
`20261013T165401Z-V7-attention-engine-interface-compile-r1`.
Icarus compiled all child RTL, the shell and bench using a declaration-only
HLS interface. The resulting artifact was never simulated. Warnings cover
missing module time units and Icarus's broad `always_comb` sensitivity;
there were no compilation errors.

- Source archive: `bbbfde87991950d9c46fa974c92db3652ce660a052f3e250034d947ad31e1038`.
- Terminal archive: `4d7fd96f12c6801aefbed4f1bcc0648a52ef24c296207fe56399c044c54c5f2b`.

Both archives are verified on D: and NFS; the earlier compile attempt
remains preserved separately.

The live HLS parent generated a real 35-port top with SHA-256
`bcfe59a688ad701aa5f18642e9f076affc99d6f40877fd7b54019a9ae38efc48`.
`tools/bind_ntrpu_attention_hls_ports_r1.py` accepted its exact widths,
directions, `sequence_r` alias and synchronous active-high reset. Generated
`ap_done` and `ap_ready` both assert in state 133. The zero-latency adapter
has SHA-256 `2cc690b77e2711ededf90f16cc162e52b9bebebd7d0b5b15694f76f2ead99707`.
Port inspection does not approve the still-running C/RTL bank.

## Independent reviewer and next dependencies

`tools/check_ntrpu_attention_engine_r1.py` reconstructs AXI writes/reads,
checks causal K/V consumption, continuation RAM, lease drainage, revocation
and published results, and compares successful states against Decimal90.
Its numeric adapter passed 147 archived native tiles, 15,288 state words
and 9,152 outputs at 0 ULP, rejecting 11 corruptions. Ownership metadata
was explicitly rebased for this validator preflight; those records are
not integrated RTL evidence. A separate synthetic transcript exercised the
full physical-trace reviewer and rejected all 27 corruption controls. Its
cycles and events are explicitly self-test inputs, not simulation results.
Eight damaged parent-review inputs were also rejected by the launch gate.

The prepared Vivado runner uses two CPU cores, no automatic wave recording,
and a 5,400-second total limit for the 155-query bank and OOC synthesis.
All 33 generated files are frozen in bundle
`fac22639a23ad2ba27815a677e2e6daf00bd0d701319a46aaae45f4c0e1fe697`.

The queue waits for reviewed/backed attention HLS and memory Vivado parents.
Before acquiring the shared tool slot, it compares every generated file
with the terminal HLS archive and verifies the source, binding and parent
review hashes. A failed parent stops launch. Simulation output, compiled
inputs, numeric states, corruption controls and synthesis reports require
independent review; each terminal result is archived before releasing the
slot. Routed timing and full inference remain separate milestones.
