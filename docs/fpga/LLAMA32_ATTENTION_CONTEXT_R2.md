# Attention context R2: registered cancellation

R2 supersedes R1 for composition with `ntrpu_attention_kv_path_r1`.
The private Q, two state banks, final output, epoch authorization and drain
contracts remain as described in [R1](LLAMA32_ATTENTION_CONTEXT_R1.md).

## Integration defect and correction

The first composed trial, `20261013T152712Z-V7-attention-memory-native-r1`,
failed its first valid query. R1 drove `query_cancel_valid` from combinational
`fault_now`. Updating a write-coverage bitmap while its port remained asserted
could briefly signal a duplicate write. Feedback through KV cancellation and
`compute_abort` sustained that cancellation after the write port changed.

R2 drives cancellation from registered `fault`. This breaks that combinational
feedback and delays cancellation propagation by one clock. Immediate epoch
guards on private RAM/results remain in effect. The original source and failed
archive (`69c5e392ff2c06d487ad95421031e3c0896d1408b090ed9175f68c56c7f92e74`)
are preserved.

## Evidence

Standalone native RTL run `20261013T153924Z-V7-attention-context-native-r2`
passed three seeds: 207 cases, 12 resets and 26,664 compared Q/state/result
words. Each suite rejected 15 evidence corruptions. Its D:/NFS archive is
`247f8b3643f921eef1a8b39ff5a9c15a59f30a6df3ec1845e77fd88a448511c4`;
independent review is
`4953af344094872255b5c8a1ecf343d7bd290fa06a500c8c0de3e1d48bae3915`.

The composed KV/context baseline
`20261013T154112Z-V7-attention-memory-native-r1` passed four suites. Seven real
RTL mutations were detected, including reverting this registered cancellation
to R1. Mutation review:
`ec47748cabbe8a44d1a96762b86d849cc38da34e1468da37499ae25322887b68`.
See [composition scope](LLAMA32_ATTENTION_MEMORY_R1.md).

## Qualification boundary

Queue R2 after the reviewed and backed-up KV path R3. Run three XSIM suites
and OOC synthesis for `xcvu5p-flva2104-2-i`, with two CPU cores, a 1,200-second
limit and a 5 ns clock. Require source hashes, independent trace checks,
zero unresolved black boxes/latches, timing diagnostics and verified backup.
Synthesis estimates do not establish routed timing. The arithmetic peer is
external; generated attention HLS, physical DDR and full model inference
remain separate work.
