# Attention memory R1: KV path and private context

`ntrpu_attention_memory_r1` connects the frozen KV path R1 to context R2.
It exposes cache bind/append, 512-bit AXI, captured Q commands, private output
reads and the external `ap_ctrl_hs`/RAM arithmetic ports. A query owns its KV
lease until the arithmetic peer and physical reads drain. Only a successful,
complete query with a live epoch authorizes publication.

## Validation scope

The native bench writes physical memory through AXI before reading each causal
K/V tile. Its deterministic checksum peer consumes every Q/K/V word, reads
continuation state, writes 104 state words and produces 64 final words. It is
**not the generated attention core** and does not measure attention throughput.

Run `20261013T154112Z-V7-attention-memory-native-r1` passed seeds 1/7/42 with
watchdog 32 and seed 1 with production watchdog 65,536. Each suite has:

- 61 queries, 139 appends, four coordinated resets and 57 arithmetic jobs.
- 80,384 K/V, 3,648 Q, 624 continuation and 2,560 result words compared.
- 565,248 published write bytes and 322,432 physical read bytes.
- All 32 query heads, layers 0/15, one/two/three tiles, concurrent append/read,
  4 KiB splits, failed writes/retry, read error, timeout, cancellation, epoch
  revocation, invalid metadata, partial state/output and arithmetic errors.

The independent checker reconstructs memory from AW/W/B and AR/R, verifies
GQA mapping, checks checksum/state evolution, and forbids publication before
lease release. Reset cases interrupt AR, held arithmetic start and held result.
Cycle totals are 114,229 / 114,264 / 114,293 / 179,764; these include the
checksum peer, backpressure and host reads, not physical DDR latency.

## Reproduction and evidence

Use `tools/run_ntrpu_attention_memory_native_r1.py` on Linux with Icarus,
a frozen `SNAPSHOT_FILES.json`, a new output directory and its source SHA.
Review a log with:

```bash
python tools/check_ntrpu_attention_memory_r1.py seed-1.log --seed 1 --watchdog 32
```

D:/NFS archive:
`c6211c5eb33ec995a26a48cd8fdf7e5339c7b1ce2b94c6355905dea832fdf638`.
Seven source mutations fail simulation after successful compilation: plane
swap, wrong head/key/result address, early finish, ignored cancellation and
the [R1 cancellation feedback defect](LLAMA32_ATTENTION_CONTEXT_R2.md).
The independent reviewer additionally rejects 25 trace corruptions per suite.
Its archived launcher differs only in later CRLF handling for mutant copies;
canonical RTL and simulation runner hashes match exactly.

## Remaining integration

Require XSIM and OOC synthesis after context R2 and the KV path dependency.
Then bind the generated attention HLS ports and validate actual arithmetic
against real-model references. Out-of-bounds private reads currently surface
as generic cancellation status 7 at the context boundary; a finer diagnostic
is future work. Routed timing, physical DDR, PCIe and full inference are open.
