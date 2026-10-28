# DS41 prefill: evaluate only the logits that are consumed

2026-10-15. Completed on the full DeepSeek-V4.1-Flash Q8_0/MXFP4 model,
H100 80 GB SXM, CUDA 13.1.115 / GCC 14.4.0. Physical GPU1 selected through
`CUDA_VISIBLE_DEVICES=1`. This measures prefill, not Brandon training or
autoregressive decode throughput.

## Change

`Session::forward(token, compute_logits=true)` now accepts an explicit request
to omit the terminal hyper-connection collapse, output norm and LM head.
Every layer, raw/compressed KV state, index state, Engram history and position
still advance. An omitted result returns `nullptr`; `logits()` also returns
`nullptr` so a caller cannot accidentally consume a previous position's logits.
The ordinary one-argument API continues to compute logits at every position.

The CLI defaults to logits at the final prompt position only. Set
`DS41_PREFILL_ALL_LOGITS=1` for the old schedule. Stage-sum diagnostics
(`DS41_DUMP_STAGES`) retain all prompt logits, even with the control variable
set to zero. `DS41_PROFILE=1` reports prompt token count, head evaluation count,
mode and elapsed prefill time. Single-token and empty prompts still evaluate
one head; the latter first obtains the existing BOS fallback.

The GPU upload/input-buffer correction from the preceding audit is shared by
both comparison arms. This experiment isolates the terminal-head change.
Model weights, kernels, precision, token IDs and cache budget are identical.

## Measured performance

One model and cache per process. State resets precede timing. Both schedules
are warmed before measurement, then each pair alternates order (all/last,
last/all, all/last, last/all). Each measured prefill has 16 identical token IDs.
Eight measurements; four comparisons:

| Pair | All-position heads, s | Last-position head, s | Throughput change |
| --- | ---: | ---: | ---: |
| 1 | 4.369038 | 4.348289 | +0.477% |
| 2 | 4.368013 | 4.350923 | +0.393% |
| 3 | 4.355520 | 4.341575 | +0.321% |
| 4 | 4.360416 | 4.338335 | +0.509% |
| Mean latency | **4.363247** | **4.344781** | **+0.425%** |

Throughput calculated from mean latency: **3.6670 → 3.6826 prompt tokens/s**.
Mean saving: **18.466 ms per 16-token prompt**. All four pairs favor omission;
four pairs are not a general statistical guarantee. Startup, loading and
decode are excluded from these paired timings.

Head evaluations fall **16 → 1 (93.75% fewer)**. This avoids a logical
**19,857,408,000 bytes (18.494 GiB)** of head-weight reads and
**7,756,800 bytes (7.397 MiB)** of logits D2H payload per prompt. These counts
come from tensor dimensions and executed-head counts, not hardware traffic
counters. They are not savings in allocated VRAM.

Both modes retain the same LM-head weights, output buffers and configured
62 GiB expert cache. The cache reports 10,622 slots of the same size; no new
peak-VRAM measurement was taken. The end-of-process expert counters include
validation and warmup: 159,840 slice lookups, 153,447 hits, zero evictions.
They must not be presented as a per-window cache hit rate.

## Numerical and behavioral checks

All checks passed with explicit nonzero failure exits; the numerical integration
tests use `memcmp` for every requested logit vector, plus finite checks.

* Six-layer F32 fixture: 18 prompt tokens, raw window 16, compression ratios
  0/2/1 and sparse top-k 4. Final-only and mixed-position schedules reproduce
  reference logits bit for bit. Both reproduce four greedy continuation steps
  and a reset/single-token replay. This crosses the small fixture's window,
  compression and sparse-selection boundaries.
* Complete 40-layer model: the same checks for a 16-token prompt and four-token
  continuation. Both omission schedules pass; all eight measured prompt
  outputs also match the reference bit for bit. This does not cover the full
  model's 128-token window rollover or long-context sparse top-k boundary.
* CLI integration: the six-token prompt `The capital of France is` produces
  identical dumped logits and the same four-token text, `Paris. It is`, with
  6 versus 1 head evaluations. The two cold-process prefill timings were
  3.966521 and 3.949960 s; they are preserved but not used for the headline gain.
* CLI edges on the fixture: one token and empty prompt use one head; stage
  diagnostics use all three heads in a three-token prompt. Missing fixture
  returns the explicit skip code 77. The configured fixture CTest did not skip.
* Previous GPU primitive regressions: all four CTest cases pass again.

Separately, the previous attention-upload correction was exercised on the real
model at token ID 0. CPU-attention versus GPU-attention logits differ by at
most **3.33786011e-6**, relative L2 **2.60817e-7**, and have identical top eight.
The analysis checks `max_abs < 1e-3`, relative L2 `< 1e-4` and matching top one;
these are diagnostic gates for this single input, not a long-context quality
criterion. This comparison is not bitwise equal; the head-omission comparison
is bitwise equal. No resident-upload failure was logged in the new GPU runs.

## Workload coordination

The runner acquired the existing pipeline, exclusive-training and correctness
locks, observed completed Brandon stages and no GPU processes twice, then ran
the bounded suite. The last GPU case completed at **08:08:35 UTC**. The next
Brandon threshold stage entered `running` at **08:08:44 UTC**; its observed
training worker start times were **08:08:50 UTC**. These observations and the
admission snapshot are preserved. Later CLI edge tests were CPU-only.

The normal remote source/build used by other work was not overwritten. The
validated executable is in the isolated experiment's `build/ntransformer`.

## Reproduction and artifacts

Build `test_ds41_prefill` and `ntransformer` normally with CUDA enabled. Fixtures
and model assets are external:

```sh
# Tiny CPU regression; CTest exits 77 if DS41_TEST_MODEL is absent.
DS41_GPU=0 DS41_TEST_MODEL=/path/to/tiny-deepseek41.gguf \
  ctest --test-dir build -V -R '^test_ds41_prefill$'

# Full-model checks followed by four warm alternating benchmark pairs.
CUDA_VISIBLE_DEVICES=1 DS41_VRAM_GB=62 \
  ./build/test_ds41_prefill /path/to/model-00001-of-00010.gguf 16 4

# CLI control versus optimized schedule; keep every other argument identical.
DS41_PROFILE=1 DS41_PREFILL_ALL_LOGITS=1 ./build/ntransformer -m /path/to/model.gguf -p 'The capital of France is' -n 4 --temperature 0
DS41_PROFILE=1 DS41_PREFILL_ALL_LOGITS=0 ./build/ntransformer -m /path/to/model.gguf -p 'The capital of France is' -n 4 --temperature 0
```

* Local: this directory, with `analysis.json`, logs, raw output logits,
  commands, fixture generator, `source.tar` and SHA-256 manifest.
* Remote: `/home/ubuntu/ntransformer-h100/ds41-prefill-20261015`.
* Persistent: `/lambda/nfs/btiny/ntransformer-training/deepseek-prefill-20261015`.
* Fixture SHA-256:
  `83bae5061d15e7f77b930e501fbe76932b30217ff390addec72dcd05ecc9e19a`.
  The preserved generator depends on the reference fork's `gguf-py`; its
  original introductory comment is stale, but its actual ratios are 0/0/2/2/1/1.
* Reference fork/model header inventory: the preceding audit's `layout.json`
  and `RESULTS.md`. This experiment does not add hashes of the 473 GiB payload.

The small wall-time gain, despite removing almost all prompt LM heads, directs
the next optimization toward the remaining layer work. A warm phase trace is
needed before choosing between grouped expert dispatch and moving attention,
shared-expert and hyper-connection state into longer GPU execution segments.
