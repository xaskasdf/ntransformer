# DeepSeek-V4.1-Flash: end-to-end NTransformer benchmark

Completed on 2026-10-15. Actual tokenized prose, batch size 1, one physical H100 (GPU 1); the other H100 remained unused. This measures the current DS41 engine through `Session`, linked to the exact library from the validated shared-expert build. DS41 dispatch returns before the generic CLI benchmark handler, so `--benchmark` currently runs ordinary DS41 generation instead of that benchmark.

## Results

Each context runs in a fresh process: one initial pass, then three identical requests reusing the weight cache. Attention/history state is reset for every request. Reused-cache rates below are total tokens divided by total time across the three repetitions.

| Prompt tokens | Initial prefill tok/s | Reused prefill tok/s | Initial decode tok/s | Reused decode tok/s | Reused TTFT (s) | Decode p95 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2.455 | 6.067 | 3.520 | 6.229 | 5.275 | 172.96 |
| 128 | 4.157 | 6.162 | 4.411 | 6.178 | 20.771 | 187.83 |
| 512 | 4.749 | 6.052 | 4.132 | 5.364 | 84.605 | 211.14 |

Each request performs **64 decode forward steps** after prefill. Prefill produces the first prediction; the 64 subsequent steps produce 64 more, giving 65 predicted IDs (including any control tokens). `completion.txt` decodes all of them. Decode includes forward execution and greedy selection; it excludes copying logits into the validation buffer and report/file I/O. `decode_wall_seconds` also records the validation-buffer copy overhead. **EOS is recorded but ignored as a stopping condition**, including any post-EOS continuation. This is a fixed-work throughput benchmark, not a serving latency or instruction-following evaluation.

EOS observations: 32-token prompt: no EOS in 65 predictions; 128-token prompt: no EOS in 65 predictions; 512-token prompt: EOS after 32 non-EOS predictions; reused pre-EOS prefix rate 5.627 tok/s. The pre-EOS rate excludes the first prediction and EOS detection; it is a prefix of the same fixed-work replays, not a separate EOS-stopping cache experiment.

TTFT starts with the model/head initialized and the prompt tokenized; it includes prompt evaluation and the first greedy selection. Initial runs have empty expert caches and lazy backbone/shared-weight uploads. They **do not establish cold-storage performance**: host page cache was not flushed. All passes rebuild KV/compression/Engram state. Decode in the initial pass already benefits from the preceding prefill.

| Prompt | Initial TTFT (s) | Setup + initial TTFT (s) | Reused prefill range (s) | Reused decode range (s) | Decode p50 (ms) |
|---:|---:|---:|---:|---:|---:|
| 32 | 13.036 | 15.853 | 5.266–5.279 | 10.270–10.281 | 161.66 |
| 128 | 30.789 | 31.259 | 20.729–20.796 | 10.334–10.373 | 158.89 |
| 512 | 107.806 | 108.277 | 84.553–84.671 | 11.924–11.944 | 186.23 |

Setup includes GGUF metadata/mmap initialization, tokenizer construction/encoding, prompt artifacts and initial head upload. It excludes executable/loader startup. Percentiles use individual token samples, with linear interpolation: 192 measured decode steps per context. Three repetitions in one process per context are a descriptive sample, not independent process replications or a confidence interval.

## Expert-cache pressure

| Prompt | Reused prefill hits | Reused decode hits | Prefill H2D per request (GiB) | Decode H2D per 64 steps (GiB) | Prefill / decode evictions per request |
|---:|---:|---:|---:|---:|---:|
| 32 | 82.55% | 87.93% | 23.46 | 32.46 | 4020 / 5562 |
| 128 | 86.55% | 90.01% | 72.33 | 26.86 | 12393 / 4602 |
| 512 | 89.09% | 82.45% | 234.68 | 47.19 | 40209 / 8085 |

Hits count individual gate/up/down slices, not whole expert triplets. Transfer totals are instrumented routed-expert H2D bytes; they exclude initial shared/backbone/head uploads, activations, and other traffic. They do not measure storage reads. Every miss is checked against the 6,266,880-byte slice size, and every phase against 720 cache lookups per token.

Repeated requests still evict experts: “reused cache” does not mean all hits. The prior 16-token synthetic prompt plus 4 decode steps fit the cache and achieved roughly 7.66 / 7.54 tok/s. Those figures use different inputs and cache conditions and are not a same-work speed comparison with this benchmark. This result motivates measuring multi-GPU expert residency/ownership and transfer overlap; it does not establish their speedup.

## Execution phases

Mean seconds per reused request. These top-level host wall spans partition Session forwards; subphase details are in `summary.json`. GPU/CPU timing spans are not a hardware utilization profile.

| Prompt | Phase | MoE | Attention | Engram | HC | Other |
|---:|---|---:|---:|---:|---:|---:|
| 32 | prefill | 2.759 | 1.409 | 0.090 | 0.431 | 0.584 |
| 32 | decode | 5.036 | 2.946 | 0.181 | 0.873 | 1.225 |
| 128 | prefill | 10.367 | 5.911 | 0.366 | 1.748 | 2.379 |
| 128 | decode | 4.797 | 3.230 | 0.187 | 0.885 | 1.249 |
| 512 | prefill | 39.326 | 26.966 | 1.492 | 7.084 | 9.736 |
| 512 | decode | 5.662 | 3.906 | 0.187 | 0.893 | 1.273 |

## Configuration and verification

- Model: `DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf`, 10 shards, 473.069 GiB on disk. Mixed storage: MXFP4 routed experts, Q8_0 shared/backbone weights and BF16 LM head. Shard names, sizes and mtimes are recorded; the full 473+ GiB payload was not rehashed.
- 40 layers, hidden dimension 5,120, 384 routed experts with 6 selected per layer, one shared expert; vocabulary 129,280. One request at a time; prefill is currently sequential token GEMV execution.
- One NVIDIA H100 80GB HBM3, physical GPU 1, logical device 0 through `CUDA_VISIBLE_DEVICES=1`. No P2P, second-GPU compute or direct GPU–NVMe path participates in this DS41 benchmark.
- Expert budget 62 GiB: 10,622 slices. BF16 head, Q8 backbone and shared experts are resident outside this cache.
- Intel Xeon Platinum 8480+, 52 visible vCPUs, one NUMA node, about 442 GiB host RAM; persistent CPU worker pool. CUDA toolkit 13.1.115, GCC 14.4.0, driver 580.178.04.
- `DS41_GPU=1 DS41_GPU_ATTN=1 DS41_VRAM_GB=62 DS41_PROFILE=1 DS41_SHARED_GPU=1 DS41_MOE_BATCH=1 DS41_GROUPED_TAIL=1 DS41_HC_DIRECT=1 DS41_MOE_OVERLAP=1`. Detailed CUDA event tracing/timing disabled. Phase profiling remains enabled for all runs.
- Self-authored ASCII technical prose in `corpus.txt`, tokenized with the model GGUF vocabulary/merges (`joyai-llm`) and BOS. Take the first 32, 128 or 512 tokens. This is raw text continuation, without a chat template, and greedy argmax without penalties. Exact IDs and decoded prefixes are stored per case.
- All 12 passes completed, 100,838,400 logits checked finite. For each context, all three repetitions matched all 65 full output vectors and predicted IDs from the initial pass bit for bit. This is within-implementation replay validation, not a new comparison against llama.cpp or the CPU reference.
- Every measured phase used GPU routed experts and GPU shared experts without CPU fallback. Counters also verified four routed launches, three shared launches and two grouped tail launches per layer/token.
- GPU telemetry sampled once per second; 211 compute-process observations found no competing GPU process. Pipeline/exclusive/GPU locks were held. The failed, already exited `formulation-audit-v1` audit was accepted as terminal after checking its PID was gone; its status was not changed.

| GPU | Observed maximum VRAM (MiB) | Mean sampled GPU utilization | Mean sampled power (W) |
|---:|---:|---:|---:|
| 0 | 1 | 0.0% | 69.9 |
| 1 | 72173 | 41.1% | 146.5 |

| Prompt | Process peak RSS (GiB) | Major page faults, entire process |
|---:|---:|---:|
| 32 | 89.64 | 80,079 |
| 128 | 112.21 | 57,064 |
| 512 | 155.85 | 100,655 |

Samples span the entire suite, including setup and gaps; maxima are observed samples, not guaranteed instantaneous peaks. CPU and major/minor page-fault statistics for each process are in `*-resources.txt`. The unused GPU’s idle power is not charged to a claim of two-GPU throughput.

## Reproduction and artifacts

The benchmark helper is `tools/benchmark_ds41.cpp` in the working tree and `benchmark_ds41.cpp` in this artifact. `compile-command.json` records the exact link command against the validated engine static library. `engine-source.tar` preserves its source snapshot, including CMake settings. A representative invocation after linking is:

```bash
CUDA_VISIBLE_DEVICES=1 DS41_GPU=1 DS41_GPU_ATTN=1 DS41_VRAM_GB=62 \
DS41_PROFILE=1 DS41_SHARED_GPU=1 DS41_MOE_BATCH=1 DS41_GROUPED_TAIL=1 \
DS41_HC_DIRECT=1 DS41_MOE_OVERLAP=1 \
./benchmark_ds41 /path/to/DeepSeek-V4.1-Flash-Q8_0-00001-of-00010.gguf \
  corpus.txt 128 64 3 p128-d64 p128-d64
```

Use the recorded CUDA runtime library path when necessary. Start a fresh process for each prompt length; do not mix the initial pass with the three reused-cache passes. `run.py` is the exact host-specific orchestration script, with admission checks and telemetry. `analyze.py` verifies counters, timing lengths, finite saved logits, greedy outputs and replay coverage; `report.py` generates this report.

Benchmark executable SHA-256: `5321fd9d19dca78dc42be4887c6b7ed68b38b4cfe3161259d30a3008087ae065`.
Engine static library SHA-256: `b07eefc14bc2ab9db8deb69280a7acd6357581b07f24b0e2e6306a60f14257a5`.
Existing NTransformer CLI SHA-256: `eca368b24fa9dfab4d884bcd55fc6460e2ecdb13b2be97f9353cadec24686ce0` (unchanged from the shared-expert validation).

Local destination: `docs/benchmarks/h100-deepseek-2026-10-15/end-to-end/`.
Remote working directory: `/home/ubuntu/ntransformer-h100/ds41-bench-20261015`.
Persistent copy: `/lambda/nfs/btiny/ntransformer-training/deepseek-benchmark-20261015`.

Raw JSONL timings, per-phase cache deltas, all prompt/prediction IDs, generated text, sampled telemetry, resource logs, source snapshot, executable and SHA-256 manifest accompany this report. No model weights are bundled.
