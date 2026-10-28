# Matched two-H100 DeepSeek comparison

Same ten GGUF shards, prompt IDs and 64 consumed continuation IDs, same two H100 80 GB SXM5 GPUs and host. Two independent launches per case, counterbalanced engine order; two warmup requests and three measured requests per launch. Each request resets sequence/KV state and retains the engine's weight cache. Rates below use the median of six warm phase times.

NTransformer uses owner-GPU expert execution, cooperative Q8 backbone, FP32 routed arithmetic and SLRU80 with 62 GiB expert cache per GPU. LRU is a separate context-512 control. llama.cpp uses its pinned native layer split and balanced routed-weight placement, chosen in a bounded eight-candidate pilot before main measurements. Available hardware is equal; allocated VRAM and arithmetic are measured/reported rather than forced equal.

Selected llama configuration: `{"cpu_moe": 20, "op_offload": false, "threads": 52}`; ubatch 512, 1:1 layer split, F32 KV requested. Pilot score is warm prefill plus projected 64-step decode at context 128. The separate narrative is excluded from this selection. This is not an exhaustive search.

| Workload | NT decode tok/s | llama decode tok/s | llama / NT | NT prefill tok/s | llama prefill tok/s | Warm major faults NT / llama |
|---|---:|---:|---:|---:|---:|---:|
| p32 | 23.5329 | 33.7132 | 1.433x | 25.2461 | 84.0887 | 0 / 0 |
| p128 | 21.6057 | 33.8955 | 1.569x | 23.8474 | 95.9958 | 0 / 0 |
| p512 | 18.1975 | 31.5693 | 1.735x | 20.2261 | 104.5600 | 0 / 0 |
| heldout-p512 | 18.2967 | 31.4071 | 1.717x | 20.2081 | 104.5951 | 0 / 0 |

The summary retains the paired launch ratios, all six times, whole warm-request latency and default-LRU control. Two launches do not establish a population confidence interval. Any nonzero major-fault count flags residual paging in the measured warm envelope; input-block counters are also retained.

## Numerical evidence

Each engine consumes the same fixed tokens regardless of its own predictions. All 65 vocabulary vectors are checked for finiteness in every request. Cross-engine bitwise equality is not required. The following values are the mean across the two independent launches; KL is NT to llama. This small likelihood probe does not certify equivalent downstream quality or instruction following.

| Workload | NT target NLL | llama target NLL | Mean KL | Argmax agreement |
|---|---:|---:|---:|---:|
| p32 | 2.169373 | 1.757762 | 0.975198 | 72.31% |
| p128 | 2.244304 | 2.137048 | 0.513642 | 72.31% |
| p512 | 2.533017 | 2.128646 | 0.620707 | 78.46% |
| heldout-p512 | 1.463863 | 1.370942 | 0.423722 | 66.15% |

## Memory and first-request diagnostics

| Case | Peak GPU 0 / 1 MiB | Peak RSS MiB | First request s | First-request major faults | Repeat bitwise |
|---|---:|---:|---:|---:|---|
| nt-slru-p32-r0 | 64017 / 72505 | 90983 | 9.184 | 0 | True |
| nt-slru-p32-r1 | 64017 / 72505 | 91028 | 9.227 | 0 | True |
| llama-p32-r0 | 74479 / 73919 | 285600 | 2.485 | 205 | True |
| llama-p32-r1 | 74479 / 73919 | 285600 | 2.503 | 166 | True |
| nt-slru-p128-r0 | 64017 / 72505 | 112606 | 14.922 | 0 | True |
| nt-slru-p128-r1 | 64017 / 72505 | 112630 | 14.873 | 0 | True |
| llama-p128-r0 | 74475 / 73915 | 285600 | 3.470 | 217 | True |
| llama-p128-r1 | 74475 / 73915 | 285600 | 3.475 | 214 | True |
| nt-slru-p512-r0 | 64017 / 72511 | 151785 | 36.834 | 640 | True |
| nt-slru-p512-r1 | 64017 / 72511 | 151682 | 36.408 | 0 | True |
| llama-p512-r0 | 74481 / 73921 | 285600 | 7.245 | 248 | True |
| llama-p512-r1 | 74481 / 73921 | 285600 | 7.602 | 266 | True |
| nt-lru-p512-r0 | 64017 / 72511 | 151708 | 36.423 | 0 | True |
| nt-lru-p512-r1 | 64017 / 72511 | 151795 | 36.507 | 0 | True |
| nt-slru-heldout-p512-r0 | 64017 / 72511 | 148683 | 36.246 | 0 | True |
| nt-slru-heldout-p512-r1 | 64017 / 72511 | 148676 | 36.198 | 0 | True |
| llama-heldout-p512-r0 | 74481 / 73921 | 285600 | 9.567 | 7264 | True |
| llama-heldout-p512-r1 | 74481 / 73921 | 285600 | 7.328 | 249 | True |

The first request is not a controlled cold-storage comparison: the global OS file cache is never flushed. Page faults/I/O counters cover each phase across process threads; decode timing excludes validation copies while its resource envelope includes them. VRAM peaks are one-second samples; RSS is the process high-water mark. Setup, initialization, file output and validation are excluded from phase throughput and recorded separately.

## Provenance and reproduction

The engine library/source identities, harnesses, predeclared plan, input-token JSONs, selected pilot and rejected/OOM pilots, commands, logits, telemetry, per-phase counters and analysis are preserved. The NT engine library is exactly the version that passed the preceding 52 CTests; only its benchmark harness gains resource counters. The llama harness gains balanced two-GPU placement, full-logit output and the same counters. GPU admission holds shared training locks and requires both GPUs idle; a foreign GPU process appearing later aborts only this benchmark. Model identity records shard sizes/mtime rather than newly hashing the entire 473 GiB payload.
