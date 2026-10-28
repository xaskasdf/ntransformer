# DeepSeek expert-cache replacement experiment

Two H100 80 GB SXM5 GPUs, owner-GPU expert execution over NVLink, 62 GiB expert cache per GPU, cooperative Q8 backbone, FP32 routed arithmetic. The candidate is SLRU with 80% protected capacity, declared before the access trace and held-out workload were measured. The default stays LRU.

The candidate does not meet all predeclared performance/transfer criteria; it is not accepted as a general replacement for LRU.

Acceptance requires at least 3% improvement in both decode and whole warm-request time, plus fewer warm decode uploads and separated warm timing ranges, on both context-512 corpora. Contexts 32/128 must not regress by more than 2% in whole-request time. These are experimental thresholds, not a general guarantee.

## Controlled inference measurements

Identical token IDs and arithmetic; 64 teacher-forced continuation steps; one cold request and three warm repeats. Variant order alternates between workloads. Trace recording is disabled for throughput measurements. The held-out corpus is a separate original narrative prepared before the trace was inspected.

| Workload | LRU decode tok/s | SLRU decode tok/s | Decode gain | Whole warm-request gain | H2D MiB/token LRU -> SLRU |
|---|---:|---:|---:|---:|---:|
| p32 | 23.4145 | 23.3751 | -0.17% | -0.07% | 0.000 -> 0.000 |
| p128 | 21.5839 | 21.5686 | -0.07% | -0.22% | 0.000 -> 0.000 |
| p512 | 17.6081 | 18.0755 | 2.65% | 5.08% | 68.917 -> 43.704 |
| heldout-p512 | 17.9735 | 18.1311 | 0.88% | 2.92% | 54.349 -> 37.540 |

All 33,612,800 logits match bitwise between policies, with finite and exact repeats within each process. This covers 260 prediction positions; it is not a task-quality certification.

## Cold requests, prefill and memory

| Workload | Policy | Warm prefill tok/s | Cold request s | Cold H2D GiB | Warm request H2D GiB | GPU 0 / GPU 1 peak MiB | RSS peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| p32 | lru | 25.0780 | 9.253 | 79.370 | 0.000 | 64017 / 72505 | 90982 |
| p32 | slru | 25.0187 | 9.218 | 79.370 | 0.000 | 64017 / 72505 | 90980 |
| p128 | lru | 23.8141 | 14.915 | 100.277 | 0.000 | 64017 / 72505 | 112695 |
| p128 | slru | 23.7515 | 14.932 | 100.277 | 0.000 | 64017 / 72505 | 112647 |
| p512 | lru | 19.0827 | 36.705 | 139.235 | 57.203 | 64017 / 72511 | 151874 |
| p512 | slru | 20.1186 | 36.578 | 139.270 | 31.029 | 64017 / 72511 | 151858 |
| heldout-p512 | lru | 19.3930 | 36.569 | 135.681 | 51.425 | 64017 / 72511 | 148872 |
| heldout-p512 | slru | 20.0274 | 50.347 | 135.733 | 30.501 | 64017 / 72511 | 148805 |

VRAM values are maxima of one-second samples; RSS is time(1) high-water memory. Whole-request time adds measured prefill and decode phases, excluding setup, file output and validation. This exposes any transfer savings purchased by slower prefill.

## Offline replay of actual ordered accesses

| Protected capacity | Warm decode H2D MiB/token | First-use misses | Recurrent misses |
|---:|---:|---:|---:|
| 0% | 68.917 | 0 | 2214 |
| 50% | 61.913 | 0 | 1989 |
| 80% | 43.704 | 0 | 1404 |
| 90% | 43.704 | 0 | 1404 |

The 0% replay reproduces every recorded LRU hit/miss. Replay uses the same C++ metadata implementation as inference; both measured policies reproduce the corresponding predicted phase upload bytes exactly. The 50% and 90% rows are diagnostic simulations, not measured speedups or tuning on the held-out corpus.

## Ownership and limits

New entries enter probation. A hit in a later operation promotes a slice to the protected queue; excess protected entries return to probation. Same-operation duplicates do not count as reuse. Eviction prefers unreserved probation entries, then unreserved protected entries. Every returned slot is reserved by an operation epoch until the next synchronous operation, after both CUDA streams have drained. No future routing decisions or token identities enter the policy.

GPU weights remain MXFP4, cache capacities and peer ownership are unchanged, and no extra GPU weight buffer is allocated. Replacement metadata remains on the CPU. Avoided misses remove their pageable-to-pinned staging and H2D copy; remaining misses and CPU dispatch are still present. This is not complete CPU bypass.

## Validation and reproduction

52 CTests passed without skips/failures. CPU tests check an independent LRU model, slot contents, reservations across demotions, all-reserved rejection, duplicates, scans, changing working sets and invalidation. CUDA integration includes both peer placements, arithmetic variants and callback recovery. Raw trace files contain a four-u64 header (magic, slots, slice bytes, record count) followed by key/batch/hit u64 triples.

`prepare.py`, `trace.py`, `run.py`, `queue.py`, `analyze.py`, `frozen-inputs.json`, build/source identity, command logs, token IDs, logits, telemetry and trace/replay evidence are archived. `DS41_CACHE_POLICY=slru` opts in; `DS41_CACHE_PROTECTED_PERCENT` defaults to 80. LRU remains the default and rollback control.
