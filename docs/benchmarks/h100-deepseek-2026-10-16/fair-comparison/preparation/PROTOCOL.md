# Matched two-H100 DeepSeek benchmark: prepared protocol

This run replaces the approximate comparison between current two-GPU NTransformer and historical one-GPU llama.cpp measurements. GPU measurements are pending admission; preparation and synthetic validation are not performance results.

## Fixed work and available hardware

- Same ten DeepSeek-V4.1-Flash GGUF shards on `ubuntu@<h100-host>`.
- Both H100 80 GB SXM5 GPUs visible in the same order: `CUDA_VISIBLE_DEVICES=1,0`; same host and available 52 logical CPUs. Record the NVLink topology, UUIDs, driver and toolchain.
- Exactly the same prompt and continuation token IDs for both engines. Contexts 32, 128 and 512 on the original corpus, plus context 512 on a separate narrative.
- One sequence, 64 consumed continuation tokens and 65 prediction vectors per request. Ignore EOS for fixed-work timing. Targets are corpus tokens, independent of either engine's argmax.
- Both engines use their native execution and arithmetic. Equal hardware availability does not force equal VRAM allocations or identical scheduling. Report actual VRAM and RSS.

## Configurations declared before measurements

NTransformer: owner-GPU expert execution, cooperative Q8 projections, FP32 routed arithmetic, 62 GiB expert cache per GPU and opt-in SLRU with 80% protected capacity. Measure default LRU separately at context 512. The engine library is identical to the preceding version that passed 52 CTests; the benchmark harness adds resource counters.

llama.cpp: pinned fork commit `5210c7c5ed61dddaee6ed476623abf4b63093d16`, native 1:1 layer split, F32 KV requested and ubatch 512. A CPU-only placement test verifies that each device gets the same number of resident routed-expert layers. Splitting includes the output layer, so the repeating-layer boundary is 0–20 / 21–39, rather than 0–19 / 20–39.

An eight-candidate GPU pilot evaluates:

- 20 or 22 CPU-resident routed-expert layers, leaving 10 or 9 GPU-resident layers per device.
- 26 or 52 CPU threads.
- Native operation offload enabled or disabled through the fork's recorded threshold setting.

Pilot work is the original context-128 prefix plus 16 continuation tokens, with one warmup and two measured repeats. Select the lowest median `prefill_seconds + 4 * decode_seconds`, projecting a 64-step request. Explicit allocation OOM makes that pilot infeasible; other runtime failures stop the job. The separate narrative is excluded from selection. This is a bounded pilot, not exhaustive tuning.

## Order, warmup and timing

- Two independent process launches per workload and engine; reverse engine order in the second launch and alternate the initial order across workloads.
- Each process makes five requests: rounds 0 and 1 warm the engine; rounds 2–4 are the three measured requests.
- Clear sequence/KV state per request and retain each engine's weight cache.
- Record prefill, decode, first-token latency, setup, individual token timings and whole-request timing. Primary throughput uses median phase time across the six measured repeats.
- Capture process-wide major/minor page faults and input/output block counters separately for prefill and decode, outside token timers. Flag measured warm phases that still page.
- Never flush the global OS file cache. First-request timings are diagnostic, not a controlled cold-storage comparison.
- GPU memory/power telemetry and compute-process observations accompany every run. A foreign GPU process appearing after admission stops only this benchmark.

The main matrix comprises 16 engine/workload/launch cases plus two LRU controls, after the eight pilot cases.

## Numerical evidence

Check finiteness of every prediction vector on every request. Preserve full-vocabulary logits, consumed IDs, repeat stability and cross-launch hashes. Compute target NLL, NT-to-llama KL and argmax agreement. Cross-engine bitwise equality is not required; these short likelihood probes do not establish equivalent downstream quality or instruction following.

## Queue and artifacts

The worker acquires the shared training locks, requires no active Brandon stage and checks that both GPUs are idle twice before starting. A paused stage without a live recorded process is inactive; its locks and both GPU checks still apply. It waits without terminating other jobs and retries admission beyond one hour when necessary. Source, binaries, libraries, scripts, corpus/token files, model shard metadata and GPU identity are checked against the frozen preparation before execution. Shard identity uses size/mtime; this preparation does not rehash 473 GiB of weights.

- Remote worker: `/home/ubuntu/ntransformer-h100/ds41-fair-20261016/queue.py`.
- Persistent preparation/results: `/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016`.
- Local results: this directory, with `STATUS.md` and `queue-status.json` tracking collection.
- A detached local collector downloads unique payloads after completion, verifies every payload hash, reproduces analysis and updates `docs/DEEPSEEK_PERFORMANCE_GAP.md`.

The Linux worker continues if the local collector disconnects. Collection resumes when the collector is running and SSH is reachable; the remote and Lambda copies preserve the results meanwhile.
