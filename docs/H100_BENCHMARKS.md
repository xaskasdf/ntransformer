# H100 CUDA P2P benchmarks

On 13 October 2026, NTransformer ran Brandon Mini 108M and Llama 3.1 8B Q8_0
on a pair of H100 SXM5 GPUs while a separate distributed training job occupied
both devices. Native CUDA P2P caching kept layer weights in the second GPU's
VRAM and transferred them into the compute GPU's double buffers. With 16 of
Llama's 32 layers cached remotely, decode was **24.75 tokens/s**, compared with
**12.5 tokens/s** for the same 16 layers cached in pinned host RAM. All eight
Llama runs generated identical text.

These are measurements under shared GPU load. The P2P backend expands weight
capacity; kernels and the KV cache remain on the compute GPU.

## Hardware and build

| Item | Configuration |
| --- | --- |
| GPUs | 2 NVIDIA H100 80GB HBM3, SXM5, MIG disabled |
| GPU interconnect | `nvidia-smi topo -m`: NV18 |
| Host links | PCIe generation 5, x16 on both GPUs |
| Host | Intel Xeon Platinum 8480+, 52 logical CPUs, 442 GiB RAM |
| Operating system | Ubuntu 22.04.5, Linux 5.15.0-143-generic |
| NVIDIA driver | 580.178.04 |
| CUDA toolkit | 13.1.115 |
| Host compiler | conda-forge GCC/G++ 14.4.0 |
| CMake | 4.4.4, Ninja generator |
| Build options | Release for inference, Debug for tests; `USE_GPUNVME=OFF`, `NTRANSFORMER_BUILD_NTRPU_SIM=OFF` |
| Shared workload | Brandon Mini distributed pretraining, about 21,312 MiB per GPU; initial utilization 96–98% |

The isolated toolchain and binaries reside under
`/home/ubuntu/ntransformer-h100`. The training environment and NVIDIA driver
were left in place. The inference build includes native SM90 kernels.

## Brandon Mini results

The checkpoint contains **107,604,288 parameters**: 30 layers, hidden size 576,
9 query heads, 3 KV heads, intermediate size 1472 and an 8,192-token SentencePiece
vocabulary. The stable `pilot-sft-smoke-v2/best.pt` checkpoint was exported on
CPU using [export_brandon_gguf.py](../tools/export_brandon_gguf.py). Q/K rows are
permuted from Brandon's adjacent RoPE pairs to the split-half layout used by
NTransformer. Export manifests record checkpoint and tokenizer hashes.

The prompt was `La inteligencia artificial`; generation used 32 tokens, context
256, greedy sampling, repetition penalty 1.0 and seed 42. Each table entry is
the median of three separate processes. Decode statistics count the 31 steps
after the first generated token.

| Format | GPU 0 tokens/s | GPU 1 tokens/s | Sampled process VRAM per GPU |
| --- | ---: | ---: | ---: |
| F16 | 158.7 | 157.3 | 764 MiB |
| Q8_0 | 163.5 | 165.6 | 650 MiB |

Three simultaneous F16 pairs achieved 164.8–166.7 tokens/s on GPU 0 and
162.4–165.8 tokens/s on GPU 1. Those were independent processes. Their decode
rates exclude model load and are separate from throughput over the entire
process lifetime. The tiny tiered run promoted all 30 layers to local VRAM.

For numerical validation, a CPU PyTorch reference provided all 8,192 logits at
22 fixed input positions, including a batched prefill check. Each format matched
the reference's most probable token at all 22 positions.

| Format | Maximum absolute logit error vs PyTorch | Relative L2 error |
| --- | ---: | ---: |
| F32 | 0.001415 | 0.0000954 |
| F16 | 0.004574 | 0.0002818 |
| Q8_0 | 0.103134 | 0.0062100 |

The F16 run on GPU 1 and the local tiered F16 run reproduced the same results.
These checks use identical input IDs; they do not measure text quality.

## Transfer bandwidth

CUDA events timed eight copies of 64 MiB after one warm-up copy. Rates below are
decimal GB/s of payload, measured under the same shared training load. D2D
payload accounts for one copy's byte count; memory traffic also includes writes.
P2P directions were measured separately.

| Transfer | GPU or direction | Payload GB/s |
| --- | --- | ---: |
| Pinned host to GPU | GPU 0 | 55.041 |
| Pinned host to GPU | GPU 1 | 54.807 |
| GPU to pinned host | GPU 0 | 53.756 |
| GPU to pinned host | GPU 1 | 54.018 |
| Local D2D | GPU 0 | 1395.543 |
| Local D2D | GPU 1 | 1388.727 |
| CUDA P2P | GPU 1 to GPU 0 | 304.365 |
| CUDA P2P | GPU 0 to GPU 1 | 222.277 |

## Native CUDA peer cache

`--peer-gpu` adds a peer VRAM tier between local VRAM and pinned RAM. Each block
holds the seven attention/FFN weight tensors of a layer, aligned to 256 bytes.
The cache checks CUDA peer access, uses `cudaMalloc` on the peer device and
loads weights from GGUF once. Per forward pass, one contiguous
`cudaMemcpyPeerAsync` stages each remote layer into the compute GPU's alternating
buffers. CUDA events protect buffer reuse and make compute wait for completed
transfers. The cache preserves the caller's current device during allocation,
upload, copying and teardown.

The peer budget reserves 2 GiB of currently free peer VRAM. Layers exceeding
that budget continue into the existing RAM/mmap tiers. `--local-layers` and
`--peer-layers` cap their respective resident counts for controlled comparisons.
Without those caps, local VRAM is filled first and only overflow uses the peer.
Device IDs refer to the order in `CUDA_VISIBLE_DEVICES`.

This path builds with `USE_GPUNVME=OFF`. CUDA P2P moves weights already in GPU
memory; initial file loading uses the existing GGUF/host path. The CUDA runtime
documents [peer access](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-runtime-api/group__CUDART__PEER.html)
and [asynchronous peer copies](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-runtime-api/group__CUDART__MEMORY.html).
Delta-encoded weights are rejected with this backend.

## Llama 8B test

The larger test used
[QuantFactory's Meta Llama 3.1 8B Instruct Q8_0 GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf),
downloaded with `hf download` and verified with `hf cache verify`. Eight runs
generated 128 tokens each from `The capital of France is`, with context 256,
temperature 0, repetition penalty 1.0 and seed 42. Each mode ran twice; the
table reports the median of the CLI's 127-step decode rates.

The measured file is 8,540,770,848 bytes at revision
`b6d5cca03f341fd97b7657420bd60e070835b7e5`; its SHA256 is
`e240fae33a33be0a820880e81239938f7a28e49010487a14a9f04ac672fa48cc`.

| Layer weights | Local / peer / RAM layers | Decode tokens/s | Process VRAM GPU 0 / GPU 1 |
| --- | --- | ---: | --- |
| All resident | 32 / 0 / 0 | 32.25 | 8420 / 0 MiB |
| Half in pinned RAM | 16 / 0 / 16 | 12.50 | 5120 / 518 MiB |
| Half in peer VRAM | 16 / 16 / 0 | 24.75 | 5120 / 4070 MiB |
| All blocks in peer VRAM | 0 / 32 / 0 | 19.20 | 1568 / 7622 MiB |

The host comparison initializes a peer CUDA context to use the same layer-cap
controls; its 518 MiB on GPU 1 contains no cached layer blocks. Memory values
come from sampling this process's entries in `nvidia-smi`, excluding the training
processes. The CLI's global VRAM total includes all processes.

Half-peer runs issued **2,048 P2P copies and transferred 442 GiB** each. All-peer
runs issued **4,096 copies and transferred 884 GiB** each. Generated-text SHA256
was identical across all eight runs. P2P caching was **1.98x faster than pinned
RAM** in this comparison, while all-local weights remained fastest. Moving all
layer blocks to the peer reduced compute-GPU process memory from 8,420 to
1,568 MiB. The training loop continued; its logged throughput decreased during
the intensive runs, so these rates include workload contention.

## Validation and reproduction

Debug CTest passed `test_tensor`, `test_gemm` and `test_peer_cache` with both
GPUs visible and no FAIL/SKIP diagnostics. The peer test checks allocations,
partial uploads, bounds, event-ordered copies, numerical kernel output, counters,
device restoration and reinitialization. The Q8_0 tensor-size assertion was
corrected to its actual 34-byte block size.

Six Brandon configurations checked 180,224 logits each: F16 and Q8_0 with all
or half the layers remote, plus local/peer/RAM and local/peer/mmap mixtures.
Every logit and the final prefill logits matched the corresponding resident run
exactly in these tests.
An additional check reversed the physical GPU order and used the peer setter
with the default `load()` streaming argument; its logits also matched exactly.

```bash
# Model assets stay outside the repository.
hf download QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
  --revision b6d5cca03f341fd97b7657420bd60e070835b7e5 --local-dir /models/llama-8b
hf cache verify QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF \
  --revision b6d5cca03f341fd97b7657420bd60e070835b7e5 --local-dir /models/llama-8b

# Use the standard CUDA build, with USE_GPUNVME=OFF.
CUDA_VISIBLE_DEVICES=0,1 ./build/ntransformer \
  -m /models/llama-8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
  --peer-gpu 1 --local-layers 16 \
  -p "The capital of France is" -n 128 -c 256 -t 0 --repeat-penalty 1.0

# Repeat all four cache configurations and save per-process metrics.
python3 scripts/benchmark_peer_cache.py --binary ./build/ntransformer \
  --model /models/llama-8b/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
  --output build/peer-benchmark --gpu 0 --peer 1 --local-layers 16 \
  --tokens 128 --context 256 --repeats 2

# In an assertion-enabled build, expose both devices to the peer test.
CUDA_VISIBLE_DEVICES=0,1 ctest --test-dir build-debug -V
```

Recorded evidence is in [benchmarks/h100-2026-10-13](benchmarks/h100-2026-10-13):
`brandon_smoke.json`, `brandon_logits.json`, `bandwidth.jsonl`, `peer_parity.json`,
`llama_8b_peer.json`, `llama_8b_source.json`, `ctest_p2p.txt` and training snapshots. Complete remote logs
remain under `/home/ubuntu/ntransformer-h100/logs`; the initial compressed
archive is `build/h100/results.tar.gz` in the measurement workspace; the P2P
archive with the tested source is `build/h100/results-p2p.tar.gz`.
