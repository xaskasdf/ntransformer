# ntransformer Inference Pipeline

## High-Level Token Generation

```
  Prompt: "Hello, how are you?"
       │
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  TOKENIZER.encode()                                         │
  │  "Hello, how are you?" → [128000, 9906, 11, 1268, 527, 499, 30] │
  │                           BOS─┘                              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
       ┌─────────────────────────┘
       │  7 tokens
       ▼
  ╔═══════════════════════════════════════════════════════════╗
  ║  PREFILL PHASE  (process entire prompt in one forward)    ║
  ║                                                           ║
  ║  forward(tokens=[128000,9906,11,1268,527,499,30],         ║
  ║          seq_len=7, start_pos=0)                          ║
  ║                                                           ║
  ║  ┌─────────────┐                                          ║
  ║  │ Embed all 7  │  CPU lookup → H2D copy                  ║
  ║  │ tokens       │  token_embd.weight[token_id] → GPU      ║
  ║  └──────┬──────┘                                          ║
  ║         │ hidden_state: [7 × 4096]                        ║
  ║         ▼                                                  ║
  ║  ┌─────────────────────────────────────────────────┐      ║
  ║  │  Layer 0..79  (see Layer Detail below)           │      ║
  ║  │  Each layer: RMSNorm → Attention → Add           │      ║
  ║  │              RMSNorm → FFN(SwiGLU) → Add         │      ║
  ║  │  KV cache filled at positions [0..6]             │      ║
  ║  └──────┬──────────────────────────────────────────┘      ║
  ║         │ hidden_state: [7 × 4096]                        ║
  ║         ▼                                                  ║
  ║  ┌─────────────┐                                          ║
  ║  │ Take LAST    │  hidden_state[6] (last token only)      ║
  ║  │ token hidden │                                          ║
  ║  └──────┬──────┘                                          ║
  ║         │ [4096]                                           ║
  ║         ▼                                                  ║
  ║  ┌─────────────┐                                          ║
  ║  │ RMSNorm     │  output_norm                              ║
  ║  └──────┬──────┘                                          ║
  ║         │ [4096]                                           ║
  ║         ▼                                                  ║
  ║  ┌─────────────┐                                          ║
  ║  │ LM Head     │  output_weight @ hidden → logits          ║
  ║  │ (GEMV)      │  [4096] → [128256]                        ║
  ║  └──────┬──────┘                                          ║
  ║         │ logits: [128256]                                 ║
  ╚═════════╪═════════════════════════════════════════════════╝
            │
            ▼
  ┌──────────────────────────────────┐
  │  SAMPLER                          │
  │  1. Temperature scaling (/0.7)    │
  │  2. Repeat penalty (prev tokens)  │
  │  3. Top-K filter (keep 40)        │
  │  4. Softmax → probabilities       │
  │  5. Top-P nucleus (cumsum ≥ 0.9)  │
  │  6. Categorical sample            │
  │  → token_id = 358 ("I")           │
  └──────────────────┬───────────────┘
                     │
       ┌─────────────┘
       │
       ▼
  ╔═══════════════════════════════════════════════════════════╗
  ║  DECODE LOOP  (one token at a time, autoregressive)       ║
  ║                                                           ║
  ║  pos = 7                                                  ║
  ║                                                           ║
  ║  ┌─── Iteration 1 ────────────────────────────────────┐  ║
  ║  │ forward(tokens=[358], seq_len=1, start_pos=7)       │  ║
  ║  │   Embed 1 token                                     │  ║
  ║  │   Layer 0..79:                                      │  ║
  ║  │     Attention: Q from new token,                    │  ║
  ║  │                K,V from cache[0..7] (8 positions)   │  ║
  ║  │     Update cache[7] with new K,V                    │  ║
  ║  │   RMSNorm → LM Head → logits                       │  ║
  ║  │ Sampler → token = 2846 ("'m")     pos=8            │  ║
  ║  └─────────────────────────────────────────────────────┘  ║
  ║  ┌─── Iteration 2 ────────────────────────────────────┐  ║
  ║  │ forward(tokens=[2846], seq_len=1, start_pos=8)      │  ║
  ║  │   Attention: cache[0..8] (9 positions)              │  ║
  ║  │   Update cache[8]                                   │  ║
  ║  │ Sampler → token = 264 (" doing")   pos=9           │  ║
  ║  └─────────────────────────────────────────────────────┘  ║
  ║  ┌─── Iteration 3 ────────────────────────────────────┐  ║
  ║  │ forward(tokens=[264], seq_len=1, start_pos=9)       │  ║
  ║  │   Attention: cache[0..9] (10 positions)             │  ║
  ║  │ Sampler → token = 128009 (EOS) → STOP              │  ║
  ║  └─────────────────────────────────────────────────────┘  ║
  ╚═══════════════════════════════════════════════════════════╝
       │
       ▼
  Output: "I'm doing"
  Stats: Prefill 7 tok @ 43 tok/s, Decode 3 tok @ 40 tok/s
```

## Single Layer Detail (Resident Mode)

```
  hidden_state [seq_len × 4096]    (from previous layer or embedding)
       │
       ├──────── save as residual ────────┐
       │                                   │
       ▼                                   │
  ┌──────────┐                             │
  │ RMSNorm  │  attn_norm.weight           │
  └────┬─────┘                             │
       │                                   │
       ▼                                   │
  ┌──────────────────────────────────┐     │
  │  ATTENTION                        │     │
  │                                   │     │
  │  Q = Wq @ input   [4096→4096]    │     │
  │  K = Wk @ input   [4096→1024]    │     │
  │  V = Wv @ input   [4096→1024]    │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Apply RoPE to Q, K               │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Store K,V → cache[pos]           │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Scores = Q @ K_cache^T / √128   │     │
  │  (GQA: 32 heads, 8 KV heads,     │     │
  │   group=4 heads share same KV)    │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Weights = Softmax(Scores)        │     │
  │  (causal mask during prefill)     │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Attn_out = Weights @ V_cache     │     │
  │       │                           │     │
  │       ▼                           │     │
  │  Output = Wo @ Attn_out [4096→4096]│     │
  └────────────┬─────────────────────┘     │
               │                            │
               ▼                            │
        ┌──────────┐                        │
        │   ADD    │◄───────────────────────┘
        └────┬─────┘   hidden_state += attention_output
             │
             ├──────── save as residual ────────┐
             │                                   │
             ▼                                   │
        ┌──────────┐                             │
        │ RMSNorm  │  ffn_norm.weight            │
        └────┬─────┘                             │
             │                                   │
             ▼                                   │
        ┌──────────────────────────────────┐     │
        │  FFN (SwiGLU)                     │     │
        │                                   │     │
        │  gate = Wgate @ input [4096→14336]│     │
        │  up   = Wup   @ input [4096→14336]│     │
        │       │                           │     │
        │       ▼                           │     │
        │  SiLU(gate) * up                  │     │
        │       │                           │     │
        │       ▼                           │     │
        │  out  = Wdown @ silu  [14336→4096]│     │
        └────────────┬─────────────────────┘     │
                     │                            │
                     ▼                            │
              ┌──────────┐                        │
              │   ADD    │◄───────────────────────┘
              └────┬─────┘   hidden_state += ffn_output
                   │
                   ▼
            hidden_state [seq_len × 4096]  → next layer
```

## Streaming Mode: SLEP Triple Pipeline

```
                    TIME ──────────────────────────────────────────────►

  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 1: Prefetch to staging  (CPU worker / gpu-nvme-direct)       │
  │                                                                     │
  │   [  memcpy L0→stg[0]  ] [  memcpy L2→stg[0]  ] [  memcpy L4→stg[0]  ]
  │                  [  memcpy L1→stg[1]  ] [  memcpy L3→stg[1]  ] ...
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │  STAGE 2: H2D transfer  (PCIe DMA, async on STREAM_TRANSFER)       │
  │                                                                     │
  │          [ stg[0]→gpu[0] ] [ stg[0]→gpu[0] ] [ stg[0]→gpu[0] ]
  │                   [ stg[1]→gpu[1] ] [ stg[1]→gpu[1] ] ...
  │                                                                     │
  ├─────────────────────────────────────────────────────────────────────┤
  │  STAGE 3: GPU compute  (on STREAM_COMPUTE)                          │
  │                                                                     │
  │                  [ Compute L0 ] [ Compute L1 ] [ Compute L2 ] ...
  │                  │ attn_norm   │              │
  │                  │ attention   │              │
  │                  │ +residual   │              │
  │                  │ ffn_norm    │              │
  │                  │ ffn(swiglu) │              │
  │                  │ +residual   │              │
  └─────────────────────────────────────────────────────────────────────┘

  Double-buffer slots: gpu_buf_[0] and gpu_buf_[1] alternate per layer

  Synchronization events:
    transfer_done_[slot]  ── Compute waits for this before reading weights
    compute_done_[slot]   ── Transfer waits for this before overwriting buffer


  Per-token timeline for 70B (80 layers):
  ┌────────────────────────────────────────────────────────────────────┐
  │                                                                    │
  │  Layer:  0    1    2    3   ...  78   79                           │
  │                                                                    │
  │  Source     ┌──────────────────────────────────────┐               │
  │  (where     │                                      │               │
  │   weights   │  mmap (page cache / NVMe cold)       │               │
  │   come      │      OR                              │               │
  │   from):    │  gpu-nvme-direct (NVMe → staging)    │               │
  │             └──────────────────────────────────────┘               │
  │                          │                                         │
  │                          ▼                                         │
  │  staging    ═══[stg0]═══[stg1]═══[stg0]═══[stg1]═══              │
  │                          │                                         │
  │                     PCIe H2D DMA                                   │
  │                          │                                         │
  │                          ▼                                         │
  │  GPU buf    ═══[gpu0]═══[gpu1]═══[gpu0]═══[gpu1]═══              │
  │                          │                                         │
  │                     GPU compute                                    │
  │                          │                                         │
  │                          ▼                                         │
  │  KV cache   [pos N updated in each layer's attention]              │
  │                                                                    │
  │  After layer 79:                                                   │
  │    RMSNorm → LM Head → logits → Sampler → next token              │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

## Data Path Comparison

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  RESIDENT MODE  (model fits in VRAM)                            │
  │                                                                 │
  │  Load once:  NVMe → mmap → H2D → VRAM                          │
  │  Inference:  VRAM only (936 GB/s bandwidth)                     │
  │                                                                 │
  │    ┌─────┐      ┌──────┐      ┌─────┐                          │
  │    │NVMe │─────►│ RAM  │─────►│VRAM │◄── GPU reads at 936 GB/s │
  │    │     │ once │(mmap)│ H2D  │     │    per token              │
  │    └─────┘      └──────┘      └─────┘                           │
  │                                                                 │
  │  8B Q8_0: 40 tok/s decode                                       │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  STREAMING MODE (mmap path)  — model does NOT fit in VRAM       │
  │                                                                 │
  │  Every token: NVMe → page cache → CPU memcpy → staging → H2D   │
  │                                                                 │
  │    ┌─────┐      ┌──────┐  memcpy  ┌───────┐  DMA  ┌─────┐     │
  │    │NVMe │─────►│ page │─────────►│staging│──────►│GPU  │     │
  │    │     │ 3 GB/s│cache│ 6 GB/s   │(pinned)│ 13GB/s│buf[]│     │
  │    └─────┘      └──────┘ ▲ CPU    └───────┘       └─────┘     │
  │                           │ bottleneck!                         │
  │                                                                 │
  │  70B Q6_K: 0.02 tok/s (page cache cold)                        │
  │            0.9 tok/s  (page cache warm, 8B only)                │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  STREAMING MODE (gpu-nvme-direct)  — GPU-autonomous I/O         │
  │                                                                 │
  │  Every token: GPU doorbell → NVMe DMA → staging → GPU compute   │
  │                                                                 │
  │    ┌─────┐  NVMe DMA  ┌───────┐                                │
  │    │NVMe │────────────►│staging│──── GPU reads directly          │
  │    │     │  3.35 GB/s  │(pinned)│   (no CPU memcpy!)             │
  │    └─────┘   ▲         └───────┘                                │
  │              │                                                   │
  │        GPU writes                                                │
  │        doorbell to                                               │
  │        NVMe BAR0                                                 │
  │        (MMIO)                                                    │
  │                                                                 │
  │  CPU: idle (not in data path)                                    │
  │                                                                 │
  │  70B Q6_K: 0.06 tok/s (B450 Gen3)                               │
  │            0.12 tok/s (B550 Gen4, estimated)                     │
  └─────────────────────────────────────────────────────────────────┘
```

## KV Cache Growth During Generation

```
  Prompt: "Hello, how are you?"  (7 tokens)

  After prefill:
    KV cache: [───filled───]░░░░░░░░░░░░░░░░░░░░░░░░  pos 0-6 filled
               0  1  2  3  4  5  6                     4096 max

  Decode token 1 ("I"):
    KV cache: [───filled────█]░░░░░░░░░░░░░░░░░░░░░░  pos 7 added
               0  1  2  3  4  5  6  7

  Decode token 2 ("'m"):
    KV cache: [───filled─────██]░░░░░░░░░░░░░░░░░░░░  pos 8 added
               0  1  2  3  4  5  6  7  8

  Attention at pos 8 reads ALL of cache[0..8]:
    Q = embed(token_8)
    K = cache_K[0..8]   ← grows with each token
    V = cache_V[0..8]   ← grows with each token
    score = Q @ K^T / √128
    output = softmax(score) @ V

  KV cache per layer: max_seq × n_kv_heads × head_dim × sizeof(float)
                    = 4096    × 8           × 128      × 4
                    = 16 MB per layer, × 80 layers = 1.28 GB total
```

## Memory Budget (70B Q6_K, Streaming Mode on 24GB VRAM)

```
  RTX 3090 VRAM: 24 GB
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │      │                                          │
  │  │ 1.4GB│  KV cache (80 layers × 16MB, ctx=4096)  │
  │  │      │                                          │
  │  ████████                                          │
  │  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │1.3GB│  Double GPU buffers (2 × 675MB)           │
  │  │     │  Only 1 layer loaded at a time per slot   │
  │  ██████                                            │
  │  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │0.5GB│  Workspace (attention + FFN intermediates) │
  │  ███                                               │
  │  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │0.3GB│  Output weight + norms + embedding lookup │
  │  ██                                                │
  │  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │misc │  Logits buf, positions, CUDA overhead     │
  │  █                                                 │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │  │        ~20 GB FREE                          │   │
  │  │        (unused in streaming mode)           │   │
  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
  │                                                    │
  │  Total used: ~3.5 GB  ← fits easily in 24GB!      │
  │  vs Resident 70B: ~54 GB  ← impossible             │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

## Pipeline Depth Tuning

The streaming pipeline uses N buffer slots to overlap H2D transfers with GPU compute.

### How N-buffer pipelining works

With N=2 (default double-buffer):
```
iter i:   [H2D layer i] [compute layer i-1]
iter i+1: [H2D layer i+1] [compute layer i]
```

With N=3 (triple-buffer, for Gen5 x16):
```
iter i:   [H2D layer i] [H2D layer i+1] [compute layer i-2]
iter i+1: ...
```

The third slot only helps if H2D transfer time < compute time, meaning the GPU is
waiting on the transfer. On Gen5 x16 (~63 GB/s), a 670 MB layer takes ~11ms vs
~14ms compute — so prefetching one layer ahead improves utilization. On Gen3/Gen4 x8
(6.5–31 GB/s), H2D time >> compute time and a second buffer is already wasted.

### Setting pipeline depth

```bash
# Auto (recommended) — reads PCIe bandwidth from sysfs at startup
./ntransformer --streaming -m model.gguf -p "Hello"

# Force 2 buffers
./ntransformer --streaming --n-buffers 2 -m model.gguf -p "Hello"

# Force 3 buffers (Gen5 x16 only)
./ntransformer --streaming --n-buffers 3 -m model.gguf -p "Hello"

# Via environment variable
NT_PIPELINE_DEPTH=3 ./ntransformer --streaming -m model.gguf -p "Hello"
```

Auto-detection logs the chosen depth at startup:
```
TierConfig: PCIe Gen4 x8 = 31.0 GB/s (detected)
Pipeline depth: 2 (PCIe 31.0 GB/s autodetect)
```
