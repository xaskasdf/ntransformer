# NTransformer - Motor de Inferencia LLM de Alta Eficiencia en Memoria

## El Problema
Correr Llama 70B a calidad Q8 en una RTX 3090 (24GB VRAM).
- Llama 70B FP16: ~140GB (6x sobre presupuesto)
- Llama 70B INT8: ~70GB (3x sobre presupuesto)
- Llama 70B INT4: ~35GB (1.5x sobre presupuesto)

**La solucion:** Un motor de inferencia C++/CUDA desde cero que combina 6 tecnicas innovadoras para hacer factible lo "imposible".

## Estado Actual
**Phase 1 (Foundation) - CODE COMPLETE, pendiente compilacion/testing en RTX 3090.**
- 43 archivos, ~5,600 lineas C++/CUDA
- Pipeline completo: GGUF loading -> tokenizacion -> transformer forward -> sampling -> texto
- **Siguiente paso:** `git pull`, build, debug, hacer que corra Llama 7B

## Setup de Desarrollo
- **Build:** CMake 3.24+, CUDA Toolkit 12.x, C++20, Linux
- **Sin dependencias externas** mas alla de CUDA Toolkit (no PyTorch, no cuBLAS)
- **Modelos test:** Llama 7B GGUF (Phase 1), Llama 70B GGUF (Phase 2+)

### Build Commands
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Tests
./test_tensor
./test_gemm
# Run
./ntransformer -m /path/to/model.gguf -p "Hello" -n 128
./ntransformer -m /path/to/model.gguf --chat
./ntransformer -m /path/to/model.gguf --benchmark -n 64
```

---

## Presupuesto de Memoria (24GB VRAM)

| Componente | Solo SLEP | + Sparsity | Full Stack |
|-----------|-----------|------------|------------|
| Layer buffer A (compute) | 428 MB | 200 MB | 200 MB |
| Layer buffer B (prefetch) | 428 MB | 200 MB | 200 MB |
| Hot neuron cache | 0 | 3,500 MB | 3,500 MB |
| Embeddings + LM head | 500 MB | 500 MB | 500 MB |
| KV-cache FP16 | 1,250 MB | 1,250 MB | -- |
| KV-cache RotateKV 2-bit | -- | -- | 160 MB |
| CUDA workspace | 512 MB | 512 MB | 512 MB |
| Activations | 1,000 MB | 1,000 MB | 1,000 MB |
| OS/driver | 1,500 MB | 1,500 MB | 1,500 MB |
| **Total** | **~5.6 GB** | **~8.8 GB** | **~7.9 GB** |

**Conclusion: Factible con margen amplio dentro de 24GB.**

---

## Las 6 Innovaciones

### 1. SLEP (Streaming Layer Execution with Predictive Prefetch)
- No cargar modelo entero en VRAM
- Double-buffer: mientras capa N computa, capa N+1 se transfiere via PCIe
- PCIe 4.0: ~25 GB/s -> capa INT4 (~428MB) en ~17ms
- Solo 2 slots de capa en VRAM a la vez

### 2. Adaptive Mixed Precision
- Analisis de sensibilidad por capa
- Capas criticas (primeras/ultimas): INT8/FP16
- Capas medias: INT4/INT3
- Promedio ~4.5 bits con calidad equivalente a Q8 uniforme

### 3. MLA Retrofit (TransMLA)
- Convertir GQA existente a Multi-head Latent Attention via SVD
- KV-cache reducido 5-10x vs GQA estandar
- Near-lossless con calibracion

### 4. RotateKV (KV-Cache Cuantizado)
- Walsh-Hadamard transform redistribuye outliers
- Cuantizar KV-cache a 2-4 bits near-lossless
- 1,280MB -> 170MB para 70B @ 4K context

### 5. Neuromorphic Sparsity (PowerInfer-style)
- ~10% neuronas "hot" (frecuentemente activadas) permanecen en GPU
- Neuronas "cold" en CPU, prefetch on-demand con predictor
- Solo 10-20% del modelo residente en GPU

### 6. Speculative Decoding (Self-Speculative)
- Usar primeras N capas como "draft model"
- Generar K tokens draft, verificar en batch con modelo completo
- 2-3x speedup en decode

---

## Arquitectura del Proyecto

### Namespace y Convenciones
- Todo en `namespace nt`, kernels CUDA en `namespace nt::cuda`
- Launchers CUDA: `launch_*()` (e.g., `launch_rmsnorm`, `launch_gemv`)
- Ops con C-linkage: `nt_cuda_*()` (para .cpp compilados sin nvcc)
- Macros de error: `NT_CHECK(cond, msg)`, `NT_CUDA_CHECK(err)` — ambos llaman `abort()`

### Estructura de Directorios
```
src/
├── core/
│   ├── types.h          # DType enum, BlockQ4_0/Q8_0/Q4_K/Q6_K structs, GGUF constants
│   ├── tensor.h/cpp     # Tensor multi-device con views y zero-copy
│   ├── allocator.h/cpp  # Pool allocator VRAM/pinned RAM (best-fit)
│   └── device.h/cu      # GPU management, 3 streams, CUDA C-linkage wrappers
├── cuda/
│   ├── kernels.h        # TODAS las declaraciones de launchers
│   ├── rmsnorm.cu       # RMSNorm fusionado (single-pass, warp reduction)
│   ├── rotary.cu        # RoPE (Llama-style y interleaved)
│   ├── softmax.cu       # Online softmax + masked variant
│   ├── gemm.cu          # GEMV cuantizados: Q4_0, Q8_0, Q4_K_M, F16, F32 + SiLU*mul
│   ├── attention.cu     # Flash Attention decode + prefill causal con GQA
│   └── elementwise.cu   # add, add_inplace, copy (residual connections)
├── model/
│   ├── config.h/cpp     # ModelConfig parseado de GGUF metadata
│   ├── loader.h/cpp     # GGUF v2/v3 parser con mmap zero-copy
│   ├── norm.h/cpp       # RMSNorm wrapper
│   ├── attention.h/cpp  # GQA attention con KV cache + RoPE
│   ├── ffn.h/cpp        # SwiGLU FFN (gate -> SiLU * up -> down)
│   └── transformer.h/cpp # Transformer completo con residual connections
├── inference/
│   ├── tokenizer.h/cpp  # BPE tokenizer (SentencePiece-style desde GGUF vocab)
│   ├── sampler.h/cpp    # Top-k, top-p, temperature, repeat penalty
│   └── engine.h/cpp     # Pipeline de generacion con streaming output
├── utils/
│   ├── timer.h          # CPU timer (RAII ScopedTimer)
│   ├── logger.h         # Logging multi-nivel (DEBUG/INFO/WARN/ERROR)
│   └── profiler.h/cpp   # Profiler basado en CUDA events
├── main.cpp             # CLI entry point
├── memory/              # (Phase 2) SLEP, offloader, prefetcher
├── quant/               # (Phase 3) RotateKV, adaptive precision
include/
│   └── ntransformer.h   # Public C API
tests/
│   ├── test_tensor.cpp  # Tests de tensor (CPU, GPU, views, transfer)
│   └── test_gemm.cpp    # Tests de kernels (GEMV F32, Q4_0, SiLU, RMSNorm)
scripts/                 # (Phase 3+) convert_model.py, benchmark.py
```

### Data Flow (Forward Pass)
```
tokens[seq_len] (CPU)
  -> embed_tokens(): CPU lookup + H2D copy -> hidden_state[seq_len, hidden_size] (GPU)
  -> for each layer:
      RMSNorm(hidden) -> residual_buf
      Attention(residual_buf, KV cache, RoPE, GQA) -> residual_buf
      hidden += residual_buf
      RMSNorm(hidden) -> residual_buf
      FFN(SwiGLU: gate->SiLU * up -> down) -> residual_buf
      hidden += residual_buf
  -> RMSNorm(last_token_hidden)
  -> LM head GEMV -> logits[vocab_size] (GPU)
  -> D2H copy -> sampling (CPU) -> next token
```

### Buffers en GPU
| Buffer | Shape | Notas |
|--------|-------|-------|
| hidden_buf | [max_seq, hidden_size] | Hidden state principal, F32 |
| residual_buf | [max_seq, hidden_size] | Temp para norm->attn/ffn |
| k_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 |
| v_cache | [n_layers, max_seq, n_kv_heads, head_dim] | F32 |
| workspace | [max(attn_ws, ffn_ws)] | Compartido entre attn y ffn |
| logits_buf | [vocab_size] | Salida final F32 |
| positions_gpu | [max_seq] | IDs de posicion I32 |

### Tensor Names GGUF (Llama)
```
token_embd.weight                    # Embedding table
output.weight                        # LM head (puede compartir con embedding)
output_norm.weight                   # RMSNorm final
blk.{i}.attn_norm.weight             # Pre-attention norm
blk.{i}.attn_q.weight               # Query projection
blk.{i}.attn_k.weight               # Key projection
blk.{i}.attn_v.weight               # Value projection
blk.{i}.attn_output.weight          # Output projection
blk.{i}.ffn_norm.weight             # Pre-FFN norm
blk.{i}.ffn_gate.weight             # SwiGLU gate
blk.{i}.ffn_up.weight               # SwiGLU up
blk.{i}.ffn_down.weight             # SwiGLU down
```

### Formatos de Cuantizacion Soportados
| Formato | Block Size | Bytes/Block | Layout |
|---------|-----------|-------------|--------|
| Q4_0 | 32 weights | 18 bytes | FP16 scale + 16 nibble bytes |
| Q8_0 | 32 weights | 36 bytes | F32 scale + 32 int8 |
| Q4_K_M | 256 weights | 144 bytes | FP16 d/dmin + 12 sub-scales + 128 nibbles |
| Q6_K | 256 weights | 210 bytes | 128 ql + 64 qh + 16 scales + FP16 d |
| F16 | 1 | 2 bytes | IEEE 754 half |
| F32 | 1 | 4 bytes | IEEE 754 float |

### CUDA Streams
- `STREAM_COMPUTE` (0): Ejecucion de kernels
- `STREAM_TRANSFER0` (1): Phase 2 SLEP buffer A
- `STREAM_TRANSFER1` (2): Phase 2 SLEP buffer B

### Constantes Tecnicas
```
GGUF_MAGIC       = 0x46475547
GGUF alignment   = 32 bytes (o lo que diga metadata)
RoPE theta       = 10000.0 (default Llama)
RMSNorm eps      = 1e-5
CUDA target SMs  = 80, 86, 89, 90
```

---

## Limitaciones Conocidas (Phase 1)

1. **Quantized embedding lookup no implementado** — cae a zeros con warning. F32 y F16 embeddings funcionan. La mayoria de GGUF usa F16 embeddings.
2. **Embedding lookup en CPU** — CPU dequant + H2D copy. OK para Phase 1, necesitara kernel GPU para prefill batch grande.
3. **GEMV por token en prefill** — attention.cpp hace loop de GEMV por token en vez de GEMM batch. Prefill lento pero correcto.
4. **Sin recovery de errores** — `NT_CHECK` llama `abort()`. Tensor faltante = crash.
5. **Chat mode stateless** — cada turno es independiente, sin historial de conversacion.

---

## Plan de Implementacion Detallado

### Phase 1: Foundation ✅ (code complete, pendiente compilacion)
**Objetivo:** Pipeline minimo funcional que corra Llama 7B.
**Validacion:** Correr Llama 7B GGUF, comparar output con llama.cpp.
**Target:** 30-50 tok/s decode, ~4GB VRAM, 500+ tok/s prefill.

### Phase 2: SLEP (Streaming Layer Execution)
**Objetivo:** Correr Llama 70B via streaming de capas.

**Como funciona:**
- Modelo 70B no cabe en VRAM -> mantener pesos en RAM del sistema
- Double-buffer: buffer A ejecuta capa N en GPU mientras buffer B transfiere capa N+1 via PCIe
- PCIe 4.0 x16: ~25 GB/s -> capa Q4 de 70B (~428MB) en ~17ms
- Solo 2 slots de capa en VRAM + KV cache + activaciones

**Archivos a crear:**
1. `src/memory/streamer.h/cpp` — Motor de double-buffer con CUDA events para sync
2. `src/memory/offloader.h/cpp` — Politica CPU<->GPU: que cargar, cuando liberar
3. `src/memory/kv_cache.h/cpp` — Manager de KV-cache con eviction para contexto largo
4. `src/memory/prefetcher.h/cpp` — Prefetch basico de siguiente capa
5. Modificar `src/model/transformer.h/cpp` — Integrar SLEP en el forward loop
6. `tests/test_streamer.cpp`

**Cambios al forward loop:**
```
Antes (Phase 1):  todas las capas en VRAM, iterar
Despues (Phase 2): prefetch capa 0 -> for each layer: compute + prefetch_next -> sync
```

**Validacion:** Correr Llama 70B Q4 en RTX 3090, medir VRAM < 8GB, ~0.7-1 tok/s decode.

### Phase 3: Advanced Quantization
**Objetivo:** RotateKV + Adaptive precision para mejor calidad y contexto largo.

**RotateKV:**
- Problema: KV cache en FP16 consume demasiado para contexto largo
- Solucion: Walsh-Hadamard transform redistribuye outliers -> cuantizar a INT2-INT4
- KV cache de 1.3GB -> 170MB para 70B @ 4K context
- Permite contexto 4K -> 16K+ sin explotar VRAM

**Adaptive Mixed Precision:**
- Analizar sensibilidad de cada capa (medido por impacto en perplexity)
- Capas criticas (primeras ~4, ultimas ~4): mantener en INT8 o FP16
- Capas intermedias: bajar a INT4 o INT3
- Promedio ~4.5 bits/peso con calidad equivalente a Q8 uniforme

**Archivos a crear:**
1. `src/cuda/hadamard.cu` — Kernel Walsh-Hadamard transform
2. `src/quant/kv_quant.h/cpp` — RotateKV: transform + quant/dequant KV cache
3. `src/quant/adaptive.h/cpp` — Asignacion de precision por capa
4. `src/quant/quantize.h/cpp` — Utilidades de cuantizacion de pesos
5. `src/quant/dequant.h/cpp` — Kernels de dequant rapidos
6. `scripts/convert_model.py` — Conversion HF -> GGUF con sensitivity analysis

**Validacion:** KV-cache 1.3GB -> 170MB, contexto 4K->16K+, perplexity +0.1 max.

### Phase 4: Novel Architectures
**Objetivo:** MLA, SSM, Sparsity, Speculative decoding.

**TransMLA (Multi-head Latent Attention):**
- Descomponer Wk y Wv via SVD: Wk ≈ Uk * Sk, Wv ≈ Uv * Sv
- Cachear la representacion latente (mucho mas pequena) en vez de K,V completos
- Reduccion 5-10x de KV cache adicional a RotateKV

**Neuromorphic Sparsity (PowerInfer-style):**
- Observacion: en FFN, ~10% de neuronas se activan >90% del tiempo ("hot neurons")
- Hot neurons permanecen en GPU (~3.5GB para 70B)
- Cold neurons en CPU, predictor decide cuales prefetchear por token
- Reduce datos transferidos por PCIe ~5x -> SLEP mucho mas rapido

**Self-Speculative Decoding:**
- Usar primeras N capas del modelo como "draft model" (sin modelo extra)
- Draft genera K tokens candidatos rapido (menos capas = mas rapido)
- Modelo completo verifica batch de K tokens en un solo forward
- Acceptance rate ~70-80% -> speedup efectivo 2-3x en decode

**Archivos a crear:**
1. Modificar `src/model/attention.h/cpp` — MLA retrofit via SVD
2. `src/model/ssm.h/cpp` — Mamba/SSM block (optional)
3. `src/cuda/ssm.cu` — Selective scan kernel
4. `src/cuda/sparsity.cu` — Sparse FFN kernel
5. Modificar `src/memory/prefetcher.h/cpp` — Neural predictor para hot/cold neurons
6. `src/inference/speculative.h/cpp` — Self-speculative decoding loop

**Validacion:** 1.5-3 tok/s en 70B, VRAM < 10GB.

### Phase 5: Polish
**Objetivo:** Optimizacion, benchmarks, API publica.

**Archivos:**
1. Optimizar kernels CUDA (occupancy, shared memory tiling, vectorized loads)
2. `scripts/benchmark.py` — Benchmarks automatizados vs llama.cpp
3. `include/ntransformer.h` — C API completa
4. `src/utils/profiler.h/cpp` — Profiler detallado por kernel
5. Documentacion final

---

## Performance Targets

| Metrica | Phase 1 (7B) | Phase 2 (70B) | Phase 3 (+Quant) | Phase 4 (Full) |
|---------|-------------|--------------|-----------------|----------------|
| VRAM | 4 GB | 6 GB | 5 GB | 8 GB |
| Decode tok/s | 30-50 | 0.7-1.0 | 0.8-1.2 | 1.5-3.0 |
| Prefill tok/s | 500+ | 20-40 | 25-50 | 40-80 |
| Calidad (PPL) | Baseline | Baseline | +0.05-0.1 | +0.1-0.2 |
| Contexto max | 4K | 4K | 16K+ | 16K+ |

---

## Riesgos y Mitigaciones

1. **SLEP bottleneck PCIe:** PCIe 4.0 puede ser cuello de botella -> Sparsity reduce datos transferidos 5x
2. **Rendimiento kernels GEMV:** Debe igualar llama.cpp -> Partir de disenos probados, optimizar iterativamente
3. **TransMLA calidad:** SVD puede perder calidad -> Opcional, validar perplexity antes de habilitar
4. **Predictor de neuronas:** Baja precision degrada output -> Threshold conservador (predecir mas de las necesarias)
5. **Compatibilidad GGUF:** Formato evoluciona -> Target GGUF v3, leer archivos existentes de llama.cpp

---

## CLI Reference

```
./ntransformer [options] -m <model.gguf>

  -m, --model <path>       Path to GGUF model file (required)
  -p, --prompt <text>      Prompt text
  -n, --n-tokens <int>     Max tokens to generate (default: 256)
  -t, --temperature <float> Temperature (default: 0.7)
  --top-k <int>            Top-K sampling (default: 40)
  --top-p <float>          Top-P nucleus sampling (default: 0.9)
  --repeat-penalty <float> Repeat penalty (default: 1.1)
  --seed <int>             Random seed (default: 42)
  --benchmark              Run benchmark mode
  --chat                   Interactive chat mode
  -v, --verbose            Verbose output
```

## Documentacion Adicional
- `DEVELOPMENT.md` — Log de progreso detallado, status por archivo, decisiones de diseno
