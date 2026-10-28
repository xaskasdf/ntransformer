# GLM-5.3-Flash decode fast-path — port del playbook de ter

Diseño para el decode device-residente con cudaGraph, rescatado del análisis
de `D:\projects\ter` (ter_cuda_forward_packed.cu y variantes). Ejecutable por
fases, cada una con verificación de paridad greedy ("The capital of France is"
→ " Paris. In French, Paris is spelled \"Paris\", but").

## Contexto medido (por qué esto y cuánto da)

- Decode actual: 0.24-0.26 tok/s. Desglose por capa (~93 ms): fills de
  expertos 40-75 ms (bursty, disco a 1.34 GiB/s instantáneo), serial
  CPU+launches ~70 ms (~40 syncs/capa).
- Techo del disco: 3.5 GB/s → el fill es inmovable sin predicción de expertos
  (routing uniforme la descarta, ver experimentos GLM5_PREFETCH).
- Este plan ataca los ~70 ms seriales → objetivo 0.30-0.38 tok/s.

## Fases

### Fase A — estado device-residente, sin graphs (glm5_gpu2.cu nuevo)

Buffers persistentes en VRAM: x[4096], streams[4x4096], scratch KDA
(xq/xk/xv/cq/ck/cv 8192, decay 8192, cout 8192, gated 8192), MLA (qr 1536,
q 16384, kv 512, q_abs 64x512, attn_out 16384), conv/ssm state ya migrados,
latent cache ya en VRAM. Los pesos ya suben via glm5_resident.

Kernels nuevos (patrones de ter):
- k_hc_collapse / k_hc_post_mix / k_sinkhorn4 (ter: state en device,
  escalas device-pointer — packed.cu:40-62)
- k_conv_silu (3 qkv x 8192, estado en VRAM)
- k_gdn_step: 1 bloque por cabeza, estado 128x128 en VRAM (update in-place,
  decay/beta computados in-kernel)
- k_out_gate (rmsnorm por cabeza + sigmoid)
- k_mla_attn: softmax sobre latentes ≤ pos + wv_b (smem de tamaño FIJO max =
  requisito para graphs, ter packed.cu:127; a ctx≤2048 el indexer selecciona
  todo → sin kernel de selección, solo causal)
- k_router_f32 → k_sigmoid_top8 (1 bloque, 288 logits, selección in-kernel,
  ids a memoria device)
- fusions: k_rmsnorm_w, k_silu_clamp (existe), k_dense_swiglu

Regla de la Fase A: CERO readbacks intermedios. Solo: token id (input) y
logits/next-token (output). El MoE: el router escribe ids a device; el HOST
lee los 8 ids (D2H de 32 B, único sync) → gestiona slots del cache existente
(H2D de slices) → kernels de expertos leen pesos via TABLA DE PUNTEROS
device-side (uint64 ptrs[43 capas][8][3]) actualizada por el host antes del
replay de la parte de expertos — el idioma de ter token_embed_lookup_k
(packed.cu:97-104).

### Fase B — cudaGraph por sub-bloque

- G1_l = attention+norm+router (captura global mode, ter packed.cu:729-752).
  Requisitos: smem fijo, sin D2H dentro, punteros estables (tabla de expertos).
- Host: lee ids de G1_l → fills (rings ReadFile + H2D a slots) → actualiza
  tabla → G2_l = expertos+shared+post_mix.
- El replay de G2 es async: el host arranca fills de nada (secuencial), pero
  los launches desaparecen (~40→2 por capa).

### Fase C — dp4a int8xint8 (activaciones cuantizadas on-the-fly)

- k_quant_absmax (activación → int8 + escala device-float, ter packed.cu:40)
- k_q8xq8_gemv: dp4a con bloques Q8_0 del peso (el peso ya ES int8; solo la
  activación se cuantiza) → ~2x en kernels de GEMV
- fusión silu_clamp→quant (ter packed.cu:311-344)
- lm_head: two-pass argmax (ter bitnet.cu:449-489, 150μs→5μs)

## Presupuesto de VRAM (24 GB, WDDM margen ~1 GB — CRÍTICO)

residents ~6.6 + cache ≤ 13.3 + scratch Fase A (~200 MB) = ≤ 20.5 GB.
NO subir cache a 16+ GB: thrashing silencioso (medido: 19 GB → 0.02 tok/s,
16 GB → 0.05 tok/s; el margen actual free-8.5 GB da 13.28 GB y 0.24 tok/s).

## Referencias ter (file:line)

- Graph capture/replay: ter-cuda\ter_cuda_forward_packed.cu:729-752
- Estado device (pos/token/KV): packed.cu:64-104, 556-562
- Escalas device-pointer: packed.cu:40-62, 357-358
- GEMV v11 dp4a warp-coop: packed.cu:346-397
- v12 (LDG.128, X en smem, 8 cols/warp): ter\cuda\ter_cuda_forward_packed_v12.cu:192-316
- rmsnorm+quant / silu+quant: packed.cu:269-344
- Two-pass argmax: ter-cuda\ter_cuda_forward_bitnet.cu:449-489
- smem fijo para graphs: packed.cu:118-129
