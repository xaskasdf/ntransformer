# DeepSeek hyper-connection projection experiment

One physical H100 (GPU 1), the same 10 GGUF shards, FP32 activations, prompt token IDs and 62 GiB expert cache. Each case runs one initial request and three warm repeats. Reported rates divide total warm work by total warm time; the initial request, including cold weight uploads, is excluded. No CUPTI tracing.

| Case | Prefill tok/s | Decode tok/s | Decode ratio | All 65 logit vectors match control |
|---|---:|---:|---:|---|
| control-p128 | 6.6515 | 6.6806 | 1.0000 | True |
| control-p32 | 6.5607 | 6.7508 | 1.0000 | True |
| cooperative-p32 | 7.4297 | 7.6830 | 1.1381 | True |
| defaults-p128 | 7.6151 | 7.6372 | 1.1432 | True |
| defaults-p32 | 7.4424 | 7.6908 | 1.1392 | True |
| gpu-p32 | 6.8518 | 7.0618 | 1.0461 | True |
| serial-p32 | 7.0541 | 7.2856 | 1.0792 | True |

| Case | HC norm ms/token | HC projection ms/token | HC mix ms/token | GPU / pool / serial calls per token | Sampled peak GPU GiB |
|---|---:|---:|---:|---|---:|
| control-p128 | 2.2205 | 23.4212 | 0.2461 | 0 / 80 / 0 | 70.6376953125 |
| control-p32 | 2.1948 | 22.9637 | 0.2439 | 0 / 80 / 0 | 70.6376953125 |
| cooperative-p32 | 1.5909 | 5.3258 | 0.1865 | 80 / 0 / 0 | 70.7919921875 |
| defaults-p128 | 1.7017 | 5.4022 | 0.1908 | 80 / 0 / 0 | 70.7919921875 |
| defaults-p32 | 1.5954 | 5.3398 | 0.1836 | 80 / 0 / 0 | 70.7919921875 |
| gpu-p32 | 1.5859 | 17.1623 | 0.1945 | 80 / 0 / 0 | 70.7919921875 |
| serial-p32 | 1.5233 | 11.5144 | 0.2198 | 0 / 0 / 80 | 70.6376953125 |

## Placement and controls

`DS41_HC_PROJECTION_GPU=1` keeps the 80 HC projection matrices resident on GPU and computes their matrix-vector products there. The actual matrices are F32 [20,480 inputs, 24 outputs], totaling 157,286,400 bytes (150 MiB), in addition to the preceding router implementation. No new activation or weight quantization is introduced. GPU allocation/shape/type failures retain the CPU fallback.

`DS41_HC_PROJECTION_GPU=0` selects the previous CPU implementation. `DS41_HC_PROJECTION_SERIAL=1` also bypasses its worker pool and directly calls the same row-dot routine; this is a diagnostic control. The GPU switch takes precedence when both are enabled. Each command JSON records its complete DS41 environment.

The first GPU implementation (`gpu-p32`) reuses the precise BF16/F32 primitive with two FP32 accumulators per AVX2 lane and the original reduction order, one eight-thread block per row. The cooperative implementation instead uses 256 threads to stage weight and input tiles of 2,048 values into 16 KiB of shared memory per block. The same eight lanes then extend the same FMA chains across tiles. Barriers protect each tile before reuse. This changes memory scheduling, without reassociating the sums; the independent scalar-order path remains covered.

Final defaults are `DS41_HC_PROJECTION_GPU=1` and `DS41_HC_COOPERATIVE=1`. `DS41_HC_PROJECTION_SERIAL` remains off. Setting the GPU switch to zero restores CPU projection; setting the cooperative switch to zero restores the first GPU kernel. Cooperative staging applies only to projections with at most 32 rows and at least 2,048 columns; larger routers keep their previous kernel.

Moving this operation alone adds 80 input uploads, 80 tiny output readbacks and 80 host completion waits per token. Logical input/output payload is 6,561,280 bytes/token (6.257 MiB), excluding the initial weight upload. Normalization, sigmoid/Sinkhorn coefficients, collapse and residual mixing remain on CPU. This is a measured placement step toward GPU-resident state, not elimination of the CPU boundary.

## Validation and interpretation

The initial router/HC test checks 60 F32/BF16 projections against AVX2 and independent scalar-order references. The final test expands this to 72, adding a 4,115-column partial-tile case, and exercises both cooperative and direct kernels. Shapes include the actual HC matrix, tails, changing inputs, scratch-buffer growth/reuse, disabled CUDA behavior and bounds. CTest also covers shared experts, backbone projections, MoE overlap, HC collapse/mix and causal prefill/reset using an external tiny fixture. Inspect ctest logs for the actual pass count.

The initial three cases use `initial-source/` and `initial-environment.json`; `cooperative-p32` uses `cooperative-source/` and `cooperative-environment.json`. The `defaults-*` cases and `control-p128` use the final `source/` and `environment.json`. The last change selects the validated defaults, with unchanged arithmetic. Exact build commands, environments and binary/library hashes are retained for each stage.

The benchmark checks finite logits and within-process repeat identity. Full-output hashes compare the prefill prediction plus all 64 decode predictions (8,403,200 FP32 values per case). `summary.json` also compares them with the previous resident-router baseline. This is a regression check for these inputs, not a general quality evaluation; identical logits are useful evidence, not a prerequisite for future optimizations.

HC norm/projection/mix are nested host wall intervals across both sublayers. They overlap the existing top-level HC/other totals and must not be added to them. They include CPU submission and waits, not isolated device kernel time. Memory peaks are one-second telemetry samples, not allocator high-water marks. Three warm requests in one process do not establish independent process-level variance.

Raw logs, outputs, telemetry, source/build hashes and the standalone analysis accompany this report. Model weights are external and are not bundled.
