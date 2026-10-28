import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')

def main():
    s=json.loads((ROOT/'audit-summary.json').read_text())
    assert all('corrected_vs_llama' in row for row in s.values())
    lines=['# DeepSeek HC layout correction and prefill audit','',
        'The large cross-engine distribution discrepancy was mainly caused by a destination/source transpose in NTransformer hyper-connection mixing. The correction changes the directed residual connections to match the pinned llama.cpp graph. It retains the model weights, FP32 routed arithmetic, GPU kernels, P2P expert execution and cache configuration.','',
        '## Cause and correction','',
        'Sinkhorn internally used row-major `matrix[dst * hc + src]`. Its final flat copy populated `comb`, but the consumer reads the GGUF/ggml layout `comb[dst + src * hc]`. This applied the transposed mixing matrix. Both orientations retain the same row/column sums, so those checks and direct-versus-thread-pool equality did not detect the error. The corrected producer explicitly exports destination-fast storage.','',
        'The intended operation is `out[dst, d] = post[dst] * branch[d] + sum_src C[dst, src] * residual[src, d]`. The producer and consumer now agree on how `C` is stored.','',
        'Reference: `reference/deepseek4.cpp`, `build_hc_sinkhorn` and `build_hc_post`, from pinned fork commit `5210c7c5ed61dddaee6ed476623abf4b63093d16`. Only `ds41_forward.cpp.o` differs between the baseline and corrected static libraries; `object-validation.json` verifies all archive members.','',
        '## Same-token numerical results','',
        'Each workload uses the exact fair-benchmark prompt and 64 continuation tokens, with 65 full-vocabulary prediction vectors. Corrected runs use one independent process and five requests; every request is finite and repeats bitwise. The rows cover 260 prediction positions from two corpora, not a downstream task-quality certification. KL is NTransformer to native batched llama.cpp.','',
        '| Workload | Old NT NLL | Corrected NT NLL | llama NLL | Old KL | Corrected KL | Corrected argmax agreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for w,row in s.items():
        old=row['old_vs_llama'];new=row['corrected_vs_llama']
        lines.append(f"| {w} | {old['nll_a']:.6f} | {new['nll_a']:.6f} | {new['nll_b']:.6f} | {old['kl']:.6f} | {new['kl']:.6f} | {100*new['argmax_agreement']:.2f}% |")
    lines+=['','The residual differences remain measurable. The serial-versus-batched llama control itself changes distributions; that places the remaining discrepancy in context but does not prove all residual error is rounding or establish equal task quality.','',
        '## Prefill control and remaining work','',
        'The same llama binary, two-GPU placement, 52 CPU threads and disabled operation-offload setting were rerun with ubatch 1 instead of 512. The control uses one warmup and one measured request per workload; historical native numbers use six warm requests. This isolates an execution-path change but is a diagnostic rather than a counterbalanced speedup estimate. Warm control phases recorded zero major page faults and input blocks.','',
        '| Workload | NT original prefill s | llama native prefill s | llama ubatch-1 prefill s | llama native-to-serial KL | Corrected NT-to-serial KL |',
        '|---|---:|---:|---:|---:|---:|']
    for w in ['p32','p512']:
        medians={}
        for engine in ['nt-slru','llama']:
            seconds=[]
            for launch in [0,1]:
                rows=[json.loads(l) for l in (FAIR/f'{engine}-{w}-r{launch}'/'measurements.jsonl').read_text().splitlines()]
                seconds += [r['seconds'] for r in rows if r.get('round') in [2,3,4] and r.get('phase')=='prefill']
            medians[engine]=statistics.median(seconds)
        row=s[w]
        lines.append(f"| {w} | {medians['nt-slru']:.6f} | {medians['llama']:.6f} | {row['serial_times']['prefill']:.6f} | {row['llama_native_vs_serial']['kl']:.6f} | {row['corrected_vs_llama_serial']['kl']:.6f} |")
    lines+=['','NTransformer prefill still calls `Session::forward` once per prompt token. Each token traverses all layers. The native llama graph performs layer operations over multiple prompt tokens. The control confirms that batching contributes substantially to the prefill gap, while llama also retains an advantage on the single-token path.','',
        'At p512 in the original fair run, median warm host intervals for launch 0 were 9.952 s attention, 7.428 s MoE, 1.404 s Engram, 2.152 s HC and 4.366 s other, out of 25.307 s total. These are host intervals containing submissions/waits; nested stage times must not be summed with their parents or treated as GPU busy time. The HC top-level interval covers only the attention-side HC work; FFN-side HC work is accounted under other.','',
        'A p512 warm prefill executes 20,480 QKV readback waits, 40,960 HC projections and 20,480 router projections. QKV alone returns 140 MiB of intermediate data to the CPU. The code also synchronizes when returning dense projection outputs and the attention tail. Small transfer volume does not remove the cost of these repeated dependencies.','',
        'The next implementation should provide a layer-wise prompt path with a bounded token tile, dense GEMMs and experts grouped by owner GPU. Keep HC/norm/router/attention intermediates on the device; precompute token-history-dependent Engram row IDs once per prompt and stage those rows asynchronously. Preserve P2P execution and explicit expert-block reservation. Causal compressed-state publication and the delayed HC pre-mix must be checked against the corrected serial path before benchmarking. This audit does not implement that prefill path.','',
        '## Validation and timing scope','',
        'The asymmetric stochastic-matrix regression fails with the former export and passes with the correction. Release and ASan/UBSan runs pass four directed layout cases, 36 independent double-precision Sinkhorn cases and 84 existing direct/pool cases. The guarded outputs and asymmetric basis residuals test producer-to-consumer behavior, including production epsilon and 20 iterations.','',
        'Corrected phase times are retained in audit-summary.json as diagnostics. They are not a paired speedup claim: the correction uses one process per workload, and CPU-only validation ran concurrently with part of the p512 numerical case. timing-scope.json records overlapping commands. GPU monitoring found no foreign GPU processes; that monitor cannot exclude CPU-only interference. No kernel or precision change was made.','',
        'Historical cache traces, predicted tokens and precision-selector calibration used the pre-correction graph. They remain historical measurements, not correctness references for the corrected model; recompute them before accepting new cache or arithmetic policies.','',
        'Artifacts include inputs by reference to the locally preserved fair benchmark, full logits, raw per-round measurements, command environments, before/after sources, test results and GPU telemetry. The local collector verifies every payload hash and reproduces audit-summary.json from the full logits.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
