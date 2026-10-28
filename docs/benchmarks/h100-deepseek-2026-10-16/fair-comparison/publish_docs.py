import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/benchmarks/h100-deepseek-2026-10-16/fair-comparison'
START='<!-- ds41-fair-comparison -->'
END='<!-- /ds41-fair-comparison -->'

def main():
    s=json.loads((DATA/'summary.json').read_text())
    proof=json.loads((DATA/'local-verification.json').read_text())
    lines=[START,'## Matched two-H100 comparison','',
        '[Verified matched report](benchmarks/h100-deepseek-2026-10-16/fair-comparison/REPORT.md). Both engines use the same two H100s and fixed corpus tokens: two independent launches with reversed engine order, two warmups and three measured requests per launch. Native execution/precision is preserved. The bounded llama pilot selects a balanced layer placement, CPU thread count and operation-offload setting before the main cases. The NT SLRU path is opt-in; LRU is measured separately at context 512.', '',
        '| Workload | NT decode tok/s | llama decode tok/s | llama / NT | NT prefill tok/s | llama prefill tok/s | Warm major faults NT / llama |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for w,cases in s['workloads'].items():
        nt,ll=cases['nt-slru'],cases['llama'];m=s['comparisons'][w]
        lines.append(f"| {w} | {nt['decode_tokens_per_second']:.4f} | {ll['decode_tokens_per_second']:.4f} | {m['llama_over_nt_decode']:.3f}x | {nt['prefill_tokens_per_second']:.4f} | {ll['prefill_tokens_per_second']:.4f} | {nt['warm_major_page_faults']} / {ll['warm_major_page_faults']} |")
    lines+=['','The report includes paired launch ratios, target NLL, distribution differences, repeat stability, VRAM/RSS and per-phase page-fault/I/O counters. Equal tokens and hardware do not establish equivalent task quality. Initial-request timings remain diagnostic because OS file-cache state is not reset; any residual warm paging is explicitly flagged.', '',
        f"{proof['payload_files_verified']} payload hashes verified locally; analysis reproduced with 1e-12 floating-point tolerance. Archive SHA-256: `{proof['archive_sha256']}`. Persistent copy: `/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016`.",END]
    path=ROOT/'docs/DEEPSEEK_PERFORMANCE_GAP.md';text=path.read_text(encoding='utf-8');section='\n'.join(lines)
    if START in text:
        begin=text.index(START);finish=text.index(END,begin)+len(END);text=text[:begin]+section+text[finish:]
    else:text=text.rstrip()+'\n\n'+section+'\n'
    path.write_text(text,encoding='utf-8',newline='\n')
    (DATA/'STATUS.md').write_text('# Completed\n\nResults downloaded, payload hashes verified and analysis reproduced locally. See [REPORT.md](REPORT.md).\n',encoding='utf-8',newline='\n')
    destination='/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016/repository-docs'
    subprocess.run(['ssh','ubuntu@<h100-host>','mkdir -p '+destination],check=True,timeout=30)
    subprocess.run(['scp','-q',str(path),'ubuntu@<h100-host>:'+destination+'/'],check=True,timeout=90)

if __name__=='__main__':main()
