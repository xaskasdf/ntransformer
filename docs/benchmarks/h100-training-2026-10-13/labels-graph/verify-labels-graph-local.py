import collections
import difflib
import gzip
import hashlib
import json
import sys
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents
            if (p/'scripts/analyze_training_trace.py').is_file()
            and (p/'python/ntransformer_training').is_dir())
dest = root/'docs/benchmarks/h100-training-2026-10-13/labels-graph'
sys.path.insert(0,str(root/'scripts'))
from analyze_training_trace import analyze

records,kernel_counts = {},{}
for name in ('strict-control','strict-labels','strict-graph'):
    for rank in range(2):
        folder = dest/('profile-'+name)
        trace = folder/f'trace-rank{rank}.json.gz'
        actual = analyze(trace)
        expected = json.loads((folder/f'trace-summary-rank{rank}.json').read_text())
        actual.pop('trace');expected.pop('trace')
        assert actual == expected,(name,rank)
        with gzip.open(trace,'rt',encoding='utf-8') as handle:
            events = json.load(handle)['traceEvents']
        kernels = collections.Counter(e['name'] for e in events if e.get('cat') == 'kernel')
        label = f'{name}/rank{rank}'
        kernel_counts[label] = kernels
        launches = {n:d['calls'] for n,d in actual['runtime'].items() if 'Launch' in n}
        records[label] = dict(summary_exact=True,kernel_count=sum(kernels.values()),
            cpu_operations=actual['categories']['cpu_op'],host_launches=launches,
            graph_launches=launches.get('cudaGraphLaunch',0),
            matrix_kernel_names={n:c for n,c in kernels.items() if any(s in n.lower() for s in ('gemm','xmma','nvjet_','cublas'))},
            loss_kernels={n:c for n,c in kernels.items() if 'target_log_probability' in n})
        print(label,json.dumps({k:v for k,v in records[label].items() if k not in ('matrix_kernel_names','loss_kernels')}),flush=True)
differences = {}
for a,b in (('strict-control','strict-labels'),('strict-labels','strict-graph')):
    for rank in range(2):
        ca,cb = kernel_counts[f'{a}/rank{rank}'],kernel_counts[f'{b}/rank{rank}']
        differences[f'{a}-to-{b}/rank{rank}'] = {n:cb[n]-ca[n] for n in sorted(ca.keys()|cb.keys()) if ca[n] != cb[n]}
manifest=json.loads((dest/'SHA256SUMS.json').read_text())
for name,expected in manifest.items():
    with (dest/name).open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest() == expected,name
sources=[]
unrelated_build_changes={}
for p in (dest/'source-snapshot').rglob('*'):
    relative = p.relative_to(dest/'source-snapshot')
    if p.is_file() and not str(relative).startswith('docs'):
        if p.name == 'CMakeLists.txt' and p.read_bytes() != (root/relative).read_bytes():
            diff=list(difflib.unified_diff(p.read_text().splitlines(),(root/relative).read_text().splitlines(),
                                          fromfile='executed',tofile='current'))
            changes=[line for line in diff if line.startswith(('+','-')) and not line.startswith(('+++','---'))]
            assert changes and all(line.startswith('+    src/model/ds41/') for line in changes),diff
            unrelated_build_changes[str(relative)]=diff
            continue
        assert p.read_bytes() == (root/relative).read_bytes(),str(relative)
        sources.append(relative.as_posix())
result=dict(payload_files_verified=len(manifest),executed_sources_exact=sources,
            unrelated_current_build_changes=unrelated_build_changes,
            trace_reanalysis=records,kernel_differences=differences)
(dest/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n')

results=json.loads((dest/'results.json').read_text())
aggregates={}
for policy in ('strict','normal'):
    groups={kind:[results[f'{policy}-{kind}{suffix}']['tokens_per_second'] for suffix in ('','-repeat')]
            for kind in ('control','labels','graph')}
    means={kind:sum(values)/len(values) for kind,values in groups.items()}
    aggregates[policy]=dict(samples=groups,means=means,
        ratios={kind:means[kind]/means['control'] for kind in ('labels','graph')},
        graph_vs_labels=means['graph']/means['labels'],
        paired_graph_gains=[g/c-1 for g,c in zip(groups['graph'],groups['control'])],
        paired_label_gains=[g/c-1 for g,c in zip(groups['labels'],groups['control'])])
aggregates['peer_ratio']=results['peer-labels']['tokens_per_second']/results['peer-control']['tokens_per_second']
(dest/'aggregates.json').write_text(json.dumps(aggregates,indent=2)+'\n')
print(json.dumps(aggregates,indent=2),flush=True)
