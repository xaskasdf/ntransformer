import hashlib, json, statistics, tarfile
from pathlib import Path
workspace=Path(__file__).resolve().parents[2]
dest=workspace/'docs/benchmarks/h100-deepseek-2026-10-15/hc-direct'
archive=workspace/'build/ds41-hc/hc-direct.tar.gz'
record=json.loads((dest/'archive.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['sha256']
with tarfile.open(archive) as t:
    for member in t.getmembers():
        assert member.isfile() and Path(member.name).name==member.name,member.name
        (dest/member.name).write_bytes(t.extractfile(member).read())
verified=[]
for line in (dest/'SHA256SUMS').read_text().splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected,name
    verified.append(name)
analysis=json.loads((dest/'analysis.json').read_text())
for name,expected in analysis['source_sha256'].items():
    assert hashlib.sha256((workspace/name).read_bytes()).hexdigest()==expected,name
data=[(dest/(label+'.bin')).read_bytes() for label in analysis['exact']['runs']]
assert len(data[0])==5*129280*4 and all(d==data[0] for d in data)
assert hashlib.sha256(data[0]).hexdigest()==analysis['exact']['sha256']
assert (dest/'cpu-before.bin').read_bytes()==(dest/'cpu-after.bin').read_bytes()
for phase in ['prefill','decode']:
    means={}
    for arm in ['pool','direct']:
        rows=[r for r in analysis['rows'] if r['phase']==phase and r['round']>=0 and r['arm']==arm]
        assert len(rows)==6 and all(r['expert_hits']==r['expert_lookups']==r['tokens']*720 and r['expert_evictions']==0 for r in rows)
        assert all(r['gpu_calls']==r['tokens']*40 and r['cpu_calls']==0 for r in rows)
        assert all(r['overlapped_calls']==r['gpu_calls'] for r in rows)
        means[arm]=statistics.mean(r['seconds'] for r in rows)
        assert means[arm]==analysis['summary'][phase][arm]['mean_seconds']
    assert means['pool']/means['direct']==analysis['summary'][phase]['speedup']
assert '100% tests passed out of 6' in (dest/'cpu-tests.log').read_text()
assert '100% tests passed out of 1' in (dest/'tiny-prefill.log').read_text()
assert 'PASS HC direct/pool/scalar' in (dest/'hc-final-test.log').read_text()
assert 'PENDING_' not in (dest/'RESULTS.md').read_text()
result=dict(archive_sha256=record['sha256'],verified_payload_files=len(verified),
    current_source_hashes_match=True,cross_binary_logits_exact=True,means_recomputed=True,
    warm_cache_deltas_verified=True,all_routed_calls_gpu=True,all_candidate_calls_overlapped=True)
(dest/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
