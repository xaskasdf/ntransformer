import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

root=Path('/home/ubuntu/ntransformer-h100')
roots=[root/'training/operational-20261013',root/'training/operational-ablation-20261013']
python='/home/ubuntu/brandon-mini/.venv/bin/python'
for out in roots:
    assert json.loads((out/'status.json').read_text())['state']=='complete'

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

subprocess.run([python,str(root/'source/scripts/compare_training_experiment.py'),
    '--reference',str(roots[0]/'native-default'),'--candidate',str(roots[1]/'combined-default'),
    '--source-checkpoint','/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt',
    '--output',str(roots[1]/'combined-default-comparison.json')],check=True)
sys.path.insert(0,str(root/'source/scripts'))
from analyze_training_trace import analyze

for out in roots:
    cases=json.loads((out/'cases.json').read_text())
    results={}
    for label,mode,flags in cases:
        results[label]=json.loads((out/label/'summary.json').read_text())
        results[label]['checkpoint_sha256']=sha(out/label/'last.pt')
        audit=out/(label+'-exact.json')
        if audit.exists():
            results[label]['audit']=json.loads(audit.read_text())
            assert results[label]['audit']['status']=='passed'
    for trace in sorted(out.glob('profile-*/trace-rank*.json')):
        data=analyze(trace)
        trace.with_name(trace.stem.replace('trace-','trace-summary-')+'.json').write_text(json.dumps(data,indent=2)+'\n')
        with trace.open('rb') as source, gzip.open(str(trace)+'.gz','wb',compresslevel=1) as target:
            shutil.copyfileobj(source,target)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(
        library_sha256=sha(root/'build-training/libntransformer_training.so'),
        source_checkpoint='/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/pretrain/last.pt',
        throughput_scope='100 updates; first 10 excluded; continuous window with audit/logging',
        profile_scope='Separate runs; offsets 11 and 12 only; excluded from throughput claims'),indent=2)+'\n')
    shutil.copy2(root/'source/scripts/analyze_training_trace.py',out/'analyze_training_trace.py')
    shutil.copy2(__file__,out/'seal-operational-review.py')
    portable=[path for path in sorted(out.rglob('*')) if path.is_file()
              and path.suffix in ('.json','.jsonl','.log','.py','.cu','.txt','.gz','.yaml','.md')
              and path.name not in ('results.tar.gz','archive_ready.json','SHA256SUMS.json')
              and not (path.name.startswith('trace-rank') and path.suffix=='.json')]
    (out/'SHA256SUMS.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in portable},indent=2)+'\n')
    with tarfile.open(out/'results.tar.gz','w:gz') as archive:
        for path in [*portable,out/'SHA256SUMS.json']:
            archive.add(path,arcname=path.relative_to(out).as_posix())
    ready=dict(archive_sha256=sha(out/'results.tar.gz'),files=len(portable)+1)
    (out/'archive_ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    print(out.name,ready,flush=True)
