import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
ROOT=Path(__file__).resolve().parent
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-audit-20261016')
sys.path.insert(0,str(FAIR))
import common as b

def main():
    assert json.loads((ROOT/'status.json').read_text())['state']=='complete-hc'
    destination=ROOT/'results';destination.mkdir(exist_ok=False)
    def copy(source,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name in ['control-plan.json','hc-plan.json','hc-build.json','object-validation.json','regression.json',
        'audit-summary.json','status.json','admission.json','telemetry.csv','compute-processes.jsonl',
        'hc-run.log','controls.log','prepare-hc.log','validation.log','hc-before.log',
        'run_controls.py','prepare_hc.py','run_hc.py','analyze_audit.py','validate_hc.py','seal_audit.py','report_audit.py','REPORT.md']:
        copy(ROOT/name,name)
    for directory in ['validation',*['nt-hc-'+w for w in ['p32','p128','p512','heldout-p512']],
        'llama-serial-p32','llama-serial-p512']:
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and source.suffix not in ['.o'] and not source.name.startswith('test-hc-'):
                copy(source,source.relative_to(ROOT))
    for pattern in ['nt-hc-*-command.json','llama-serial-*-command.json','nt-hc-*-resources.txt','nt-hc-*.log','llama-serial-*.log']:
        for source in ROOT.glob(pattern):copy(source,source.name)
    for name in ['src/model/ds41/ds41_hc.h','src/model/ds41/ds41_forward.cpp']:
        copy(ROOT/'hc-source'/name,'source/'+name)
        copy(FAIR/'nt-source'/name,'baseline-source/'+name)
    copy(ROOT/'test_ds41_hc.cpp','source/tests/test_ds41_hc_initial.cpp')
    copy(ROOT/'test_ds41_hc_extended.cpp','source/tests/test_ds41_hc.cpp')
    for name in ['deepseek4.cpp','deepseek41.cpp']:
        copy(FAIR/'llama-source/src/models'/name,'reference/'+name)
    for name in ['admission.json','compute-processes.jsonl','telemetry.csv']:
        copy(PERSIST/'controls'/name,'control-observations/'+name)
    copy(FAIR/'environment.json','baseline-environment.json')
    copy(FAIR/'archive.json','baseline-archive.json')
    # Preserve a precise record of host validation overlapping numerical runs.
    validation=[]
    for source in (ROOT/'validation').glob('*-command.json'):
        record=json.loads(source.read_text());validation.append(dict(name=source.name,**record))
    cases=[]
    for source in ROOT.glob('nt-hc-*-command.json'):
        record=json.loads(source.read_text())
        begin=datetime.datetime.fromisoformat(record['utc_start']).timestamp();end=begin+record['wall_seconds']
        overlap=[]
        for v in validation:
            vb=datetime.datetime.fromisoformat(v['utc_start']).timestamp();ve=vb+v['wall_seconds']
            if max(begin,vb)<min(end,ve):overlap.append(v['name'])
        cases.append(dict(case=source.name,host_validation_overlaps=overlap))
    (destination/'timing-scope.json').write_text(json.dumps(dict(
        scope='Timing diagnostics, not paired performance claims. CPU validation overlaps are disclosed; GPU process monitoring does not detect CPU-only competitors.',
        cases=cases),indent=2)+'\n')
    files=[p for p in sorted(destination.rglob('*')) if p.is_file()]
    manifest={p.relative_to(destination).as_posix():b.sha(p) for p in files}
    (destination/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(ROOT/'audit-results.tar.gz','w:gz') as archive:
        for p in [*files,destination/'sha256.json']:archive.add(p,arcname=p.relative_to(destination),recursive=False)
    info=dict(sha256=b.sha(ROOT/'audit-results.tar.gz'),bytes=(ROOT/'audit-results.tar.gz').stat().st_size,payload_files=len(files))
    (destination/'archive.json').write_text(json.dumps(info,indent=2)+'\n')
    shutil.copytree(destination,PERSIST/'results',dirs_exist_ok=True)
    for name,digest in manifest.items():assert b.sha(PERSIST/'results'/name)==digest
    shutil.copy2(ROOT/'audit-results.tar.gz',PERSIST/'audit-results.tar.gz')
    assert b.sha(PERSIST/'audit-results.tar.gz')==info['sha256']
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
