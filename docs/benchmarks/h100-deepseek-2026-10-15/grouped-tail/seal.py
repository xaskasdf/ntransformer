import datetime, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
root=Path('/home/ubuntu/ntransformer-h100/ds41-tail-20261015')
host=root.parent
assert json.loads((root/'status.json').read_text())['state']=='complete'
analysis=json.loads((root/'analysis.json').read_text())
for name,expected in analysis['source_sha256'].items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,name
for path in root.glob('*-command.json'):
    assert json.loads(path.read_text())['returncode']==0,path
for name in ['ds41_gpu.cu','ds41_gpu.h']:
    shutil.copy2(host/'ds41-hc-20261015/source/src/model/ds41'/name,root/('before-'+name))
shutil.copy2(host/'ds41-audit-20261015/layout.json',root/'model-layout.json')
environment={}
for name,command in [
    ('cuda',[str(host/'env/bin/nvcc'),'--version']),
    ('compiler',[str(host/'env/bin/x86_64-conda-linux-gnu-g++'),'--version']),
    ('gpu',['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv']),
    ('cpu',['lscpu']),
    ('reference',['git','-C','/home/ubuntu/llama-ds41','rev-parse','HEAD'])]:
    environment[name]=subprocess.check_output(command,text=True)
environment['fixture_sha256']=hashlib.sha256(Path('/home/ubuntu/tiny-deepseek41.gguf').read_bytes()).hexdigest()
environment['binary_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
    ['profile-candidate','build/ntransformer','build/test_ds41_backbone','build/test_ds41_prefill','build/test_ds41_moe_overlap']}
environment['previous_binary_sha256']=hashlib.sha256((host/'ds41-hc-20261015/profile-candidate').read_bytes()).hexdigest()
(root/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
observations=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    stages={p.parent.name:json.loads(p.read_text()) for p in Path('/lambda/nfs/btiny/brandon-mini/stages').glob('*/status.json')},
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
(root/'after-observation.json').write_text(json.dumps(observations,indent=2)+'\n')
with tarfile.open(root/'source.tar','w') as t:t.add(root/'source',arcname='.',recursive=True)
files=sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.cpp','.cu','.h','.json','.log','.bin','.md','.tar'})
files=[p for p in files if p.name not in ['archive.json','local-verification.json']]
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
files.append(root/'SHA256SUMS')
archive=root/'grouped-tail.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in files:t.add(p,arcname=p.name,recursive=False)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
dest=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-tail-20261015')
dest.mkdir(parents=True,exist_ok=True)
for name in ['grouped-tail.tar.gz','SHA256SUMS','RESULTS.md','OPTIMIZATION.md','analysis.json','environment.json']:
    shutil.copy2(root/name,dest/name)
assert hashlib.sha256((dest/archive.name).read_bytes()).hexdigest()==digest
record=dict(sha256=digest,archive=archive.name,files=len(files),bytes=archive.stat().st_size,persistent=str(dest))
for path in [root,dest]:(path/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
