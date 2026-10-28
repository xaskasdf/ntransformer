import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import common as b

ROOT=b.ROOT
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016')

def main():
    assert not (ROOT/'queue-status.json').exists(), 'queue already initialized'
    assert json.loads((ROOT/'status.json').read_text())['state']=='prepared'
    for name,digest in json.loads((ROOT/'expected-upload.json').read_text()).items():
        assert b.sha(ROOT/name)==digest,name
    files=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix in ['.py','.cpp','.h'] or p.name in [
        'plan.json','environment.json','cpu-validation.json','model-identity.json','nt-engine-environment.json',
        'nt-engine-cpu-validation.json','nt-engine-ctest.log','analysis-validation.log','llama-source.tar',
        'bench-nt','bench-llama','libntransformer_lib.a']) or p.is_file() and p.name.startswith('workload-')]
    files += [p for p in (ROOT/'nt-source').rglob('*') if p.is_file()]
    files += [p for p in (ROOT/'llama-lib').glob('*') if p.is_file()]
    frozen={p.relative_to(ROOT).as_posix():b.sha(p) for p in files}
    b.save('frozen-inputs.json',frozen);b.verify()
    with tarfile.open(ROOT/'preparation.tar.gz','w:gz') as archive:
        for name in [*frozen,'frozen-inputs.json']:
            archive.add(ROOT/name,arcname=name,recursive=False)
    info=dict(sha256=b.sha(ROOT/'preparation.tar.gz'),bytes=(ROOT/'preparation.tar.gz').stat().st_size,
        frozen_files=len(frozen),scope='Runnable preparation, CPU placement/analysis validation only; GPU benchmark pending')
    b.save('preparation.json',info)
    PERSIST.mkdir(parents=True,exist_ok=True)
    for name in ['preparation.tar.gz','preparation.json','frozen-inputs.json','plan.json','cpu-validation.json','environment.json']:
        shutil.copy2(ROOT/name,PERSIST/name);assert b.sha(ROOT/name)==b.sha(PERSIST/name)
    with (ROOT/'queue.log').open('w') as log:
        process=subprocess.Popen([str(b.PYTHON),str(ROOT/'queue.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    b.save('launch.json',dict(pid=process.pid,preparation=info))
    print(json.dumps(dict(pid=process.pid,preparation=info)),flush=True)

if __name__=='__main__':main()
