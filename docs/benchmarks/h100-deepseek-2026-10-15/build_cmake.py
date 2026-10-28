import fcntl
import json
import os
from pathlib import Path
import subprocess
import tarfile

root = Path('/home/ubuntu/ntransformer-h100/ds41-audit-20261015')
source = root/'cmake-source'
source.mkdir(exist_ok=True)
with tarfile.open(root/'cmake-source.tar') as archive:
    archive.extractall(source, filter='data')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', DS41_GPU='1')
prefix = '/home/ubuntu/ntransformer-h100/env'
env['PATH'] = prefix+'/bin:'+env['PATH']
env['LD_LIBRARY_PATH'] = prefix+'/targets/x86_64-linux/lib:'+prefix+'/lib'
commands = [
    ['cmake', '-S', str(source), '-B', str(root/'cmake-build'),
     '-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF', '-DCMAKE_BUILD_TYPE=Release',
     '-DCMAKE_CXX_COMPILER='+prefix+'/bin/x86_64-conda-linux-gnu-g++',
     '-DCMAKE_CUDA_HOST_COMPILER='+prefix+'/bin/x86_64-conda-linux-gnu-gcc',
     '-DCMAKE_CUDA_COMPILER='+prefix+'/bin/nvcc'],
    ['cmake','--build',str(root/'cmake-build'),'--target','test_ds41_backbone','-j','2'],
    ['ctest','--test-dir',str(root/'cmake-build'),'-V','-R','test_ds41_backbone']]
results=[]
with open('/home/ubuntu/ntransformer-h100/gpu1-correctness.lock','a') as lock:
    for name,args in zip(['configure','cmake-build','ctest'], commands):
        if name == 'ctest':
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root/(name+'.log')).open('w') as log:
            p = subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        results.append(dict(name=name,command=args,returncode=p.returncode))
        (root/'cmake-checks.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,p.returncode,flush=True)
        if p.returncode:
            print((root/(name+'.log')).read_text()[-6000:])
            raise SystemExit(p.returncode)
print((root/'ctest.log').read_text())
