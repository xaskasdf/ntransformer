import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import shutil

ROOT=Path('/home/ubuntu/ntransformer-h100/llama-bench-20261015')
HOST=ROOT.parent
PREFIX=HOST/'env'
REFERENCE=Path('/home/ubuntu/llama-ds41')
COMMIT='5210c7c5ed61dddaee6ed476623abf4b63093d16'
ENV=dict(os.environ)
ENV['PATH']=str(PREFIX/'bin')+':'+ENV['PATH']
ENV['LD_LIBRARY_PATH']=str(PREFIX/'targets/x86_64-linux/lib')+':'+str(PREFIX/'lib')
ENV['CUDA_VISIBLE_DEVICES']='1'
CUBLAS=Path('/home/ubuntu/brandon-mini/.venv/lib/python3.12/site-packages/nvidia/cu13')
ENV['LD_LIBRARY_PATH'] += ':'+str(CUBLAS/'lib')

def status(state,**extra):
    (ROOT/'prepare-status.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra),indent=2))

def run(label,args,timeout=1800):
    status('running',case=label)
    args=list(map(str,args))
    if (ROOT/(label+'.log')).exists() and not (ROOT/(label+'-first.log')).exists():
        shutil.copy2(ROOT/(label+'.log'),ROOT/(label+'-first.log'))
    with (ROOT/(label+'.log')).open('w') as log:
        result=subprocess.run(args,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    (ROOT/(label+'-command.json')).write_text(json.dumps(dict(command=args,returncode=result.returncode),indent=2))
    if result.returncode: raise RuntimeError(label+' failed')
    print(label,'PASS',flush=True)

try:
    # Isolate cuBLAS headers: its Python package also contains CUDA 13.0
    # runtime headers, which must not shadow the compiler's 13.1 toolkit.
    cublas_include=ROOT/'cublas-include';cublas_include.mkdir(exist_ok=True)
    for header in (CUBLAS/'include').glob('cublas*.h'):
        link=cublas_include/header.name
        if not link.exists(): link.symlink_to(header)
    assert subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(REFERENCE),'status','--porcelain'],text=True).strip()
    run('source-archive',['git','-C',REFERENCE,'archive',COMMIT,'--format=tar','-o',ROOT/'llama-source.tar'])
    source=ROOT/'source';source.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'llama-source.tar') as archive: archive.extractall(source,filter='data')
    run('configure',['cmake','-S',source,'-B',ROOT/'build','-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_COMPILER='+str(PREFIX/'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_C_COMPILER='+str(PREFIX/'bin/x86_64-conda-linux-gnu-gcc'),
        '-DCMAKE_CUDA_HOST_COMPILER='+str(PREFIX/'bin/x86_64-conda-linux-gnu-gcc'),
        '-DCMAKE_CUDA_COMPILER='+str(PREFIX/'bin/nvcc'),'-DCMAKE_CUDA_ARCHITECTURES=90',
        '-DCUDA_cublas_LIBRARY='+str(CUBLAS/'lib/libcublas.so.13'),
        '-DCUDA_cublasLt_LIBRARY='+str(CUBLAS/'lib/libcublasLt.so.13'),
        '-DCMAKE_CUDA_FLAGS=-I'+str(cublas_include),
        '-DCMAKE_CXX_FLAGS=-I'+str(cublas_include),
        '-DGGML_CUDA=ON','-DGGML_CUDA_NCCL=OFF','-DGGML_NATIVE=ON',
        '-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON'])
    run('build',['cmake','--build',ROOT/'build','--target','llama','llama-bench','-j','4'],timeout=2400)
    status('complete',commit=COMMIT)
except BaseException as error:
    status('failed',error=repr(error));raise
