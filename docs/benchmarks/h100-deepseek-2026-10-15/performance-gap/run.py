import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015')
BASE=ROOT.parent/'llama-bench-20261015'
NT=ROOT.parent/'ds41-shared-20261015'
spec=importlib.util.spec_from_file_location('bench',BASE/'run.py'); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.ROOT=ROOT
PREFIX=b.PREFIX; CUDA=PREFIX/'targets/x86_64-linux'; CU13=b.CUBLAS
CUPTI=ROOT/'cupti-13.1/nvidia/cu13'
# Keep the measured llama libraries, with the new isolated output directory.
b.ENV['LD_LIBRARY_PATH']=':'.join(map(str,[CUPTI/'lib',BASE/'build/bin',CUDA/'lib',PREFIX/'lib',CU13/'lib']))
def cmd(name,args,**kwargs):
    result=b.command(name,args,**kwargs)
    path=ROOT/(name+'-command.json')
    record=json.loads(path.read_text())
    record['environment'].update({k:v for k,v in b.ENV.items() if k.startswith('GAP_')})
    path.write_text(json.dumps(record,indent=2)+'\n')
    return result
def main():
    for name in ['32','128']:
        shutil.copy2(BASE/f'workload-p{name}.json',ROOT/f'workload-p{name}.json')
    shutil.copy2(ROOT.parent/'ds41-bench-20261015/corpus.txt',ROOT/'corpus.txt')
    # Only CUPTI headers are exposed, avoiding CUDA 13.0 runtime headers shadowing 13.1.
    headers=ROOT/'cupti131-include';headers.mkdir(exist_ok=True)
    for p in (CUPTI/'include').glob('*.h'):
        if p.name.startswith(('cupti','generated_','Openacc')) or p.name == 'cuda_stdint.h':
            dest=headers/p.name
            if not dest.exists():dest.symlink_to(p)
    compiler=PREFIX/'bin/x86_64-conda-linux-gnu-g++'
    common=[compiler,'-std=c++20','-O3','-march=native','-rdynamic','-I'+str(CUDA/'include'),'-I'+str(headers)]
    cmd('compile-trace',common+['-c',ROOT/'trace.cpp','-o',ROOT/'trace.o'])
    link=[ROOT/'trace.o','-L'+str(CUPTI/'lib'),'-l:libcupti.so.13']
    cmd('compile-nt',common+['-I'+str(NT/'source/src'),'-I'+str(NT/'source/include'),ROOT/'benchmark_ds41.cpp',NT/'build/libntransformer_lib.a',*link,
        '-L'+str(PREFIX/'lib'),'-L'+str(CUDA/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'bench-nt'])
    source=BASE/'source'
    cmd('compile-llama',common+['-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I'+str(source/'vendor'),ROOT/'benchmark_llama_ds41.cpp',*link,
        '-L'+str(BASE/'build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-L'+str(CUDA/'lib/stubs'),'-lcuda','-lpthread','-o',ROOT/'bench-llama'])
    b.save('environment.json',dict(base_environment=json.loads((BASE/'environment.json').read_text()),
        source_sha256={p.name:b.sha(p) for p in ROOT.glob('*.cpp')},binaries={p.name:b.sha(p) for p in ROOT.glob('bench-*')},
        cupti_library=str(CUPTI/'lib/libcupti.so.13'),cupti_sha256=b.sha(CUPTI/'lib/libcupti.so.13')))
    if '--prepare' in __import__('sys').argv: return
    with b.admitted():
        baseline_env=b.ENV.copy()
        def llama(label,n,trace=False,extra=None,repeats=3):
            b.ENV=baseline_env.copy();b.ENV.update(extra or {})
            (ROOT/label).mkdir(exist_ok=True)
            if trace:b.ENV['GAP_TRACE']=str(ROOT/label)
            cmd(label,[ROOT/'bench-llama',b.MODEL,ROOT/f'workload-p{n}.json',ROOT/label,repeats,52,31,512],required=not trace)
        llama('llama-control-p32',32)
        llama('llama-no-graphs-p32',32,extra={'GGML_CUDA_DISABLE_GRAPHS':'1'})
        llama('llama-control-p128',128)
        llama('llama-no-op-offload-p128',128,extra={'GGML_OP_OFFLOAD_MIN_BATCH':'1000000'})
        label='nt-trace-p32';(ROOT/label).mkdir(exist_ok=True)
        b.ENV=baseline_env.copy()
        b.ENV.update(DS41_GPU='1',DS41_GPU_ATTN='1',DS41_VRAM_GB='62',DS41_PROFILE='1',DS41_SHARED_GPU='1',DS41_MOE_BATCH='1',DS41_GROUPED_TAIL='1',DS41_HC_DIRECT='1',DS41_MOE_OVERLAP='1',GAP_TRACE=str(ROOT/label))
        cmd(label,[ROOT/'bench-nt',b.MODEL,ROOT/'corpus.txt',32,64,2,ROOT/label,label],required=False)
        b.ENV=baseline_env.copy()
        llama('llama-trace-p32',32,True,repeats=2)
    b.save('after.json',dict(compute_apps=b.capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])))
    b.save('performance-complete.json',dict(complete=True,trace_returncodes={name:json.loads((ROOT/(name+'-command.json')).read_text())['returncode'] for name in ['nt-trace-p32','llama-trace-p32']}))
    b.status('complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:b.status('failed',error=repr(e));raise

