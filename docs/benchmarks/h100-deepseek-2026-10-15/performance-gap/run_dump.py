import importlib.util
import json
from pathlib import Path
import time
ROOT=Path('/home/ubuntu/ntransformer-h100/ds41-gap-20261015')
spec=importlib.util.spec_from_file_location('gap',ROOT/'run.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
b=g.b
def main():
    # Do not compete with performance probes during compilation or tracing.
    deadline=time.monotonic()+1800
    while '--prepare' not in __import__('sys').argv and not (ROOT/'performance-complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('performance probes')
        time.sleep(5)
    b.status('preparing-numerical')
    compiler=b.PREFIX/'bin/x86_64-conda-linux-gnu-g++'
    include=['-I'+str(g.NT/'source/src'),'-I'+str(g.NT/'source/include'),'-I'+str(g.NT/'source/src/model/ds41')]
    common=[compiler,'-std=gnu++20','-O3','-DNDEBUG','-march=native','-fPIC']
    g.cmd('compile-dump-forward',common+include+['-c',ROOT/'ds41_forward_dump.cpp','-o',ROOT/'forward-dump.o'])
    g.cmd('compile-dump-nt',common+include+[ROOT/'nt_dump.cpp',ROOT/'forward-dump.o',g.NT/'build/libntransformer_lib.a',
        '-L'+str(b.PREFIX/'lib'),'-L'+str(g.CUDA/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'dump-nt'])
    g.cmd('compile-dump-nt-control',common+include+[ROOT/'nt_dump.cpp',g.NT/'build/libntransformer_lib.a',
        '-L'+str(b.PREFIX/'lib'),'-L'+str(g.CUDA/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'dump-nt-control'])
    s=g.BASE/'source'
    g.cmd('compile-dump-llama',common+['-I'+str(s/'include'),'-I'+str(s/'ggml/include'),'-I'+str(s/'vendor'),ROOT/'llama_dump.cpp',
        '-L'+str(g.BASE/'build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-L'+str(g.CUDA/'lib/stubs'),'-lcuda','-lpthread','-o',ROOT/'dump-llama'])
    if '--prepare' in __import__('sys').argv: return
    w=json.loads((ROOT/'workload-p32.json').read_text());token=w['prompt_ids'][0]
    b.save('workload-single.json',dict(prompt_ids=[token],continuation_ids=[w['prompt_ids'][1]],expected_prediction_ids=[0,0]))
    with b.admitted():
        env=b.ENV.copy()
        (ROOT/'nt-stages').mkdir(exist_ok=True)
        b.ENV.update(DS41_GPU='1',DS41_GPU_ATTN='1',DS41_VRAM_GB='62',DS41_PROFILE='1',DS41_SHARED_GPU='1',DS41_MOE_BATCH='1',DS41_GROUPED_TAIL='1',DS41_HC_DIRECT='1',DS41_MOE_OVERLAP='1',GAP_DUMP=str(ROOT/'nt-stages'))
        g.cmd('nt-stages',[ROOT/'dump-nt',b.MODEL,token])
        (ROOT/'nt-single-control').mkdir(exist_ok=True);b.ENV['GAP_DUMP']=str(ROOT/'nt-single-control')
        g.cmd('nt-single-control',[ROOT/'dump-nt-control',b.MODEL,token])
        b.ENV=env.copy();b.ENV['GAP_DUMP']=str(ROOT/'llama-stages');(ROOT/'llama-stages').mkdir(exist_ok=True)
        g.cmd('llama-stages',[ROOT/'dump-llama',b.MODEL,ROOT/'workload-single.json',ROOT/'llama-single',1,52,31,1])
        b.ENV=env.copy()
        g.cmd('llama-single-control',[ROOT/'bench-llama',b.MODEL,ROOT/'workload-single.json',ROOT/'llama-single-control',1,52,31,1])
    b.save('after-numerical.json',dict(compute_apps=b.capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])))
    b.status('complete',numerical=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:b.status('failed',error=repr(e));raise
