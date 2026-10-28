import json
from pathlib import Path
import shutil
import subprocess
import sys

FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(FAIR))
import common as b

def main():
    b.verify();b.ROOT=ROOT
    source=ROOT/'hc-source'
    shutil.copytree(FAIR/'nt-source',source)
    for name in ['ds41_forward.cpp','ds41_hc.h']:shutil.copy2(ROOT/name,source/'src/model/ds41'/name)
    shutil.copy2(ROOT/'test_ds41_hc.cpp',source/'tests/test_ds41_hc.cpp')
    compiler=b.PREFIX/'bin/x86_64-conda-linux-gnu-g++'
    flags=['-std=c++20','-O3','-march=native','-fPIC','-I'+str(source/'src'),'-I'+str(source/'include')]
    libraries=['-L'+str(b.PREFIX/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread']
    b.command('compile-hc-test',[compiler,*flags,source/'tests/test_ds41_hc.cpp',FAIR/'libntransformer_lib.a',*libraries,'-o',ROOT/'test-hc-fixed'])
    b.command('test-hc-fixed',[ROOT/'test-hc-fixed'])
    result=subprocess.run([str(ROOT/'test-hc-before')],env=b.ENV,capture_output=True,text=True)
    assert result.returncode!=0 and 'transposes destination and source' in result.stderr
    b.save('regression.json',dict(original_exit=result.returncode,original_diagnostic=result.stderr,corrected_exit=0))
    # Rebuild only the changed C++ unit; retain every GPU kernel and other
    # object from the exact, verified fair-comparison library.
    b.command('compile-hc-forward',[compiler,*flags,'-DNDEBUG','-c',source/'src/model/ds41/ds41_forward.cpp','-o',ROOT/'ds41_forward.cpp.o'])
    library=ROOT/'libntransformer-hc.a';shutil.copy2(FAIR/'libntransformer_lib.a',library)
    b.command('archive-hc',[b.PREFIX/'bin/x86_64-conda-linux-gnu-ar','r',library,ROOT/'ds41_forward.cpp.o'])
    b.command('compile-hc-bench',[compiler,*flags,FAIR/'benchmark_ds41.cpp',library,*libraries,'-o',ROOT/'bench-nt-hc'])
    b.save('hc-build.json',dict(binary=b.sha(ROOT/'bench-nt-hc'),library=b.sha(library),
        baseline_library=b.sha(FAIR/'libntransformer_lib.a'),
        sources={name:b.sha(source/name) for name in ['src/model/ds41/ds41_forward.cpp','src/model/ds41/ds41_hc.h','tests/test_ds41_hc.cpp']},
        changed_object='ds41_forward.cpp.o',gpu_kernels_unchanged=True))
    b.status('prepared-hc')

if __name__=='__main__':main()
