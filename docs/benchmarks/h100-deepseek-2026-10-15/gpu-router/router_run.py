import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
FLOW=ROOT.parent/'ds41-flow-20261015'
spec=importlib.util.spec_from_file_location('flow_run',FLOW/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.ROOT=ROOT;r.b.ROOT=ROOT

def main():
    source=ROOT/'source'
    if '--prepare' in sys.argv:
        if source.exists():raise RuntimeError('source already exists')
        shutil.copytree(FLOW/'source',source)
        for name in ['ds41_gpu.cu','ds41_gpu.h','ds41_shared.cu','ds41_shared.h','ds41_forward.cpp']:
            shutil.copy2(ROOT/name,source/'src/model/ds41'/name)
        shutil.copy2(ROOT/'test_ds41_router.cpp',source/'tests/test_ds41_router.cpp')
        with (source/'CMakeLists.txt').open('a') as f:f.write('''
add_executable(test_ds41_router tests/test_ds41_router.cpp)
target_link_libraries(test_ds41_router PRIVATE ntransformer_lib)
add_test(NAME test_ds41_router COMMAND test_ds41_router)
add_test(NAME test_ds41_router_disabled COMMAND ${CMAKE_COMMAND} -E env DS41_GPU=0 $<TARGET_FILE:test_ds41_router> disabled)
''')
        r.b.command('configure',[r.PREFIX/'bin/cmake','-S',source,'-B',ROOT/'build','-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+str(r.PREFIX/'bin/x86_64-conda-linux-gnu-g++'),
            '-DCMAKE_CUDA_HOST_COMPILER='+str(r.PREFIX/'bin/x86_64-conda-linux-gnu-gcc'),'-DCMAKE_CUDA_COMPILER='+str(r.PREFIX/'bin/nvcc')])
        r.b.command('build',[r.PREFIX/'bin/cmake','--build',ROOT/'build','-j8','--target','ntransformer','test_ds41_router',
            'test_ds41_backbone','test_ds41_shared','test_ds41_moe_overlap'])
        r.b.command('compile-harness',[r.PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
            '-I'+str(source/'src'),'-I'+str(source/'include'),ROOT/'benchmark_ds41.cpp',ROOT/'build/libntransformer_lib.a',
            '-L'+str(r.PREFIX/'lib'),'-L'+str(r.CUDA/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'bench'])
        r.b.save('environment.json',dict(base_experiment=str(FLOW),
            source_sha256={str(p.relative_to(source)):r.b.sha(p) for p in source.rglob('*') if p.is_file()},
            binary_sha256=r.b.sha(ROOT/'bench'),library_sha256=r.b.sha(ROOT/'build/libntransformer_lib.a')))
        shutil.copy2(FLOW/'corpus.txt',ROOT/'corpus.txt')
        r.b.status('prepared');return
    with r.b.admitted():
        with (ROOT/'telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            try:
                r.env()
                r.b.command('ctest',[r.PREFIX/'bin/ctest','--test-dir',ROOT/'build','-V','-R','test_ds41_(router|shared|backbone|moe_overlap)'])
                if any(t in (ROOT/'ctest.log').read_text() for t in ['FAIL','SKIP','Not Run']):raise RuntimeError('test diagnostic failure')
                r.DEFAULT['DS41_ROUTER_GPU']='0';r.case('control-p32',[0]*4)
                r.DEFAULT['DS41_ROUTER_GPU']='1';r.case('router-p32',[0]*4)
                r.case('router-schedule-p32',[0,1,1,0])
            finally:monitor.terminate();monitor.wait(timeout=10)
    r.b.status('initial-complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:r.b.status('failed',error=repr(e));raise
