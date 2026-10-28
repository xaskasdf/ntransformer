import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
HOST=ROOT.parent
BASE=HOST/'ds41-shared-20261015'
spec=importlib.util.spec_from_file_location('bench',HOST/'llama-bench-20261015/run.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.ROOT=ROOT
PREFIX=b.PREFIX
CUDA=PREFIX/'targets/x86_64-linux'
b.ENV['LD_LIBRARY_PATH']=':'.join(map(str,[CUDA/'lib',PREFIX/'lib']))
b.ENV.update(DS41_GPU='1',DS41_GPU_ATTN='1',DS41_PROFILE='1',DS41_VRAM_GB='62',
    DS41_SHARED_GPU='1',DS41_MOE_BATCH='1',DS41_GROUPED_TAIL='1',DS41_HC_DIRECT='1',DS41_MOE_OVERLAP='1')
FLAGS=['DS41_STAGING_POOL','DS41_SHARED_EARLY','DS41_QKV_READBACK_BATCH','DS41_MXFP4_REGISTER_LUT']
DEFAULT=b.ENV.copy()

def prepare():
    source=ROOT/'source'
    if source.exists():raise RuntimeError('isolated source already exists')
    shutil.copytree(BASE/'source',source)
    overlay={
        'ds41_gpu.cu':'src/model/ds41/ds41_gpu.cu',
        'ds41_gpu.h':'src/model/ds41/ds41_gpu.h',
        'test_ds41_backbone.cpp':'tests/test_ds41_backbone.cpp',
        'test_ds41_moe_overlap.cpp':'tests/test_ds41_moe_overlap.cpp',
        'test_ds41_shared.cpp':'tests/test_ds41_shared.cpp'}
    for name,dest in overlay.items():shutil.copy2(ROOT/name,source/dest)
    # Same three automated control variants as the local CMake edit, without
    # copying unrelated workspace changes into the validated remote snapshot.
    with (source/'CMakeLists.txt').open('a') as f:
        f.write('''
add_test(NAME test_ds41_backbone_prologue_serial COMMAND ${CMAKE_COMMAND} -E env DS41_QKV_READBACK_BATCH=0 $<TARGET_FILE:test_ds41_backbone> prologue)
add_test(NAME test_ds41_moe_overlap_flow_control COMMAND ${CMAKE_COMMAND} -E env DS41_STAGING_POOL=0 DS41_MXFP4_REGISTER_LUT=0 DS41_VRAM_GB=1 $<TARGET_FILE:test_ds41_moe_overlap>)
add_test(NAME test_ds41_shared_late COMMAND ${CMAKE_COMMAND} -E env DS41_SHARED_EARLY=0 DS41_VRAM_GB=2 $<TARGET_FILE:test_ds41_shared>)
''')
    b.command('configure',[PREFIX/'bin/cmake','-S',source,'-B',ROOT/'build','-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+str(PREFIX/'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER='+str(PREFIX/'bin/x86_64-conda-linux-gnu-gcc'),'-DCMAKE_CUDA_COMPILER='+str(PREFIX/'bin/nvcc')])
    b.command('build',[PREFIX/'bin/cmake','--build',ROOT/'build','-j8','--target',
        'ntransformer','test_ds41_backbone','test_ds41_shared','test_ds41_moe_overlap'])
    b.command('compile-harness',[PREFIX/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
        '-I'+str(source/'src'),'-I'+str(source/'include'),ROOT/'benchmark_ds41.cpp',ROOT/'build/libntransformer_lib.a',
        '-L'+str(PREFIX/'lib'),'-L'+str(CUDA/'lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread','-o',ROOT/'bench'])
    b.save('environment.json',dict(gpu=b.capture(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv']),
        compiler=b.capture([PREFIX/'bin/x86_64-conda-linux-gnu-g++','--version']),toolkit=b.capture([PREFIX/'bin/nvcc','--version']),
        source_sha256={str(p.relative_to(source)):b.sha(p) for p in source.rglob('*') if p.is_file()},
        binary_sha256=b.sha(ROOT/'bench'),base_library_sha256=b.sha(BASE/'build/libntransformer_lib.a'),library_sha256=b.sha(ROOT/'build/libntransformer_lib.a')))
    shutil.copy2(HOST/'ds41-bench-20261015/corpus.txt',ROOT/'corpus.txt')

def env(flags=None,**extra):
    b.ENV=DEFAULT.copy();b.ENV.update(dict(zip(FLAGS,map(str,flags or [1]*4))));b.ENV.update(extra)

def tests():
    env()
    b.command('ctest',[PREFIX/'bin/ctest','--test-dir',ROOT/'build','-V','-R','test_ds41_(shared|backbone|moe_overlap)'])
    log=(ROOT/'ctest.log').read_text()
    if 'FAIL' in log or 'SKIP' in log or 'Not Run' in log:raise RuntimeError('test diagnostic failure')
    pairs=[]
    for shape in ['small','real','max']:
        for variant,flags in [('control',[0]*4),('candidate',[1]*4)]:
            env(flags,DS41_VRAM_GB='2',DS41_MOE_TRACE='1')
            name=f'moe-{shape}-{variant}'
            b.command(name,[ROOT/'build/test_ds41_moe_overlap',shape,ROOT/(name+'.f32')])
        pairs.append((f'moe-{shape}-control.f32',f'moe-{shape}-candidate.f32'))
    for variant,flags in [('control',[0]*4),('candidate',[1]*4)]:
        env(flags,DS41_VRAM_GB='2')
        b.command('prologue-'+variant,[ROOT/'build/test_ds41_backbone','prologue',ROOT/('prologue-'+variant+'.f32')])
        b.command('shared-real-'+variant,[ROOT/'build/test_ds41_shared','real',ROOT/('shared-real-'+variant+'.f32')])
    pairs += [('prologue-control.f32','prologue-candidate.f32'),('shared-real-control.f32','shared-real-candidate.f32')]
    results={a:dict(candidate=c,bytes=(ROOT/a).stat().st_size,bitwise_equal=(ROOT/a).read_bytes()==(ROOT/c).read_bytes()) for a,c in pairs}
    b.save('test-comparison.json',results)
    if not all(r['bitwise_equal'] for r in results.values()):raise RuntimeError('numerical difference requires review')

def case(name,flags,n=32,repeats=3):
    env(flags,DS41_BENCH_ALL_LOGITS='1')
    b.command(name,['/usr/bin/time','-v','-o',ROOT/(name+'-resources.txt'),ROOT/'bench',b.MODEL,ROOT/'corpus.txt',n,64,repeats,ROOT/name,name])

def main():
    if '--prepare' in sys.argv:prepare();b.status('prepared');return
    with b.admitted():
        with (ROOT/'telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            try:
                tests()
                case('control-p32',[0,0,0,0])
                case('candidate-p32',[1,1,1,1])
                case('lut-only-p32',[0,0,0,1])
            finally:monitor.terminate();monitor.wait(timeout=10)
    b.status('initial-complete')

if __name__=='__main__':
    try:main()
    except BaseException as e:b.status('failed',error=repr(e));raise
