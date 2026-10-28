import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ds41-hc-gpu-20261015'
spec = importlib.util.spec_from_file_location('bench', ROOT.parent / 'llama-bench-20261015/run.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
b.ROOT = ROOT
PREFIX = b.PREFIX; CUDA = PREFIX / 'targets/x86_64-linux'
b.ENV['LD_LIBRARY_PATH'] = ':'.join(map(str, [CUDA / 'lib', PREFIX / 'lib']))
b.ENV.update(DS41_GPU='1', DS41_GPU_ATTN='1', DS41_PROFILE='1', DS41_VRAM_GB='62')
DEFAULT = b.ENV.copy()

def prepare():
    source = ROOT / 'source'
    if not source.exists():
        shutil.copytree(BASE / 'source', source)
    for name in ['ds41_gpu.cu', 'ds41_gpu.h', 'ds41_attention.cu', 'ds41_attention.h', 'ds41_forward.cpp', 'ds41_forward.h']:
        shutil.copy2(ROOT / name, source / 'src/model/ds41' / name)
    for name in ['test_ds41_attention.cpp', 'test_ds41_backbone.cpp']:
        shutil.copy2(ROOT / name, source / 'tests' / name)
    shutil.copy2(ROOT / 'CMakeLists.txt', source / 'CMakeLists.txt')
    b.command('configure', [PREFIX / 'bin/cmake', '-S', source, '-B', ROOT / 'build', '-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-gcc'), '-DCMAKE_CUDA_COMPILER=' + str(PREFIX / 'bin/nvcc')])
    b.command('build', [PREFIX / 'bin/cmake', '--build', ROOT / 'build', '-j8', '--target', 'ntransformer',
                       'test_ds41_attention', 'test_ds41_router', 'test_ds41_backbone', 'test_ds41_shared',
                       'test_ds41_moe_overlap', 'test_ds41_prefill', 'test_ds41_hc'])
    b.command('compile-harness', [PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O3', '-march=native',
        '-I' + str(source / 'src'), '-I' + str(source / 'include'), ROOT / 'benchmark_ds41.cpp', ROOT / 'build/libntransformer_lib.a',
        '-L' + str(PREFIX / 'lib'), '-L' + str(CUDA / 'lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread', '-o', ROOT / 'bench'])
    b.save('environment.json', dict(base_experiment=str(BASE),
        gpu=b.capture(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv']),
        compiler=b.capture([PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '--version']), toolkit=b.capture([PREFIX / 'bin/nvcc', '--version']),
        source_sha256={str(p.relative_to(source)): b.sha(p) for p in source.rglob('*') if p.is_file()},
        binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a')))
    shutil.copy2(BASE / 'corpus.txt', ROOT / 'corpus.txt')
    b.status('prepared')

def case(name, resident, n=32):
    b.ENV = DEFAULT.copy()
    b.ENV.update(DS41_ATTENTION_GPU=str(resident), DS41_BENCH_ALL_LOGITS='1')
    b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
                     b.MODEL, ROOT / 'corpus.txt', n, 64, 3, ROOT / name, name])

def main():
    if '--prepare' in sys.argv:
        prepare(); return
    with b.admitted():
        with (ROOT / 'telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                b.ENV = DEFAULT.copy()
                b.ENV.update(DS41_ATTENTION_GPU='1', DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf')
                b.command('ctest', [PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
                                    'test_ds41_(attention|router|shared|backbone|moe_overlap|prefill|hc)'])
                assert not any(t in (ROOT / 'ctest.log').read_text() for t in ['FAIL', 'SKIP', 'Not Run'])
                case('control-p32', 0)
                case('resident-p32', 1)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('initial-complete')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        b.status('failed', error=repr(e)); raise
