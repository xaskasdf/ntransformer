import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ds41-attention-20261015'
spec = importlib.util.spec_from_file_location('common', ROOT.parent / 'llama-bench-20261015/run.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
b.ROOT = ROOT
PREFIX = b.PREFIX
CUDA = PREFIX / 'targets/x86_64-linux'
b.ENV['LD_LIBRARY_PATH'] = ':'.join(map(str, [CUDA / 'lib', PREFIX / 'lib']))
b.ENV.update(DS41_GPU='1', DS41_GPU_ATTN='1', DS41_PROFILE='1', DS41_VRAM_GB='62', DS41_BENCH_ALL_LOGITS='1')
DEFAULT = b.ENV.copy()

def main():
    source = ROOT / 'source'
    if not source.exists(): shutil.copytree(BASE / 'source', source)
    shutil.copy2(BASE / 'corpus.txt', ROOT / 'corpus.txt')
    b.command('compile-pilot', [PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O3', '-march=native',
        '-I' + str(source / 'src'), '-I' + str(source / 'include'), ROOT / 'benchmark_ds41.cpp', BASE / 'build/libntransformer_lib.a',
        '-L' + str(PREFIX / 'lib'), '-L' + str(CUDA / 'lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread', '-o', ROOT / 'bench-pilot'])
    b.save('pilot-environment.json', dict(base=str(BASE), binary_sha256=b.sha(ROOT / 'bench-pilot'),
        harness_sha256=b.sha(ROOT / 'benchmark_ds41.cpp'), library_sha256=b.sha(BASE / 'build/libntransformer_lib.a')))
    with b.admitted():
        with (ROOT / 'pilot-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for name, flags in [('pilot-control', {}), ('pilot-trace', {'DS41_MOE_TRACE': '1'})]:
                    b.ENV = DEFAULT | flags
                    b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench-pilot',
                        b.MODEL, ROOT / 'corpus.txt', 32, 64, 3, ROOT / name, name])
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('pilot-complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        b.status('failed', error=repr(error)); raise
