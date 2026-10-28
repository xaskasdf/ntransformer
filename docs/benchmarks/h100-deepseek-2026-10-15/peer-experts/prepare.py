import importlib.util
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'ds41-moe-pipeline-20261015'
spec = importlib.util.spec_from_file_location('common', ROOT.parent / 'llama-bench-20261015/run.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
b.ROOT = ROOT
PREFIX = b.PREFIX
CUDA = PREFIX / 'targets/x86_64-linux'
b.ENV['LD_LIBRARY_PATH'] = ':'.join(map(str, [CUDA / 'lib', PREFIX / 'lib']))
b.ENV.update(CUDA_VISIBLE_DEVICES='1,0', DS41_GPU='1', DS41_GPU_ATTN='1', DS41_PROFILE='1',
             DS41_VRAM_GB='62', DS41_PEER_VRAM_GB='62', DS41_BENCH_ALL_LOGITS='1', DS41_BENCH_TEACHER_FORCING='1')
DEFAULT = b.ENV.copy()

def main():
    if not (ROOT / 'source').exists(): shutil.copytree(BASE / 'source', ROOT / 'source')
    shutil.copy2(BASE / 'corpus.txt', ROOT / 'corpus.txt')
    for rel in ['src/model/ds41/ds41_gpu.cu', 'src/model/ds41/ds41_gpu.h', 'tests/test_ds41_mxfp4.cpp',
                'tests/test_ds41_moe_overlap.cpp', 'CMakeLists.txt', 'tools/benchmark_ds41.cpp']:
        target = ROOT / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / Path(rel).name, target)
    b.command('configure', [PREFIX / 'bin/cmake', '-S', ROOT / 'source', '-B', ROOT / 'build',
        '-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-gcc'), '-DCMAKE_CUDA_COMPILER=' + str(PREFIX / 'bin/nvcc')])
    b.command('build', [PREFIX / 'bin/cmake', '--build', ROOT / 'build', '-j8', '--target', 'ntransformer',
        'test_ds41_mxfp4', 'test_ds41_attention', 'test_ds41_router', 'test_ds41_backbone', 'test_ds41_shared',
        'test_ds41_moe_overlap', 'test_ds41_prefill', 'test_ds41_hc'])
    b.command('compile-harness', [PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O3', '-march=native',
        '-I' + str(ROOT / 'source/src'), '-I' + str(ROOT / 'source/include'), ROOT / 'benchmark_ds41.cpp', ROOT / 'build/libntransformer_lib.a',
        '-L' + str(PREFIX / 'lib'), '-L' + str(CUDA / 'lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread', '-o', ROOT / 'bench'])
    b.save('environment.json', dict(base=str(BASE),
        gpu=b.capture(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv']),
        topology=b.capture(['nvidia-smi', 'topo', '-m']),
        compiler=b.capture([PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '--version']), toolkit=b.capture([PREFIX / 'bin/nvcc', '--version']),
        source_sha256={str(f.relative_to(ROOT / 'source')): b.sha(f) for f in (ROOT / 'source').rglob('*')
                       if f.is_file() and '__pycache__' not in f.parts},
        binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a')))
    b.status('prepared')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise
