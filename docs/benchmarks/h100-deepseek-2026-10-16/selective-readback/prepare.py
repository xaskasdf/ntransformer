import json
from pathlib import Path
import shutil
import sys

FAIR = Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
AUDIT = FAIR.parent / 'ds41-audit-20261016'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FAIR))
import common as b

def main():
    b.verify(); b.ROOT = ROOT
    source = ROOT / 'source'
    link_only = '--link-only' in sys.argv
    if not link_only:
        shutil.copytree(AUDIT / 'hc-source', source)
        for name in ['ds41_forward.cpp', 'ds41_gpu.cu', 'ds41_gpu.h']:
            shutil.copy2(ROOT / name, source / 'src/model/ds41' / name)
        shutil.copy2(ROOT / 'test_ds41_backbone.cpp', source / 'tests/test_ds41_backbone.cpp')
    shutil.copy2(FAIR / 'benchmark_resources.h', ROOT / 'benchmark_resources.h')
    compiler = b.PREFIX / 'bin/x86_64-conda-linux-gnu-g++'
    flags = ['-std=c++20', '-O3', '-march=native', '-fPIC', '-I' + str(source / 'src'), '-I' + str(source / 'include')]
    libraries = ['-L' + str(b.PREFIX / 'targets/x86_64-linux/lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread']
    cuda_flags = ['-std=c++20', '-O3', '-DNDEBUG', '-Xcompiler=-march=native', '-Xcompiler=-fPIC', '--use_fast_math']
    for arch in [80, 86, 89, 90]:
        cuda_flags += [f'--generate-code=arch=compute_{arch},code=[compute_{arch},sm_{arch}]']
    library = ROOT / 'libntransformer-readback.a'
    if not link_only:
        b.command('compile-gpu', [b.PREFIX / 'bin/nvcc', '-ccbin', b.PREFIX / 'bin/x86_64-conda-linux-gnu-gcc',
            *cuda_flags, '-I' + str(source / 'src'), '-I' + str(source / 'include'), '-c', source / 'src/model/ds41/ds41_gpu.cu', '-o', ROOT / 'ds41_gpu.cu.o'])
        b.command('compile-forward', [compiler, *flags, '-DNDEBUG', '-c', source / 'src/model/ds41/ds41_forward.cpp', '-o', ROOT / 'ds41_forward.cpp.o'])
        shutil.copy2(AUDIT / 'libntransformer-hc.a', library)
        b.command('archive', [b.PREFIX / 'bin/x86_64-conda-linux-gnu-ar', 'r', library, ROOT / 'ds41_forward.cpp.o', ROOT / 'ds41_gpu.cu.o'])
    b.command('compile-bench-v2', [compiler, *flags, ROOT / 'benchmark_ds41.cpp', library, *libraries, '-o', ROOT / 'bench-nt'])
    for name in ['backbone', 'attention', 'hc']:
        b.command('compile-test-' + name, [compiler, *flags, '-I' + str(b.PREFIX / 'targets/x86_64-linux/include'),
            source / f'tests/test_ds41_{name}.cpp', library, *libraries, '-o', ROOT / ('test-' + name)])
    b.save('build.json', dict(binary=b.sha(ROOT / 'bench-nt'), library=b.sha(library),
        baseline_library=b.sha(AUDIT / 'libntransformer-hc.a'),
        sources={str(p.relative_to(ROOT)): b.sha(p) for p in [source / 'src/model/ds41/ds41_gpu.cu',
            source / 'src/model/ds41/ds41_gpu.h', source / 'src/model/ds41/ds41_forward.cpp',
            source / 'src/model/ds41/ds41_hc.h', source / 'tests/test_ds41_backbone.cpp', ROOT / 'benchmark_ds41.cpp']}))
    b.status('built')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.ROOT = ROOT; b.status('build-failed', error=repr(error)); raise
