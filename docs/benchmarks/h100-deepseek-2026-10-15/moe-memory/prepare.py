import shutil
import pilot as p
b, ROOT, PREFIX = p.b, p.ROOT, p.PREFIX
try:
    if not (ROOT / 'initial-source').exists():
        shutil.copytree(ROOT / 'source', ROOT / 'initial-source')
        shutil.copy2(ROOT / 'benchmark_ds41.cpp', ROOT / 'benchmark_pilot.cpp')
    shutil.copy2(ROOT / 'benchmark_ds41.cpp.next', ROOT / 'benchmark_ds41.cpp')
    for name in ['ds41_gpu.cu', 'ds41_gpu.h']:
        shutil.copy2(ROOT / name, ROOT / 'source/src/model/ds41' / name)
    for name in ['test_ds41_mxfp4.cpp', 'test_ds41_moe_overlap.cpp']:
        shutil.copy2(ROOT / name, ROOT / 'source/tests' / name)
    shutil.copy2(ROOT / 'CMakeLists.txt', ROOT / 'source/CMakeLists.txt')
    b.command('configure', [PREFIX / 'bin/cmake', '-S', ROOT / 'source', '-B', ROOT / 'build', '-DNTRANSFORMER_BUILD_NTRPU_SIM=OFF',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-g++'),
        '-DCMAKE_CUDA_HOST_COMPILER=' + str(PREFIX / 'bin/x86_64-conda-linux-gnu-gcc'), '-DCMAKE_CUDA_COMPILER=' + str(PREFIX / 'bin/nvcc')])
    b.command('build', [PREFIX / 'bin/cmake', '--build', ROOT / 'build', '-j8', '--target', 'ntransformer',
        'test_ds41_mxfp4', 'test_ds41_attention', 'test_ds41_router', 'test_ds41_backbone', 'test_ds41_shared',
        'test_ds41_moe_overlap', 'test_ds41_prefill', 'test_ds41_hc'])
    b.command('compile-harness', [PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O3', '-march=native',
        '-I' + str(ROOT / 'source/src'), '-I' + str(ROOT / 'source/include'), ROOT / 'benchmark_ds41.cpp', ROOT / 'build/libntransformer_lib.a',
        '-L' + str(PREFIX / 'lib'), '-L' + str(p.CUDA / 'lib'), '-lcudart', '-lcudadevrt', '-lcudart_static', '-ldl', '-lrt', '-lpthread', '-o', ROOT / 'bench'])
    b.save('environment.json', dict(base_experiment=str(p.BASE),
        gpu=b.capture(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total', '--format=csv']),
        compiler=b.capture([PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '--version']), toolkit=b.capture([PREFIX / 'bin/nvcc', '--version']),
        source_sha256={str(f.relative_to(ROOT / 'source')): b.sha(f) for f in (ROOT / 'source').rglob('*') if f.is_file()},
        binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a')))
    b.status('prepared')
except BaseException as error:
    b.status('failed', error=repr(error)); raise
