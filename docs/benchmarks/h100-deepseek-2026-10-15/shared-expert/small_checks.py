from common import *
import common
common.status=lambda *args,**kwargs:None
run('compile-probe',[prefix/'bin/x86_64-conda-linux-gnu-g++','-std=c++20','-O3','-march=native',
    '-I'+str(source/'src'),'-I'+str(source/'include'),root/'probe.cpp',root/'build/libntransformer_lib.a',
    '-L'+str(prefix/'lib'),'-L'+str(prefix/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static',
    '-ldl','-lrt','-lpthread','-o',root/'probe'])
with (host/'gpu1-correctness.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=4096
    (root/'small-observation.json').write_text(json.dumps(dict(free_mib=free,note='Correctness only, concurrent Brandon permitted',
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)),indent=2))
    for shape in ['small','real','max']:
        run('small-check-'+shape,[root/'build/test_ds41_shared',shape],{'DS41_VRAM_GB':'2'})
