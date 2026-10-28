from common import *
import common
common.status=lambda *a,**k:None
run('build-exp64',['cmake','--build',root/'build','--target','ntransformer','test_ds41_shared','test_ds41_prefill','test_ds41_backbone','test_ds41_moe_overlap','-j','4'])
compile_profile('candidate')
with (host/'gpu1-correctness.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for shape in ['small','real','max']:
        run('exp64-check-'+shape,[root/'build/test_ds41_shared',shape],{'DS41_VRAM_GB':'2'})
