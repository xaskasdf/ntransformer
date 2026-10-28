from common import *
import common
common.status=lambda *a,**k:None
run('build-test-diagnostic',['cmake','--build',root/'build','--target','test_ds41_shared','-j','4'])
with (host/'gpu1-correctness.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run('diagnostic-max',[root/'build/test_ds41_shared','max'],{'DS41_VRAM_GB':'2'})
