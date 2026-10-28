from common import *
import shutil,tarfile,hashlib
try:
    shutil.copytree(old/'source',source,dirs_exist_ok=True)
    with tarfile.open(root/'changes.tar') as t:t.extractall(source,filter='data')
    configure()
    run('build',['cmake','--build',root/'build','--target','ntransformer','test_ds41_shared','test_ds41_backbone','test_ds41_moe_overlap','test_ds41_prefill','-j','4'])
    compile_profile('candidate')
    run('tiny-prefill',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_prefill$'],{'DS41_GPU':'0','DS41_TEST_MODEL':'/home/ubuntu/tiny-deepseek41.gguf'})
    run('cpu-cross-binary',['/home/ubuntu/brandon-mini/.venv/bin/python',root/'cpu_checks.py'])
    with admitted('experiment',wait_seconds=1200):
        run('gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_shared|test_ds41_backbone|test_ds41_moe_overlap'])
        for shape in ['real','max']:
            run('shared-'+shape,[root/'build/test_ds41_shared',shape,root/('shared-'+shape+'.bin')],{'DS41_VRAM_GB':'2'})
        for label,mode in [('cpu1','0'),('gpu1','1'),('gpu2','1'),('cpu2','0')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_SHARED_GPU':mode,'DS41_MOE_TRACE':'0'})
        for label,mode in [('trace-cpu','0'),('trace-gpu','1')]:
            run(label,[root/'profile-candidate',model,'16','2',root/(label+'.bin')],{'DS41_SHARED_GPU':mode,'DS41_MOE_TRACE':'1'})
        run('previous',[old/'profile-candidate',model,'16','1',root/'previous.bin'])
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
