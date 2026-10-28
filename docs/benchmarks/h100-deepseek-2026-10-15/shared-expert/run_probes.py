from common import *
try:
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    run('build-strict-tests',['cmake','--build',root/'build','--target','test_ds41_shared','-j','4'])
    with admitted('context-probes',wait_seconds=1200):
        run('strict-shared-tests',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_shared'])
        for shape in ['real','max']:
            run('strict-'+shape,[root/'build/test_ds41_shared',shape],{'DS41_VRAM_GB':'2'})
        run('exp32',[root/'profile-candidate-exp32',model,'16','1',root/'exp32.bin'],{'DS41_SHARED_GPU':'1','DS41_MOE_TRACE':'0'})
        for label,mode in [('probe-cpu','0'),('probe-gpu','1')]:
            run(label,[root/'probe',model,root/(label+'.bin')],{'DS41_SHARED_GPU':mode,'DS41_MOE_TRACE':'0'},timeout=600)
    run('compare',['/home/ubuntu/brandon-mini/.venv/bin/python',root/'compare.py'])
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
