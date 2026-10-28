from common import *
import shutil, tarfile
try:
    shutil.copytree(old/'source',source,dirs_exist_ok=True)
    with tarfile.open(root/'changes.tar') as t:t.extractall(source,filter='data')
    configure()
    run('build-trace',['cmake','--build',root/'build','--target','test_ds41_moe_overlap','-j','4'])
    compile_profile('trace-baseline')
    with admitted('trace-baseline',wait_seconds=600):
        run('trace-baseline',[root/'profile-trace-baseline',model,'16','2',root/'trace-baseline.bin'],{'DS41_MOE_TRACE':'1'})
    status('trace-baseline-complete')
except Exception as e:status('failed',error=repr(e));raise
