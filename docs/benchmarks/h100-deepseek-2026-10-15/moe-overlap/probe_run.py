from common import *
import shutil, tarfile
try:
    shutil.copytree(old/'source',source)
    with tarfile.open(root/'instrumentation.tar') as t:t.extractall(source,filter='data')
    configure()
    run('build-probe',['cmake','--build',root/'build','--target','ntransformer_lib','-j','2'])
    compile_profile('probe')
    with admitted('probe'):
        run('probe',[root/'profile-probe',model,'16','2',root/'probe.bin'])
    status('probe-complete')
except Exception as e:status('failed',error=repr(e));raise
