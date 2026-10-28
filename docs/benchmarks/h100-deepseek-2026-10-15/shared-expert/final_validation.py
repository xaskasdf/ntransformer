from common import *
import shutil,runpy,hashlib
try:
    assert json.loads((root/'status.json').read_text())['state']=='complete'
    # Preserve the complete first measurement and its numerical diagnostics.
    names=['gpu-tests','shared-real','shared-max','cpu1','cpu2','gpu1','gpu2','trace-cpu','trace-gpu','previous','real-state','probe-gpu','compare']
    for name in names:
        for suffix in ['.log','.bin','-command.json']:
            path=root/(name+suffix)
            if path.exists():shutil.copy2(path,root/('initial-'+path.name))
    for name in ['analysis.json','numerics.json']:
        shutil.copy2(root/name,root/('initial-'+name))
    run('build-final',['cmake','--build',root/'build','--target','ntransformer','test_ds41_shared','test_ds41_prefill','test_ds41_backbone','test_ds41_moe_overlap','-j','4'])
    compile_profile('candidate')
    runpy.run_path(str(root/'compile_probe.py'),run_name='__main__')
    with admitted('final-validation',wait_seconds=1200):
        run('gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_shared|test_ds41_backbone|test_ds41_moe_overlap'])
        for shape in ['real','max']:
            run('shared-'+shape,[root/'build/test_ds41_shared',shape,root/('shared-'+shape+'.bin')],{'DS41_VRAM_GB':'2'})
        for label,mode in [('gpu1','1'),('cpu1','0'),('cpu2','0'),('gpu2','1')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_SHARED_GPU':mode,'DS41_MOE_TRACE':'0'})
        for label,mode in [('trace-gpu','1'),('trace-cpu','0')]:
            run(label,[root/'profile-candidate',model,'16','2',root/(label+'.bin')],{'DS41_SHARED_GPU':mode,'DS41_MOE_TRACE':'1'})
        # Previous binary and CPU context probe are unchanged independent controls.
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
        run('probe-gpu',[root/'probe',model,root/'probe-gpu.bin'],{'DS41_SHARED_GPU':'1','DS41_MOE_TRACE':'0'})
        for name in ['gpu1','cpu1','probe-gpu']:
            assert (root/(name+'.bin')).read_bytes()==(root/('initial-'+name+'.bin')).read_bytes(),name
    run('compare',['/home/ubuntu/brandon-mini/.venv/bin/python',root/'compare.py'])
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
