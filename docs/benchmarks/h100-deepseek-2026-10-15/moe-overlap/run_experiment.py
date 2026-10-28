from common import *
import hashlib, shutil, tarfile
try:
    with tarfile.open(root/'changes.tar') as t:t.extractall(source,filter='data')
    configure()
    run('build',['cmake','--build',root/'build','--target','ntransformer','test_ds41_moe_overlap',
        'test_ds41_prefill','test_ds41_backbone','-j','2'])
    compile_profile('candidate')
    run('tiny-prefill',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_prefill$'],
        {'DS41_GPU':'0','DS41_TEST_MODEL':'/home/ubuntu/tiny-deepseek41.gguf'})
    with admitted('experiment'):
        run('gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_moe_overlap|test_ds41_backbone'])
        for label,mode in [('serial1','0'),('overlap1','1'),('overlap2','1'),('serial2','0')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_MOE_OVERLAP':mode})
        # Compare with the preceding unmodified CPU-chunk binary too.
        run('previous',[old/'profile-candidate',model,'16','1',root/'previous.bin'])
        # Real-model session state/head schedules with the new default overlap.
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
        names=['probe','previous','serial1','overlap1','overlap2','serial2']
        data=[(root/(name+'.bin')).read_bytes() for name in names]
        assert len(data[0])==5*129280*4 and all(d==data[0] for d in data),'cross-binary logits differ'
        (root/'exact.json').write_text(json.dumps(dict(bitwise_equal=True,logit_vectors=5,
            floats_per_vector=129280,sha256=hashlib.sha256(data[0]).hexdigest(),runs=names),indent=2))
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
