from common import *
import hashlib
try:
    with admitted('experiment',wait_seconds=900):
        run('gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_moe_overlap|test_ds41_backbone'])
        for label,mode in [('serial1','0'),('overlap1','1'),('overlap2','1'),('serial2','0')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_MOE_OVERLAP':mode})
        run('previous',[old/'profile-candidate',model,'16','1',root/'previous.bin'])
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
        names=['probe','previous','serial1','overlap1','overlap2','serial2']
        data=[(root/(name+'.bin')).read_bytes() for name in names]
        assert len(data[0])==5*129280*4 and all(d==data[0] for d in data),'cross-binary logits differ'
        (root/'exact.json').write_text(json.dumps(dict(bitwise_equal=True,logit_vectors=5,
            floats_per_vector=129280,sha256=hashlib.sha256(data[0]).hexdigest(),runs=names),indent=2))
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
