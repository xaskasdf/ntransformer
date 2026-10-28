from common import *
import hashlib
try:
    shutil_status = root/'status.json'
    if shutil_status.exists():
        (root/'initial-admission-status.json').write_text(shutil_status.read_text())
    run('hc-final-test',[root/'build/test_ds41_hc'])
    with admitted('experiment',wait_seconds=1200):
        run('microbench',[root/'build/test_ds41_hc','--benchmark'])
        for label,mode in [('pool1','0'),('direct1','1'),('direct2','1'),('pool2','0')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_HC_DIRECT':mode})
        run('previous',[old/'profile-candidate',model,'16','1',root/'previous.bin'])
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
        names=['previous','pool1','direct1','direct2','pool2']
        data=[(root/(name+'.bin')).read_bytes() for name in names]
        assert len(data[0])==5*129280*4 and all(d==data[0] for d in data),'cross-binary logits differ'
        (root/'exact.json').write_text(json.dumps(dict(bitwise_equal=True,logit_vectors=5,
            floats_per_vector=129280,sha256=hashlib.sha256(data[0]).hexdigest(),runs=names),indent=2))
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
