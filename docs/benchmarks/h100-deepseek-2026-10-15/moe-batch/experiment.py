from common import *
import hashlib
try:
    # The original diagnostic process has its own bounded admission wait.
    # Do not compete with it or replace its records while it is active.
    deadline=time.monotonic()+1200
    while Path('/proc/2481759/cmdline').exists() and b'trace_baseline.py' in Path('/proc/2481759/cmdline').read_bytes():
        if time.monotonic()>deadline:raise TimeoutError('original trace process still active')
        time.sleep(5)
    run('cpu-cross-binary',['/home/ubuntu/brandon-mini/.venv/bin/python',root/'cpu_checks.py'])
    with admitted('experiment',wait_seconds=2400):
        if not (root/'trace-baseline-command.json').exists():
            run('trace-baseline',[root/'profile-trace-baseline',model,'16','2',root/'trace-baseline.bin'],{'DS41_MOE_TRACE':'1'})
        for label,mode in [('serial1','0'),('batched1','1'),('batched2','1'),('serial2','0')]:
            run(label,[root/'profile-candidate',model,'16','3',root/(label+'.bin')],{'DS41_MOE_BATCH':mode,'DS41_MOE_TRACE':'0'})
        run('trace-batched',[root/'profile-candidate',model,'16','2',root/'trace-batched.bin'],{'DS41_MOE_BATCH':'1','DS41_MOE_TRACE':'1'})
        run('previous',[old/'profile-candidate',model,'16','1',root/'previous.bin'])
        run('real-state',[root/'build/test_ds41_prefill',model,'16','0'])
        names=['previous','trace-baseline','trace-batched','serial1','batched1','batched2','serial2']
        data=[(root/(name+'.bin')).read_bytes() for name in names]
        assert len(data[0])==5*129280*4 and all(d==data[0] for d in data),'cross-binary logits differ'
        (root/'exact.json').write_text(json.dumps(dict(bitwise_equal=True,logit_vectors=5,
            floats_per_vector=129280,sha256=hashlib.sha256(data[0]).hexdigest(),runs=names),indent=2))
    status('complete')
except Exception as e:status('failed',error=repr(e));raise
