# Small correctness checks are permitted alongside Brandon. No speed claims.
from common import *
import common
common.status=lambda *args, **kwargs: None
try:
    with (host/'gpu1-correctness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<4096:raise BlockingIOError('less than 4 GiB free on GPU1')
        observation=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_mib=free,
            note='Correctness only, concurrent Brandon work permitted; no throughput measurement',
            processes_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
        run('small-gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_moe_overlap|test_ds41_backbone'],timeout=120)
        observation['processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
        (root/'small-gpu-observation.json').write_text(json.dumps(observation,indent=2))
except Exception as e:
    (root/'small-gpu-failure.json').write_text(json.dumps(dict(error=repr(e))));raise
