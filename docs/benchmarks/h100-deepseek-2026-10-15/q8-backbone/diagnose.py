import fcntl
import prepare as p
b, ROOT = p.b, p.ROOT
# The user permits small correctness probes on a free GPU alongside Brandon.
# This run is diagnostic only; run.py repeats it after full admission.
with (ROOT.parent / 'gpu1-correctness.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    b.ENV = p.DEFAULT | dict(CUDA_VISIBLE_DEVICES='1')
    apps = b.capture(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'])
    assert 'GPU-67e3ec29-9bbc-da80-baf1-8f982193ee6c' not in apps
    # Keep the queue's status.json separate from this temporary diagnostic.
    b.status = lambda *args, **kwargs: None
    b.command('q8-diagnostic', [ROOT / 'build/test_ds41_q8', 'bench'])
