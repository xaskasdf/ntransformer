from common import *
import common,hashlib
common.status=lambda *args,**kwargs:None
for suffix in ['.log','-command.json']:
    (root/('synthetic-max-0'+suffix)).rename(root/('synthetic-max-0-attempt1'+suffix))
with (host/'gpu1-correctness.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=4096
    observation=dict(free_mib=free,note='Correctness only; may overlap Brandon',processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True))
    (root/'small-gpu-observation.json').write_text(json.dumps(observation,indent=2))
    run('gpu-tests',['ctest','--test-dir',root/'build','-V','-R','test_ds41_backbone|test_ds41_moe_overlap'])
    exact={}
    for shape,hidden in [('small',256),('real',5120),('max',8192)]:
        for mode in (['0','1'] if shape=='max' else []):
            label='synthetic-'+shape+'-'+mode
            run(label,[root/'build/test_ds41_moe_overlap',shape,root/(label+'.bin')],{'DS41_VRAM_GB':'2' if shape=='max' else '1','DS41_MOE_BATCH':mode,'DS41_MOE_TRACE':'1'})
        data=(root/('synthetic-'+shape+'-0.bin')).read_bytes()
        assert len(data)==9*hidden*4 and data==(root/('synthetic-'+shape+'-1.bin')).read_bytes()
        exact[shape]=dict(bitwise_equal=True,floats=9*hidden,sha256=hashlib.sha256(data).hexdigest())
    (root/'synthetic-exact.json').write_text(json.dumps(exact,indent=2))
run('tiny-prefill',['ctest','--test-dir',root/'build','-V','-R','^test_ds41_prefill$'],{'DS41_GPU':'0','DS41_TEST_MODEL':'/home/ubuntu/tiny-deepseek41.gguf'})
(root/'prepare-complete.json').write_text(json.dumps(dict(complete=True)))
