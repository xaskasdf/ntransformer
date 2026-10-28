import json
from pathlib import Path
import shutil
import subprocess
import sys
import router_run as rr
r=rr.r;ROOT=rr.ROOT

def main():
    if '--prepare' in sys.argv:
        if json.loads((ROOT/'status.json').read_text())['state']!='initial-complete':raise RuntimeError('initial job not complete')
        shutil.copytree(ROOT/'source',ROOT/'initial-source')
        shutil.copy2(ROOT/'environment.json',ROOT/'initial-environment.json')
        # Preserve the initial binary remotely; payload contains its hash.
        shutil.copy2(ROOT/'bench',ROOT/'bench-initial')
        for name in ['ds41_gpu.cu','ds41_gpu.h','ds41_forward.cpp']:
            shutil.copy2(ROOT/'final-input'/name,ROOT/'source/src/model/ds41'/name)
        shutil.copy2(ROOT/'final-input/test_ds41_moe_overlap.cpp',ROOT/'source/tests/test_ds41_moe_overlap.cpp')
        cmake=ROOT/'source/CMakeLists.txt'
        text=cmake.read_text().replace('test_ds41_moe_overlap_flow_control','test_ds41_moe_overlap_flow_experimental').replace('DS41_STAGING_POOL=0 DS41_MXFP4_REGISTER_LUT=0','DS41_STAGING_POOL=1 DS41_MXFP4_REGISTER_LUT=1')
        cmake.write_text(text)
        r.b.command('final-build',[r.PREFIX/'bin/cmake','--build',ROOT/'build','-j8','--target','ntransformer',
            'test_ds41_router','test_ds41_backbone','test_ds41_shared','test_ds41_moe_overlap','test_ds41_prefill'])
        compile_record=json.loads((ROOT/'compile-harness-command.json').read_text())
        r.b.command('final-compile-harness',compile_record['command'])
        r.b.save('environment.json',dict(base_experiment=str(ROOT.parent/'ds41-flow-20261015'),
            source_sha256={str(p.relative_to(ROOT/'source')):r.b.sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()},
            binary_sha256=r.b.sha(ROOT/'bench'),library_sha256=r.b.sha(ROOT/'build/libntransformer_lib.a')))
        r.b.status('final-prepared');return
    with r.b.admitted():
        with (ROOT/'final-telemetry.csv').open('w') as log:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv','--loop-ms=1000'],stdout=log,stderr=subprocess.STDOUT)
            try:
                r.b.ENV=r.DEFAULT.copy()
                r.b.ENV['DS41_TEST_MODEL']='/home/ubuntu/tiny-deepseek41.gguf'
                r.b.command('final-ctest',[r.PREFIX/'bin/ctest','--test-dir',ROOT/'build','-V','-R','test_ds41_(router|shared|backbone|moe_overlap|prefill)'])
                if any(t in (ROOT/'final-ctest.log').read_text() for t in ['FAIL','SKIP','Not Run']):raise RuntimeError('test diagnostic failure')
                def defaults(name,n):
                    r.b.ENV=r.DEFAULT.copy();r.b.ENV['DS41_BENCH_ALL_LOGITS']='1'
                    r.b.command(name,['/usr/bin/time','-v','-o',ROOT/(name+'-resources.txt'),ROOT/'bench',r.b.MODEL,ROOT/'corpus.txt',n,64,3,ROOT/name,name])
                defaults('defaults-p32',32)
                # Reverse configuration order at the longer prompt.
                defaults('defaults-p128',128)
                r.DEFAULT['DS41_ROUTER_GPU']='0';r.case('control-p128',[0]*4,128)
            finally:monitor.terminate();monitor.wait(timeout=10)
    r.b.status('complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:r.b.status('failed',error=repr(e));raise
