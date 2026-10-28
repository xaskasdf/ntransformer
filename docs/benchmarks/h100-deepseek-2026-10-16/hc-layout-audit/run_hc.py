import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(FAIR))
import common as b
from admission import admitted
import run as fair_run

def main():
    b.verify();b.ROOT=ROOT;fair_run.ROOT=ROOT
    build=json.loads((ROOT/'hc-build.json').read_text())
    assert b.sha(ROOT/'bench-nt-hc')==build['binary']
    ar=b.PREFIX/'bin/x86_64-conda-linux-gnu-ar'
    old,new=FAIR/'libntransformer_lib.a',ROOT/'libntransformer-hc.a'
    members=b.capture([ar,'t',old]).splitlines()
    assert members==b.capture([ar,'t',new]).splitlines()
    changed=[]
    for name in members:
        before=subprocess.check_output([str(ar),'p',str(old),name])
        after=subprocess.check_output([str(ar),'p',str(new),name])
        if before!=after:changed.append(name)
    assert changed==['ds41_forward.cpp.o'],changed
    b.save('object-validation.json',dict(objects=len(members),changed=changed))
    b.save('hc-plan.json',dict(workloads=['p32','p128','p512','heldout-p512'],
        independent_launches=1,repeats=4,measured_rounds=[2,3,4],
        scope='Numerical correction with timing diagnostics against prior matched baseline; not paired speedup proof',
        change='HC mixing matrix exported with destination as the fastest dimension',
        tokens='Identical corpus inputs and 64 teacher-forced continuation tokens from fair benchmark',
        arithmetic='Unchanged FP32 routed arithmetic; unchanged GPU kernels'))
    with admitted(b),fair_run.monitoring():
        b.ENV=b.BASE_ENV | dict(DS41_GPU='1',DS41_GPU_ATTN='1',DS41_PROFILE='1',
            DS41_Q8_COOPERATIVE='1',DS41_PEER_MODE='execute',DS41_MXFP4_COMPUTE='fp32',
            DS41_VRAM_GB='62',DS41_PEER_VRAM_GB='62',DS41_BENCH_ALL_LOGITS='1',
            DS41_BENCH_TEACHER_FORCING='1',DS41_CACHE_POLICY='slru',DS41_CACHE_PROTECTED_PERCENT='80',DS41_CACHE_TRACE='0')
        for w in ['p32','p128','p512','heldout-p512']:
            workload=json.loads((FAIR/('workload-'+w+'.json')).read_text());label='nt-hc-'+w
            b.command(label,['/usr/bin/time','-v','-o',ROOT/(label+'-resources.txt'),ROOT/'bench-nt-hc',b.MODEL,
                FAIR/workload['corpus'],len(workload['prompt_ids']),64,4,ROOT/label,label])
            assert json.loads((ROOT/label/'prompt-ids.json').read_text())==workload['prompt_ids']
            assert json.loads((ROOT/label/'target-ids.json').read_text())==workload['target_ids']
            rows=[json.loads(line) for line in (ROOT/label/'measurements.jsonl').read_text().splitlines()]
            assert len([r for r in rows if r['kind']=='completion'])==5
            assert all(r['repeat_bitwise_equal'] for r in rows if r['kind']=='completion')
            for r in rows:
                if r['kind']=='phase':assert r['peer_fallback_calls']==r['routed_cpu_calls']==0
            subprocess.run([str(b.PYTHON),str(ROOT/'analyze_audit.py')],check=True,cwd=ROOT)
    b.status('complete-hc')

if __name__=='__main__':
    try:main()
    except BaseException as error:b.ROOT=ROOT;b.status('failed-hc',error=repr(error));raise
