import json
from pathlib import Path
import sys
FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(FAIR))
import common as b

def main():
    b.ROOT=ROOT/'validation';b.ROOT.mkdir(exist_ok=False)
    compiler=b.PREFIX/'bin/x86_64-conda-linux-gnu-g++'
    source=ROOT/'hc-source'
    flags=['-std=c++20','-march=native','-I'+str(source/'src'),'-I'+str(source/'include')]
    libraries=['-L'+str(b.PREFIX/'targets/x86_64-linux/lib'),'-lcudart','-lcudadevrt','-lcudart_static','-ldl','-lrt','-lpthread']
    for mode,extra in [('release',['-O3','-DNDEBUG']),('sanitized',['-O1','-g','-fno-omit-frame-pointer','-fsanitize=address,undefined'])]:
        executable=b.ROOT/('test-hc-'+mode)
        b.command('compile-'+mode,[compiler,*flags,*extra,ROOT/'test_ds41_hc_extended.cpp',ROOT/'libntransformer-hc.a',*libraries,'-o',executable])
        b.command('run-'+mode,[executable])
    b.save('validation.json',dict(release_pass=True,asan_ubsan_pass=True,
        source_sha256=b.sha(ROOT/'test_ds41_hc_extended.cpp'),header_sha256=b.sha(source/'src/model/ds41/ds41_hc.h'),
        directed_layout_cases=4,double_precision_cases=36,existing_direct_pool_cases=84,
        scope='CPU HC producer/consumer and independent Sinkhorn oracle; no GPU work'))

if __name__=='__main__':main()
