"""Run actual post-O HLS C++ and deliberate implementation mutants on CPU2/3."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in hashes.items(): assert digest(a.source/n)==h,n
    sys.path.insert(0,str(a.source/'tools'));import check_ntrpu_post_attention_r1 as checker
    jobs=checker.fixture(a.source/'fixture')
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[-2:]
    os.sched_setaffinity(0,affinity);a.output.mkdir(parents=True)
    now=lambda:datetime.now(timezone.utc).isoformat();start=time.monotonic()
    invocation=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,cpu_affinity=affinity,
        scope='Native HLS C++/AMD math only; no RTL, AXI or FPGA timing claim.')
    inv=a.output/'invocation.json';inv.write_text(json.dumps(invocation,indent=2)+'\n')
    libs=['/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim',
          '/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1']
    env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(libs)+':'+env.get('LD_LIBRARY_PATH','')
    cores=['post_attention_r1','residual_fp32_r1','norm_q8_abi1_r1','rmsnorm_fp32_r1','quantize_q8_0']
    includes=[a.source/'fpga/hls'/n for n in cores+['swiglu']]
    includes.append(Path('/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include'))
    sources=[a.source/'fpga/hls'/n/('ntrpu_'+n+'.cpp') for n in cores]
    sources.append(a.source/'tests/test_ntrpu_hls_post_attention_r1.cpp')
    common=['g++','-std=c++14','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas',
            '-DNTRPU_USE_HLS_MATH',*['-I'+str(q) for q in includes]]
    link=['-L'+libs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46']
    phases=[];errors=[];numeric=None;rejected=[];mutants=[]
    def execute(name,cmd,seconds=120):
        tick=time.monotonic()
        with (a.output/(name+'.log')).open('x') as f:
            try:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds).returncode
            except subprocess.TimeoutExpired:code=124;f.write('\nTIMEOUT\n')
        phases.append(dict(name=name,command=cmd,exit_code=code,seconds=time.monotonic()-tick))
        return code
    try:
        binary=a.output/'post-attention-amd'
        assert execute('compile',common+list(map(str,sources))+link+['-o',str(binary)])==0
        assert execute('positive',[str(binary),str(a.source/'fixture/llama32_post_attention_r1.bin')])==0
        text=(a.output/'positive.log').read_text()
        numeric=checker.review(text,jobs);rejected=checker.controls(text,jobs);assert len(rejected)==11
        original=sources[0].read_text()
        variants={
            'early_residual_publication':original.replace('    const uint32_t norm_status',
                '    for (uint32_t i=0;i<elements;++i) residual_output[i]=residual[i];\n    const uint32_t norm_status',1),
            'residual_lsb':original.replace('residual_output[i] = residual[i];','residual_output[i] = residual[i] ^ 1;'),
            'abi_lsb':original.replace('ffn_input[i] = packed[i];','ffn_input[i] = packed[i] ^ 1;'),
            'wrong_branch':original.replace('residual_input, branch, residual, elements','residual_input, residual_input, residual, elements'),
            'wrong_epsilon':original.replace('residual, norm_weight, packed, elements, epsilon_bits','residual, norm_weight, packed, elements, 0'),
        }
        for name,mutated in variants.items():
            assert mutated!=original
            path=a.output/(name+'.cpp');path.write_text(mutated);exe=a.output/name
            assert execute(name+'-compile',common+[str(path)]+list(map(str,sources[1:]))+link+['-o',str(exe)])==0
            code=execute(name+'-test',[str(exe),str(a.source/'fixture/llama32_post_attention_r1.bin')])
            assert code==1 and 'FAIL:' in (a.output/(name+'-test.log')).read_text()
            mutants.append(dict(name=name,sha256=digest(path),exit_code=code))
        # Deliberately wrong expected data must fail the executable itself.
        raw=bytearray((a.source/'fixture/llama32_post_attention_r1.bin').read_bytes())
        raw[12+16+3*8192]^=1
        bad=a.output/'bad-reference.bin';bad.write_bytes(raw)
        assert execute('bad-reference',[str(binary),str(bad)])==1
        headers=subprocess.check_output(common+['-MM']+list(map(str,sources)),env=env,text=True)
        (a.output/'headers.d').write_text(headers)
        header_hashes={str(q):digest(q) for n in headers.replace('\\\n',' ').split() if (q:=Path(n)).is_file()}
        linked=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
        (a.output/'runtime-libraries.txt').write_text(linked)
        runtime={m[1]:digest(Path(m[1])) for m in re.finditer(r'=> (/[^ ]+)',linked) if Path(m[1]).is_file()}
    except Exception as exc: errors.append(repr(exc));header_hashes={};runtime={}
    invocation.update(finished_utc=now(),duration_seconds=time.monotonic()-start,exit_code=1 if errors else 0)
    inv.write_text(json.dumps(invocation,indent=2)+'\n')
    report=dict(status='DIAGNOSTIC_POST_ATTENTION_NATIVE_AMD_R1' if errors else 'PASS_POST_ATTENTION_NATIVE_AMD_R1',
        invocation=invocation,errors=errors,numerical=numeric,rejected_controls=rejected,mutants=mutants,
        phases=phases,source_files=hashes,headers=header_hashes,runtime_libraries=runtime,
        compiler=subprocess.check_output(['g++','--version'],text=True))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','errors','numerical','rejected_controls','mutants')}),flush=True)


if __name__=='__main__':main()
