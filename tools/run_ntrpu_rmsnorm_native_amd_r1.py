"""Native AMD-header preflight; this does not invoke Vitis or qualify RTL."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    binding=json.loads((a.source/'BINDING.json').read_text());parent=Path(binding['source']['source'])
    assert hashlib.sha256(Path(binding['source']['source_zip']).read_bytes()).hexdigest()==binding['source']['source_sha256']
    for n,h in binding['input_hashes'].items():assert hashlib.sha256((parent/n).read_bytes()).hexdigest()==h,n
    own=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in own.items():assert hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h,n
    affinity=sorted(os.sched_getaffinity(0))[-2:];os.sched_setaffinity(0,affinity)
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    now=lambda:datetime.now(timezone.utc).isoformat()
    inv=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,cpu_affinity=affinity,
        scope='Native GCC/AMD sqrt library preflight only; no Vitis/C-RTL or physical qualification.')
    ip=a.output/'invocation.json';ip.write_text(json.dumps(inv,indent=2)+'\n')
    includes=[parent/'fpga/hls/rmsnorm_fp32_r1',parent/'fpga/hls/swiglu',Path('/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include')]
    libs=['/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim','/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1']
    env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(libs)+':'+env.get('LD_LIBRARY_PATH','')
    binary=a.output/'rmsnorm-native-amd'
    common=['g++','-std=c++14','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas','-DNTRPU_USE_HLS_MATH',*['-I'+str(p) for p in includes]]
    cpp=parent/'fpga/hls/rmsnorm_fp32_r1/ntrpu_rmsnorm_fp32_r1.cpp'
    command=common+[str(cpp),str(parent/'tests/test_ntrpu_hls_rmsnorm_r1.cpp'),'-L'+libs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46','-o',str(binary)]
    phases=[]
    def execute(phase,cmd,seconds):
        before=time.monotonic()
        with (a.output/(phase+'.log')).open('x') as f:
            try:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds).returncode
            except subprocess.TimeoutExpired:code=124;f.write('\nTIMEOUT\n')
        phases.append(dict(phase=phase,command=cmd,exit_code=code,seconds=time.monotonic()-before));return code
    status='DIAGNOSTIC_RMSNORM_NATIVE_AMD_R1';review=None;controls=[];headers={};runtime={}
    if execute('compile',command,120)==0:
        fixture=parent/'fixture'/'llama32_norm_r1.bin'
        ok=execute('positive',[str(binary),str(fixture)],60)
        bad=bytearray(fixture.read_bytes());bad[12+12+8192+8192]^=128
        altered=a.output/'negative-reference.bin';altered.write_bytes(bad)
        negative=execute('negative',[str(binary),str(altered)],60)
        if ok==0 and negative==1:
            sys.path.insert(0,str(parent/'tools'));import check_ntrpu_rmsnorm_r1 as checker
            positive=(a.output/'positive.log').read_text();review=checker.review(positive,parent/'fixture');controls=checker.controls(positive,parent/'fixture')
            text=(a.output/'negative.log').read_text()
            assert '\nFAIL: ' in text and 'NTRPU_RMSNORM_PASS' not in text
            status='PASS_RMSNORM_NATIVE_AMD_R1'
        dependencies=subprocess.run(common+['-MM',str(cpp)],capture_output=True,text=True,timeout=60,env=env)
        (a.output/'headers.d').write_text(dependencies.stdout)
        assert dependencies.returncode==0
        for name in dependencies.stdout.replace('\\\n',' ').split()[1:]:
            q=Path(name)
            if q.is_file():headers[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
        linked=subprocess.check_output(['ldd',str(binary)],text=True,env=env);(a.output/'runtime-libraries.txt').write_text(linked)
        for line in linked.splitlines():
            m=re.search(r'=> (/[^ ]+)',line)
            if m and Path(m[1]).is_file():runtime[m[1]]=hashlib.sha256(Path(m[1]).read_bytes()).hexdigest()
    inv.update(finished_utc=now(),duration_seconds=time.monotonic()-started,exit_code=0 if status.startswith('PASS') else 1)
    ip.write_text(json.dumps(inv,indent=2)+'\n')
    report=dict(status=status,invocation=inv,binding=binding,source_files=own,numerical=review,controls=controls,commands=phases,
        compiler=subprocess.check_output(['g++','--version'],text=True),header_dependencies=headers,runtime_libraries=runtime,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,phases=[{k:v for k,v in q.items() if k!='command'} for q in phases],headers=len(headers))),flush=True)
if __name__=='__main__':main()
