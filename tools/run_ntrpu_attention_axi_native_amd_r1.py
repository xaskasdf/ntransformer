"""Native AMD-math validation of production-partition attention inputs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for name,sha in hashes.items():assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha
    available=sorted(os.sched_getaffinity(0));affinity=available[2:4] if len(available)>=4 else available[-2:]
    os.sched_setaffinity(0,affinity);a.output.mkdir(parents=True);started=time.monotonic()
    now=lambda:datetime.now(timezone.utc).isoformat()
    inv=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,cpu_affinity=affinity,
        scope='Native GCC/AMD exp backend, 143 cases with 64-token tiles. No Vitis C/RTL, AXI transaction or timing qualification.')
    ip=a.output/'invocation.json';ip.write_text(json.dumps(inv,indent=2)+'\n')
    libs=['/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim',
          '/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1']
    env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(libs)+':'+env.get('LD_LIBRARY_PATH','')
    includes=[a.source/'fpga/hls/attention_tile_fp32_r1',a.source/'fpga/hls/swiglu',
              Path('/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include')]
    common=['g++','-std=c++14','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas',
            '-DNTRPU_USE_HLS_MATH',*['-I'+str(d) for d in includes]]
    cpp=a.source/'fpga/hls/attention_tile_fp32_r1/ntrpu_attention_tile_fp32_r1.cpp'
    binary=a.output/'attention-fixed64-amd'
    command=common+[str(cpp),str(a.source/'tests/test_ntrpu_hls_attention_fixture_r1.cpp'),
        '-L'+libs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46','-o',str(binary)]
    phases=[]
    def execute(name,cmd,seconds):
        t=time.monotonic()
        with (a.output/(name+'.log')).open('x') as f:
            try:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=seconds).returncode
            except subprocess.TimeoutExpired:code=124;f.write('\nTIMEOUT\n')
        phases.append(dict(phase=name,command=cmd,exit_code=code,seconds=time.monotonic()-t));return code
    status='DIAGNOSTIC_ATTENTION_AXI_NATIVE_AMD_R1';numerical=None;controls=[];headers={};runtime={}
    if execute('compile',command,120)==0:
        fixture=a.source/'fixture/native-fixture/llama32_attention_r1.bin'
        positive=execute('positive',[str(binary),str(fixture)],120)
        bad=bytearray(fixture.read_bytes());offset=12+40+256+256+256
        struct.pack_into('<I',bad,offset,struct.unpack_from('<I',bad,offset)[0]^0x00800000)
        negative_fixture=a.output/'negative-reference.bin';negative_fixture.write_bytes(bad)
        negative=execute('negative',[str(binary),str(negative_fixture)],120)
        if positive==0 and negative==1:
            sys.path.insert(0,str(a.source/'tools'));import check_ntrpu_attention_axi_reference_r1 as checker
            jobs=checker.fixture(a.source/'fixture/native-fixture');text=(a.output/'positive.log').read_text()
            numerical=checker.review(text,jobs);controls=checker.controls(text,jobs)
            assert 'FAIL: Output exceeds one ULP from offline reference' in (a.output/'negative.log').read_text()
            assert len(controls)==12
            status='PASS_ATTENTION_AXI_NATIVE_AMD_R1'
        dependencies=subprocess.run(common+['-MM',str(cpp)],env=env,capture_output=True,text=True,timeout=60)
        assert dependencies.returncode==0;(a.output/'headers.d').write_text(dependencies.stdout)
        for name in dependencies.stdout.replace('\\\n',' ').split()[1:]:
            path=Path(name)
            if path.is_file():headers[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        linked=subprocess.check_output(['ldd',str(binary)],env=env,text=True)
        (a.output/'runtime-libraries.txt').write_text(linked)
        for line in linked.splitlines():
            match=re.search(r'=> (/[^ ]+)',line)
            if match and Path(match[1]).is_file():runtime[match[1]]=hashlib.sha256(Path(match[1]).read_bytes()).hexdigest()
    inv.update(finished_utc=now(),duration_seconds=time.monotonic()-started,exit_code=0 if status.startswith('PASS') else 1)
    ip.write_text(json.dumps(inv,indent=2)+'\n')
    report=dict(status=status,invocation=inv,source_files=hashes,commands=phases,numerical=numerical,
        rejected_controls=controls,header_dependencies=headers,runtime_libraries=runtime,
        compiler=subprocess.check_output(['g++','--version'],text=True),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,phases=[{k:v for k,v in r.items() if k!='command'} for r in phases],numerical=numerical)),flush=True)


if __name__=='__main__':main()
