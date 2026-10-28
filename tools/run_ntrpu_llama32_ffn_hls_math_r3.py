"""Compile only the AMD-math translation unit as C++14, matching run_hls.tcl.

Driver and other kernels stay C++20. No kernel or numerical reference changes.
This is native code linked against AMD headers, not a Vitis csim/cosim result.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import os
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    library_dirs=["/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim", "/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1"]
    os.environ["LD_LIBRARY_PATH"]=":".join(library_dirs)+":"+os.environ.get("LD_LIBRARY_PATH","")
    binding=json.loads((args.source/'BINDING.json').read_text())
    original=Path(binding['original_source'])
    fixture=original/'fixture'
    for name,sha in binding['original_files'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest()==sha,name
    own=json.loads((args.source/'SNAPSHOT_FILES.json').read_text())
    for name,sha in own.items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==sha
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    inv=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(args.source),
             scope='Full FFN native AMD-math C++ check; no RTL or Vitis csim/cosim')
    (args.output/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
    kernels=('q8_block_mac','quantize_q8_0','swiglu','swiglu_q8_block')
    includes=['-I'+str(original/'fpga/hls'/name) for name in kernels]
    common=['-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas','-DNTRPU_USE_HLS_MATH',*includes]
    obj=args.output/'swiglu-hls.o';binary=args.output/'ffn-native-hls-r3'
    commands=[['g++','-std=c++14',*common,'-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include',
               '-c',str(original/'fpga/hls/swiglu/ntrpu_swiglu.cpp'),'-o',str(obj)],
              ['g++','-std=c++20',*common,str(original/'tests/test_ntrpu_llama32_ffn_native_r1.cpp'),
               *[str(original/'fpga/hls'/name/('ntrpu_'+name+'.cpp')) for name in kernels if name!='swiglu'],
               str(obj),'-L'+library_dirs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46','-o',str(binary)]]
    results=[];status='DIAGNOSTIC_LLAMA32_FFN_HLS_MATH_R3'
    for index,command in enumerate(commands):
        with (args.output/f'compile-{index}.log').open('x') as stream:
            try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:stream.write('\nTIMEOUT=180\n');code=124
        results.append(dict(phase='compile',index=index,command=command,exit_code=code))
        if code:break
    else:
        for mode in ('positive','negative'):
            command=[str(binary),str(fixture)]+(['--inject-bad-reference'] if mode=='negative' else [])
            before=time.monotonic()
            with (args.output/(mode+'.log')).open('x') as stream:
                try:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
                except subprocess.TimeoutExpired:stream.write('\nTIMEOUT=180\n');code=124
            results.append(dict(phase=mode,command=command,exit_code=code,wall_seconds=time.monotonic()-before))
        positive=(args.output/'positive.log').read_text();negative=(args.output/'negative.log').read_text()
        cases=re.findall(r'^NTRPU_LLAMA32_FFN_NATIVE_CASE sample=([123]) dim=2048 hidden=8192 mac_blocks=1572864 reference_values=28672 activation_records=256 bit_exact=1 cpu_seconds=([0-9.e+-]+)$',positive,re.M)
        if (results[-2]['exit_code']==0 and results[-1]['exit_code']==1 and
            [v[0] for v in cases]==['1','2','3'] and positive.splitlines().count('NTRPU_LLAMA32_FFN_NATIVE_PASS samples=3 reference_values=86016 mac_blocks=4718592 math=hls')==1 and
            'NTRPU_LLAMA32_FFN_NATIVE_FAIL' not in positive and
            re.fullmatch(r'NTRPU_LLAMA32_FFN_NATIVE_FAIL gate byte=0 actual=\d+ expected=\d+\n',negative)):
            status='PASS_LLAMA32_FFN_HLS_MATH_NATIVE_R3'
    inv.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),duration_seconds=time.monotonic()-start,
               exit_code=0 if status.startswith('PASS') else 1)
    (args.output/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
    # Record every installed header dependency reached by the actual AMD TU.
    dependency_command=['g++','-std=c++14',*common,'-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include',
                        '-MM',str(original/'fpga/hls/swiglu/ntrpu_swiglu.cpp')]
    dep=subprocess.run(dependency_command,capture_output=True,text=True,timeout=60)
    (args.output/'headers.d').write_text(dep.stdout)
    headers={}
    if dep.returncode==0:
        for name in dep.stdout.replace('\\\n',' ').split()[1:]:
            path=Path(name)
            if path.is_file():headers[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    runtime={}
    if binary.exists():
        linked=subprocess.check_output(['ldd',str(binary)],text=True)
        (args.output/'runtime-libraries.txt').write_text(linked)
        for line in linked.splitlines():
            match=re.search(r'=> (/[^ ]+)',line)
            if match and Path(match[1]).is_file():runtime[match[1]]=hashlib.sha256(Path(match[1]).read_bytes()).hexdigest()
    report=dict(status=status,invocation=inv,commands=results,binding=binding,source_files=own,math_runtime_libraries=runtime,
                compiler=subprocess.check_output(['g++','--version'],text=True),
                header_dependencies=headers,header_dependency_exit_code=dep.returncode,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None,
                scope=inv['scope'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,phases=[{k:v for k,v in r.items() if k!='command'} for r in results],
                         headers=len(headers))),flush=True)


if __name__=='__main__':main()
