"""Native C++/NumPy cross-check; retain every command and diagnostic."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace


def run_bounded(command, stream):
    try:
        return subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    except subprocess.TimeoutExpired:
        stream.write('\nNTRPU_NATIVE_COMMAND_TIMEOUT_SECONDS=180\n')
        stream.flush()
        return SimpleNamespace(returncode=124)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((args.source/'SNAPSHOT_FILES.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==expected,name
    invocation=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    source=str(args.source),scope='Native C++ kernel-source cross-check; not RTL, cosim, timing, or inference')
    (args.output/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    kernels=('q8_block_mac','quantize_q8_0','swiglu','swiglu_q8_block')
    cpp=[str(args.source/'fpga/hls'/name/('ntrpu_'+name+'.cpp')) for name in kernels]
    includes=['-I'+str(args.source/'fpga/hls'/name) for name in kernels]
    compiler=subprocess.check_output(['g++','--version'],text=True)
    (args.output/'compiler.txt').write_text(compiler)
    results=[]
    start=time.monotonic()
    for math in ('stdlib','hls'):
        binary=args.output/('ffn-native-'+math)
        command=['g++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas',
                 *includes,str(args.source/'tests/test_ntrpu_llama32_ffn_native_r1.cpp'),*cpp,'-o',str(binary)]
        if math=='hls':command += ['-DNTRPU_USE_HLS_MATH','-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include','-lgmp']
        entry=dict(math=math,compile_command=command)
        with (args.output/(math+'-compile.log')).open('x') as stream:
            compile_result=run_bounded(command,stream)
        entry['compile_exit_code']=compile_result.returncode
        if compile_result.returncode:
            entry.update(status='COMPILE_DIAGNOSTIC');results.append(entry);continue
        entry['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        for mode in ('positive','negative'):
            command=[str(binary),str(args.source/'fixture')]
            if mode=='negative':command+=['--inject-bad-reference']
            before=time.monotonic()
            with (args.output/(math+'-'+mode+'.log')).open('x') as stream:
                result=run_bounded(command,stream)
            entry[mode]=dict(command=command,exit_code=result.returncode,wall_seconds=time.monotonic()-before)
        positive=(args.output/(math+'-positive.log')).read_text()
        negative=(args.output/(math+'-negative.log')).read_text()
        cases=re.findall(r'^NTRPU_LLAMA32_FFN_NATIVE_CASE sample=([123]) dim=2048 hidden=8192 mac_blocks=1572864 reference_values=28672 activation_records=256 bit_exact=1 cpu_seconds=([0-9.e+-]+)$',positive,re.M)
        ok=(entry['positive']['exit_code']==0 and [r[0] for r in cases]==['1','2','3'] and
            positive.splitlines().count('NTRPU_LLAMA32_FFN_NATIVE_PASS samples=3 reference_values=86016 mac_blocks=4718592 math='+math)==1 and
            'NTRPU_LLAMA32_FFN_NATIVE_FAIL' not in positive and entry['negative']['exit_code']==1 and
            re.fullmatch(r'NTRPU_LLAMA32_FFN_NATIVE_FAIL gate byte=0 actual=\d+ expected=\d+\n',negative) is not None)
        entry.update(status='PASS' if ok else 'NUMERICAL_DIAGNOSTIC',cases=cases)
        results.append(entry)
    invocation.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      duration_seconds=time.monotonic()-start,exit_code=0 if all(r['status']=='PASS' for r in results) else 1)
    (args.output/'invocation.json').write_text(json.dumps(invocation,indent=2)+'\n')
    report=dict(status='PASS_LLAMA32_FULL_FFN_NATIVE_R1' if invocation['exit_code']==0 else 'DIAGNOSTIC_LLAMA32_FULL_FFN_NATIVE_R1',
                results=results,invocation=invocation,source_files=hashes,
                reference_manifest_sha256=hashlib.sha256((args.source/'fixture/manifest.json').read_bytes()).hexdigest(),
                scope=invocation['scope'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],results=results)),flush=True)


if __name__=='__main__':main()
