"""Bounded actual-kernel CPU check of attention concatenation through FFN input."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and not args.output.exists()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = json.loads((args.source/'SNAPSHOT_FILES.json').read_text())
    for name, sha in hashes.items(): assert digest(args.source/name) == sha, name
    assert hashes['fixture/manifest.json'] == '73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'
    review = json.loads((args.source/'reference-review.json').read_text())
    assert review['status'] == 'PASS_ATTENTION_OUTPUT_R1_INDEPENDENT_CPU_REFERENCE'
    assert review['manifest_sha256'] == hashes['fixture/manifest.json'] and len(review['rejected_controls']) == 13
    available = sorted(os.sched_getaffinity(0)); affinity = available[2:4] if len(available) >= 4 else available[-2:]
    os.sched_setaffinity(0, affinity)
    args.output.mkdir(parents=True); started = time.monotonic()
    now = lambda: datetime.now(timezone.utc).isoformat()
    invocation = dict(started_utc=now(), source=str(args.source), source_sha256=args.source_sha, cpu_affinity=affinity,
        scope='Actual C++ kernels on CPU with AMD RMS math. No RTL/AXI, cycle, timing or full inference qualification.')
    ip = args.output/'invocation.json'; ip.write_text(json.dumps(invocation, indent=2)+'\n')
    libs = ['/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim',
            '/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1']
    env = os.environ.copy(); env['LD_LIBRARY_PATH'] = ':'.join(libs)+':'+env.get('LD_LIBRARY_PATH','')
    cores = ['q8_block_mac','quantize_q8_0','residual_fp32_r1','norm_q8_abi1_r1','rmsnorm_fp32_r1']
    includes = [args.source/'fpga/hls'/n for n in cores+['swiglu']]
    includes.append(Path('/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include'))
    sources = [args.source/'fpga/hls'/n/('ntrpu_'+n+'.cpp') for n in cores]
    sources.append(args.source/'tests/test_ntrpu_llama32_attention_output_native_r1.cpp')
    common = ['g++','-std=c++14','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas',
              '-DNTRPU_USE_HLS_MATH',*['-I'+str(p) for p in includes]]
    binary = args.output/'attention-output-amd'
    command = common+list(map(str,sources))+['-L'+libs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46','-o',str(binary)]
    phases = []

    def execute(name, cmd, seconds):
        t = time.monotonic()
        with (args.output/(name+'.log')).open('x') as stream:
            try: code = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds).returncode
            except subprocess.TimeoutExpired: code = 124; stream.write('\nTIMEOUT\n')
        phases.append(dict(phase=name,command=cmd,exit_code=code,seconds=time.monotonic()-t))
        return code

    status = 'DIAGNOSTIC_ATTENTION_OUTPUT_NATIVE_AMD_R1'; outputs = {}; headers = {}; runtime = {}
    if execute('compile',command,120) == 0:
        positive_dir = args.output/'positive'; positive_dir.mkdir()
        negative_dir = args.output/'negative'; negative_dir.mkdir()
        positive = execute('positive',[str(binary),str(args.source/'fixture'),str(positive_dir)],120)
        negative = execute('negative',[str(binary),str(args.source/'fixture'),str(negative_dir),'--inject-bad-reference'],120)
        if positive == 0 and negative == 1:
            assert (args.output/'positive.log').read_text().splitlines()[-1] == 'ATTENTION_OUTPUT_NATIVE_PASS samples=4 mac_blocks=524288 projection=8192 residual=8192 math=hls'
            assert (args.output/'negative.log').read_text().strip() == 'ATTENTION_OUTPUT_NATIVE_FAIL O FP64 byte=0'
            assert not list(negative_dir.iterdir())
            for sample in range(1,5):
                for suffix in ('attention-abi1','output-fp64','output-fp32','residual-fp32','ffn-input-abi1'):
                    name = f'sample-{sample:03d}-{suffix}.bin'; path = positive_dir/name
                    assert path.read_bytes() == (args.source/'fixture'/name).read_bytes(), name
                    outputs[name] = digest(path)
            assert set(p.name for p in positive_dir.iterdir()) == set(outputs)
            status = 'PASS_ATTENTION_OUTPUT_NATIVE_AMD_R1'
        dep = subprocess.run(common+['-MM']+list(map(str,sources)),env=env,capture_output=True,text=True,timeout=60)
        assert dep.returncode == 0; (args.output/'headers.d').write_text(dep.stdout)
        for name in dep.stdout.replace('\\\n',' ').split():
            path = Path(name)
            if path.is_file(): headers[str(path)] = digest(path)
        linked = subprocess.check_output(['ldd',str(binary)],env=env,text=True)
        (args.output/'runtime-libraries.txt').write_text(linked)
        for line in linked.splitlines():
            match = re.search(r'=> (/[^ ]+)',line)
            if match and Path(match[1]).is_file(): runtime[match[1]] = digest(Path(match[1]))
    invocation.update(finished_utc=now(),duration_seconds=time.monotonic()-started,exit_code=0 if status.startswith('PASS') else 1)
    ip.write_text(json.dumps(invocation,indent=2)+'\n')
    report = dict(status=status,invocation=invocation,source_files=hashes,commands=phases,actual_outputs=outputs,
        header_dependencies=headers,runtime_libraries=runtime,reference_review=review,
        compiler=subprocess.check_output(['g++','--version'],text=True),binary_sha256=digest(binary) if binary.exists() else None)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=status,actual_outputs=len(outputs),phases=[{k:v for k,v in p.items() if k!='command'} for p in phases])),flush=True)


if __name__ == '__main__': main()
