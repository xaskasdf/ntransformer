"""Bounded, single-attempt native compile/execution; never launches Vivado."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['compile-stdlib','compile-amd','run-stdlib','run-amd','negative-amd'])
    p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[1];folder=source.parent
    library_dirs=['/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim','/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1']
    os.environ['LD_LIBRARY_PATH']=':'.join(library_dirs)+':'+os.environ.get('LD_LIBRARY_PATH','')
    manifest=json.loads((source/'MANIFEST.json').read_text())
    assert set(os.sched_getaffinity(0))<={2,3}
    assert all(sha(source/n)==h for n,h in manifest['files'].items())
    started=folder/(a.action+'.started.json');done=folder/(a.action+'.json')
    assert not started.exists() and not done.exists(),'Attempt exists: inspect, do not relaunch'
    variant='stdlib' if a.action.endswith('stdlib') else 'amd'
    binary=folder/('forward-'+variant)
    if a.action.startswith('compile'):
        assert not binary.exists()
        dirs=[source/'fpga/hls'/n for n in manifest['kernel_directories']]
        command=['g++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas']
        command += ['-I'+str(d) for d in dirs]
        if variant=='amd':command+=['-DNTRPU_USE_HLS_MATH','-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include']
        command += [str(source/'tests/test_ntrpu_llama32_forward_native_r2.cpp')]
        command += [str(d/('ntrpu_'+d.name+'.cpp')) for d in dirs]
        if variant=='amd':command+=['-lgmp']
        command+=['-o',str(binary)]
        if variant=='amd':
            # AMD 2026.1 math headers use constructor syntax rejected in C++20.
            # Keep the tested harness C++20; compile unchanged HLS units C++17.
            obj=folder/'amd-objects';obj.mkdir(exist_ok=False)
            flags=['g++','-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas']
            flags+=['-I'+str(d) for d in dirs]
            common=flags+['-DNTRPU_USE_HLS_MATH','-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include']
            steps=[common+['-std=c++17','-MMD','-c']+[str(d/('ntrpu_'+d.name+'.cpp')) for d in dirs],
                   flags+['-std=c++20','-c',str(source/'tests/test_ntrpu_llama32_forward_native_r2.cpp'),'-o','harness.o'],
                   ['g++','harness.o']+['ntrpu_'+d.name+'.o' for d in dirs]+['-L'+library_dirs[0],'-lhlsmc++-GCC46','-lhlsm-GCC46','-o',str(binary)]]
            command=['bash','-e','-c','cd '+shlex.quote(str(obj))+'\n'+'\n'.join(shlex.join(step) for step in steps)]
        model_verified=False
    else:
        assert a.model.stat().st_size==manifest['model_bytes'] and sha(a.model)==manifest['model_sha256']
        compile_record=json.loads((folder/('compile-'+variant+'.json')).read_text())
        assert compile_record['exit_code']==0 and sha(binary)==compile_record['binary_sha256']
        for name,digest in compile_record['header_dependencies'].items():assert sha(name)==digest
        for name,digest in compile_record['runtime_libraries'].items():assert sha(name)==digest
        command=[str(binary),str(a.model),str(source/'tensors.tsv'),str(source/'reference'),str(folder/(a.action+'-output'))]
        if a.action.startswith('negative'):command+=['--inject-last-logit']
        model_verified=True
    record=dict(action=a.action,command=command,start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        model_verified=model_verified,model_sha256=manifest['model_sha256'],cpu_affinity=sorted(os.sched_getaffinity(0)),
        timeout_seconds=120,vivado_launched=False)
    with started.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    before=time.monotonic()
    with (folder/(a.action+'.log')).open('xb') as log:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=5s','120s']+command,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode,wall_seconds=time.monotonic()-before,
        source_unchanged=all(sha(source/n)==h for n,h in manifest['files'].items()),
        log_sha256=sha(folder/(a.action+'.log')),binary_sha256=sha(binary) if binary.exists() else None,
        compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0])
    headers={};runtime={}
    if binary.exists():
        linked=subprocess.check_output(['ldd',str(binary)],text=True)
        (folder/(a.action+'-runtime.txt')).write_text(linked)
        for line in linked.splitlines():
            match=re.search(r'=> (/[^ ]+)',line)
            if match and Path(match[1]).is_file():runtime[match[1]]=sha(match[1])
        for dep in (folder/'amd-objects').glob('*.d'):
            for name in dep.read_text().replace('\\\n',' ').split()[1:]:
                if Path(name).is_file():headers[name]=sha(name)
    record.update(header_dependencies=headers,runtime_libraries=runtime)
    out=folder/(a.action+'-output')
    record['outputs']={str(q.relative_to(out)):sha(q) for q in sorted(out.glob('*.bin'))} if out.exists() else {}
    with done.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))

if __name__=='__main__':main()
