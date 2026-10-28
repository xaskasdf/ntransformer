"""Native stdlib/AMD kernel tests on CPU2/3; never claims RTL qualification."""
import argparse,datetime,hashlib,json,os,re,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[2:4])
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    for n,h in hashes.items():assert hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h,n
    a.output.mkdir(parents=True,exist_ok=False);out=a.output/'output';out.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    start=time.monotonic();inv=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=sorted(os.sched_getaffinity(0)),native_only=True,hardware_slot=False,commands=[],status='running')
    ip=a.output/'invocation.json';ip.write_text(json.dumps(inv,indent=2)+'\n')
    cores=('q8_block_mac','quantize_q8_0','swiglu','swiglu_q8_block','residual_fp32_r1')
    includes=['-I'+str(a.source/'fpga/hls'/n) for n in cores]
    files={n:str(a.source/'fpga/hls'/n/('ntrpu_'+n+'.cpp')) for n in cores}
    lib='/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/lib/csim'
    os.environ['LD_LIBRARY_PATH']=lib+':/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/lnx64/tools/fpo_v7_1:'+os.environ.get('LD_LIBRARY_PATH','')
    common=['-O2','-fno-fast-math','-ffp-contract=off','-Wno-unknown-pragmas',*includes]
    test=str(a.source/'tests/test_ntrpu_llama32_post_ffn_native_r1.cpp')
    errors=[];libraries={};headers={}
    def execute(label,cmd,limit=180):
        before=time.monotonic()
        with (out/(label+'.log')).open('wb') as f:
            try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=limit).returncode
            except subprocess.TimeoutExpired:code=124
        inv['commands'].append(dict(log=label+'.log',command=cmd,exit_code=code,seconds=time.monotonic()-before))
        ip.write_text(json.dumps(inv,indent=2)+'\n');return code
    assert execute('compiler-version',['g++','--version'])==0
    for mode in ('stdlib','hls'):
        binary=out/('post-ffn-'+mode);extra=[]
        if mode=='hls':
            flags=['-DNTRPU_USE_HLS_MATH','-I/lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/include']
            obj=out/'swiglu-amd.o'
            compile_cmd=['g++','-std=c++14',*common,*flags,'-c',files['swiglu'],'-o',str(obj)]
            if execute('hls-math-compile',compile_cmd):errors.append('AMD TU compile');continue
            dep=subprocess.run(['g++','-std=c++14',*common,*flags,'-MM',files['swiglu']],capture_output=True,text=True,timeout=60)
            assert dep.returncode==0
            (out/'headers.d').write_text(dep.stdout)
            for n in dep.stdout.replace('\\\n',' ').split()[1:]:
                q=Path(n)
                if q.is_file():headers[str(q)]=sha(q)
            extra=['-DNTRPU_USE_HLS_MATH',str(obj),'-L'+lib,'-lhlsmc++-GCC46','-lhlsm-GCC46']
        cmd=['g++','-std=c++20',*common,test,*[files[n] for n in cores if mode=='stdlib' or n!='swiglu'],*extra,'-o',str(binary)]
        if execute(mode+'-compile',cmd):errors.append(mode+' compile');continue
        linked=subprocess.check_output(['ldd',str(binary)],text=True);(out/(mode+'-libraries.txt')).write_text(linked)
        for n in re.findall(r'=> (/[^ ]+)',linked):
            q=Path(n)
            if q.is_file():libraries[n]=sha(q)
        for label,option,expected in [('positive','',0),('bad-reference','--inject-bad-reference',1),('bad-input','--inject-bad-input',1),('bad-residual','--inject-bad-residual',1)]:
            cmd=[str(binary),str(a.source/'fixture')]+([option] if option else [])
            code=execute(mode+'-'+label,cmd)
            text=(out/(mode+'-'+label+'.log')).read_text()
            if code!=expected:errors.append(mode+' '+label+' exit')
            if not option:
                cases=re.findall(r'^NTRPU_POST_FFN_NATIVE_CASE sample=([1-4]) dim=2048 hidden=8192 mac_blocks=1572864 reference_values=30720 activation_records=256 bit_exact=1 cpu_seconds=([0-9.e+-]+)$',text,re.M)
                if [s for s,v in cases]!=list('1234') or text.splitlines().count('NTRPU_POST_FFN_NATIVE_PASS samples=4 reference_values=122880 mac_blocks=6291456 math='+mode)!=1:errors.append(mode+' positive coverage')
                if 'NTRPU_POST_FFN_NATIVE_FAIL' in text:errors.append(mode+' positive diagnostic')
            elif not text.startswith('NTRPU_POST_FFN_NATIVE_FAIL '+('second residual' if label=='bad-residual' else 'gate')+' byte=') or 'NTRPU_POST_FFN_NATIVE_PASS' in text:errors.append(mode+' negative diagnostic '+label)
    status='PASS_POST_FFN_NATIVE_R1' if not errors else 'DIAGNOSTIC_POST_FFN_NATIVE_R1'
    inv.update(status=status,errors=errors,exit_code=int(bool(errors)),finished_utc=now(),duration_seconds=time.monotonic()-start)
    ip.write_text(json.dumps(inv,indent=2)+'\n')
    report=dict(status=status,invocation=inv,source_files=hashes,header_dependencies=headers,runtime_libraries=libraries,
        scope='Native kernel arithmetic on four post-O reference prefixes, exact ABI1 input and second residual. No RTL, routed timing, tokens/s or full-model inference qualification.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(inv),flush=True)


if __name__=='__main__':main()
