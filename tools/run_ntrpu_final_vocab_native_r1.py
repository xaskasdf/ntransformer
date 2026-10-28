"""Native final-norm/head protocol, mutation and trace checks; no arithmetic execution."""
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import check_ntrpu_final_vocab_handoff_r1 as checker
from run_ntrpu_final_vocab_handoff_mutants_r1 import mutations

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists() and {2,3}<=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{2,3})
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    assert sha(a.source.with_suffix('.zip'))==a.source_sha
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    assert all(sha(a.source/n)==h for n,h in hashes.items())
    a.output.mkdir(parents=True);out=a.output/'output';out.mkdir()
    info=dict(source=str(a.source),source_sha256=a.source_sha,cpu_affinity=[2,3],commands=[],
        started_utc=datetime.now(timezone.utc).isoformat(),arithmetic_executed=False,errors=[],suites=[],mutants=[])
    def save():
        (a.output/'invocation.json').write_text(json.dumps(info,indent=2)+'\n')
    def execute(name,cmd):
        started=time.monotonic()
        with (out/name).open('xb') as log:
            try:code=subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
            except subprocess.TimeoutExpired:code=124
        info['commands'].append(dict(log=name,command=cmd,exit_code=code,seconds=time.monotonic()-started));save()
        return code
    rtl=a.source/'fpga/rtl/ntrpu_final_vocab_handoff_r1.sv'
    tb=a.source/'fpga/tb/tb_ntrpu_final_vocab_handoff_r1.sv'
    def compile_bank(name,path):
        binary=out/(name+'.vvp')
        code=execute(name+'-compile.log',['iverilog','-g2012','-Wall','-s','tb_ntrpu_final_vocab_handoff_r1','-o',str(binary),str(path),str(tb)])
        assert code==0,('Compile failure',name)
        return binary
    save()
    try:
        assert execute('version.log',['iverilog','-V'])==0
        binary=compile_bank('baseline',rtl)
        for seed in (1,7,42):
            name='seed-'+str(seed)+'.log'
            assert execute(name,['vvp',str(binary),'+SEED='+str(seed)])==0,('Baseline failed',seed)
            text=(out/name).read_text();metrics=checker.review(text,seed);controls=checker.controls(text,seed)
            info['suites'].append(dict(metrics=metrics,controls=controls));save()
        for name,text in mutations(rtl.read_text()).items():
            path=out/(name+'.sv');path.write_text(text)
            binary=compile_bank(name,path)
            code=execute(name+'-test.log',['vvp',str(binary),'+SEED=1'])
            lines=(out/(name+'-test.log')).read_text().splitlines()
            failure=next((line for line in lines if line.startswith('FATAL:') and 'FVH_FAIL' in line),None)
            caught=code==1 and failure is not None and not any(line.startswith('FVH_PASS') for line in lines)
            if failure and any(word in failure.lower() for word in ('timeout','deadline')):caught=False
            info['mutants'].append(dict(name=name,rtl_sha256=sha(path),caught=caught,exit_code=code,diagnostic=failure));save()
            if not caught:info['errors'].append('Mutation lacked a semantic rejection: '+name)
    except Exception as exc:info['errors'].append(repr(exc))
    info.update(exit_code=int(bool(info['errors'])),finished_utc=datetime.now(timezone.utc).isoformat(),
        frozen_sources_unchanged=all(sha(a.source/n)==h for n,h in hashes.items()))
    save();print(json.dumps(info),flush=True)
    return info['exit_code']

if __name__=='__main__':raise SystemExit(main())
