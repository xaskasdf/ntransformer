"""Bounded functional comparison against a reviewed routed checkpoint/netlist."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_read_return import ROOT,SOURCES as BASE_SOURCES,sha,save
from prepare_ntrpu_read_return_netlist import prepare
SOURCES=BASE_SOURCES+['tools/run_ntrpu_read_return_netlist.py','tools/prepare_ntrpu_read_return_netlist.py','tools/check_ntrpu_read_return.py']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    for name in ('physical','physical-review','reference','reference-review'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    physical=a.physical.resolve();reference=a.reference.resolve();pr=json.loads(a.physical_review.read_text());rr=json.loads(a.reference_review.read_text())
    assert pr['status'] in ('READ_RETURN_ROUTED_INTERNAL_TIMING_PASS','READ_RETURN_ROUTED_INTERNAL_TIMING_OPEN') and rr['status']=='READ_RETURN_PREFLIGHT_REVIEWED' and rr['common']==0 and rr['native_profile']==0
    assert pr['result_sha256']==sha(physical/'result.json') and rr['result_sha256']==sha(reference/'result.json')
    for previous in (physical,reference):
        for n,h in json.loads((previous/'FILES.json').read_text()).items():assert sha(previous/n)==h
        result=json.loads((previous/'result.json').read_text());assert result['sources']==result['sources_after']
        assert all(sha(ROOT/n)==h for n,h in result['sources'].items() if n.endswith('.sv'))
    net=physical/'suite/routed.v';donor_hash=sha(net);assert pr['checkpoint_sha256']==sha(physical/'suite/routed.dcp')
    shutil.copy2(reference/'suite/native_stub.sv',suite/'native_stub.sv');shutil.copy2(reference/'suite/ntrpu_read_return_binding.svh',suite/'ntrpu_read_return_binding.svh')
    shutil.copy2(a.physical_review,out/'PHYSICAL_REVIEW.json');shutil.copy2(a.reference_review,out/'REFERENCE_REVIEW.json')
    prepare(ROOT,suite,net)
    sources={n:sha(ROOT/n) for n in SOURCES};generated={p.name:sha(p) for p in suite.iterdir() if p.is_file()}
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    cmd=['vivado','-mode','batch','-notrace','-source',str(suite/'run.tcl')]
    r=dict(status='RUNNING',experiment='read_return_netlist',sources=sources,generated=generated,netlist_sha256=donor_hash,
           physical_result_sha256=sha(physical/'result.json'),reference_result_sha256=sha(reference/'result.json'),
           command=cmd,timeout=900,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES},
             generated_after={n:sha(suite/n) for n in generated},donor_unchanged=sha(net)==donor_hash)
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] and generated==r['generated_after'] and r['donor_unchanged'] else 'FAILED_OR_CHANGED'
    for p in (suite/'project').rglob('*.log'):
        dest=out/'tool_logs'/p.relative_to(suite/'project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'result.json',r);save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
if __name__=='__main__':main()
