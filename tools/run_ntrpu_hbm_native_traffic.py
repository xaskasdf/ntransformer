"""Compile and exercise an isolated two-stack HBM model with bounded resources."""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_hbm_native_traffic.py','tools/ntrpu_hbm_native_traffic.tcl',
         'tools/compile_ntrpu_hbm_native_incremental.py',
         'tools/ntrpu_hbm_native_traffic_run.tcl','tools/prepare_ntrpu_hbm_native_traffic.py',
         'tools/prepare_ntrpu_hbm_native_link.py','tools/prepare_ntrpu_hbm_cdc.py',
         'fpga/tb/ntrpu_hbm_native_traffic_body.svh']
SIMDIR='suite/project/hbm_native_traffic.sim/sim_1/behav/xsim'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('compile','traffic'),required=True)
    ap.add_argument('--prior',type=Path);ap.add_argument('--review',type=Path)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES}
    r=dict(status='PREPARING',experiment='hbm_native_traffic',stage=a.stage,sources=sources,
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),load_before=os.getloadavg(),timeout=900)
    save(out/'result.json',r)
    donor=None;inputs={}
    if a.stage=='compile':
        cwd=out
        if a.prior:
            assert a.review
            command=['python3','-B',str(ROOT/'tools/compile_ntrpu_hbm_native_incremental.py'),
                     str(out/'suite'),str(a.prior),str(a.review)]
        else:
            command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_hbm_native_traffic.tcl'),
                     '-log','vivado.log','-journal','vivado.jou','-tclargs',str(out/'suite')]
    else:
        prior=json.loads((a.prior/'result.json').read_text());review=json.loads(a.review.read_text())
        assert prior['sources']==prior['sources_after']==sources and review['status']=='NATIVE_TRAFFIC_SNAPSHOT_REVIEWED'
        inputs=json.loads((a.prior/'SNAPSHOT_INPUTS.json').read_text());donor=a.prior/SIMDIR
        cwd=out/'runtime';cwd.mkdir()
        for n,v in inputs.items():
            source=donor/n;assert source.resolve().is_relative_to(donor.resolve()) and sha(source)==v['sha256']
            target=cwd/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==v['sha256']
        shutil.copy2(a.prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json');shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
        shutil.copy2(ROOT/'tools/ntrpu_hbm_native_traffic_run.tcl',cwd/'run.tcl')
        command=['xsim','tb_ntrpu_hbm_native_traffic_behav','-tclbatch','run.tcl','-log','simulate.log']
        r.update(prior_result_sha256=sha(a.prior/'result.json'),manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'))
    r.update(status='RUNNING',command=command);save(out/'result.json',r)
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
             sources_after={n:sha(ROOT/n) for n in SOURCES})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] else 'FAILED_OR_CHANGED'
    if a.stage=='compile':
        sim=out/SIMDIR
        selected=[p for p in [*sim.glob('*.mem'),sim/'xsim.ini',
                  *(sim/'xsim.dir/tb_ntrpu_hbm_native_traffic_behav').glob('xsim*')] if p.is_file()]
        save(out/'SNAPSHOT_INPUTS.json',{p.relative_to(sim).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in selected})
        for p in (out/'suite/project').rglob('*.log'):
            dest=out/'tool_logs'/p.relative_to(out/'suite/project');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        generated=[p for p in (out/'suite').rglob('*') if p.is_file() and 'project' not in p.relative_to(out/'suite').parts]
        r['generated']={p.relative_to(out).as_posix():sha(p) for p in generated}
    else:
        r['donor_unchanged']=all(sha(donor/n)==v['sha256'] for n,v in inputs.items())
        r['runtime_inputs_unchanged']=all(sha(cwd/n)==v['sha256'] for n,v in inputs.items())
        if not r['donor_unchanged'] or not r['runtime_inputs_unchanged']:r['status']='INPUTS_CHANGED'
        for p in cwd.glob('*'):
            if p.is_file() and p.name in ('simulate.log','run.tcl','progress.tsv'):
                (out/'evidence').mkdir(exist_ok=True);shutil.copy2(p,out/'evidence'/p.name)
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
        and not set(p.relative_to(out).parts)&{'project','runtime'} and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}),flush=True)
    return 0 if r['status']=='EXPORTED_PENDING_REVIEW' else 1

if __name__=='__main__':raise SystemExit(main())
