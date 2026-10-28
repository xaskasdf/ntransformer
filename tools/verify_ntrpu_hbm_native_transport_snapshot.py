"""Load-test a completed XSIM snapshot after its Vivado parent timed out.

Keep the original timeout rejected. Copy its frozen evidence and executable
to a new batch, validate loading at time zero, and record separate provenance.
This does not claim calibration or traffic; those require the native run.
"""
import argparse,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path

SIMDIR='suite/project/hbm_adapter.sim/sim_1/behav/xsim'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('prior',type=Path);a=ap.parse_args()
    out=a.output.resolve();prior=a.prior.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    old=json.loads((prior/'result.json').read_text());files=json.loads((prior/'FILES.json').read_text())
    assert old['stage']=='compile' and old['experiment']=='hbm_native_transport'
    assert old['status']=='FAILED_OR_CHANGED' and old['timed_out'] and old['returncode']==-15
    assert old['sources']==old['sources_after'] and all(sha(prior/n)==h for n,h in files.items())
    text=(prior/'console.log').read_text();assert 'Built simulation snapshot tb_ntrpu_hbm_native_transport_behav' in text
    assert not any(x.lstrip().startswith(('ERROR:','Fatal:')) for x in text.splitlines())
    inputs=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text());assert len(inputs)==15
    for n in files:
        if n.startswith(('suite/','tool_logs/')):
            target=out/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(prior/n,target)
    shutil.copy2(prior/'result.json',out/'TIMED_OUT_PARENT.json')
    shutil.copy2(prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json')
    cwd=out/SIMDIR;cwd.mkdir(parents=True)
    for n,v in inputs.items():
        source=prior/SIMDIR/n;assert sha(source)==v['sha256']
        dest=cwd/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(dest)==v['sha256']
    tcl='''run 0 ns
foreach path {/tb_ntrpu_hbm_native_transport/clk /tb_ntrpu_hbm_native_transport/hbm/memory/inst/TWO_STACK.u_hbm_top/init_seq_complete_0 /tb_ntrpu_hbm_native_transport/dut/started} {
    if {[llength [get_objects -quiet $path]]!=1} {error "Missing native composition object $path"}
}
puts "NTRPU_NATIVE_SNAPSHOT_LOAD_COMPLETE time=[current_time -s]"
quit
'''
    (cwd/'load.tcl').write_text(tcl)
    command=['xsim','tb_ntrpu_hbm_native_transport_behav','-tclbatch','load.tcl','-log','load.log']
    r=dict(status='RUNNING',experiment='hbm_native_transport',stage='compile',operation='verify_completed_snapshot_after_parent_timeout',
           sources=old['sources'],sources_after=old['sources_after'],cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=120,
           prior_result_sha256=sha(prior/'result.json'),prior_manifest_sha256=sha(prior/'SNAPSHOT_INPUTS.json'),
           verifier_sha256=sha(Path(__file__)),command=command)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=120)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    donor_ok=all(sha(prior/SIMDIR/n)==v['sha256'] for n,v in inputs.items())
    copy_ok=all(sha(cwd/n)==v['sha256'] for n,v in inputs.items())
    evidence=out/'load_evidence';evidence.mkdir()
    for name in ('load.tcl','load.log'):shutil.copy2(cwd/name,evidence/name)
    passed=code==0 and not expired and donor_ok and copy_ok and 'NTRPU_NATIVE_SNAPSHOT_LOAD_COMPLETE time=0' in (cwd/'load.log').read_text()
    recovery=dict(status='SNAPSHOT_LOAD_VERIFIED' if passed else 'SNAPSHOT_LOAD_FAILED',original_parent_accepted=False,
                  donor_unchanged=donor_ok,snapshot_unchanged=copy_ok,calibration_executed=False,traffic_executed=False,
                  prior_result_sha256=r['prior_result_sha256'],snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'))
    save(out/'RECOVERY.json',recovery)
    r.update(status='EXPORTED_PENDING_REVIEW' if passed else 'FAILED_OR_CHANGED',returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3))
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts})
    print(json.dumps(r),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
