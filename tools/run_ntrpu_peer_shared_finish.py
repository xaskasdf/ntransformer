"""Resume the completed placement of a bounded shared-source route attempt."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_peer_read import sha,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--placed-review',type=Path,required=True)
    a=ap.parse_args();prior=a.prior.resolve();out=a.output.resolve()
    donor=json.loads((prior/'result.json').read_text());placed=json.loads(a.placed_review.read_text())
    assert donor['experiment']=='peer_shared_physical' and donor['stage']=='route' and donor['status']!='RUNNING'
    assert donor['sources']==donor['sources_after'] and donor['donor_unchanged']
    assert all(sha(ROOT/n)==h for n,h in donor['sources'].items())
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    assert placed['status']=='PLACED_CHECKPOINT_ONLY_NOT_ROUTING_APPROVAL'
    assert placed['sources']==donor['sources'] and placed['synthesis_result_sha256']==donor['prior_result_sha256']
    dcp=prior/'suite/placed.dcp';digest=sha(dcp);assert digest==placed['checkpoint_sha256']
    synth=json.loads((prior/'PRIOR_REVIEW.json').read_text());assert synth['status']=='SHARED_SYNTHESIS_REVIEWED'
    assert synth['result_sha256']==donor['prior_result_sha256']==sha(prior/'DONOR_RESULT.json')
    out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    shutil.copyfile(dcp,out/'input.dcp');shutil.copyfile(a.placed_review,out/'PLACED_PROVENANCE.json')
    shutil.copyfile(prior/'result.json',out/'RESUME_RESULT.json')
    for n in ('PRIOR_REVIEW.json','DONOR_RESULT.json'):shutil.copyfile(prior/n,out/n)
    for p in (prior/'suite').iterdir():
        if p.is_file() and p.suffix!='.dcp':shutil.copyfile(p,suite/p.name)
    flow=(ROOT/'tools/ntrpu_peer_read_physical.tcl').read_text()
    tail=flow[flow.index('    phys_opt_design -directive Explore'):]
    header='''set output [file normalize [lindex $argv 0]]
if {[catch {
    set_param general.maxThreads 8
    open_checkpoint [file join $output .. input.dcp]
'''
    for line in (ROOT/'fpga/constraints/ntrpu_hbm_startup_cdc.xdc').read_text().splitlines():
        if line.startswith('set ntrpu_status_'):header+='    '+line+'\n'
    (out/'flow.tcl').write_text(header+tail.replace('PEER_READ_PHYSICAL_EXPORTED','SHARED_PHYSICAL_EXPORTED'))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8]);os.nice(10);os.environ['OMP_NUM_THREADS']='8'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    cmd=['vivado','-mode','batch','-notrace','-source',str(out/'flow.tcl'),'-tclargs',str(suite)]
    result=dict(status='RUNNING',experiment='peer_shared_physical',stage='route',resumed=True,
                resume_result_sha256=sha(prior/'result.json'),prior_result_sha256=donor['prior_result_sha256'],
                input_dcp_sha256=digest,sources=sources,command=cmd,timeout=1800,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    result.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
                  sources_after={n:sha(ROOT/n) for n in sources},donor_unchanged=sha(dcp)==digest and sha(prior/'result.json')==result['resume_result_sha256'])
    result['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and result['sources_after']==sources and result['donor_unchanged'] and 'SHARED_PHYSICAL_EXPORTED' in (out/'console.log').read_text() else 'FAILED_OR_CHANGED'
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}))


if __name__=='__main__':main()
