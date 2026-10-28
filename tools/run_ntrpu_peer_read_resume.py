"""Resume a verified completed checkpoint without repeating earlier stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from run_ntrpu_read_return import ROOT, sha, save


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--checkpoint',choices=('latest','synthesized','optimized','placed'),default='latest')
    a=ap.parse_args();prior=a.prior.resolve();out=a.output.resolve()
    donor=json.loads((prior/'result.json').read_text())
    assert donor['experiment']=='peer_read_physical' and donor['status']!='RUNNING'
    assert donor['sources']==donor['sources_after']
    for n,h in json.loads((prior/'FILES.json').read_text()).items():assert sha(prior/n)==h,n
    for n,h in donor['sources'].items():assert sha(ROOT/n)==h,n
    reviewed=json.loads((prior/'PRIOR_REVIEW.json').read_text())
    assert reviewed['status']=='PEER_READ_SINGLE_SPAN_REVIEWED' and reviewed['full'] and reviewed['pair']
    assert reviewed['result_sha256']==donor['prior_result_sha256']
    stage=a.checkpoint
    if stage=='latest':
        stage=next(name for name in ('placed','optimized','synthesized') if (prior/'suite'/(name+'.dcp')).exists())
    dcp=prior/'suite'/(stage+'.dcp');input_hash=sha(dcp);donor_hash=sha(prior/'result.json')
    out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    shutil.copyfile(dcp,out/'input.dcp')
    shutil.copyfile(prior/'result.json',out/'DONOR_RESULT.json')
    shutil.copyfile(prior/'PRIOR_REVIEW.json',out/'PRIOR_REVIEW.json')
    if (prior/'CUT_REVIEW.json').exists():
        shutil.copyfile(prior/'CUT_REVIEW.json',out/'CUT_REVIEW.json')
    for name in ['clocks.xdc','ntrpu_peer_read_physical.sv','synthesis_utilization.rpt','synthesis_cdc.rpt']:
        shutil.copyfile(prior/'suite'/name,suite/name)
    flow=(ROOT/'tools/ntrpu_peer_read_physical.tcl').read_text()
    first={'synthesized':'    opt_design','optimized':'    place_design','placed':'    phys_opt_design'}[stage]
    tail=flow[flow.index(first):]
    if stage=='synthesized' and 'write_checkpoint [file join $output optimized.dcp]' not in tail:
        tail=tail.replace('    place_design','    write_checkpoint [file join $output optimized.dcp]\n    place_design',1)
    if stage!='placed' and 'write_checkpoint [file join $output placed.dcp]' not in tail:
        tail=tail.replace('    route_design','    write_checkpoint [file join $output placed.dcp]\n    route_design',1)
    header='''set output [file normalize [lindex $argv 0]]
if {[catch {
    set_param general.maxThreads 4
    open_checkpoint [file join $output .. input.dcp]
'''
    # Recover Tcl variables for reporting; retain the checkpoint's constraints.
    xdc=(ROOT/'fpga/constraints/ntrpu_hbm_startup_cdc.xdc').read_text()
    for line in xdc.splitlines():
        if line.startswith('set ntrpu_status_'):header+='    '+line+'\n'
    (out/'resume.tcl').write_text(header+tail)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);os.nice(10);os.environ['OMP_NUM_THREADS']='4'
    sources={n:sha(ROOT/n) for n in [*donor['sources'],'tools/run_ntrpu_peer_read_resume.py']}
    command=['vivado','-mode','batch','-notrace','-source',str(out/'resume.tcl'),'-tclargs',str(suite)]
    result=dict(status='RUNNING',experiment='peer_read_physical',resumed=True,resume_stage=stage,
                sources=sources,command=command,timeout=1200,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),
                prior_result_sha256=donor['prior_result_sha256'],donor_result_sha256=donor_hash,input_dcp_sha256=input_hash)
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    result.update(returncode=code,timed_out=expired,seconds=round(time.monotonic()-start,3),
        sources_after={n:sha(ROOT/n) for n in sources},donor_unchanged=sha(dcp)==input_hash and sha(prior/'result.json')==donor_hash)
    result['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and result['sources_after']==sources and result['donor_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
                          if p.is_file() and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}))


if __name__=='__main__':main()
