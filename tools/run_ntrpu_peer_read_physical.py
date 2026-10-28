"""Route the reviewed dual-clock composition; external IO and board remain open."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_read_return import ROOT,SOURCES as BASE_SOURCES,sha,save
from run_ntrpu_peer_read import NEW
SOURCES=list(dict.fromkeys(BASE_SOURCES+NEW+['tools/run_ntrpu_peer_read_physical.py','tools/ntrpu_peer_read_physical.tcl','fpga/constraints/ntrpu_hbm_startup_cdc.xdc']))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--cpus',type=int,choices=range(1,9),default=2)
    ap.add_argument('--timeout',type=int,default=1200)
    ap.add_argument('--cut-review',type=Path)
    a=ap.parse_args()
    assert 0<a.timeout<=3600
    prior=a.prior.resolve();pr=json.loads(a.review.read_text());r0=json.loads((prior/'result.json').read_text())
    assert pr['status']=='PEER_READ_SINGLE_SPAN_REVIEWED' and pr['full'] and pr['pair'] and pr['cases']==21
    assert pr['result_sha256']==sha(prior/'result.json') and r0['sources']==r0['sources_after']
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    assert all(sha(ROOT/n)==h for n,h in r0['sources'].items())
    variant=json.loads((ROOT/'VARIANT.json').read_text())
    cut_review=None
    if variant.get('return_cut'):
        assert a.cut_review is not None,'Return cut needs its independent queue review'
        cut_review=json.loads(a.cut_review.read_text())
        assert cut_review['status']=='PEER_READ_CUT_FUNCTIONAL_REVIEWED'
        assert cut_review['result_sha256']==sha(prior/'result.json')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'suite').mkdir();shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    declaration=(ROOT/'fpga/experimental/ntrpu_peer_read_pair.sv').read_text().split('module ntrpu_peer_read_pair (',1)[1].split('\n);',1)[0]
    wrapper=('`timescale 1ns/1ps\nmodule ntrpu_peer_read_physical ('+declaration+'\n);\n'
             'wire core_clk,core_mem_clk;\n'
             "BUFGCE source_clock(.I(clk),.CE(1'b1),.O(core_clk));\n"
             "BUFGCE memory_clock(.I(mem_clk),.CE(1'b1),.O(core_mem_clk));\n"
             'ntrpu_peer_read_pair core(.clk(core_clk),.mem_clk(core_mem_clk),.*);\nendmodule\n')
    (out/'suite/ntrpu_peer_read_physical.sv').write_text(wrapper)
    for n in ('hbm_read_width_config.tsv','hbm_read_protocol_config.tsv','hbm_read_cdc_config.tsv'):shutil.copy2(prior/'suite'/n,out/('prior_'+n))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.cpus]);os.nice(10);os.environ['OMP_NUM_THREADS']=str(a.cpus)
    if cut_review:save(out/'CUT_REVIEW.json',cut_review)
    sources={n:sha(ROOT/n) for n in SOURCES}
    command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_peer_read_physical.tcl'),'-tclargs',str(out/'suite'),str(a.cpus)]
    r=dict(status='RUNNING',experiment='peer_read_physical',sources=sources,prior_result_sha256=sha(prior/'result.json'),
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),command=command,timeout=a.timeout,load_before=os.getloadavg())
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and 'project' not in p.relative_to(out).parts and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
if __name__=='__main__':main()
