"""Measure placement/routing of the isolated prototype; external IO stays unqualified."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_dual_ingress_packed import ROOT,sha,save


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);prior=a.prior.resolve()
    previous=json.loads((prior/'result.json').read_text());review=json.loads(a.review.read_text())
    assert review['status']=='PACKED_DUAL_INGRESS_SYNTHESIS_REVIEWED' and review['serial']==0
    assert review['result_sha256']==sha(prior/'result.json')
    inputs=json.loads((prior/'FILES.json').read_text());assert all(sha(prior/n)==h for n,h in inputs.items())
    assert previous['sources']==previous['sources_after']
    design=['fpga/experimental/ntrpu_dual_ingress_packed.sv','fpga/experimental/ntrpu_dual_allocator.sv']
    assert all(sha(ROOT/n)==previous['sources'][n] for n in design)
    sources={n:sha(ROOT/n) for n in design+['tools/run_ntrpu_dual_ingress_packed_route.py','tools/run_ntrpu_dual_ingress_packed.py']}
    donor=prior/'suite/dual_ingress_packed.dcp';donor_hash=sha(donor);shutil.copy2(donor,out/'input.dcp')
    shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    tcl='''set_param general.maxThreads 2
if {[catch {
    open_checkpoint input.dcp
    if {[llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks source]]!=5} {error "Wrong clock contract"}
    opt_design
    place_design
    phys_opt_design
    route_design
    report_utilization -file utilization.rpt
    report_timing_summary -report_unconstrained -file timing.rpt
    check_timing -verbose -file check_timing.rpt
    report_route_status -file route_status.rpt
    report_drc -file drc.rpt
    write_checkpoint routed.dcp
    write_verilog -mode funcsim routed.v
    set f [open summary.tsv w]
    foreach mode {max min} {
        puts $f "${mode}_slack_ns\\t[get_property SLACK [get_timing_paths -delay_type $mode -max_paths 1]]"
    }
    puts $f "latches\\t[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ LD*}]]"
    puts $f "blackboxes\\t[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts PACKED_DUAL_INGRESS_ROUTED_EXPORTED
exit 0
'''
    (out/'route.tcl').write_text(tcl)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[-4:-2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    cmd=['vivado','-mode','batch','-notrace','-source','route.tcl']
    r=dict(status='RUNNING',experiment='dual_ingress_packed_route',sources=sources,prior_result_sha256=sha(prior/'result.json'),
           input_dcp_sha256=donor_hash,timeout=900,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),command=cmd)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,
             donor_unchanged=sha(donor)==donor_hash,sources_after={n:sha(ROOT/n) for n in sources})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['donor_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and not p.name.startswith('.')})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)


if __name__=='__main__':main()
