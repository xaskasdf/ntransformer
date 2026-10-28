"""Read-only bus-skew and clock audit of a reviewed routed transport checkpoint."""
import argparse, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=['tools/run_ntrpu_hbm_route_audit.py']
TCL='''set_param general.maxThreads 2
if {[catch {
    open_checkpoint [lindex $argv 0]
    report_bus_skew -file bus_skew.rpt
    report_clock_utilization -file clocks.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file timing.rpt
    report_exceptions -coverage -file exceptions.rpt
    report_route_status -file route_status.rpt
    set f [open global_buffers.tsv w]
    foreach c [lsort [get_cells -hier -filter {REF_NAME =~ BUFG*}]] {
        set input [get_nets -of_objects [get_pins $c/I]]
        set output [get_nets -of_objects [get_pins $c/O]]
        puts $f "$c\\t[get_property REF_NAME $c]\\t[get_property LOC $c]\\t$input\\t$output\\t[get_clocks -of_objects [get_pins $c/O]]"
    }
    close $f
    close_design
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
puts NTRPU_HBM_ROUTE_AUDIT_COMPLETE
exit 0
'''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    r0=json.loads((a.prior/'result.json').read_text());review=json.loads(a.review.read_text())
    files=json.loads((a.prior/'FILES.json').read_text());source=a.prior/'suite/physical_route.dcp'
    assert r0['stage']=='route' and r0['sources']==r0['sources_after'] and review['routed'] and review['evidence_validated']
    assert sha(source)==files['suite/physical_route.dcp']==review['checkpoint_sha256']
    dest=out/'input.dcp';shutil.copy2(source,dest);shutil.copy2(a.review,out/'PRIOR_REVIEW.json')
    (out/'run.tcl').write_text(TCL)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
    r=dict(status='RUNNING',experiment='hbm_route_audit',sources={n:sha(ROOT/n) for n in SOURCES},
           input_sha256=sha(dest),prior_result_sha256=sha(a.prior/'result.json'),
           timeout=300,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0))
    save(out/'result.json',r);start=time.monotonic();expired=False
    cmd=['vivado','-mode','batch','-notrace','-source','run.tcl','-log','vivado.log','-journal','vivado.jou','-tclargs',str(dest)]
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=300)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,command=cmd,
             sources_after={n:sha(ROOT/n) for n in SOURCES},input_after=sha(dest),donor_after=sha(source))
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] and r['input_sha256']==r['input_after']==r['donor_after'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds')}),flush=True)
if __name__=='__main__':main()
