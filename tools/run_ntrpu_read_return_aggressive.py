"""Aggressively refine only the routed physical checkpoint; preserve RTL and timing budgets."""
import argparse,json,os,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_read_return_physical import ROOT,SOURCES as PHYSICAL_SOURCES,sha,save
SOURCES=PHYSICAL_SOURCES+['tools/run_ntrpu_read_return_refine.py','tools/run_ntrpu_read_return_aggressive.py']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    prior=a.prior.resolve();pr=json.loads(a.review.read_text());r0=json.loads((prior/'result.json').read_text())
    assert pr['status']=='READ_RETURN_ROUTED_INTERNAL_TIMING_OPEN' and pr['evidence_validated']
    assert pr['result_sha256']==sha(prior/'result.json') and r0['sources']==r0['sources_after']
    assert all(sha(prior/n)==h for n,h in json.loads((prior/'FILES.json').read_text()).items())
    assert all(sha(ROOT/n)==h for n,h in r0['sources'].items())
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);suite=out/'suite';suite.mkdir()
    donor=prior/'suite/routed.dcp';donor_hash=sha(donor);assert donor_hash==pr['checkpoint_sha256'];shutil.copy2(donor,out/'input.dcp')
    shutil.copy2(a.review,out/'INPUT_PHYSICAL_REVIEW.json');shutil.copy2(prior/'PRIOR_REVIEW.json',out/'PRIOR_REVIEW.json')
    for n in ('clocks.xdc','ntrpu_read_return_physical.sv','hbm_read_width_config.tsv','hbm_read_protocol_config.tsv','hbm_read_cdc_config.tsv'):
        shutil.copy2(prior/'suite'/n,suite/n)
    original=(ROOT/'tools/ntrpu_read_return_physical.tcl').read_text()
    reports=original[original.index('    write_checkpoint [file join $output routed.dcp]'):original.index('    close_project')]
    tcl='''set_param general.maxThreads 2
set output [file normalize suite]
if {[catch {
    open_checkpoint input.dcp
    if {[get_property PERIOD [get_clocks transport_clk]]!=5 || [get_property PERIOD [get_clocks memory_clk]]!=2.222} {error "Changed clock contract"}
    set ntrpu_status_internal_d [get_pins -quiet -hier -regexp {(^|.*/)startup/(fault_meta_reg\\[[0-1]\\]|ready_meta_reg\\[2\\])/D}]
    set ntrpu_status_apb_d [get_pins -quiet -hier -regexp {(^|.*/)startup/ready_meta_reg\\[[0-1]\\]/D}]
    phys_opt_design -directive AggressiveExplore
    route_design -directive AggressiveExplore
'''+reports+'''    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts READ_RETURN_REFINED_EXPORTED
exit 0
'''
    (out/'refine.tcl').write_text(tcl)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={n:sha(ROOT/n) for n in SOURCES};command=['vivado','-mode','batch','-notrace','-source',str(out/'refine.tcl')]
    r=dict(status='RUNNING',experiment='read_return_physical',refinement='post_route_aggressive_explore',sources=sources,
           prior_result_sha256=r0['prior_result_sha256'],input_result_sha256=sha(prior/'result.json'),input_dcp_sha256=donor_hash,
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),command=command,timeout=900)
    save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES},donor_unchanged=sha(donor)==donor_hash)
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==r['sources_after'] and r['donor_unchanged'] else 'FAILED_OR_CHANGED'
    save(out/'result.json',r);save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('status','seconds','returncode')}),flush=True)
if __name__=='__main__':main()
