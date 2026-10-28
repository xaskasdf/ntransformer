"""Review internal dual-clock timing and CDC separately from unqualified IO."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_dual_hbm import sha
from check_ntrpu_hbm_audit import cdc,report_checks

def timing_summary(text):
    match=re.search(r'^\s*WNS\(ns\)[^\n]*\n\s*[- ]+\n\s*([^\n]+)',text,re.M)
    assert match is not None,'Missing design timing summary'
    values=list(map(float,match[1].split()));assert len(values)==12
    result={}
    for i,name in enumerate(('setup','hold','pulse_width')):
        worst,total,failed,endpoints=values[i*4:i*4+4]
        assert endpoints>0 and endpoints.is_integer() and failed.is_integer() and 0<=failed<=endpoints
        result[name]=dict(worst_slack_ns=worst,total_slack_ns=total,failing_endpoints=int(failed),endpoints=int(endpoints))
    return result

def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='dual_hbm_physical'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    prior=json.loads((out/'PRIOR_REVIEW.json').read_text());assert prior['status']=='DUAL_HBM_PREFLIGHT_REVIEWED' and prior['cases']==11 and prior['policy']==1
    assert prior['result_sha256']==r['prior_result_sha256']
    if r.get('refinement'):
        base=json.loads((out/'INPUT_PHYSICAL_REVIEW.json').read_text())
        assert base['evidence_validated'] and base['result_sha256']==r['input_result_sha256'] and r['donor_unchanged']
        assert base['checkpoint_sha256']==r['input_dcp_sha256']==sha(out/'input.dcp')
    suite=out/'suite';s={k:float(v) for k,v in (line.split('\t') for line in (suite/'summary.tsv').read_text().splitlines())}
    assert s['latches']==s['blackboxes']==0 and s['startup_async_registers']==21
    assert s['period_transport_clk']==5 and s['period_memory_clk']==2.222
    assert s['internal_status_pins']==3 and s['external_status_pins']==2
    clocks=(suite/'clocks.xdc').read_text();assert 'set_clock_groups' not in clocks
    assert 'set_clock_uncertainty -setup 0.100' in clocks and 'set_clock_uncertainty -hold 0.050' in clocks
    wrapper=(suite/'ntrpu_dual_hbm_physical.sv').read_text();assert wrapper.count('BUFGCE ')==2
    counts=cdc(suite/'cdc.rpt');assert set(counts)<={'CDC-3','CDC-9'} and sum(counts.values())>0
    text=(suite/'cdc.rpt').read_text()
    for endpoint in ('fault_meta_reg[0]/D','fault_meta_reg[1]/D','ready_meta_reg[2]/D'):
        assert any(endpoint in line and 'CDC-3' in line and 'Max Delay Datapath Only' in line for line in text.splitlines())
    coverage=report_checks(suite/'check_timing.rpt')
    assert all(coverage[k]==0 for k in ('no_clock','unconstrained_internal_endpoints','loops','latch_loops','multiple_clock'))
    drc=(suite/'drc.rpt').read_text();rows=re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error)\s*\|[^|]+\|\s*(\d+)\s*\|',drc,re.M)
    assert sum(int(n) for _,_,n in rows)==int(re.search(r'Checks found:\s+(\d+)',drc)[1]),'Unparsed DRC'
    assert all(level=='Warning' and rule in ('PDCN-1569','RTSTAT-10') for rule,level,n in rows),'Unreviewed DRC'
    routing={k:int(v) for k,v in re.findall(r'# of (.+?)\.+\s*:\s*(\d+)',(suite/'route_status.rpt').read_text())}
    assert routing['fully routed nets']==routing['routable nets']>0 and routing['nets with routing errors']==0
    skew=(suite/'bus_skew.rpt').read_text();skews=[tuple(map(float,x)) for x in re.findall(r'^\s+(?:Slow|Fast)\s+([.0-9]+)\s+([.0-9]+)\s+([-0-9.]+)\s*$',skew,re.M)]
    assert len(skews)==12 and all(req==2.222 and value>=0 and abs(req-value-slack)<.002 for req,value,slack in skews)
    slacks={k:v for k,v in s.items() if k.startswith(('max_','min_'))}
    global_timing=timing_summary((suite/'timing.rpt').read_text())
    closed=all(v>=0 for v in slacks.values()) and all(v[2]>=0 for v in skews) and all(v['worst_slack_ns']>=0 and v['failing_endpoints']==0 and v['total_slack_ns']==0 for v in global_timing.values())
    util=(suite/'utilization.rpt').read_text()
    def count(name):return float(re.search(r'\|\s*'+re.escape(name)+r'\s*\|\s*([\d.]+)',util)[1])
    return dict(status='DUAL_HBM_ROUTED_INTERNAL_TIMING_'+('PASS' if closed else 'OPEN'),evidence_validated=True,
                result_sha256=sha(out/'result.json'),seconds=r['seconds'],clock_slack_ns=slacks,cdc=counts,global_timing=global_timing,
                luts=count('CLB LUTs'),registers=count('CLB Registers'),bram36_equivalent=count('Block RAM Tile'),
                minimum_bus_skew_slack_ns=min(v[2] for v in skews),bus_skew_constraints=len(skews),coverage=coverage,
                drc_findings=rows,drc_clean=not rows,fully_routed_nets=routing['fully routed nets'],
                checkpoint_sha256=sha(suite/'routed.dcp'),native_hbm_in_design=False,external_io_qualified=False,
                post_route_simulated=False,board_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
