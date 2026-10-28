"""Review routed internal timing separately from unqualified external IO."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_dual_ingress import sha


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='dual_ingress_route' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['sources']==r['sources_after'] and r['returncode']==0 and not r['timed_out'] and r['donor_unchanged']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    previous=json.loads((out/'PRIOR_REVIEW.json').read_text())
    assert previous['status']=='DUAL_INGRESS_SYNTHESIS_REVIEWED' and previous['serial']==0
    assert previous['result_sha256']==r['prior_result_sha256'] and sha(out/'input.dcp')==r['input_dcp_sha256']
    summary={k:float(v) for line in (out/'summary.tsv').read_text().splitlines() for k,v in [line.split('\t')]}
    assert summary['latches']==summary['blackboxes']==0
    assert summary['max_slack_ns']>=0 and summary['min_slack_ns']>=0
    routing=(out/'route_status.rpt').read_text()
    assert int(re.search(r'nets with routing errors\.*\s*:\s*(\d+)',routing)[1])==0
    full=int(re.search(r'fully routed nets\.*\s*:\s*(\d+)',routing)[1]);assert full>0
    assert 'Design State : Fully Routed' in (out/'drc.rpt').read_text() and 'Checks found: 0' in (out/'drc.rpt').read_text()
    checks=(out/'check_timing.rpt').read_text()
    for name in ('no_clock','unconstrained_internal_endpoints','loops','latch_loops'):assert 'checking '+name+' (0)' in checks
    missing={name:int(re.search(r'checking '+name+r' \((\d+)\)',checks)[1]) for name in ('no_input_delay','no_output_delay')}
    timing=(out/'timing.rpt').read_text();assert 'period=5.000ns' in timing
    for value in (summary['max_slack_ns'],summary['min_slack_ns']):assert f'{value:.3f}ns' in timing
    utilization=(out/'utilization.rpt').read_text()
    def count(name):return int(re.search(r'\|\s*'+re.escape(name)+r'\s*\|\s*(\d+)',utilization)[1])
    return dict(status='DUAL_INGRESS_ROUTED_INTERNAL_TIMING_PASS',setup_ns=summary['max_slack_ns'],hold_ns=summary['min_slack_ns'],
                fully_routed_nets=full,luts=count('CLB LUTs'),registers=count('CLB Registers'),bram36_equivalent=count('Block RAM Tile'),
                missing_external_delays=missing,external_io_qualified=False,routed_netlist_simulated=False,board_qualified=False,
                result_sha256=sha(out/'result.json'),seconds=r['seconds'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
