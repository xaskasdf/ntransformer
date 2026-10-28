"""Audit the extra setup requirement and the restored nominal timing contract."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_read_return_physical import timing_summary
from check_ntrpu_read_frames import sha


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['refinement']=='post_route_extra_setup_100ps'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and not r['timed_out'] and r['returncode']==0
    assert r['sources']==r['sources_after'] and r['donor_unchanged']
    files=json.loads((out/'FILES.json').read_text())
    assert all(sha(out/n)==h for n,h in files.items())
    s=out/'suite';flow=(out/'refine.tcl').read_text()
    strict='set_clock_uncertainty -setup 0.200 [get_clocks memory_clk]'
    nominal='set_clock_uncertainty -setup 0.100 [get_clocks memory_clk]'
    assert flow.count(strict)==flow.count(nominal)==1
    assert flow.index(strict)<flow.index('phys_opt_design')<flow.index('route_design')
    assert flow.index('route_design')<flow.index('guard_timing.rpt')<flow.index(nominal)
    assert flow.index(nominal)<flow.index('write_checkpoint')<flow.index('report_timing_summary -report_unconstrained')
    values={k:float(v) for k,v in (line.split('\t') for line in (s/'guardband.tsv').read_text().splitlines())}
    assert set(values)=={'input_nominal_setup_ns','input_guard_setup_ns','guard_setup_ns','nominal_setup_ns'}
    assert abs(values['input_nominal_setup_ns']-values['input_guard_setup_ns']-0.100)<0.002
    assert abs(values['nominal_setup_ns']-values['guard_setup_ns']-0.100)<0.002
    for name,uncertainty in [('guard_memory_max.rpt',0.200),('memory_clk_max.rpt',0.100)]:
        text=(s/name).read_text()
        observed=[float(v) for v in re.findall(r'User Uncertainty\s+\(UU\):\s*([.0-9]+)ns',text)]
        assert observed and all(v==uncertainty for v in observed)
    guard=timing_summary((s/'guard_timing.rpt').read_text())
    final=timing_summary((s/'timing.rpt').read_text())
    assert abs(final['setup']['worst_slack_ns']-values['nominal_setup_ns'])<0.002
    return dict(status='READ_CUT_EXTRA_SETUP_AUDITED',result_sha256=sha(out/'result.json'),
                extra_setup_ns=0.100,**values,guard_global_timing=guard,
                guard_met=all(v['worst_slack_ns']>=0 and v['failing_endpoints']==0 for v in guard.values()),
                nominal_constraints_restored=True,clocks_changed=False,board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args()
    print(json.dumps(review(a.output),indent=2))
