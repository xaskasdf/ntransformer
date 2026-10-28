"""Audit routed public transactions and match their outcomes to the RTL reference."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_read_frames import sha,replay,controls

def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='read_frames_routed' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['returncode']==0 and not r['timed_out'] and r['donor_unchanged']
    assert r['sources']==r['sources_after'] and r['generated']==r['generated_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    p=json.loads((out/'PHYSICAL_REVIEW.json').read_text());ref=json.loads((out/'REFERENCE_REVIEW.json').read_text())
    assert p['status']=='READ_FRAMES_ROUTED_INTERNAL_TIMING_PASS' and p['result_sha256']==r['physical_result_sha256']
    assert ref['status']=='READ_FRAMES_PREFLIGHT_REVIEWED' and ref['result_sha256']==r['reference_result_sha256']
    suite=out/'suite';meta=json.loads((suite/'COMPARISON.json').read_text())
    assert meta['netlist_sha256']==meta['isolated_sha256']==r['netlist_sha256']==sha(suite/'route_isolated.v')
    assert not any(meta[k] for k in ('sdf','force','netlist_edited','vendor_primitives_edited','reset_phase_guard','cycle_equivalence'))
    bench=(suite/'tb_ntrpu_read_return.sv').read_text()
    assert 'ntrpu_read_return_physical dut(.*);' in bench and 'dut.channels' not in bench
    assert 'predicted' not in bench and 'force ' not in bench
    lines=(suite/'simulation.log').read_text().splitlines()
    assert not any(re.match(r'\s*(ERROR:|Fatal:|FAIL )',s) for s in lines)
    counts=replay(lines,routed=True)
    reference=out/'REFERENCE_SIMULATION.log';assert sha(reference)==r['reference_trace_sha256']
    expected=replay(reference.read_text().splitlines())
    assert counts['transaction_digest']==expected['transaction_digest']
    assert counts['cases']==12 and counts['completions']==56 and counts['quarantined_commands']==4
    elaborate=next((out/'tool_logs').rglob('elaborate.log')).read_text()
    assert 'Compiling module read_return_routed.ntrpu_read_return_physical' in elaborate
    return dict(status='READ_FRAMES_ROUTED_PUBLIC_TRANSACTIONS_PASS',result_sha256=sha(out/'result.json'),
        seconds=r['seconds'],**counts,negative_controls=controls(lines,routed=True),
        netlist_sha256=r['netlist_sha256'],reference_result_sha256=r['reference_result_sha256'],
        reference_trace_sha256=r['reference_trace_sha256'],transaction_outcomes_match=True,
        exact_cycle_equivalence=False,formal_equivalence=False,sdf=False,board_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
