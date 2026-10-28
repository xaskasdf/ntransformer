"""Review cycle comparisons and the original independent payload/lease replay."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_read_return import sha,replay,controls
def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='read_return_netlist'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out'] and r['donor_unchanged']
    assert r['sources']==r['sources_after'] and r['generated']==r['generated_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    p=json.loads((out/'PHYSICAL_REVIEW.json').read_text());ref=json.loads((out/'REFERENCE_REVIEW.json').read_text())
    assert p['status'] in ('READ_RETURN_ROUTED_INTERNAL_TIMING_PASS','READ_RETURN_ROUTED_INTERNAL_TIMING_OPEN') and p['result_sha256']==r['physical_result_sha256']
    assert ref['status']=='READ_RETURN_PREFLIGHT_REVIEWED' and ref['result_sha256']==r['reference_result_sha256']
    suite=out/'suite';comparison=json.loads((suite/'COMPARISON.json').read_text())
    assert comparison['netlist_sha256']==comparison['isolated_sha256']==r['netlist_sha256']==sha(suite/'route_isolated.v')
    assert comparison['renamed_modules']==0 and not comparison['sdf'] and not comparison['force'] and comparison['latency_edges']==0
    log=(suite/'simulation.log').read_text();assert not re.search(r'(?im)^\s*(Fatal:|ERROR:|FAIL )',log)
    counts=replay(log.splitlines());negative=controls(log.splitlines(),0)
    if counts['revision']>=1:
        assert comparison.get('injection_cursor_sampling')=='falling_memory_edge'
        bench=(suite/'tb_ntrpu_read_return.sv').read_text()
        assert 'always @(negedge mem_clk) injection_cursor=native_reads[0];' in bench
        assert '(injection_cursor==inject_index)' in bench and '(native_reads[0]==inject_index)' not in bench
    final=re.findall(r'^DM ALL_PASS source=(\d+) memory=(\d+) resets=(\d+)$',log,re.M);assert len(final)==1
    src,mem,resets=map(int,final[0]);assert src>0 and mem>0 and resets==12
    starts={int(case):int(ts) for case,ts in re.findall(r'^RR CASE (\d+) (\d+)$',log,re.M)}
    ends={int(case):int(ts) for case,ts in re.findall(r'^RR PASS (\d+) (\d+)$',log,re.M)}
    case_samples={}
    for kind,count in (('SOURCE',src),('MEMORY',mem)):
        values=[tuple(map(int,x)) for x in re.findall(r'^DM '+kind+r' (\d+) (\d+)$',log,re.M)]
        assert [c for ts,c in values]==list(range(1,count+1)) and all(b[0]>a[0] for a,b in zip(values,values[1:])),kind+' missing comparisons'
        case_samples[kind]={case:sum(starts[case]<=ts<=ends[case] for ts,c in values) for case in starts}
        assert len(case_samples[kind])==20 and all(case_samples[kind].values()),kind+' unobserved scenario'
    assert len(re.findall(r'^DM RESET ',log,re.M))==resets
    elaborate=next((out/'tool_logs').rglob('elaborate.log')).read_text()
    assert 'Compiling module read_return_routed.ntrpu_read_return_physical' in elaborate
    assert 'Compiling module xil_defaultlib.ntrpu_hbm_read_return' in elaborate
    for kind,needle in (('PAYLOAD','data'),('READY','cmd_ready')):
        damaged=(suite/f'negative_{kind}.log').read_text()
        assert 'DM_MISMATCH '+needle in damaged and 'DM ALL_PASS' not in damaged
    return dict(status='READ_RETURN_ROUTED_FUNCTIONAL_REGRESSION_PASS',result_sha256=sha(out/'result.json'),seconds=r['seconds'],
                internal_timing_status=p['status'],**counts,source_samples=src,memory_samples=mem,cold_reset_checks=resets,case_samples=case_samples,negative_controls=negative,
                comparator_injections=['PAYLOAD','READY'],netlist_sha256=r['netlist_sha256'],formal_equivalence=False,
                injection_cursor_sampling=comparison.get('injection_cursor_sampling','legacy_active_edge'),
                sdf=False,native_hbm_in_design=False,board_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
