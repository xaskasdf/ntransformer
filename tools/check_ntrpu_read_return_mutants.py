"""Require the intended RTL fault and its diagnostic, not just any failed run."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_read_return import sha

def review(local):
    out=local/'output';r=json.loads((out/'result.json').read_text());m=json.loads((local/'MUTATION.json').read_text())
    assert r['experiment']=='read_return' and r['status']=='FAILED_OR_CHANGED'
    assert r['returncode']!=0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    assert m['base_sha256']!=m['mutant_sha256']==r['sources'][m['source']]
    assert sha(local/'sources'/m['source'])==m['mutant_sha256']
    log=(out/'suite/simulation.log').read_text();assert 'RR ALL_PASS' not in log
    cases=[int(n) for n in re.findall(r'^RR PASS (\d+) ',log,re.M)]
    if m['mutation']=='late_error':
        assert cases==list(range(1,11))
        assert re.search(r'Fatal: RR_FAIL case=11 .* DATA_IDENTITY',log)
        assert any(s.startswith('RR NR ') and s.split()[6]=='2' and s.split()[7]=='1' for s in log.splitlines())
        detection='late error wrongly published payload; bench DATA_IDENTITY rejected it'
    elif m['mutation']=='ready_path':
        assert cases==[] and 'RR CASE 1 ' in log
        assert re.search(r'Fatal: RR_FAIL case=1 .* READY_COMBINATORIAL_DEPENDENCE',log)
        detection='response inputs changed READY between clock edges; dedicated boundary probe rejected it'
    else:
        assert m['mutation']=='rid_mask' and cases==list(range(1,13))
        assert 'RR_WATCHDOG' in log
        assert any(s.startswith('RR NR ') and s.split()[5]=='32' for s in log.splitlines())
        assert 'RR CASE 13 ' in log and 'RR CASE 14 ' not in log
        detection='upper RID bit accepted; required quarantine never asserted; bench watchdog rejected it'
    return dict(status='READ_RETURN_RTL_MUTANT_REJECTED',mutation=m['mutation'],detection=detection,
                result_sha256=sha(out/'result.json'),mutation_sha256=sha(local/'MUTATION.json'),seconds=r['seconds'],
                outer_timeout=False,qualified_rtl_modified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('local',type=Path);a=ap.parse_args();print(json.dumps(review(a.local),indent=2))
