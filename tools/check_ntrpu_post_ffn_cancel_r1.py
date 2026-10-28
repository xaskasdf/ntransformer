"""Review full handoff protocol plus the two-sided cancellation comparison."""
import re
import check_ntrpu_post_ffn_handoff_r1 as protocol

def review(text,seed):
    summary=protocol.review(text,seed)
    rows=re.findall(r'^PFCANCEL_PASS seed=(\d+) jobs=(\d+) delayed_samples=(\d+) compared_edges=(\d+)$',text,re.M)
    assert len(rows)==1
    actual,jobs,delayed,edges=map(int,rows[0])
    assert actual==seed and jobs==522 and delayed>=22
    assert summary['cycles']-32<=edges<summary['cycles']
    return dict(summary,compared_outputs=43,delayed_cancel_samples=delayed,compared_edges=edges,
        maximum_cancel_sampling_edges=1,public_result_revocation_immediate=True,
        scope='R2/R3 RTL comparison with controlled peers across 522 jobs. No arithmetic, physical timing or throughput qualification.')

def controls(text,seed):
    rejected=protocol.controls(text,seed)
    for key,value in [('seed',seed+1),('jobs',521),('delayed_samples',0),('compared_edges',0)]:
        altered=re.sub(r'(^PFCANCEL_PASS [^\n]*?\b'+key+r'=)\d+',lambda m:m[1]+str(value),text,flags=re.M)
        assert altered!=text
        try:review(altered,seed)
        except (AssertionError,KeyError,ValueError):rejected.append('comparison-'+key)
        else:raise AssertionError('Accepted altered comparison '+key)
    altered='\n'.join(s for s in text.splitlines() if not s.startswith('PFCANCEL_PASS'))+'\n'
    try:review(altered,seed)
    except AssertionError:rejected.append('missing-comparison')
    else:raise AssertionError('Accepted missing comparison')
    return rejected
