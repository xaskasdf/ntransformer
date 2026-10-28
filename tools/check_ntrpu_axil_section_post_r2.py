"""Full conservation regression plus actual guard alarm and held orphan."""
import re
from check_ntrpu_axil_section_post import review as review_previous


def review(text, period, expect_revoke=1, stop_source=1):
    if re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:)', text, re.M):
        raise ValueError('simulator failure')
    result = review_previous(text, period, expect_revoke, stop_source)
    for case in result['fault_cases']:
        rows = case['events']
        ident = case['case']
        if rows[0]['guard_sources'] != 0:
            raise ValueError('clean guard baseline')
        expected = [1, 2, 12][ident]
        if any(r['fault_mgmt'] != (4 if ident == 2 and stop_source == 1
                                  and r['name'] in ['AFTER_PROPAGATION', 'FRESH_ID_BLOCKED']
                                  else expected) for r in rows[2:]):
            raise ValueError('exact management source composition')
        if rows[-2]['guard_sources'] != 0:
            raise ValueError('all actual guard source reports removed')
        if ident == 2:
            if any(r['guard_sources'] != 1-stop_source for r in [rows[2], rows[3]]) or rows[-1]['guard_sources'] != 1:
                raise ValueError('orphan B guard source')
            if any(r['raw_bvalid'] != (0 if stop_source == 1 and r['name'] in ['AFTER_PROPAGATION', 'FRESH_ID_BLOCKED'] else 1)
                   or r['raw_bready'] != 0 or r['protected_bvalid'] != 0
                   for r in rows[2:]):
                raise ValueError('raw orphan remains below protected manager boundary')
        elif any(r['guard_sources'] != 0 for r in rows):
            raise ValueError('no unexpected guard alarm')
    result['actual_response_guard'] = True
    result['direct_h_injection_bypasses_guard'] = True
    return result
