"""Check the real section composition, including separated admission sources."""
from check_ntrpu_axil_fault_r3 import review as review_faults


def review(text, period, expect_revoke=1, stop_source=1):
    result = review_faults(text, period, expect_revoke, stop_source)
    for case in result['fault_cases']:
        rows = case['events']
        if rows[0]['gate_sources'] != 0:
            raise ValueError('clean admission source baseline')
        if rows[1]['gate_sources'] != (2 if case['case'] == 0 else 0):
            raise ValueError('registered admission fault source identity')
        if rows[-2]['gate_sources'] != 0:
            raise ValueError('both actual admission source reports removed')
    result['actual_admission_certificate_composition'] = True
    return result
