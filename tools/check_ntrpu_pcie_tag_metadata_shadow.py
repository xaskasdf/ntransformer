"""Validate full R7 shadow coverage and real omission/order mutation failures."""
import re
from check_ntrpu_pcie_request_inventory_r5 import review as protocol_review


def review(logs, vectors, manifest, baseline_log):
    normal = logs['normal/simulate.log']
    protocol = protocol_review(normal, vectors, manifest)
    if normal.splitlines().count('META_SHADOW_MODE mode=0') != 1:
        raise ValueError('Wrong normal shadow mode')
    if normal.splitlines().count('META_SHADOW_SUITE_PASS rows=22339 scenarios=68 mode=0') != 1:
        raise ValueError('Incomplete shadow suite')
    rows = [line for line in normal.splitlines() if line.startswith('META_SHADOW_ROW ')]
    protocol_rows = [line for line in normal.splitlines() if line.startswith('PCIE_INV_ROW ')]
    baseline_rows = [line for line in baseline_log.splitlines() if line.startswith('PCIE_INV_ROW ')]
    if protocol_rows != baseline_rows or len(rows) != 22339:
        raise ValueError('R7 output changed or shadow rows missing')
    for index, (shadow, protocol_row) in enumerate(zip(rows, protocol_rows)):
        scenario = re.search(r' scenario=(\d+) ', protocol_row)[1]
        if shadow != f'META_SHADOW_ROW index={index} scenario={scenario} words=256 queries=4 mode=0':
            raise ValueError('Missing table/query comparison or wrong order')
    mutants = []
    for name, mode, failure_row, failure in [
        ('omit_rq', 1, 4, 'META_SHADOW_MISMATCH row=4 tag=0 mode=1'),
        ('reverse_writes', 2, 211, 'META_SHADOW_LOOKUP_MISMATCH row=211 mode=2')]:
        log = logs[name + '/simulate.log']
        if log.splitlines().count(f'META_SHADOW_MODE mode={mode}') != 1:
            raise ValueError('Mutation was not enabled')
        fatals = [line for line in log.splitlines() if line.startswith('Fatal:')]
        if fatals != ['Fatal: ' + failure] or 'META_SHADOW_SUITE_PASS' in log:
            raise ValueError('Expected actual mutant failure missing')
        before = [line for line in log.splitlines() if line.startswith('PCIE_INV_ROW ')]
        if before != baseline_rows[:failure_row]:
            raise ValueError('Reference changed before mutation was caught')
        matched = [line for line in log.splitlines() if line.startswith('META_SHADOW_ROW ')]
        expected = [line.replace(' mode=0', f' mode={mode}') for line in rows[:failure_row]]
        if matched != expected:
            raise ValueError('Mutation prefix is incomplete')
        mutants.append(dict(mode=name, detected_at_row=failure_row, failure=failure))
    return dict(status='PASS_PCIE_METADATA_R7_SHADOW_AND_MUTATIONS', protocol=protocol,
        rows=len(rows), scenarios=68, table_word_comparisons=len(rows)*256,
        lookup_slots_compared=len(rows)*4, all_r7_emitted_states_identical=True,
        actual_mutants=mutants, active_pipeline_integrated=False)
