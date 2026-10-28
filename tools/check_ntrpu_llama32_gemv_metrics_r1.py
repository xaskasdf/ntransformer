"""Validate complete Llama slice metrics using the existing GEMV counter checker."""
import re
import check_ntrpu_gemv_metrics as counters


def records(text, prefix, fields):
    result = []
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        tokens = line[len(prefix):].split(' ')
        if not tokens or any(not re.fullmatch(r'[a-z][a-z0-9_]*=\d+', token) for token in tokens):
            raise ValueError('Invalid counter syntax')
        pairs = [token.split('=') for token in tokens]
        if len({key for key, value in pairs}) != len(pairs):
            raise ValueError('Duplicate counter field')
        row = {key: int(value) for key, value in pairs}
        if set(row) != fields:
            raise ValueError('Missing or extra counter fields')
        result.append(row)
    return result


def review(text, manifest):
    case_fields = {'id', 'rows', 'columns', 'scenario', 'status', 'cycles',
                   'first_result', 'bursts', 'outputs'}
    cases = records(text, 'NTRPU_GEMV_CASE ', case_fields)
    metrics = records(text, 'NTRPU_GEMV_METRIC ', counters.FIELDS)
    jobs = manifest['jobs']
    expected_ids = [job['id'] for job in jobs]
    if expected_ids != list(range(1, 29)):
        raise ValueError('Expected all 28 Llama jobs')
    if [row['id'] for row in cases] != expected_ids or [row['id'] for row in metrics] != expected_ids:
        raise ValueError('Missing, duplicate or unordered job records')
    if re.findall(r'^NTRPU_GEMV_VARIANT lanes=(\d+) layout=(\d+)\s*$', text, re.M) != [('2', '2')]:
        raise ValueError('Dual packed implementation required')
    for case, job in zip(cases, jobs):
        expected = dict(rows=job['rows'], columns=job['k_blocks'] * 32, scenario=0,
                        status=0, outputs=job['rows'])
        if any(case[key] != value for key, value in expected.items()):
            raise ValueError('Case differs from the actual fixture')
    result = counters.review(text, 2)
    if result['status'] != 'PASS' or result['successful_jobs'] != 28:
        raise ValueError('Counter reconciliation: ' + repr(result['errors']))
    result['cases'] = cases
    result['scope'] = ('All 28 Llama slice jobs in RTL simulation; phase cycles partition each job. '
                       'Busy, wait and stall counters overlap. Bytes/cycle excludes inter-job gaps. '
                       'No hardware frequency, DDR bandwidth, tokens/s or full-model inference claim.')
    return result


def corruption_controls(text, manifest):
    review(text, manifest)
    line = next(line for line in text.splitlines() if line.startswith('NTRPU_GEMV_METRIC id=1 '))
    variants = [('missing_record', text.replace(line, '', 1)),
                ('duplicate_record', text.replace(line, line + '\n' + line, 1)),
                ('duplicate_field', text.replace(line, line + ' id=1', 1)),
                ('extra_field', text.replace(line, line + ' extra=0', 1)),
                ('negative_counter', text.replace(line, re.sub(r'idle=\d+', 'idle=-1', line), 1))]
    for name, field, value in [('wrong_cycle_partition', 'idle', '999999'),
                               ('missing_mac_blocks', 'block_ops', '0'),
                               ('missing_payload_beats', 'r_beats', '0'),
                               ('missing_activation_writes', 'activation_writes', '0'),
                               ('missing_second_lane', 'lane1_busy', '0'),
                               ('missing_lane_overlap', 'both_busy', '0'),
                               ('wrong_record_identity', 'id', '28')]:
        changed = re.sub(r'(?<!\w)' + field + r'=\d+', field + '=' + value, line)
        variants.append((name, text.replace(line, changed, 1)))
    variants.append(('wrong_shape', text.replace('NTRPU_GEMV_CASE id=1 rows=32',
                                                'NTRPU_GEMV_CASE id=1 rows=31', 1)))
    variants.append(('wrong_layout', text.replace('NTRPU_GEMV_VARIANT lanes=2 layout=2',
                                                 'NTRPU_GEMV_VARIANT lanes=2 layout=1', 1)))
    rejected = []
    for name, changed in variants:
        if changed == text:
            raise ValueError('Mutation did not alter evidence: ' + name)
        try:
            review(changed, manifest)
        except (ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError('Accepted corrupt counter evidence: ' + name)
    if len(rejected) != 14:
        raise AssertionError('Incomplete corruption controls')
    return rejected
