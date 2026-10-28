"""Check the paired-RAM protocol plan; peer arithmetic is intentionally synthetic."""
import re

FIELDS = {
    'BEGIN': 'cycle job mode sequence epoch', 'START': 'cycle job',
    'DONE': 'cycle job status', 'ACK': 'cycle job',
    'RESIDUAL': 'cycle job index address data',
    'RESULT': 'cycle job index address data',
    'REVOKED': 'cycle job mode', 'COMPLETE': 'cycle job mode status sequence epoch',
    'ABORT_RESET': 'cycle job mode', 'RESET': 'cycle job count',
    'PASS': 'seed watchdog jobs completions results starts dones acks resets reads writes residual_results residual_writes held cycles',
}
STATUS = {
    1: 3, 2: 2, 3: 2, 4: 2, 5: 11, 6: 7, 7: 7, 8: 2, 9: 11,
    10: 11, 11: 11, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4,
    18: 5, 19: 11, 20: 4, 21: 8, 22: 7, 23: 2,
    31: 4, 32: 4, 33: 4, 34: 4, 35: 4, 36: 4, 37: 11, 38: 11,
    39: 7, 40: 2, 41: 5, 42: 4,
}


def record_word(job, address, seed):
    if address % 8 < 4:
        return int.from_bytes(bytes((job + address + byte + seed) % 255 - 127 & 255
                                    for byte in range(8)), 'little')
    if address % 8 == 4:
        return 0x3ff0000000000000 | (job * 512 + address)
    return 0


def review(text, seed):
    assert not re.search(r'(?mi)^\s*(?:FATAL|Fatal:|ERROR|POWNER_FAIL)', text), 'Diagnostic'
    events = []
    for line in text.splitlines():
        if not line.startswith('POWNER_'):
            continue
        pieces = line.split()
        tag = pieces.pop(0).removeprefix('POWNER_')
        assert tag in FIELDS, 'Unknown event'
        pairs = [piece.split('=') for piece in pieces]
        assert all(len(pair) == 2 for pair in pairs), 'Malformed field'
        assert [pair[0] for pair in pairs] == FIELDS[tag].split(), 'Schema'
        row = {}
        for key, value in pairs:
            hexadecimal = key in ('sequence', 'epoch', 'data')
            width = 8 if tag == 'RESIDUAL' and key == 'data' else 16
            assert re.fullmatch('[0-9a-f]{%d}' % width if hexadecimal else '[0-9]+', value)
            row[key] = int(value, 16 if hexadecimal else 10)
        events.append((tag, row))
    assert events and events[-1][0] == 'PASS'
    assert sum(tag == 'PASS' for tag, _ in events) == 1
    summary = events[-1][1]
    expected = dict(seed=seed, watchdog=64, jobs=85, completions=80,
                    results=23040, residual_results=92160, starts=78, dones=76,
                    acks=79, resets=6)
    assert {k: summary[k] for k in expected} == expected, 'Coverage'
    # These are frozen bank counters, not independently logged input beats.
    # Two finished peers omit ABI1 entirely; one omits residual, and the reset
    # during residual contributes 64 writes. Short/extra write faults balance.
    assert summary['reads'] == 294936 and summary['writes'] == (76 - 2) * 512
    assert summary['residual_writes'] == (76 - 1) * 2048 + 64
    assert summary['held'] == 98169
    assert 350000 < summary['cycles'] < 800000
    rows = events[:-1]
    assert rows[0] == ('RESET', dict(cycle=1, job=0, count=1))
    cycles = [row['cycle'] for _, row in rows]
    assert cycles == sorted(cycles) and summary['cycles'] > cycles[-1], 'Cycle order'
    plan = [0] + [mode for fault in range(1, 43) for mode in (fault, 0)]
    begins = [i for i, (tag, _) in enumerate(rows) if tag == 'BEGIN']
    assert len(begins) == 85 and begins[0] == 1
    reset_count = 1
    metrics = []
    for job, (offset, mode) in enumerate(zip(begins, plan), 1):
        batch = rows[offset:begins[job] if job < len(plan) else len(rows)]
        assert all(row['job'] == job for _, row in batch), 'Foreign owner'
        sequence, epoch = 0x1234567800000000 + job, 0xfedcba9800000000 + job
        begin = batch[0][1]
        assert (begin['mode'], begin['sequence'], begin['epoch']) == (mode, sequence, epoch)
        want = ['BEGIN']
        if mode not in (1, 2, 3, 4, 5, 6, 30):
            want.append('START')
        if mode == 28:
            middle = [tag for tag, _ in batch[2:-1]]
            assert sorted(middle) == ['ABORT_RESET', 'DONE']
            want.extend(middle)
        elif mode not in (1, 2, 3, 4, 5, 6, 26, 27, 30):
            want.append('DONE')
        if mode not in (2, 3, 26, 27, 28, 30):
            want.append('ACK')
        if 26 <= mode <= 30:
            if mode != 28:
                want.append('ABORT_RESET')
            want.append('RESET')
            reset_count += 1
        else:
            if STATUS.get(mode, 0) == 0:
                for index in range(2048):
                    want.append('RESIDUAL')
                    if index < 512:
                        want.append('RESULT')
            if mode in (24, 25):
                want.append('REVOKED')
            want.append('COMPLETE')
        assert [tag for tag, _ in batch] == want, f'Job {job}: ownership and paired output order'
        indices = {'RESIDUAL': 0, 'RESULT': 0}
        for tag, row in batch:
            if 'mode' in row:
                assert row['mode'] == mode
            if tag == 'DONE':
                assert row['status'] == (2 if mode == 18 else 6 if mode == 41 else 0)
            elif tag == 'RESET':
                assert row['count'] == reset_count
            elif tag in indices:
                index = indices[tag]
                address = (index * (5 if tag == 'RESIDUAL' else 3) + seed) % (2048 if tag == 'RESIDUAL' else 512)
                value = (0xbf000000 | (job * 4096 + address)) if tag == 'RESIDUAL' else record_word(job, address, seed)
                assert (row['index'], row['address'], row['data']) == (index, address, value), 'RAM bytes/address'
                indices[tag] += 1
            elif tag == 'COMPLETE':
                assert (row['status'], row['sequence'], row['epoch']) == (STATUS.get(mode, 0), sequence, epoch), 'Completion identity/status'
        tagged = {tag: row for tag, row in batch if tag not in indices}
        if 'ACK' in tagged and 'DONE' in tagged:
            assert tagged['ACK']['cycle'] > tagged['DONE']['cycle'], 'Premature source release'
        if 'START' in tagged:
            assert tagged['START']['cycle'] > begin['cycle']
        if 'COMPLETE' in tagged:
            metrics.append(dict(job=job, mode=mode, status=tagged['COMPLETE']['status'],
                                cycles_including_public_readback=tagged['COMPLETE']['cycle'] - begin['cycle'],
                                residual_words=indices['RESIDUAL'], abi_words=indices['RESULT']))
    assert reset_count == 6
    return dict(seed=seed, jobs=85, completions=80, residual_words=92160,
                abi_words=23040, resets=6, summary=summary, jobs_metrics=metrics,
                arithmetic_qualified=False, scope='Synthetic HLS and loader protocol peers; actual paired RAM owner.')


def controls(text, seed):
    def remove(tag):
        return re.sub(r'^POWNER_' + tag + r' [^\n]*\n', '', text, count=1, flags=re.M)

    variants = {f'missing_{tag.lower()}': remove(tag) for tag in
                ('BEGIN', 'START', 'DONE', 'ACK', 'RESIDUAL', 'RESULT', 'COMPLETE', 'REVOKED', 'RESET', 'ABORT_RESET')}
    variants.update({
        'residual_data': re.sub(r'(POWNER_RESIDUAL[^\n]*data=)[0-9a-f]+', r'\g<1>00000000', text, count=1),
        'abi_data': re.sub(r'(POWNER_RESULT[^\n]*data=)[0-9a-f]+', r'\g<1>0000000000000001', text, count=1),
        'residual_address': re.sub(r'(POWNER_RESIDUAL[^\n]*address=)\d+', r'\g<1>2048', text, count=1),
        'abi_address': re.sub(r'(POWNER_RESULT[^\n]*address=)\d+', r'\g<1>512', text, count=1),
        'residual_order': re.sub(r'(POWNER_RESIDUAL[^\n]*index=)0', r'\g<1>1', text, count=1),
        'abi_order': re.sub(r'(POWNER_RESULT[^\n]*index=)0', r'\g<1>1', text, count=1),
        'sequence_high': text.replace('sequence=12345678', 'sequence=02345678', 1),
        'epoch_high': text.replace('epoch=fedcba98', 'epoch=eedcba98', 1),
        'status': re.sub(r'(POWNER_COMPLETE[^\n]*status=)0', r'\g<1>1', text, count=1),
        'cycle': re.sub(r'(POWNER_ACK cycle=)\d+', r'\g<1>0', text, count=1),
        'seed': text.replace(f'PASS seed={seed} ', f'PASS seed={seed+1} ', 1),
        'coverage': text.replace('jobs=85 ', 'jobs=84 ', 1),
        'unknown': text + '\nPOWNER_UNKNOWN cycle=0 job=0\n',
        'diagnostic': text + '\nPOWNER_FAIL injected\n',
        'duplicate_pass': text + next(line for line in text.splitlines() if line.startswith('POWNER_PASS')) + '\n',
    })
    rejected = []
    for name, changed in variants.items():
        assert changed != text, name
        try:
            review(changed, seed)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Accepted ' + name)
    return rejected
