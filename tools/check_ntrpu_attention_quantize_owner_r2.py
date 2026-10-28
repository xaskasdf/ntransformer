"""Independently check the fixed 73-job protocol plan; not arithmetic evidence."""
import re


FIELDS = {
    'FOREIGN': 'cycle job mode sequence epoch',
    'HELD': 'cycle job mode', 'RESTORE': 'cycle job mode sequence epoch',
    'BEGIN': 'cycle job mode sequence epoch', 'START': 'cycle job',
    'DONE': 'cycle job status', 'ACK': 'cycle job sequence epoch',
    'RESULT': 'cycle job index data', 'REVOKED': 'cycle job mode',
    'COMPLETE': 'cycle job mode status sequence epoch',
    'ABORT_RESET': 'cycle job mode', 'RESET': 'cycle job count',
    'PASS': 'seed watchdog jobs completions results starts dones acks resets reads writes held cycles',
}
STATUS = {1:3, 2:2, 3:2, 4:2, 5:11, 6:7, 7:7, 8:2, 9:11,
          10:11, 11:11, 12:4, 13:4, 14:4, 15:4, 16:4, 17:4,
          18:5, 19:11, 20:4, 21:8, 22:7, 23:2, 33:11, 34:11}


def record_word(job, index, seed):
    if index % 8 < 4:
        return int.from_bytes(bytes((job + index + byte + seed) % 255 - 127 & 255
                                    for byte in range(8)), 'little')
    if index % 8 == 4:
        return 0x3ff0000000000000 | (job * 512 + index)
    return 0


def review(text, seed):
    assert not re.search(r'(?mi)^\s*(?:FATAL|Fatal:|ERROR|AQOWNER_FAIL)', text)
    events = []
    for line in text.splitlines():
        if not line.startswith('AQOWNER_'):
            continue
        tokens = line.split(); tag = tokens.pop(0).removeprefix('AQOWNER_')
        assert tag in FIELDS, 'Unknown event'
        pairs = [token.split('=') for token in tokens]
        assert all(len(pair) == 2 for pair in pairs), 'Malformed field'
        assert [pair[0] for pair in pairs] == FIELDS[tag].split(), 'Event schema'
        values = {}
        for name, value in pairs:
            hexadecimal = name in ('sequence', 'epoch', 'data')
            assert re.fullmatch('[0-9a-f]{16}' if hexadecimal else '[0-9]+', value)
            values[name] = int(value, 16 if hexadecimal else 10)
        events.append((tag, values))
    assert events and events[-1][0] == 'PASS'
    assert sum(tag == 'PASS' for tag, _ in events) == 1
    summary = events[-1][1]
    assert summary['seed'] == seed and summary['watchdog'] == 64
    assert {k:summary[k] for k in ('jobs','completions','results','starts','dones','acks','resets')} == dict(
        jobs=73, completions=64, results=19968, starts=66, dones=64, acks=63, resets=10)
    # Peer traffic totals are counters asserted by the frozen testbench, not per-beat traces.
    assert summary['reads'] == 115522 and summary['writes'] == 32320
    assert summary['held'] == 21629 and 180000 < summary['cycles'] < 220000
    rows = events[:-1]
    assert rows[0] == ('RESET', dict(cycle=1, job=0, count=1))
    times = [v['cycle'] for _, v in rows]
    assert times == sorted(times) and summary['cycles'] > times[-1], 'Cycle ordering'
    plan = [0] + [mode for fault in range(1,37) for mode in (fault,0)]
    begins = [i for i, (tag, _) in enumerate(rows) if tag == 'BEGIN']
    assert len(begins) == 73 and begins[0] == 1
    reset_count = 1
    for job, (offset, mode) in enumerate(zip(begins, plan), 1):
        batch = rows[offset:begins[job] if job < 73 else len(rows)]
        sequence, epoch = 0x1234567800000000 + job, 0xfedcba9800000000 + job
        assert all(v['job'] == job for _, v in batch), 'Cross-owner event'
        begin = batch[0][1]
        assert begin['mode'] == mode and begin['sequence'] == sequence and begin['epoch'] == epoch
        if mode >= 31:
            permanent = mode in (31,32,35,36)
            tags = ['BEGIN','START'] + (['FOREIGN','DONE'] if mode <= 32 else ['DONE','FOREIGN'])
            tags += ['HELD'] + (['ABORT_RESET','RESET'] if permanent else ['RESTORE','ACK','COMPLETE'])
            assert [tag for tag, _ in batch] == tags, 'Foreign release event order'
            bytag = dict(batch)
            for tag, value in batch:
                if 'mode' in value:
                    assert value['mode'] == mode
            assert bytag['DONE']['status'] == 0
            assert bytag['START']['cycle'] > begin['cycle']
            foreign = bytag['FOREIGN']
            assert foreign['sequence'] == (sequence ^ ((1 << 40) if mode in (31,33,35) else 0))
            assert foreign['epoch'] == (epoch ^ ((1 << 44) if mode in (32,34,36) else 0))
            assert bytag['HELD']['cycle'] >= max(foreign['cycle'],bytag['DONE']['cycle']) + 32, 'Foreign hold interval'
            if permanent:
                reset_count += 1
                assert bytag['RESET']['count'] == reset_count
            else:
                for tag in ('RESTORE','ACK','COMPLETE'):
                    assert bytag[tag]['sequence'] == sequence and bytag[tag]['epoch'] == epoch, 'Restored ownership identity'
                assert bytag['ACK']['cycle'] > bytag['RESTORE']['cycle']
                assert bytag['COMPLETE']['status'] == 11, 'Foreign fault must stay sticky'
            continue
        want = ['BEGIN']
        if mode not in (1,2,3,4,5,6,30):
            want.append('START')
        if mode == 28:
            # ABORT is emitted while ap_done is high; DONE and ABORT share an edge.
            middle = [tag for tag, _ in batch[2:-1]]
            assert sorted(middle) == ['ABORT_RESET', 'DONE']
            want.extend(middle)
        elif mode not in (1,2,3,4,5,6,26,27,30):
            want.append('DONE')
        if mode not in (2,3,26,27,28,30):
            want.append('ACK')
        if mode >= 26:
            if mode != 28:
                want.append('ABORT_RESET')
            want.append('RESET'); reset_count += 1
        else:
            if STATUS.get(mode, 0) == 0:
                want.extend(['RESULT'] * 512)
            if mode in (24,25):
                want.append('REVOKED')
            want.append('COMPLETE')
        assert [tag for tag, _ in batch] == want, f'Job {job}: ownership/event order'
        result_index = 0
        for tag, value in batch:
            if 'mode' in value:
                assert value['mode'] == mode
            if tag == 'ACK':
                assert value['sequence'] == sequence and value['epoch'] == epoch, 'ACK full identity'
            elif tag == 'DONE':
                assert value['status'] == (2 if mode == 18 else 0)
            elif tag == 'RESET':
                assert value['count'] == reset_count
            elif tag == 'RESULT':
                assert value['index'] == result_index
                assert value['data'] == record_word(job, result_index, seed), 'ABI record bytes'
                result_index += 1
            elif tag == 'COMPLETE':
                assert value['status'] == STATUS.get(mode, 0), 'Completion status'
                assert value['sequence'] == sequence and value['epoch'] == epoch, '64-bit identity'
        tagged = {tag: v for tag, v in batch if tag != 'RESULT'}
        if 'ACK' in tagged and 'DONE' in tagged:
            assert tagged['ACK']['cycle'] > tagged['DONE']['cycle'], 'Premature release'
        if 'START' in tagged:
            assert tagged['START']['cycle'] > begin['cycle']
    assert reset_count == 10
    return dict(seed=seed, jobs=73, completions=64, output_words=19968,
                resets=10, summary=summary, arithmetic_qualified=False)


def controls(text, seed):
    def remove(tag):
        return re.sub(r'^AQOWNER_' + tag + r' [^\n]*\n', '', text, count=1, flags=re.M)
    variants = {
        'missing_begin': remove('BEGIN'), 'missing_start': remove('START'),
        'missing_done': remove('DONE'), 'missing_ack': remove('ACK'),
        'missing_word': remove('RESULT'), 'missing_completion': remove('COMPLETE'),
        'missing_revocation': remove('REVOKED'), 'missing_reset': remove('RESET'),
        'missing_abort': remove('ABORT_RESET'),
        'word_data': re.sub(r'data=[0-9a-f]{16}', 'data=0000000000000001', text, count=1),
        'word_order': text.replace('index=0 ', 'index=1 ', 1),
        'sequence_high': text.replace('sequence=12345678', 'sequence=02345678', 1),
        'epoch_high': text.replace('epoch=fedcba98', 'epoch=eedcba98', 1),
        'status': re.sub(r'(AQOWNER_COMPLETE[^\n]*status=)0', r'\g<1>1', text, count=1),
        'cycle': re.sub(r'(AQOWNER_ACK cycle=)\d+', r'\g<1>0', text, count=1),
        'seed': text.replace(f'PASS seed={seed} ', f'PASS seed={seed+1} ', 1),
        'coverage': text.replace('jobs=73 ', 'jobs=72 ', 1),
        'unknown_event': text + '\nAQOWNER_UNKNOWN cycle=0 job=0\n',
        'diagnostic': text + '\nAQOWNER_FAIL injected\n',
        'duplicate_pass': text + next(line for line in text.splitlines() if line.startswith('AQOWNER_PASS')) + '\n',
    }
    variants.update({
        'missing_foreign': remove('FOREIGN'), 'missing_held': remove('HELD'),
        'missing_restore': remove('RESTORE'),
        'foreign_identity': re.sub(r'(AQOWNER_FOREIGN[^\n]*sequence=)[0-9a-f]{16}', r'\g<1>0000000000000000', text, count=1),
        'ack_high_identity': re.sub(r'(AQOWNER_ACK[^\n]*epoch=)fedcba98', r'\g<1>00000000', text, count=1),
        'short_foreign_hold': re.sub(r'(AQOWNER_HELD cycle=)[0-9]+', r'\g<1>0', text, count=1),
    })
    rejected = []
    for name, changed in variants.items():
        assert changed != text
        try:
            review(changed, seed)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Accepted ' + name)
    return rejected
