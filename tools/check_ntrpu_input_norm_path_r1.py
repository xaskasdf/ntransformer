"""Independent AXI payload/identity/retirement review; HLS is a replay peer."""
import hashlib
import json
from pathlib import Path
import re

PLAN = [0] * 4 + [m for mode in range(1, 32) for m in (mode, 0)]
SCHEMA = {
    'BEGIN': 'cycle job mode sample sequence epoch', 'AR': 'cycle job address beats',
    'R': 'cycle job index response data', 'B': 'cycle job index address response data',
    'START': 'cycle job', 'DONE': 'cycle job return',
    'COMPLETE': 'cycle job status hls valid read abi ar r b',
    'ABORT': 'cycle job mode', 'RESET': 'cycle job', 'REVOKE': 'cycle job mode',
    'ACK': 'cycle job valid revoked',
    'PASS': 'seed jobs publics pairs resets starts dones ar r b cycles',
}
HEX = {'sequence': 16, 'epoch': 16, 'address': 16, 'data': 128}
STATUS = {1: 2, 2: 2, 3: 11, 4: 1, 5: 3, 6: 5, 7: 1, 8: 3, 9: 7,
          10: 7, 11: 2, 12: 7, 13: 7, 20: 7, 21: 7, 22: 8, **{m: 1 for m in range(23, 32)}}
PREFLIGHT = {1, 2, 3, 4, 7, 20, *range(23, 32)}


def fixture(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['parent_manifest_sha256'] == 'f5b4413ece6c6bd9da2fa772fab69502eef3ce1c39d802afe4d1c44087af9626'
    assert manifest['model_sha256'] == '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
    contents = {name: (folder / name).read_bytes() for name in manifest['files']}
    assert all(hashlib.sha256(contents[n]).hexdigest() == h for n, h in manifest['files'].items())
    parent_raw = contents['parent-manifest.json']
    assert hashlib.sha256(parent_raw).hexdigest() == manifest['parent_manifest_sha256']
    parent = json.loads(parent_raw)
    for n, raw in contents.items():
        if n.endswith('.bin'):
            assert hashlib.sha256(raw).hexdigest() == parent['files'][n]
    values = {n: [int(v, 16) for v in contents[n].split()] for n in ('source32.hex', 'abi64.hex')}
    assert len(values['source32.hex']) == 16384 and len(values['abi64.hex']) == 2048
    for i in range(4):
        source = contents[f'sample-{i+1:03d}-embedding-fp32.bin'] + contents['attn-norm-fp32.bin']
        output = contents[f'sample-{i+1:03d}-input-abi1.bin']
        assert b''.join(v.to_bytes(4, 'little') for v in values['source32.hex'][i*4096:(i+1)*4096]) == source
        assert b''.join(v.to_bytes(8, 'little') for v in values['abi64.hex'][i*512:(i+1)*512]) == output
    return values


def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|INPATH_FAIL)', text)
    rows = []
    for line in text.splitlines():
        if not line.startswith('INPATH_'):
            continue
        parts = line.split()
        tag = parts[0][7:]
        assert tag in SCHEMA
        fields = [s.split('=') for s in parts[1:]]
        assert all(len(f) == 2 for f in fields) and [f[0] for f in fields] == SCHEMA[tag].split()
        for k, v in fields:
            assert re.fullmatch('[0-9a-fA-F]{'+str(HEX[k])+'}' if k in HEX else r'\d+', v)
        rows.append((tag, {k: int(v, 16 if k in HEX else 10) for k, v in fields}))
    return rows


def review_events(rows, values, seed):
    assert rows[0][0] == 'RESET' and rows[0][1]['job'] == 0 and rows[-1][0] == 'PASS'
    assert sum(k == 'PASS' for k, _ in rows) == 1
    summary = rows[-1][1]
    assert (summary['seed'], summary['jobs'], summary['publics'], summary['pairs'], summary['resets']) == (seed, 66, 63, 38, 5)
    buckets = {j: {k: [] for k in SCHEMA} for j in range(1, 67)}
    last_job = 0
    last_cycle = -1
    for tag, row in rows:
        if 'cycle' in row:
            assert row['cycle'] >= last_cycle
            last_cycle = row['cycle']
        if tag == 'BEGIN':
            assert row['job'] == last_job + 1
            last_job = row['job']
        if row.get('job', 0):
            assert row['job'] == last_job
            buckets[last_job][tag].append(row)
    assert last_job == 66 and summary['cycles'] > last_cycle
    totals = {k: 0 for k in ('ar', 'r', 'b', 'starts', 'dones')}
    ack_valid = 0
    for job, mode in enumerate(PLAN, 1):
        e = buckets[job]
        assert len(e['BEGIN']) == 1
        begin = e['BEGIN'][0]
        sample = (job-1) % 4
        assert (begin['mode'], begin['sample'], begin['sequence'], begin['epoch']) == (
            mode, sample, 0x1234567800000000+job, 0xfedcba9800000000+job)
        source = [0x8000000fc0+p*0x100000000+job*16384 for p in range(2)]
        output = 0x3300000fc0+job*16384
        cursor = 0
        for ar in e['AR']:
            assert cursor < 256 and ar['address'] == source[cursor//128]+(cursor % 128)*64
            assert ar['beats'] == min(128-cursor % 128, (4096-ar['address'] % 4096)//64)
            responses = e['R'][cursor:cursor+ar['beats']]
            assert len(responses) == ar['beats'] or mode == 17
            if cursor:
                assert e['R'][cursor-1]['cycle'] < ar['cycle']
            for r in responses:
                assert r['index'] == cursor and ar['cycle'] < r['cycle']
                if cursor:
                    assert e['R'][cursor-1]['cycle'] < r['cycle']
                want = sum(values['source32.hex'][sample*4096+cursor*16+l] << (l*32) for l in range(16))
                assert r['data'] == want and r['response'] == (2 if mode == 5 and cursor == 3 else 0)
                cursor += 1
        assert cursor == len(e['R'])
        for i, r in enumerate(e['B']):
            assert i < 64 and r['index'] == i and r['address'] == output+i*64
            want = sum(values['abi64.hex'][sample*512+i*8+l] << (l*64) for l in range(8))
            assert r['data'] == want and r['response'] == (2 if mode == 8 and i == 32 else 0)
            assert len(e['DONE']) == 1 and e['DONE'][0]['cycle'] < r['cycle']
            if i:
                assert e['B'][i-1]['cycle'] < r['cycle']
        if e['START']:
            assert len(e['START']) == 1 and len(e['R']) == 256 and e['R'][-1]['cycle'] < e['START'][0]['cycle']
        if e['DONE']:
            assert len(e['DONE']) == len(e['START']) == 1
            assert e['START'][0]['cycle'] < e['DONE'][0]['cycle']
            assert e['DONE'][0]['return'] == (6 if mode == 6 else 0)
        if mode in (17, 18, 19):
            assert not e['COMPLETE'] and not e['ACK']
        else:
            assert len(e['COMPLETE']) == 1
            c = e['COMPLETE'][0]
            status = STATUS.get(mode, 0)
            assert (c['status'], c['hls'], c['valid']) == (status, 6 if mode == 6 else 0, int(status == 0))
            delivered = next((r['index'] for r in e['R'] if r['response'] != 0), len(e['R']))
            assert (c['read'], c['ar'], c['r'], c['b']) == (delivered, len(e['AR']), len(e['R']), len(e['B']))
            assert c['abi'] == sum(r['response'] == 0 for r in e['B'])
            for tag in ('R', 'B', 'DONE'):
                if e[tag]:
                    assert e[tag][-1]['cycle'] < c['cycle']
            if status == 0:
                assert len(e['R']) == 256 and len(e['B']) == 64 and len(e['DONE']) == 1
            if mode == 16:
                assert not e['ACK']
            else:
                assert len(e['ACK']) == 1
                a = e['ACK'][0]
                valid = int(status == 0 and mode not in (14, 15))
                assert (a['valid'], a['revoked']) == (valid, 1-valid)
                assert c['cycle'] < a['cycle']
                ack_valid += valid
        if mode in (16, 17, 18, 19):
            assert len(e['ABORT']) == len(e['RESET']) == 1
            assert e['ABORT'][0]['mode'] == mode and e['ABORT'][0]['cycle'] < e['RESET'][0]['cycle']
        else:
            assert not e['ABORT'] and not e['RESET']
        if mode in (14, 15):
            assert len(e['REVOKE']) == 1 and e['REVOKE'][0]['mode'] == mode
            assert e['COMPLETE'][0]['cycle'] < e['REVOKE'][0]['cycle'] < e['ACK'][0]['cycle']
        else:
            assert not e['REVOKE']
        if mode in PREFLIGHT:
            assert not any(e[tag] for tag in ('AR', 'R', 'B', 'START', 'DONE'))
        if mode in (5, 9):
            assert not e['START'] and not e['B']
        for key, tag in [('ar','AR'), ('r','R'), ('b','B'), ('starts','START'), ('dones','DONE')]:
            totals[key] += len(e[tag])
    assert all(summary[k] == v for k, v in totals.items()) and ack_valid == 35
    return dict(seed=seed, jobs=66, completions=63, successful_offers=38, accepted_results=ack_valid,
                accepted_bytes=ack_valid*4096, cycles=summary['cycles'], totals=totals,
                arithmetic_qualified=False, hls_peer_replays_reference=True,
                scope='Independent AXI R/B bytes and retirement review. AW/W hold and input RAM forwarding use frozen-bank assertions.')


def review(text, values, seed):
    return review_events(parse(text), values, seed)


def controls(text, values, seed):
    rows = parse(text)
    rejected = []
    changes = [('BEGIN','sequence',1<<40), ('BEGIN','sample',1), ('AR','address',64),
               ('AR','beats',1), ('R','data',1), ('R','index',1), ('R','response',2),
               ('B','data',1), ('B','response',2), ('B','cycle',1<<40), ('DONE','return',1),
               ('COMPLETE','status',1), ('COMPLETE','read',1), ('COMPLETE','abi',1),
               ('ACK','valid',1), ('ACK','revoked',1), ('REVOKE','mode',1), ('PASS','pairs',1)]
    def reject(name, bad):
        try:
            review_events(bad, values, seed)
        except (AssertionError, KeyError, IndexError):
            rejected.append(name)
        else:
            raise AssertionError('Accepted corruption '+name)
    for tag, key, mask in changes:
        i = next(i for i, (t, r) in enumerate(rows) if t == tag)
        bad = rows.copy()
        r = dict(rows[i][1])
        r[key] ^= mask
        bad[i] = (tag, r)
        reject(tag+'.'+key, bad)
    for tag, criterion in [('B', lambda r: r['index'] == 63), ('ACK', lambda r: True), ('REVOKE', lambda r: True)]:
        i = next(i for i, (t, r) in enumerate(rows) if t == tag and criterion(r))
        reject('missing-'+tag, rows[:i]+rows[i+1:])
    i = next(i for i, (t, _) in enumerate(rows) if t == 'ACK')
    reject('duplicate-ACK', rows[:i]+[rows[i]]+rows[i:])
    assert len(rejected) == 22
    return rejected
