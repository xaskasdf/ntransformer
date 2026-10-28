"""Replay public peer payload and lifetime evidence, independently of RTL state."""
import argparse
import hashlib
import json
from pathlib import Path


def replay(text, full):
    assert 'PR ALL_PASS cases=21 full='+str(int(full)) in text
    assert not any(x in text for x in ('PR_FAIL', 'PR_WATCHDOG', 'Fatal:'))
    assert 'PR CONFIG_PRIORITY_PASS' in text and 'PR CONFIG_REVOKE_PASS' in text and 'PR GLOBAL_STOP_PASS' in text
    cases = set()
    jobs = {}
    rejected = 0
    quarantines = set()
    payload_bytes = 0
    results = 0
    for line in text.splitlines():
        p = line.split()
        if p[:2] == ['PR', 'CASE']:
            assert p[3] == 'PASS'
            cases.add(int(p[2]))
        if p[:2] == ['PR', 'REJECT']:
            rejected += 1
        if p[:2] == ['PR', 'QUARANTINE']:
            assert int(p[4]) > 0
            quarantines.add(int(p[2]))
        if len(p) < 5 or p[0] != 'PR' or p[1] not in ('CMD', 'DATA', 'LOCAL', 'DMA', 'ACK', 'RESULT'):
            continue
        kind = p[1]
        case, epoch, lane, seq = map(int, p[2:6])
        key = (epoch, lane, seq)
        if kind == 'CMD':
            assert key not in jobs
            jobs[key] = dict(address=int(p[6]), length=int(p[7])*64, received=0,
                             local=False, dma=False, ack=False, result=False)
            continue
        # A failure result may precede reader acceptance of a held command.
        if kind == 'RESULT' and int(p[6]) != 0:
            assert int(p[7]) == 0
            results += 1
            continue
        assert key in jobs, (kind, key)
        job = jobs[key]
        if kind == 'DATA':
            beat, last = int(p[6]), int(p[7])
            assert beat*64 == job['received']+64
            assert bool(last) == (beat*64 == job['length'])
            raw = bytes.fromhex(p[8])[::-1]
            assert len(raw) == 64
            for offset, value in enumerate(raw):
                address = job['address']+job['received']+offset
                assert value == ((address ^ (address>>8) ^ (address>>16) ^ (lane*91)) & 255)
            job['received'] += 64
            payload_bytes += 64
        elif kind == 'LOCAL':
            status, length = map(int, p[6:8])
            if status == 0:
                assert length == job['length'] == job['received']
                job['local'] = True
        elif kind in ('DMA', 'ACK'):
            status, length = map(int, p[6:8])
            # Fault cases deliberately inject malformed/early notifications.
            if case == 2:
                assert status == 0 and length == job['length'] == job['received']
                assert not job[kind.lower()]
                job[kind.lower()] = True
        elif kind == 'RESULT':
            status, length = map(int, p[6:8])
            assert status == 0 and length == job['length']
            assert job['local'] and job['dma'] and job['ack'] and not job['result']
            job['result'] = True
            results += 1
    assert cases == set(range(1, 22))
    assert rejected == 29
    assert quarantines == set(range(4, 22))
    assert 'PR REUSE_PASS transfers=16' in text
    assert sum(j['result'] for j in jobs.values()) == 20
    return dict(cases=len(cases), rejections=rejected, quarantines=len(quarantines),
                commands=len(jobs), result_handshakes=results, payload_bytes=payload_bytes)


def controls(text, full):
    mutations = [text.replace('PR ALL_PASS', 'PR NOT_PASS', 1),
                 '\n'.join(x for x in text.splitlines() if not x.startswith('PR ACK 2 ')),
                 '\n'.join(x for x in text.splitlines() if not x.startswith('PR LOCAL 2 ')),
                 '\n'.join(x for x in text.splitlines() if not x.startswith('PR DMA 2 ')),
                 '\n'.join(x for x in text.splitlines() if not x.startswith('PR QUARANTINE 20 '))]
    for line in text.splitlines():
        if line.startswith('PR DATA 2 '):
            p=line.split();p[-1]=format(int(p[-1],16)^1,'0128x')
            mutations.append(text.replace(line,' '.join(p),1));break
    for mutation in mutations:
        try:
            replay(mutation, full)
        except (AssertionError, KeyError, ValueError):
            continue
        raise AssertionError('Negative control accepted')
    return len(mutations)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args()
    out=a.output;r=json.loads((out/'result.json').read_text())
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out']
    assert r['sources']==r['sources_after']
    for name,digest in json.loads((out/'FILES.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    log=(out/('suite/simulation.log' if r['full'] else 'simulation.log')).read_text()
    print(json.dumps(dict(status='PEER_READ_SINGLE_SPAN_REVIEWED',full=r['full'],pair=r.get('pair',False),
        result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),
        **replay(log,r['full']),negative_controls=controls(log,r['full'])),indent=2))


if __name__=='__main__':main()
