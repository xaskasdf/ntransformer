"""Review RAM-port evidence independently of the RTL and simulator PASS marker.

Synthetic RAM consumer only: this cannot certify an HLS binding or arithmetic.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def plan():
    jobs = [(0, ([1, 2, 3, 16, 63, 64] + [1] * 26)[h], h) for h in range(32)]
    jobs += [(c, 2, 3) for c in range(1, 12)]
    jobs += [(c, 2, 7) for c in range(12, 25)]
    jobs += [(25, 2, 0), (26, 1, 0)]
    jobs += [(c, 2, 31) for c in range(27, 33)]
    jobs += [(33, 1, 0), (34, 1, 0), (35, 1, 5), (36, 1, 5), (37, 1, 0), (38, 1, 0)]
    for c in range(39, 43):
        jobs += [(c, 2, 2), (43, 1, 3)]
    jobs += [(44, 1, 0), (45, 1, 0), (46, 1, 0)]
    assert len(jobs) == 81
    return jobs


def word(job, plane, address):
    return 0x3f000000 ^ (job * 65536) ^ (plane * 32768) ^ address


def review(text, seed, watchdog):
    assert not re.search(r'(?im)^\s*(?:fatal:|error:|fail\b|AKR_FAIL)', text), 'simulator error'
    events = []
    for line in text.splitlines():
        if not line.startswith('AKR_'):
            continue
        pieces = line.split()
        assert all(re.fullmatch(r'\w+=[0-9a-f]+', p) for p in pieces[1:]), line
        pairs = [p.split('=') for p in pieces[1:]]
        assert len(dict(pairs)) == len(pairs), 'duplicate field'
        fields = {k: int(v, 16 if k in ('data', 'k', 'v') else 10) for k, v in pairs}
        events.append((pieces[0], fields))
    assert events[0] == ('AKR_START', dict(seed=seed, watchdog=watchdog))
    assert events[1] == ('AKR_RESET', dict(job=0, kind=0))
    counts = Counter(tag for tag, _ in events)
    assert counts == Counter(AKR_START=1, AKR_RESET=5, AKR_CMD=81, AKR_TILE=1824,
                             AKR_COMMIT=69, AKR_COMPUTE=45, AKR_READ=12487, AKR_DONE=77, AKR_PASS=1), counts
    expected_pass = dict(seed=seed, watchdog=watchdog, jobs=81, resets=5, commands=81,
                         tiles=1824, commits=69, computes=45, reads=12487, dones=77)
    assert events[-1] == ('AKR_PASS', expected_pass)
    groups = []
    for tag, fields in events[2:-1]:
        if tag == 'AKR_CMD':
            groups.append([])
        assert groups and fields['job'] == len(groups), 'missing/reordered command or event'
        groups[-1].append((tag, fields))
    assert len(groups) == 81
    read_words = 0
    for job, ((kind, tokens, head), group) in enumerate(zip(plan(), groups), 1):
        start = (job % 5) * 64
        command = dict(job=job, kind=kind, start=start, tokens=tokens, position=start+tokens-1,
                       layer=job % 16, head=head, seq=1000+job, epoch=100, lease=10000+job, expected=100)
        if kind == 1: command['tokens'] = 0
        if kind == 2: command['tokens'] = 65
        if kind == 3: command['position'] = start
        if kind == 4: command['position'] = 131072
        if kind == 5: command['layer'] = 16
        if kind == 6: command['head'] = 32
        if kind == 7: command['seq'] = 0
        if kind == 8: command['epoch'] = 0
        if kind == 9: command['lease'] = 0
        if kind == 10: command['expected'] = 101
        if kind == 11: command.update(start=0xffffffff, position=0)
        assert group[0] == ('AKR_CMD', command), ('descriptor', job)
        identity = {k: command[k] for k in ('job', 'seq', 'epoch', 'lease')}
        invalid = 1 <= kind <= 11
        count = 0 if invalid or kind == 33 else tokens*8
        if kind in (25, 34, 39): count = 1
        if kind in (27, 28): count = 2
        if kind == 26: count = 9
        reset = 39 <= kind <= 42
        committed = not invalid and kind != 39
        consumed = kind in (0, 31, 32, 35, 36, 37, 38, 41, 42, 43, 45)
        nreads = tokens*64+32 if kind == 0 else (64 if kind == 43 else
                    (1 if kind in (31, 32, 35, 36, 38, 41, 42) else 0))
        status = 0
        if invalid: status = 2 if 7 <= kind <= 10 else 1
        if 12 <= kind <= 26: status = 2 if 19 <= kind <= 21 else (3 if kind == 24 else 4)
        if kind in (27, 29, 31, 44): status = 7
        if kind in (28, 30, 32, 45): status = 2
        if kind in (33, 34): status = 8
        if kind in (35, 36): status = 9
        if kind == 37: status = 5
        if kind == 46: status = 4
        sequence = ['AKR_CMD'] + ['AKR_TILE']*count + (['AKR_COMMIT'] if committed else [])
        sequence += (['AKR_COMPUTE'] if consumed else []) + ['AKR_READ']*nreads
        sequence += ['AKR_RESET' if reset else 'AKR_DONE']
        assert [tag for tag, _ in group] == sequence, ('event order', job)
        memory = {}
        for ordinal, (_, actual) in enumerate(group[1:1+count]):
            plane, index = divmod(ordinal, tokens*4)
            # The extra beat deliberately wraps the row, but still has V plane.
            plane = int(plane > 0)
            data = sum(word(job, plane, index*16+lane) << (lane*32) for lane in range(16))
            expected = dict(identity, plane=plane, index=index, last=int(ordinal==tokens*8-1), data=data)
            if ordinal == 1:
                if kind == 12: expected['plane'] ^= 1
                if kind == 13: expected['index'] += 1
                if kind == 14: expected['last'] = 1
                if kind == 15: expected['seq'] += 1
                if kind == 16: expected['epoch'] += 1
                if kind == 17: expected['lease'] += 1
                if kind == 18:
                    expected['data'] = (data & ~(0xffffffff << 224)) | (0x7fc00001 << 224)
            if kind == 46 and ordinal == tokens*8-1: expected['last'] = 0
            assert actual == expected, ('tile', job, ordinal)
            for lane in range(16):
                memory[actual['plane'], actual['index']*16+lane] = (actual['data'] >> (lane*32)) & 0xffffffff
        if committed:
            report_status = 7 if kind in (27, 28, 33, 34) else (3 if kind == 24 else 0)
            completion = dict(identity, status=report_status, beats=count, bytes=count*64)
            if kind == 19: completion['seq'] += 1
            if kind == 20: completion['epoch'] += 1
            if kind == 21: completion['lease'] += 1
            if kind == 22: completion['beats'] += 1
            if kind == 23: completion['bytes'] -= 1
            assert group[1+count] == ('AKR_COMMIT', completion), ('commit', job)
        if consumed:
            assert committed and count == tokens*8
            descriptor = {k: command[k] for k in ('job','start','tokens','position','layer','head','seq','epoch','lease')}
            descriptor['kv'] = head//4
            assert group[2+count] == ('AKR_COMPUTE', descriptor), ('consumer', job)
        for index, actual in enumerate(fields for tag, fields in group if tag == 'AKR_READ'):
            assert set(actual) == {'job','ka','va','kc','vc','kvalid','vvalid','k','v'}
            if kind == 0 and index < tokens*64 or kind == 43:
                assert (actual['ka'],actual['va'],actual['kc'],actual['vc']) == (index,tokens*64-1-index,1,1)
            elif kind == 0:
                offset = index-tokens*64
                assert (actual['kc'],actual['vc']) == (int(offset%3!=0),int(offset%4!=0))
            else:
                ka,va = (15,16) if kind==38 else ((1,65) if kind in (41,42) else (0,tokens*64-1))
                assert (actual['ka'],actual['va'],actual['kc'],actual['vc']) == (ka,va,1,1)
            for p,prefix in enumerate(('k','v')):
                address,enable = actual[prefix+'a'],actual[prefix+'c']
                assert 0 <= address < tokens*64 and enable in (0,1)
                assert actual[prefix+'valid'] == enable
                # Compare the returned word against the actual accepted stream,
                # after its separately checked identity, layout, and completion.
                assert actual[prefix] == (memory[p,address] if enable else 0), ('RAM value',job,index,p)
                read_words += enable
        terminal = dict(job=job,kind=kind) if reset else dict(identity,status=status,beats=count,started=int(consumed))
        assert group[-1][1] == terminal, ('terminal', job)
    return dict(expected_pass, checked_ram_words=read_words, private_storage_bytes=32768,
                max_tokens=64, hls_binding_qualified=False, physical_timing_qualified=False)


def controls(text, seed, watchdog):
    """Alter observed results/ordering, never the accepted expected values."""
    lines=text.splitlines(); rejected=[]
    changes=[('missing_command','AKR_CMD',None),('missing_tile','AKR_TILE',None),
             ('missing_commit','AKR_COMMIT',None),('missing_compute','AKR_COMPUTE',None),
             ('missing_read','AKR_READ',None),('missing_done','AKR_DONE',None),
             ('missing_reset','AKR_RESET',None),('missing_pass','AKR_PASS',None),
             ('wrong_data','AKR_TILE',('data',1)),('wrong_plane','AKR_TILE',('plane',1)),
             ('wrong_lease','AKR_COMPUTE',('lease',1)),('wrong_head','AKR_COMPUTE',('kv',1)),
             ('wrong_k','AKR_READ',('k',1)),('wrong_v','AKR_READ',('v',1)),
             ('wrong_valid','AKR_READ',('kvalid',1)),('wrong_status','AKR_DONE',('status',1)),
             ('wrong_count','AKR_COMMIT',('beats',1)),('wrong_bytes','AKR_COMMIT',('bytes',1))]
    for label,tag,change in changes:
        altered=lines.copy();i=next(i for i,line in enumerate(altered) if line.startswith(tag+' '))
        if change is None:del altered[i]
        else:
            key,delta=change;base=16 if key in ('data','k','v') else 10
            altered[i]=re.sub(r'\b'+key+r'=([0-9a-f]+)',lambda m:key+'='+format(int(m[1],base)^delta,'x' if base==16 else 'd'),altered[i],count=1)
        try:review('\n'.join(altered),seed,watchdog)
        except (AssertionError,KeyError,ValueError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corrupt evidence: '+label)
    return rejected


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--watchdog',type=int,default=32)
    a=p.parse_args();text=a.log.read_text();result=review(text,a.seed,a.watchdog)
    result['rejected_controls']=controls(text,a.seed,a.watchdog);print(json.dumps(result,indent=2))
