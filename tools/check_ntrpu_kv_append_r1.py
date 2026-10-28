"""Independent transcript review of owned KV prefixes and physical AXI writes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

BASE = 0xa000000000
HEX = {'cache', 'region', 'bytes', 'sequence', 'epoch', 'expected_epoch', 'previous_epoch', 'data', 'address'}

def plan():
    out = [('bind', (131072, 4032, 12, 0))]
    out += [('append', (layer, 0, 0, 0)) for layer in range(16)]
    out += [('append', (0, 1, 39, 0))]
    statuses = [1, 1, 2, 2, 2] + [4]*9 + [2, 2, 6] + [3]*3 + [7]*5 + [8]*4 + [4, 7, 0]
    assert len(statuses) == 32
    out += [('append', (1, 1, scenario, status)) for scenario, status in enumerate(statuses, 1)]
    out += [('append', (1, 2, 38, 7)), ('append', (1, 2, 0, 0))]
    for scenario in range(33, 38):
        out += [('append', (2, 1 if scenario == 33 else 0, scenario, 0 if scenario == 37 else 7)),
                ('append', (2, 0, 0, 2)), ('bind', (17, 64, 0, 0))]
    out += [('bind', (3, 0, 0, 0))]
    out += [('append', (15, pos, 0, 0)) for pos in range(3)]
    out += [('append', (15, 3, 0, 1)), ('append', (15, 1, 0, 1))]
    for scenario in range(1, 12):
        out += [('bind', (17, 64, scenario, 2 if scenario >= 8 else 1)),
                ('append', (0, 0, 0, 2)), ('bind', (17, 64, 0, 0))]
    out += [('append', (15, 0, 0, 0))]
    assert len(out) == 107
    return out

def word(op, index, scenario, injected=False):
    words = [(0x3f000000 | ((op*17 + index*19 + lane*23) & 0x7fffff)) for lane in range(16)]
    if scenario == 32:
        words = [([0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff])[lane % 6] for lane in range(16)]
    if injected:
        if (scenario == 11 and index == 0) or (scenario == 30 and index == 5):
            words[0] = 0x7fc12345
        if scenario == 12 and index == 63:
            words[15] = 0x7f800000
    return sum(value << (lane*32) for lane, value in enumerate(words))

def review(text, seed, watchdog):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|FAIL )', text), 'Simulator failure'
    groups = defaultdict(list)
    header = []
    terminal = []
    for line in text.splitlines():
        if not line.startswith('NTRPU_KVA_'):
            continue
        tag, _, rest = line.partition(' ')
        tag = tag.removeprefix('NTRPU_KVA_')
        values = {}
        for token in rest.split():
            key, value = token.split('=')
            # Byte totals in completions are decimal; bind allocation bytes are hex.
            radix = 16 if key in HEX and not (key == 'bytes' and tag != 'BIND') else 10
            values[key] = int(value, radix)
        if tag == 'SEED':
            header.append(values)
        elif tag == 'PASS':
            terminal.append(values)
        else:
            assert tag in {'BIND', 'BIND_ACCEPT', 'BIND_RESULT', 'BIND_DONE', 'PREFIX', 'BEGIN', 'DATA', 'COMMIT', 'CANCEL', 'AW', 'W', 'B', 'DONE', 'PUBLICATION', 'RESET', 'CASE'}, tag
            groups[values['id']].append((tag, values))
    assert header == [dict(seed=seed, watchdog=watchdog)]
    assert len(terminal) == 1
    expected_counts = dict(binds=29, jobs=78, resets=5, successes=23)
    assert all(terminal[0][k] == v for k, v in expected_counts.items())
    assert all(terminal[0][k] > 0 for k in ('aw_stalls', 'w_stalls', 'done_stalls', 'w_before_aw'))
    assert sorted(groups) == list(range(1, 108))
    prefixes = [0]*16
    valid = False
    last_epoch = 0
    next_epoch = 0xfedcba9800000000
    base = capacity = seq = epoch = 0
    total_writes = total_aw = total_b = 0
    completed_bytes = 0
    cycles = []
    for op, (kind, args) in enumerate(plan(), 1):
        events = groups[op]
        by = defaultdict(list)
        for tag, row in events:
            by[tag].append(row)
        def one(tag):
            assert len(by[tag]) == 1, (op, tag, len(by[tag]))
            return by[tag][0]
        scenario, status = args[-2:]
        if kind == 'bind':
            cap, offset = args[:2]
            next_epoch += 1
            binding = dict(id=op, scenario=scenario, capacity=cap, cache=BASE+offset, region=BASE,
                           bytes=cap*65536+8192, sequence=0x1234567800000000+op, epoch=next_epoch,
                           expected_epoch=next_epoch, previous_epoch=last_epoch)
            if scenario == 1: binding['capacity'] = 0
            if scenario == 2: binding['capacity'] = 131073
            if scenario == 3: binding['cache'] += 1
            if scenario == 4: binding['bytes'] = offset+cap*65536-1
            if scenario == 5: binding['region'] = binding['cache']+64
            if scenario == 6: binding.update(cache=2**64-4096, region=2**64-4096, bytes=8192)
            if scenario == 7: binding.update(cache=2**64-4096, region=0, bytes=2**64-1)
            if scenario == 8: binding['sequence'] = 0
            if scenario == 9: binding.update(epoch=0, expected_epoch=0)
            if scenario == 10: binding.update(epoch=last_epoch, expected_epoch=last_epoch)
            if scenario == 11: binding['expected_epoch'] += 1
            assert one('BIND') == binding, (op, 'bind descriptor')
            accepted = one('BIND_ACCEPT')
            assert accepted['maintenance'] == 1 and accepted['cycle'] == (11 if scenario == 12 else 1)
            prefixes = [0]*16
            valid = status == 0
            base, capacity, seq, epoch = (binding[k] for k in ('cache', 'capacity', 'sequence', 'epoch'))
            if valid: last_epoch = epoch
            assert one('BIND_RESULT') == dict(id=op, status=status, valid=int(valid), bytes=0, committed=0, sequence=seq, epoch=epoch)
            result = one('BIND_DONE')
            assert result['status'] == status and result['valid'] == int(valid) and result['cycles'] >= 6
            assert not (set(by) - {'BIND', 'BIND_ACCEPT', 'BIND_RESULT', 'BIND_DONE', 'PREFIX'})
        else:
            lay, pos = args[:2]
            layer = 16 if scenario == 1 else lay
            position = pos+1 if scenario == 2 else pos
            command_seq = seq ^ ((1 << 40) if scenario == 3 else 0)
            command_epoch = epoch ^ ((1 << 40) if scenario == 4 else 0)
            expected_epoch = epoch+1 if scenario == 5 else epoch
            old = prefixes[layer] if layer < 16 else 0
            assert one('BEGIN') == dict(id=op, scenario=scenario, layer=layer, position=position, prefix=old,
                valid=int(valid), capacity=capacity, cache=base, sequence=command_seq, epoch=command_epoch, expected_epoch=expected_epoch)
            reset = 33 <= scenario <= 37
            admissible = valid and command_seq == seq and command_epoch == epoch == expected_epoch and layer < 16 and position < capacity and position == old
            count = 5 if scenario == 33 else (63 if scenario == 13 else (65 if scenario == 14 else 64))
            if not admissible: count = 0
            assert len(by['DATA']) == count
            for index, data in enumerate(by['DATA']):
                assert data == dict(id=op, index=4 if scenario == 6 and index == 5 else index % 64,
                    last=int((index == 63 and scenario != 10) or (scenario == 9 and index == 8)),
                    sequence=command_seq ^ ((1 << 40) if scenario == 7 and index == 6 else 0),
                    epoch=command_epoch ^ ((1 << 40) if scenario == 8 and index == 7 else 0),
                    data=word(op, index, scenario, True)), (op, index, 'input bytes')
            if admissible and scenario != 33:
                assert one('COMMIT') == dict(id=op, status=6 if scenario in (17, 30) else 0,
                    sequence=command_seq ^ ((1 << 40) if scenario == 15 else 0),
                    epoch=command_epoch ^ ((1 << 40) if scenario == 16 else 0), inputs=count)
            else: assert not by['COMMIT']
            full_plan = []
            index = 0
            while admissible and index < 64:
                address = base + ((lay*16+index//4)*capacity+pos)*256+(index % 4)*64
                beats = min(4-index % 4, (4096-address % 4096)//64)
                full_plan.append((address, beats))
                index += beats
            assert len(by['AW']) <= len(full_plan)
            for row, (address, beats) in zip(by['AW'], full_plan):
                assert row == dict(id=op, address=address, beats=beats), (op, 'AW layout')
            writes = len(by['W'])
            for index, row in enumerate(by['W']):
                address = base + ((lay*16+index//4)*capacity+pos)*256+(index % 4)*64
                assert row == dict(id=op, index=index, last=int(index % 4 == 3 or address % 4096 == 4032),
                                   data=word(op, index, scenario)), (op, index, 'physical W bytes')
            for index, row in enumerate(by['B']):
                resp = 2 if (scenario == 18 and index == 0) or (scenario == 19 and index == len(full_plan)-1) or scenario == 38 else 0
                bid = 1 if scenario == 20 and index == 7 else 0
                assert row == dict(id=op, ordinal=index, resp=resp, bid=bid), (op, 'B fault evidence')
            if not reset:
                assert len(by['AW']) == len(by['B'])
                assert writes == sum(row['beats'] for row in by['AW'])
            # Replay ordering independently, including W preceding AW.
            saw_commit = False
            aw = w = b = 0
            for tag, row in events:
                if tag == 'COMMIT': saw_commit = True
                if tag in ('AW', 'W'): assert saw_commit, (op, 'write before commit')
                if tag == 'AW':
                    assert aw == b, (op, 'multiple outstanding bursts')
                    aw += 1
                if tag == 'W': w += 1
                if tag == 'B':
                    assert b < aw and w == sum(n for _, n in full_plan[:b+1]), (op, 'premature B')
                    b += 1
                if tag == 'PUBLICATION':
                    assert b == aw == len(full_plan) and w == 64
                    assert not any(x['resp'] or x['bid'] for x in by['B'])
                    assert not by['CANCEL']
            cancel_expected = scenario in (21, 22, 23, 24, 25, 31, 38)
            assert len(by['CANCEL']) == int(cancel_expected)
            for row in by['CANCEL']:
                assert row['writes'] <= writes and row['responses'] <= len(by['B'])
            published = status == 0
            if published:
                assert one('PUBLICATION') == dict(id=op, committed=pos+1, responses=len(full_plan), writes=64)
                prefixes[lay] = pos+1
            else:
                assert not by['PUBLICATION']
            if reset:
                r = one('RESET')
                assert r == dict(id=op, inputs=count, writes=writes, bursts=len(by['AW']), responses=len(by['B']), prefix=prefixes[lay])
                assert not by['DONE']
                prefixes = [0]*16
                valid = False
                last_epoch = 0
            else:
                assert one('DONE') == dict(id=op, status=status, bytes=writes*64, committed=pos+1 if published else old,
                                          sequence=command_seq, epoch=command_epoch)
            result = one('CASE')
            assert {k: v for k, v in result.items() if k != 'cycles'} == dict(id=op, scenario=scenario, status=status,
                reset=int(reset), inputs=count, writes=writes, bursts=len(by['AW']), responses=len(by['B']),
                published=0 if reset else prefixes[lay])
            assert result['cycles'] > 0
            if scenario in (26, 27, 28, 29): assert result['cycles'] > watchdog
            if not reset and status == 0:
                completed_bytes += writes*64
                cycles.append(result['cycles'])
            if not reset and status not in (0, 3, 7, 8): assert not writes and not by['AW']
            total_writes += writes
            total_aw += len(by['AW'])
            total_b += len(by['B'])
        assert by['PREFIX'] == [dict(id=op, layer=layer, committed=prefixes[layer] if valid else 0, valid=int(valid)) for layer in range(16)], (op, 'prefix readback')
    assert completed_bytes == 23*4096
    return dict(status='PASS_KV_APPEND_R1', seed=seed, watchdog=watchdog, **expected_counts,
                physical_w_bytes=total_writes*64, aw_bursts=total_aw, b_responses=total_b,
                completed_publishable_bytes=completed_bytes, successful_token_cycles=dict(min=min(cycles), max=max(cycles)),
                prefix_readbacks=107*16, log_sha256=hashlib.sha256(text.encode()).hexdigest(),
                scope='One active lease, 16 layer prefixes; model AXI, no physical DDR or complete attention integration.')

def controls(text, seed, watchdog):
    lines = text.splitlines()
    mutations = {}
    def change(name, tag, fn):
        index = next(i for i, line in enumerate(lines) if line.startswith('NTRPU_KVA_'+tag+' '))
        copy = lines[:]
        copy[index] = fn(copy[index])
        mutations[name] = '\n'.join(copy)+'\n'
    change('missing_write', 'W', lambda _: '')
    change('duplicate_write', 'W', lambda line: line+'\n'+line)
    change('missing_address', 'AW', lambda _: '')
    change('truncated_address', 'AW', lambda line: re.sub(r'address=([0-9a-f]+)', lambda m: 'address='+format(int(m[1], 16)&0xffffffff, '016x'), line))
    change('wrong_physical_byte', 'W', lambda line: re.sub(r'data=([0-9a-f]+)', lambda m: 'data='+format(int(m[1], 16)^1, '0128x'), line))
    change('missing_publication', 'PUBLICATION', lambda _: '')
    change('wrong_owned_prefix', 'PREFIX', lambda line: re.sub(r'committed=\d+', 'committed=7', line))
    change('wrong_completion', 'DONE', lambda line: line.replace('status=0', 'status=3'))
    change('missing_reset', 'RESET', lambda _: '')
    change('missing_commit', 'COMMIT', lambda _: '')
    change('missing_pass', 'PASS', lambda _: '')
    mutations['erased_b_error'] = text.replace('resp=2', 'resp=0')
    mutations['fatal'] = text+'\nFATAL: deliberately corrupted evidence\n'
    rejected = []
    for name, mutated in mutations.items():
        assert mutated != text
        try: review(mutated, seed, watchdog)
        except (AssertionError, KeyError, ValueError, IndexError): rejected.append(name)
        else: raise AssertionError('Accepted corruption: '+name)
    return rejected

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--watchdog', type=int, default=32)
    args = p.parse_args()
    text = args.log.read_text()
    result = review(text, args.seed, args.watchdog)
    result['rejected_controls'] = controls(text, args.seed, args.watchdog)
    print(json.dumps(result, indent=2))
