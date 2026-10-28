"""Review physical KV writes, leased reads and publication from an RTL trace."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

BASE = 0xa000000000
HEX = {'base', 'address', 'data', 'sequence', 'epoch', 'lease', 'expected'}


def plan():
    result = []
    def add(kind, *args): result.append((kind, args))
    add('BIND', 4, 4032)
    add('LEASE', 0, 0, 0, 0, 0, 1)
    add('APPEND', 0, 0, 1, 3); add('LEASE', 0, 0, 0, 0, 0, 1); add('APPEND', 0, 0, 0, 0)
    for mode in (1, 2, 3): add('LEASE', 0, 0, 0, 0, mode, 2)
    for layer, head, pos in [(16, 0, 0), (0, 32, 0), (0, 0, 4)]: add('LEASE', 0, layer, head, pos, 0, 1)
    add('LEASE', 0, 0, 0, 0, 0, 0)
    for mode in (1, 2, 3): add('READ', 0, 1, mode, 2)
    for start, count in [(1, 1), (0, 0), (0, 65), (0, 2)]: add('READ', start, count, 0, 1)
    for mode in (1, 2, 5): add('LEASE', 1, 0, 0, 0, mode, 2)
    for mode, status in [(5, 3), (6, 7), (7, 8), (0, 0)]: add('READ', 0, 1, mode, status)
    add('READ', 0, 1, 0, 1); add('LEASE', 1, 0, 0, 0, 0, 0); add('LEASE', 0, 0, 0, 0, 4, 2)
    add('LEASE', 0, 0, 3, 0, 0, 0); add('APPEND', 0, 1, 0, 0); add('READ', 0, 1, 0, 0)
    add('READ', 1, 1, 0, 1); add('LEASE', 1, 0, 0, 0, 0, 0)
    add('APPEND', 0, 2, 2, 3); add('LEASE', 0, 0, 0, 2, 0, 1); add('APPEND', 0, 2, 0, 0)
    add('APPEND', 2, 0, 0, 0); add('LEASE', 0, 2, 0, 0, 0, 0)
    add('READ', 0, 1, 0, 0); add('LEASE', 1, 0, 0, 0, 0, 0)
    add('APPEND', 15, 0, 0, 0)
    for head in range(32):
        add('LEASE', 0, 15 if head % 2 else 0, head, 0, 0, 0)
        add('READ', 0, 1, 0, 0); add('LEASE', 1, 0, 0, 0, 0, 0)
    add('LEASE', 0, 0, 31, 2, 0, 0); add('READ', 0, 3, 0, 2)
    add('LEASE', 1, 0, 0, 0, 0, 0); add('LEASE', 0, 0, 31, 2, 0, 2)
    add('BIND', 65, 64); add('LEASE', 0, 0, 0, 0, 0, 1)
    for pos in range(65): add('APPEND', 15, pos, 0, 0)
    add('LEASE', 0, 15, 31, 64, 0, 0); add('READ', 0, 64, 0, 0); add('READ', 64, 1, 0, 0)
    add('READ', 65, 1, 0, 1); add('LEASE', 1, 0, 0, 0, 0, 0); add('LEASE', 0, 15, 0, 64, 0, 0)
    add('BIND', 1, 0); add('LEASE', 0, 15, 0, 0, 0, 1); add('APPEND', 15, 0, 0, 0)
    add('LEASE', 0, 15, 31, 0, 0, 0); add('READ', 0, 1, 0, 0); add('LEASE', 1, 0, 0, 0, 0, 0)
    for scenario in range(3):
        add('LEASE', 0, 15, 31, 0, 0, 0); add('READ', 0, 1, 8+scenario, None)
        add('BIND', 1, 64); add('LEASE', 0, 15, 31, 0, 0, 1); add('APPEND', 15, 0, 0, 0)
        add('LEASE', 0, 15, 31, 0, 0, 0); add('READ', 0, 1, 0, 0); add('LEASE', 1, 0, 0, 0, 0, 0)
    add('LEASE', 0, 15, 31, 0, 0, None); add('BIND', 1, 4032)
    add('LEASE', 0, 15, 31, 0, 0, 1); add('APPEND', 15, 0, 0, 0)
    add('LEASE', 0, 15, 31, 0, 0, 0); add('READ', 0, 1, 0, 0); add('LEASE', 1, 0, 0, 0, 0, 0)
    assert len(result) == 252
    return result


def pattern(generation, layer, position, index):
    return sum((0x3f000000 | ((generation*100003 + layer*4099 + position*97 + index*23 + lane*3) & 0x7fffff))
               << (lane*32) for lane in range(16))


def address(base, capacity, layer, plane, head, position, word=0):
    return base + (((layer*2+plane)*8+head)*capacity+position)*256 + word*64


def bursts(start, beats, maximum):
    result = []
    while beats:
        count = min(beats, maximum, (4096-start % 4096)//64)
        assert count > 0
        result.append((start, count)); start += count*64; beats -= count
    return result


def review(text, seed, watchdog):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|FAIL )', text), 'Simulator failure'
    specs = plan(); jobs = {}; counts = Counter(); memory = {}; prefix = [0]*16
    generation = capacity = base = seq = epoch = expected_epoch = last_id = current_cycle = resets = 0
    next_id = 0x1122334400000000
    active = last_lease = writer = reader = pending_lease = None
    valid = False
    terminal = None
    epoch_events = []; metrics = []; interleaved = False; waiting_acquisition = False
    passed_bytes = 0

    for line in text.splitlines():
        if not line.startswith('NTRPU_KVC_'): continue
        assert terminal is None, 'Trace after PASS'
        tag, _, rest = line.partition(' '); tag = tag.removeprefix('NTRPU_KVC_')
        row = {}
        for token in rest.split():
            key, value = token.split('='); assert key not in row, 'Duplicate field'
            row[key] = int(value, 16 if key in HEX else 10)
        counts[tag] += 1
        if tag == 'PASS':
            terminal = row; continue
        cycle = row.pop('cycle'); assert cycle >= current_cycle; current_cycle = cycle
        if tag == 'RESET':
            resets += 1; assert row == dict(count=resets)
            assert writer is None
            if reader is not None: assert reader.get('aborted')
            if pending_lease is not None: assert pending_lease.get('aborted')
            valid = False; prefix = [0]*16; active = last_lease = reader = pending_lease = None; last_id = 0
            continue
        if tag == 'EPOCH':
            assert set(row) == {'expected'} and generation == 1
            epoch_events.append(row['expected']); assert epoch_events == [epoch+1, epoch][:len(epoch_events)]
            expected_epoch = row['expected']; continue
        ident = row.pop('id'); assert 1 <= ident <= len(specs)
        kind, args = specs[ident-1]
        if tag in ('BIND', 'APPEND', 'LEASE', 'READ'):
            assert ident == len(jobs)+1 and tag == kind, (ident, 'Operation order')
            job = dict(kind=kind, args=args, begin=cycle, command=dict(row), events=Counter())
            jobs[ident] = job
            if kind == 'BIND':
                assert writer is None and reader is None and active is None and pending_lease is None
                generation += 1; capacity, offset = args; base = BASE+offset
                seq, epoch = 0x1234567800000000+generation, 0xfedcba9800000000+generation
                expected_epoch = epoch; prefix = [0]*16; valid = True; writer = job
                assert row == dict(generation=generation, capacity=capacity, base=base, sequence=seq, epoch=epoch)
            elif kind == 'APPEND':
                layer, pos, mode, status = args
                assert writer is None and valid and pos == prefix[layer]
                assert row == dict(generation=generation, layer=layer, position=pos, mode=mode, prefix=prefix[layer])
                job.update(plan=[p for plane in range(2) for head in range(8)
                                 for p in bursts(address(base, capacity, layer, plane, head, pos), 4, 4)],
                           aw=0, w=0, b=0, pending_aw=None, pending_w=[], responses=[], generation=generation)
                writer = job
            elif kind == 'LEASE':
                release, layer, head, pos, mode, status = args
                assert pending_lease is None and reader is None
                sent_id = (last_lease['lease'] if last_lease else 0) if release else next_id
                if not release: next_id += 1
                sent_seq, sent_epoch = seq, epoch
                if mode == 1: sent_seq ^= 1 << 32
                if mode == 2: sent_epoch ^= 1 << 32
                if mode == 3: sent_id = 0
                if mode == 4: sent_id = last_lease['lease'] if last_lease else 0
                if mode == 5: sent_id ^= 1 << 32
                assert row == dict(release=release, layer=layer, head=head, position=pos,
                                   sequence=sent_seq, epoch=sent_epoch, lease=sent_id)
                if writer is not None: waiting_acquisition = True
                pending_lease = job
            else:
                start, tokens, mode, status = args
                assert active is not None and reader is None and pending_lease is None
                sent_seq, sent_epoch, sent_id = seq, epoch, active['lease']
                if mode == 1: sent_seq ^= 1 << 32
                if mode == 2: sent_epoch ^= 1 << 32
                if mode == 3: sent_id ^= 1 << 32
                assert row == dict(start=start, tokens=tokens, mode=mode, sequence=sent_seq, epoch=sent_epoch, lease=sent_id)
                rejected = 0
                if (sent_seq, sent_epoch, sent_id) != (active['sequence'], active['epoch'], active['lease']) or sent_epoch != expected_epoch:
                    rejected = 2
                elif start != active['next'] or not 1 <= tokens <= 64 or start+tokens > active['position']+1:
                    rejected = 1
                assert status == rejected or (rejected == 0 and (mode in (5, 6, 7, 8, 9, 10))), (ident, 'Read admission')
                job.update(lease=dict(active), rejected=rejected, ar=0, r=0, tiles=0, pending_r=None, returned={},
                    plan=[] if rejected else [p for plane in range(2) for p in bursts(
                        address(base, capacity, active['layer'], plane, active['head']//4, start), tokens*4, 64)])
                reader = job
            continue
        assert ident in jobs
        job = jobs[ident]; job['events'][tag] += 1
        if tag == 'BIND_DONE':
            assert writer is job and kind == 'BIND' and row == dict(status=0)
            writer = None; job['done'] = cycle
        elif tag == 'AW':
            assert writer is job and kind == 'APPEND' and job['pending_aw'] is None
            assert job['aw'] == job['b'] < len(job['plan'])
            addr, beats = job['plan'][job['aw']]
            assert row == dict(address=addr, beats=beats), (ident, 'AW address/length')
            job['pending_aw'] = (addr, beats); job['aw'] += 1
        elif tag == 'W':
            assert writer is job and kind == 'APPEND'
            layer, pos, mode, status = args; addr, beats = job['plan'][job['b']]
            assert len(job['pending_w']) < beats
            assert row == dict(last=int(len(job['pending_w'])+1 == beats),
                              data=pattern(job['generation'], layer, pos, job['w'])), (ident, 'Write data/last')
            job['pending_w'].append(row['data']); job['w'] += 1
        elif tag == 'B':
            assert writer is job and kind == 'APPEND' and job['pending_aw'] is not None
            addr, beats = job['pending_aw']; assert len(job['pending_w']) == beats
            mode = args[2]; response = 2 if mode == 1 or (mode == 2 and job['b'] == 15) else 0
            assert row == dict(ordinal=job['b'], resp=response), (ident, 'B response/order')
            for i, value in enumerate(job['pending_w']): memory[addr+i*64] = value
            job['pending_aw'] = None; job['pending_w'] = []; job['b'] += 1; job['responses'].append(response)
        elif tag == 'APPEND_DONE':
            assert writer is job and kind == 'APPEND'
            layer, pos, mode, status = args
            assert job['pending_aw'] is None and not job['pending_w'] and job['aw'] == job['b']
            if status == 0:
                assert job['w'] == 64 and job['aw'] == len(job['plan']) and not any(job['responses'])
                prefix[layer] += 1; passed_bytes += 4096
            else:
                assert job['responses'][-1] == 2 and sum(v != 0 for v in job['responses']) == 1
                assert job['b'] == (1 if mode == 1 else 16)
            assert row == dict(status=status, bytes=job['w']*64, committed=prefix[layer]), (ident, 'Publication')
            writer = None; job['done'] = cycle
        elif tag in ('LEASE_DONE', 'LEASE_ABORT'):
            assert pending_lease is job and kind == 'LEASE' and writer is None, 'Grant before writer publication'
            release, layer, head, pos, mode, wanted = args; command = job['command']
            old_prefix = active['prefix'] if active else 0
            identity = tuple(command[k] for k in ('sequence', 'epoch', 'lease'))
            if release:
                status = 0 if active and identity == tuple(active[k] for k in ('sequence', 'epoch', 'lease')) else 2
                if status == 0: active = None
            else:
                assert active is None
                if not valid or identity[:2] != (seq, epoch) or epoch != expected_epoch or not identity[2] or identity[2] <= last_id:
                    status = 2
                elif layer >= 16 or head >= 32 or pos >= capacity or pos >= prefix[layer]: status = 1
                else:
                    status = 0; last_id = identity[2]; old_prefix = prefix[layer]
                    active = dict(command, prefix=old_prefix, next=0, generation=generation)
                    last_lease = active
            if tag == 'LEASE_ABORT':
                assert wanted is None and status == 0 and row == dict(active=1)
                job['aborted'] = True
            else:
                assert wanted == status and row == dict(status=status, prefix=old_prefix, active=int(active is not None)), (ident, 'Lease publication/identity')
                pending_lease = None; job['done'] = cycle
        elif tag == 'AR':
            assert reader is job and kind == 'READ' and not job['rejected'] and job['pending_r'] is None
            assert job['ar'] < len(job['plan'])
            addr, beats = job['plan'][job['ar']]
            assert row == dict(address=addr, beats=beats), (ident, 'Read address/length')
            job['pending_r'] = [addr, beats, 0]; job['ar'] += 1
            if writer and writer.get('aw', 0): interleaved = True
        elif tag == 'R':
            assert reader is job and kind == 'READ' and job['pending_r'] is not None
            addr, beats, index = job['pending_r']; location = addr+index*64
            assert location in memory, 'Read of unpublished/unwritten physical bytes'
            resp = 2 if args[2] == 5 and job['r'] == 3 else 0
            assert row == dict(address=location, resp=resp, last=int(index+1 == beats), data=memory[location]), (ident, 'Physical AXI read')
            job['returned'][location] = (row['data'], resp); job['r'] += 1
            if index+1 == beats: job['pending_r'] = None
            else: job['pending_r'][2] += 1
        elif tag == 'TILE':
            assert reader is job and kind == 'READ'
            start, tokens, mode, status = args; lease = job['lease']; seen = job['tiles']
            plane, index = divmod(seen, tokens*4); assert plane < 2
            location = address(base, capacity, lease['layer'], plane, lease['head']//4, start+index//4, index % 4)
            value = pattern(lease['generation'], lease['layer'], start+index//4, plane*32+(lease['head']//4)*4+index % 4)
            assert job['returned'].get(location) == (value, 0), 'Tile without matching physical read'
            assert row == dict(plane=plane, index=index, last=int(seen+1 == tokens*8), lease=lease['lease'], data=value)
            job['tiles'] += 1
        elif tag in ('CANCEL', 'STALL'):
            assert reader is job and kind == 'READ' and job['events'][tag] == 1
            assert (tag == 'CANCEL' and args[2] == 6 and row == {}) or (tag == 'STALL' and args[2] == 7 and row == dict(cycles=watchdog+8))
        elif tag == 'READ_DONE':
            assert reader is job and kind == 'READ' and job['pending_r'] is None
            start, tokens, mode, status = args; assert status is not None
            if status == 0:
                assert job['tiles'] == job['r'] == tokens*8 and job['ar'] == len(job['plan'])
                active['next'] += tokens
            elif status in (1, 2): assert job['ar'] == job['r'] == job['tiles'] == 0
            elif status == 3: assert any(resp for value, resp in job['returned'].values())
            elif status == 7: assert job['events']['CANCEL'] == 1
            elif status == 8: assert job['events']['STALL'] == 1
            assert row == dict(status=status, beats=job['tiles'], bytes=job['tiles']*64, next=active['next'])
            metrics.append(dict(id=ident, tokens=tokens, status=status, cycles=cycle-job['begin'],
                                ar=job['ar'], r=job['r'], output_beats=job['tiles'], reset=False))
            reader = None; job['done'] = cycle
        elif tag == 'READ_ABORT':
            assert reader is job and kind == 'READ' and args[3] is None
            phase = args[2]-8; assert row == dict(phase=phase) and not job['tiles']
            if phase == 0: assert job['ar'] == job['r'] == 0
            if phase == 1: assert job['ar'] == 1 and job['r'] == 0 and job['pending_r'] is not None
            if phase == 2: assert job['ar'] and job['r']
            job['aborted'] = True
            metrics.append(dict(id=ident, tokens=1, status=None, cycles=cycle-job['begin'],
                                ar=job['ar'], r=job['r'], output_beats=0, reset=True))
        else:
            raise AssertionError('Unknown trace tag: '+tag)

    assert len(jobs) == 252 and all(('done' in job) ^ bool(job.get('aborted')) for job in jobs.values())
    assert writer is reader is pending_lease is active is None
    assert resets == 6 and len(epoch_events) == 2 and interleaved and waiting_acquisition
    expected = dict(seed=seed, watchdog=watchdog, binds=7, appends=77, leases=109, reads=59, resets=6,
                    aw=counts['AW'], w=counts['W'], b=counts['B'], ar=counts['AR'], r=counts['R'], tiles=counts['TILE'])
    assert terminal is not None and set(terminal) == set(expected) | {'overlap', 'cycles'}
    assert all(terminal[k] == v for k, v in expected.items()) and terminal['overlap'] > 0 and terminal['cycles'] >= current_cycle
    assert counts['AW'] == counts['B'] and counts['READ_ABORT'] == 3 and counts['LEASE_ABORT'] == 1
    return dict(status='PASS_KV_CACHE_R1_TRANSCRIPT', summary=terminal, operations=len(jobs),
                publishable_write_bytes=passed_bytes, read_metrics=metrics,
                successful_read_bytes=sum(row['output_beats']*64 for row in metrics if row['status'] == 0),
                observed_interleaved_axi=True, acquisition_waited_for_publication=True,
                physical_memory_words_checked=len(memory), physical_timing_qualified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--watchdog', type=int, required=True)
    a = p.parse_args(); print(json.dumps(review(a.log.read_text(), a.seed, a.watchdog), indent=2))


if __name__ == '__main__': main()
