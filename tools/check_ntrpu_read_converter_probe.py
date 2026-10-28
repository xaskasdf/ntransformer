"""Independently reconstruct converter addresses, data, ordering and occupancy."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(address, words):
    return sum(((((address >> 2) + j) ^ 0xa5c319e7) & 0xffffffff) << (32*j)
               for j in range(words))


def replay(lines):
    events = [line.split()[1:] for line in lines if line.startswith('CP ')]
    assert events[0] == ['CONFIG', 'period_ps=2222', 'default_response_latency_cycles=48',
                         'initial_hold_cycles=256', 'native_hbm=0']
    expected = [(cap, mode, 4 if mode in (1, 2) else 1, 64, 48)
                for cap in (1, 2, 4, 8) for mode in range(4)]
    expected += [(8, 1, 1, n, 48) for n in (1, 8, 9, 64)]
    expected += [(cap, 1, 4, 64, latency) for cap in (1, 2, 4, 8) for latency in (64, 96)]
    rows = []
    active = None
    for e in events[1:]:
        kind = e[0]
        if kind == 'ALL_PASS':
            assert active is None and len(rows) == 28 and e == ['ALL_PASS', 'cases=28']
            assert e == events[-1]
            continue
        case_id = int(e[1])
        if kind == 'CASE':
            assert active is None and case_id == len(rows)+1
            cfg = {k: int(v) for k, v in (s.split('=') for s in e[2:])}
            assert tuple(cfg[k] for k in ('cap', 'mode', 'commands', 'beats', 'latency')) == expected[case_id-1]
            active = cfg
            source = deque()
            down = deque()
            native = deque()
            pending_events = []
            source_count = native_count = native_beats = outputs = completes = d_count = 0
            first_response_cycle = None
            address_cycles = []
            first_source = last_source = None
            previous_cycle = -1
        elif kind == 'PASS':
            assert active and case_id == len(rows)+1
            totals = {k: int(v) for k, v in (s.split('=') for s in e[2:])}
            assert not source and not down and not native
            ncmd, beats = active['commands'], active['beats']
            assert source_count == completes == ncmd
            assert outputs == ncmd*beats and native_beats == 2*outputs
            assert native_count == d_count == ncmd*((2*beats+15)//16)
            # Simultaneous AR and terminal R are one edge, not an artificial peak.
            deltas = {}
            for cycle, change in pending_events:
                deltas[cycle] = deltas.get(cycle, 0)+change
            occupancy = peak = area = 0
            last_cycle = first_source
            trajectory = []
            for cycle, change in sorted(deltas.items()):
                area += occupancy*(cycle-last_cycle)
                occupancy += change
                assert 0 <= occupancy <= active['cap']
                peak = max(peak, occupancy)
                trajectory.append([cycle, occupancy])
                last_cycle = cycle
            assert occupancy == 0
            before_response = sum(c < first_response_cycle for c in address_cycles)
            assert totals['peak'] == peak and totals['before'] == before_response
            assert totals['fragments'] == native_count
            assert totals['native_beats'] == native_beats and totals['outputs'] == outputs
            assert totals['span_cycles'] == last_source-first_source
            assert totals['source_stalls'] >= 0 and totals['native_stalls'] >= 0
            if active['mode'] == 2:
                assert totals['source_stalls'] > 0
            span = last_source-first_source
            rows.append(dict(case=case_id, **active, peak_outstanding=peak,
                             accepted_before_first_response=before_response,
                             source_to_last_data_cycles=span, fragments=native_count,
                             native_beats=native_beats, outputs=outputs,
                             payload_bytes=outputs*64, source_stall_cycles=totals['source_stalls'],
                             native_stall_cycles=totals['native_stalls'],
                             outstanding_cycle_area=area, trajectory=trajectory,
                             synthetic_window_gb_s=outputs*64*1000/(span*2222)))
            active = None
        else:
            assert active and case_id == len(rows)+1
            cycle = int(e[2])
            assert cycle >= previous_cycle
            previous_cycle = cycle
            if kind == 'S_AR':
                address, beats = int(e[3], 16), int(e[4])
                assert address == 0x10000+source_count*4096 and beats == active['beats']
                assert source_count < active['commands']
                source.append(dict(addr=address, beats=beats, got=0, issued=0, native_got=0))
                if first_source is None:
                    first_source = cycle
                source_count += 1
            elif kind == 'D_AR':
                address, beats = int(e[3], 16), int(e[4])
                cmd = next((c for c in source if c['issued'] < 2*c['beats']), None)
                assert cmd and address == cmd['addr']+32*cmd['issued']
                # Legal downsizing need not emit its largest fragment first.
                # E.g. 18 native beats may be split as 2+16 rather than 16+2.
                assert 1 <= beats <= min(16, 2*cmd['beats']-cmd['issued'])
                cmd['issued'] += beats
                down.append((address, beats, cmd))
                d_count += 1
            elif kind == 'P_AR':
                assert down
                address, beats = int(e[3], 16), int(e[4])
                da, dn, cmd = down.popleft()
                assert (address, beats) == (da, dn)
                native.append(dict(addr=address, beats=beats, got=0, cmd=cmd, cycle=cycle))
                native_count += 1
                address_cycles.append(cycle)
                pending_events.append((cycle, 1))
            elif kind == 'P_R':
                assert native
                f = native[0]
                if first_response_cycle is None:
                    first_response_cycle = cycle
                rid, resp, last, value = int(e[3]), int(e[4]), int(e[5]), int(e[6], 16)
                address = f['addr']+f['got']*32
                assert rid == 0 and value == payload(address, 8)
                assert resp == (2 if active['mode'] == 3 and address == 0x10000+active['beats']*64-32 else 0)
                assert last == int(f['got']+1 == f['beats'])
                assert cycle >= f['cycle']+active['latency']
                if active['mode'] == 0:
                    assert cycle >= 256
                f['got'] += 1
                f['cmd']['native_got'] += 1
                native_beats += 1
                if last:
                    native.popleft()
                    pending_events.append((cycle, -1))
            elif kind == 'S_R':
                assert source
                cmd = source[0]
                rid, resp, last, value = int(e[3]), int(e[4]), int(e[5]), int(e[6], 16)
                assert rid == 0 and value == payload(cmd['addr']+cmd['got']*64, 16)
                assert cmd['native_got'] >= 2*(cmd['got']+1)
                assert last == int(cmd['got']+1 == cmd['beats'])
                assert resp == (2 if active['mode'] == 3 and last else 0)
                cmd['got'] += 1
                outputs += 1
                last_source = cycle
                if last:
                    assert cmd['issued'] == cmd['native_got'] == 2*cmd['beats']
                    source.popleft()
                    completes += 1
            else:
                raise AssertionError('Unexpected event '+kind)
    assert active is None and len(rows) == 28 and events[-1] == ['ALL_PASS', 'cases=28']
    return rows


def negative_controls(lines):
    mutations = [('native_address', 'P_AR', 4), ('native_length', 'P_AR', 5),
                 ('native_id', 'P_R', 4), ('native_data', 'P_R', 7),
                 ('source_data', 'S_R', 7), ('source_last', 'S_R', 6),
                 ('source_error', 'S_R', 5), ('peak', 'PASS', 3)]
    passed = []
    for label, kind, column in mutations:
        altered = list(lines)
        index = next(i for i, line in enumerate(lines) if line.startswith('CP '+kind+' '))
        words = altered[index].split()
        if label == 'peak':
            words[column] = 'peak=99'
        else:
            radix = 16 if label.endswith(('address', 'data')) else 10
            value = int(words[column], radix)^1
            words[column] = format(value, 'x') if radix == 16 else str(value)
        altered[index] = ' '.join(words)
        try:
            replay(altered)
        except (AssertionError, KeyError, StopIteration):
            passed.append(label)
        else:
            raise AssertionError('Undetected mutation '+label)
    for kind in ('S_AR', 'D_AR', 'P_AR', 'P_R', 'S_R', 'PASS', 'ALL_PASS'):
        altered = list(lines)
        del altered[next(i for i, line in enumerate(lines) if line.startswith('CP '+kind+' '))]
        try:
            replay(altered)
        except (AssertionError, KeyError, StopIteration):
            passed.append('missing_'+kind)
        else:
            raise AssertionError('Undetected missing '+kind)
    return passed


def review(output):
    result = json.loads((output / 'result.json').read_text())
    assert result['experiment'] == 'read_converter_concurrency_probe'
    assert result['status'] == 'EXPORTED_PENDING_REVIEW'
    assert result['sources'] == result['sources_after'] and result['donor_unchanged']
    assert [s['stage'] for s in result['steps']] == ['compile', 'elaborate', 'simulate']
    assert all(s['returncode'] == 0 and not s['timed_out'] for s in result['steps'])
    files = json.loads((output / 'FILES.json').read_text())
    assert all(sha(output / name) == digest for name, digest in files.items())
    assert all(sha(output / 'sources' / name) == digest for name, digest in result['sources'].items())
    for name in ('hbm_read_width', 'hbm_read_protocol'):
        for suffix in ('_wrapper.v', '_config.tsv'):
            assert sha(output / (name+suffix)) == result['donor_files']['suite/'+name+suffix]
    adapter = (output / 'sources/fpga/experimental/ntrpu_hbm_read_adapter.sv').read_text()
    start = adapter.index('    hbm_read_width width_converter (')
    assert (output / 'read_converter_instances.vh').read_text() == adapter[start:adapter.index('\nendmodule', start)]+'\n'
    for stage in ('compile', 'elaborate', 'simulate'):
        assert not re.search(r'(?im)^\s*(ERROR:|Fatal:|FAIL )', (output/(stage+'.log')).read_text())
    lines = (output/'simulate.log').read_text().splitlines()
    rows = replay(lines)
    return dict(status='READ_CONVERTER_CONCURRENCY_PROBE_REVIEWED',
                result_sha256=sha(output/'result.json'), trace_sha256=sha(output/'simulate.log'),
                seconds=result['seconds'], cases=rows, negative_controls=negative_controls(lines),
                amd_conversion=True, native_hbm=False, read_guard_included=False,
                cdc_included=False, physical_timing_qualified=False,
                sustained_board_throughput=False, board_qualified=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    print(json.dumps(review(ap.parse_args().output), indent=2))
