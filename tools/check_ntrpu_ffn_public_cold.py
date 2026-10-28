"""Independently reconstruct cold MIG/FFN host, memory and ABI2 evidence.

Synthetic D32/H32, three region rotations, no runtime reset. This is not
qualification of an exact DIMM, PCIe DMA, board timing or full inference.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from check_ntrpu_ffn_dma_recovery import inputs, SPANS
from check_ntrpu_ffn_memory import reference
from check_ntrpu_ffn_public_control import CAPACITY, region_base, review_control


def parse(text):
    for line_no, line in enumerate(text.splitlines()):
        if not line.startswith('NTRPU_'):
            continue
        name = line.split()[0]
        hexkeys = {'base', 'addr', 'data', 'strb', 'hash', 'sequence', 'flags'}
        if not name.startswith('NTRPU_DMA_'):
            hexkeys.add('epoch')
        row = {k: int(v, 16 if k in hexkeys else 10)
               for k, v in re.findall(r'(\w+)=([0-9a-f]+)(?=\s|$)', line)}
        yield line_no, name, row


def review(text, seed=1, host_period_ps=4000):
    result = dict(status='FAIL', errors=[], seed=seed)
    def need(ok, why):
        if not ok:
            raise ValueError(why)
    try:
        need(host_period_ps in (4000, 6200), 'clock profile')
        need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL\b|CRITICAL WARNING:)', text), 'tool/model error')
        need('NTRPU_DDR_FFN_INCOMPLETE' not in text, 'incomplete simulation')
        rows = list(parse(text))
        by = defaultdict(list)
        for line, name, row in rows:
            by[name].append((line, row))
        def rec(name):
            return [r for _, r in by['NTRPU_' + name]]
        def one(name):
            values = rec(name)
            need(len(values) == 1, 'one ' + name)
            return values[0]
        need(one('FFN_MEMORY_SEED') == dict(seed=seed), 'seed')
        need(one('FFN_MEMORY_FORMAT') == dict(fp32=1), 'FP32 format')
        need(one('DMA_REC_CONFIG') == dict(host_period_ps=host_period_ps,
             device_period_ps=5000, mgmt_period_ps=10000, dim=32, hidden=32), 'configuration')
        summary = one('DDR_FFN_PASS')
        need(all(summary[k] == v for k, v in dict(ffns=3, fp32=96, load_bytes=9984, result_bytes=384).items()), 'DDR coverage')
        need(all(summary['rank' + str(i)] > 0 for i in range(4)), 'four physical ranks observed')
        admit = one('DDR_FFN_ADMITTED')
        need(admit['calib'] == 1 and admit['ui_rst'] == 0, 'calibrated admission')
        states = rec('DDR_FFN_STATE')
        ready = [r for r in states if r.get('calib') == 1 and r.get('ui_rst') == 0]
        need(bool(ready) and ready[0]['ps'] < admit['ps'], 'calibration precedes admission')
        calibration_ps = ready[0]['ps']
        need(all(r.get('calib') == 1 and r.get('ui_rst') == 0 for r in states if r['ps'] >= calibration_ps), 'no calibration/reset loss')
        ui = rec('DDR_FFN_UI')
        need(len(ui) == 2 and [r['sample'] for r in ui] == [1, 2], 'two initial calibrated UI edges')
        need(calibration_ps < ui[0]['ps'] < ui[1]['ps'] < admit['ps'] and ui[1]['ps'] - ui[0]['ps'] == 5000, 'observed UI period')
        acts = rec('DDR_FFN_ACT'); rank_counts = Counter()
        for r in acts:
            need(0 <= r['rank'] < 4 and r['ps'] > calibration_ps, 'physical rank ACT after calibration')
            rank_counts[r['rank']] += 1
            need(r['count'] == rank_counts[r['rank']], 'physical ACT counter')
        need(all(rank_counts[i] == summary['rank' + str(i)] for i in range(4)), 'reconstructed physical rank activity')
        ids = [1, 2, 3]
        regions = rec('DDR_FFN_REGION')
        need([(r['id'], r['buffer']) for r in regions] == [(i, b) for i in ids for b in range(8)], 'ordered regions')
        for r in regions:
            base = region_base(r['id'], r['buffer'])
            need(r == dict(id=r['id'], buffer=r['buffer'], base=base, addr=base + 4032, bytes=SPANS[r['buffer']]), 'independent region geometry')
            need(0 <= r['addr'] < r['addr'] + r['bytes'] <= CAPACITY, '128GiB range before narrowing')
        for i in ids:
            areas = sorted((r['addr'], r['addr'] + r['bytes']) for r in regions if r['id'] == i)
            need(all(a[1] <= b[0] for a, b in zip(areas, areas[1:])), 'nonoverlapping buffers')
        need(region_base(3, 0) + 4032 + SPANS[0] == 2**32, '4GiB endpoint')
        need(region_base(3, 7) + 4032 + SPANS[7] == CAPACITY, '128GiB endpoint')
        begins, ends = rec('DMA_REC_FFN_BEGIN'), rec('DMA_REC_FFN_END')
        cases = rec('FFN_MEMORY_CASE')
        for values in [begins, ends, cases]:
            need([r['id'] for r in values] == ids, 'three ordered complete FFNs')
        for start, end, case in zip(begins, ends, cases):
            need(admit['ps'] < start['ps'] < end['ps'], 'FFN lifecycle time')
            need(end['load_bytes'] == 3328 and end['result_bytes'] == 128, 'per-job transfer totals')
            need(all(case[k] == v for k, v in dict(dim=32, hidden=32, mode=0, target=0, reset=0, bytes=704, down_bytes=128).items()), 'numerical completion classification')
            need(0 < case['cycles'] < 30000, 'observed job interval')
            need(not any('p'+str(p) in case for p in range(4)), 'no invented internal phase metrics')
        final = one('FFN_MEMORY_PASS')
        need(all(final[k] == v for k, v in dict(cases=3, bytes=2112, resets=0, cancels=0).items()), 'FFN summary')
        need(final['cycles'] == sum(c['cycles'] for c in cases), 'cycle sum')

        normal, golden = inputs(), reference(32, 32, True)
        def address(ident, addr):
            for b in range(8):
                start = region_base(ident, b) + 4032
                if start <= addr < start + SPANS[b]:
                    return b, addr - start
            raise ValueError('address outside registered buffers')
        def payload(r):
            return r['data'].to_bytes(64, 'little')
        def expected(b, off):
            value = (normal if b < 4 else golden)[b][off:off + 64]
            need(len(value) == 64, 'whole independent payload beat')
            return value
        tx, active, current = {}, None, 0
        counters = Counter()
        shared_counts = defaultdict(Counter)
        physical, writes = {}, defaultdict(list)
        read_pending = write_pending = None
        shared_read_beats = []
        for line, name, r in rows:
            if name == 'NTRPU_DDR_FFN_REGION':
                current = r['id']
            elif name == 'NTRPU_DMA_BEGIN':
                need(active is None and r['epoch'] == len(tx) + 1, 'unique nonoverlapping DMA epoch')
                need(current in ids and r['ps'] >= admit['ps'] and
                     line > by['NTRPU_DDR_FFN_ADMITTED'][0][0], 'DMA after admission')
                need(1 <= r['beats'] <= 4 and r['addr'] % 64 == 0 and r['addr'] % 4096 + r['beats'] * 64 <= 4096, 'DMA burst/page')
                active = r['epoch']
                tx[active] = dict(begin=r, id=current, events=defaultdict(list), loads=[])
            elif name in {'NTRPU_DMA_' + k for k in ['AW', 'W', 'B_HELD', 'B', 'AR', 'R_HELD', 'R', 'END']}:
                need(active is not None and r['epoch'] == active, 'DMA response ownership')
                t = tx[active]; kind = name.removeprefix('NTRPU_DMA_')
                t['events'][kind].append(r)
                need(r['ps'] >= t['begin']['ps'], 'DMA event time')
                if kind in {'AW', 'W', 'B', 'AR', 'R'}:
                    counters[kind] += 1
                    need(r['count'] == counters[kind], 'monotonic DMA ' + kind)
                    need((r['ps'] - 700) % host_period_ps == host_period_ps // 2, 'actual host edge ' + kind)
                if kind == 'END':
                    need(r['aborted'] == 0, 'cold bank has no DMA abort')
                    t['end_line'] = line
                    active = None
            elif name in {'NTRPU_TRANSFER_AR', 'NTRPU_TRANSFER_AW'}:
                need(r['id'] == current and r['host'] in (0, 1), 'shared AXI ownership')
                need(r['addr'] % 64 == 0 and 1 <= r['beats'] <= 256 and r['addr'] % 4096 + r['beats'] * 64 <= 4096, 'shared AXI burst/page')
                b, off = address(current, r['addr'])
                need(off + 64 * r['beats'] <= SPANS[b], 'shared AXI full range')
                kind = name.rsplit('_', 1)[1]
                shared_counts[current][('host_' if r['host'] else '') + kind.lower()] += 1
                pending = dict(r, buffer=b, offset=off, received=0)
                if kind == 'AR':
                    need(read_pending is None, 'single outstanding shared read')
                    read_pending = pending
                else:
                    need(write_pending is None and r['beats'] <= 4, 'single shared write')
                    need((b < 4) == bool(r['host']), 'immutable input write owner')
                    write_pending = pending
            elif name in {'NTRPU_TRANSFER_LOAD', 'NTRPU_FFN_MEMORY_W'}:
                need(write_pending is not None, 'observed payload belongs to accepted AW')
                p = write_pending; host = name == 'NTRPU_TRANSFER_LOAD'
                b, off = r['buffer'], r['offset']; value = payload(r)
                need(r['id'] == current == p['id'] and p['host'] == host, 'physical write owner')
                need((b, off) == (p['buffer'], p['offset'] + p['received'] * 64), 'physical write address order')
                need(value == expected(b, off), 'independent physical write bytes')
                need(off == len(writes[current, b]) * 64, 'exactly-once contiguous buffer writes')
                writes[current, b].append(value)
                addr = region_base(current, b) + 4032 + off
                physical[addr] = value
                p['received'] += 1
                shared_counts[current]['host_w' if host else 'w'] += 1
                if host:
                    need(active is not None, 'host write has DMA epoch')
                    tx[active]['loads'].append((b, off, value))
                else:
                    need(r['strb'] == 2**64 - 1, 'compute full strobes')
            elif name == 'NTRPU_DDR_FFN_B':
                need(write_pending is not None, 'MIG B belongs to AW')
                p = write_pending
                need(r['id'] == p['id'] and r['host'] == p['host'] and r['bid'] == (9 if p['host'] else 0) and r['resp'] == 0, 'MIG B identity/response')
                need(p['received'] == p['beats'], 'MIG B after all W')
                shared_counts[current]['host_b' if p['host'] else 'b'] += 1
                if p['host']:
                    need(active is not None, 'MIG host B belongs to live DMA')
                    tx[active]['mig_b_ps'] = r['ps']
                write_pending = None
            elif name == 'NTRPU_DDR_FFN_R':
                need(read_pending is not None, 'MIG R belongs to AR')
                p = read_pending; addr = p['addr'] + p['received'] * 64
                need(r['id'] == p['id'] and r['host'] == p['host'] and r['rid'] == (9 if p['host'] else 0), 'MIG R identity')
                need(r['addr'] == addr and r['last'] == int(p['received'] + 1 == p['beats']), 'MIG R address/LAST')
                b, off = address(current, addr); value = payload(r)
                need(addr in physical and value == physical[addr] == expected(b, off), 'MIG R from accepted independent bytes')
                shared_read_beats.append(r)
                shared_counts[current]['host_r' if p['host'] else 'r'] += 1
                if p['host']:
                    need(active is not None, 'MIG host R belongs to live DMA')
                    tx[active].setdefault('mig_reads', []).append(r)
                p['received'] += 1
                if p['received'] == p['beats']:
                    read_pending = None
        need(active is None and read_pending is None and write_pending is None, 'all transactions drained')
        need(len(tx) == 54 and counters == dict(AW=48, W=156, B=48, AR=6, R=6), 'complete DMA coverage')
        for ident in ids:
            want_ops = []
            for b in range(4):
                off = 0
                while off < SPANS[b]:
                    addr = region_base(ident, b) + 4032 + off
                    beats = min(4, (SPANS[b] - off) // 64, (4096 - addr % 4096) // 64)
                    want_ops.append((0, addr, beats)); off += 64 * beats
            want_ops += [(1, region_base(ident, 7) + 4032 + off, 1) for off in (0, 64)]
            batch = [t for t in tx.values() if t['id'] == ident]
            need([(t['begin']['kind'], t['begin']['addr'], t['begin']['beats']) for t in batch] == want_ops, 'independent transfer schedule')
            need(batch[15]['events']['END'][0]['ps'] <= begins[ident - 1]['ps'] and
                 batch[15]['end_line'] < by['NTRPU_DMA_REC_FFN_BEGIN'][ident - 1][0], 'inputs loaded before START')
            for b in range(8):
                need(b''.join(writes[ident, b]) == (normal if b < 4 else golden)[b], 'complete independent buffer image')
            case = cases[ident - 1]; counts = shared_counts[ident]
            for key in ['ar', 'r', 'aw', 'w', 'b']:
                need(counts[key] == case[key], 'shared/core ' + key + ' count')
            need(counts['host_aw'] == counts['host_b'] == 16 and counts['host_w'] == 52 and counts['host_ar'] == counts['host_r'] == 2, 'shared host totals')
            done = [r for r in rec('FFN_IRQ_READ') if r['id'] == ident and r['addr'] == 0xc and r['data'] & 4]
            service = [r for r in rec('FFN_IRQ_SERVICE_REQUEST') if r['id'] == ident]
            need(bool(done) and len(service) == 1 and done[0]['time_ps'] < batch[16]['begin']['ps'] < batch[17]['events']['END'][0]['ps'] < service[0]['time_ps'], 'result reads between completion and service')
        for t in tx.values():
            b, ev, ident = t['begin'], t['events'], t['id']
            need(len(ev['END']) == 1 and all(b['ps'] <= r['ps'] <= ev['END'][0]['ps'] for values in ev.values() for r in values), 'bounded transaction events')
            want = dict(AW=1, W=b['beats'], B_HELD=1, B=1, END=1) if b['kind'] == 0 else dict(AR=1, R_HELD=b['beats'], R=b['beats'], END=1)
            need({k: len(v) for k, v in ev.items() if v} == want, 'complete DMA channel record')
            for key in ['AR', 'AW']:
                for r in ev[key]:
                    need((r['addr'], r['beats']) == (b['addr'], b['beats']), 'host address/length')
            for key in ['W', 'R']:
                for beat, r in enumerate(ev[key]):
                    buf, off = address(ident, b['addr'] + beat * 64)
                    need(r['beat'] == beat and r['last'] == int(beat + 1 == b['beats']) and payload(r) == expected(buf, off), 'host beat bytes/LAST')
                    if key == 'W':
                        need(r['strb'] == 2**64 - 1, 'host strobes')
            if b['kind'] == 0:
                need(ev['B'][0]['ps'] > max(ev['AW'][0]['ps'], ev['W'][-1]['ps'], t['mig_b_ps']), 'MIG/host write response order')
                need(ev['B'][0]['ps'] - ev['B_HELD'][0]['ps'] >= 3 * host_period_ps, 'host B backpressure')
                need(t['loads'] == [(*address(ident, b['addr'] + beat * 64), payload(r)) for beat, r in enumerate(ev['W'])], 'DMA W equals memory write')
            else:
                need(len(t.get('mig_reads', [])) == b['beats'] and not t['loads'], 'MIG read coverage')
                for mr, held, rr in zip(t['mig_reads'], ev['R_HELD'], ev['R']):
                    need(mr['ps'] < rr['ps'] and payload(mr) == payload(rr), 'MIG data arrives unchanged at host')
                    need(rr['ps'] - held['ps'] >= 2 * host_period_ps, 'host R backpressure')
        phase = ui[0]['ps'] % 5000
        for r in shared_read_beats + rec('DDR_FFN_B'):
            need(r['ps'] > admit['ps'] and r['ps'] % 5000 == phase, 'MIG responses on UI edges')
        mgmt = rec('DMA_REC_M')
        need([r['ps'] for r in mgmt] == list(range(5001, mgmt[-1]['ps'] + 1, 10000)), 'continuous management observation')
        stable = [r for r in mgmt if r['ps'] >= admit['ps']]
        need(bool(stable), 'admitted clock observations')
        anchor = stable[0]
        for r in mgmt:
            need(r['ht'] == max(0, 1 + (r['ps'] - 700 - host_period_ps // 2) // host_period_ps), 'continuous host clock')
            need(r['mt'] == 1 + (r['ps'] - 5000) // 10000, 'management clock counter')
            if r['ps'] >= admit['ps']:
                need(all(r[k] == 1 for k in ['den', 'hen', 'rst', 'hrst', 'da', 'ha', 'run', 'dr', 'mr']) and r['ext'] == r['fault'] == 0, 'cold bank remains admitted')
                need(r['dt'] - anchor['dt'] == (r['ps'] - phase) // 5000 - (anchor['ps'] - phase) // 5000, 'continuous MIG UI clock')
        need(any(r.get('mh') == 1 and r.get('rst') == 0 for r in mgmt), 'observed memory held acknowledgement')
        result.update(cases=cases, device_phase_ps=phase, calibration_ps=calibration_ps,
                      admission_ps=admit['ps'], transactions=len(tx), dma_counts=dict(counters),
                      mig_read_beats=len(shared_read_beats), computed_bytes=2112,
                      host_load_bytes=9984, host_result_bytes=384, fp32_values=96,
                      rank_activations=[summary['rank' + str(i)] for i in range(4)])
        review_control(text, seed, host_period_ps, result)
    except (ValueError, KeyError, IndexError, OverflowError) as exc:
        result['errors'].append(str(exc))
    result['status'] = 'FAIL' if result['errors'] else 'PASS'
    return result


def mutations(text):
    """Remove or corrupt independently checked observations, never DUT results."""
    out = {}
    for marker in ['DDR_FFN_PASS', 'FFN_MEMORY_PASS', 'DDR_FFN_ADMITTED', 'DDR_FFN_REGION',
                   'DMA_BEGIN', 'DMA_W', 'DMA_B', 'DMA_R', 'DMA_END', 'TRANSFER_AW',
                   'TRANSFER_AR', 'TRANSFER_LOAD', 'FFN_MEMORY_W', 'DDR_FFN_R', 'DDR_FFN_B',
                   'FFN_IRQ_READ', 'FFN_IRQ_END', 'FFN_HOST_WRITE', 'DMA_REC_M',
                   'DDR_FFN_UI', 'DDR_FFN_ACT']:
        out['missing_' + marker] = re.sub(r'^NTRPU_' + marker + r' .+\n', '', text, count=1, flags=re.M)
    for name, marker, key in [
        ('host_input', 'DMA_W', 'data'), ('host_result', 'DMA_R', 'data'),
        ('mig_result', 'DDR_FFN_R', 'data'), ('physical_input', 'TRANSFER_LOAD', 'data'),
        ('physical_output', 'FFN_MEMORY_W', 'data'), ('alias', 'DDR_FFN_REGION', 'base'),
        ('mig_id', 'DDR_FFN_R', 'rid'), ('mig_last', 'DDR_FFN_R', 'last'),
        ('mig_b_response', 'DDR_FFN_B', 'resp'), ('strobe', 'DMA_W', 'strb'),
        ('host_edge', 'DMA_W', 'ps'), ('mig_edge', 'DDR_FFN_R', 'ps'),
        ('abi_counter', 'FFN_HOST_CASE', 'reads'), ('epoch', 'FFN_HOST_CASE', 'epoch'),
    ]:
        pattern = r'(^NTRPU_' + marker + r' .*?\b' + key + r'=)([0-9a-f]+)'
        def alter(m):
            base = 16 if key in {'data', 'base', 'strb', 'epoch'} else 10
            value = int(m[2], base) ^ 1
            return m[1] + (format(value, 'x') if base == 16 else str(value))
        out[name] = re.sub(pattern, alter, text, count=1, flags=re.M)
    out['model_error'] = text + '\nERROR: injected DDR model diagnostic\n'
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--host-period-ps', type=int, default=4000)
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'), args.seed, args.host_period_ps)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
