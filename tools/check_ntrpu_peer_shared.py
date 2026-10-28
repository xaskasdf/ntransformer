"""Independent trace/ownership audit of the experimental shared-source profile.

Standard library only: this file also runs from a restored artifact bundle.
This audits simulated transactions, not PCIe ordering or physical timing.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def words(text, count, flags=0):
    value = int(text, 16)
    require(value < 1 << (count * 64 + flags), 'packed width')
    value >>= flags
    return tuple((value >> (64 * n)) & ((1 << 64) - 1)
                 for n in reversed(range(count)))


def request(text):
    w = words(text, 16)
    return dict(raw=int(text, 16), identity=w[:3], source=w[3:7], target=w[7:11],
                window=w[11], model=w[12], src=w[13], dst=w[14], size=w[15])


def lease(text):
    w = words(text, 18, 4)
    return dict(source=w[:4], target=w[4:8], src=w[8], src_size=w[9],
                published=w[10], published_size=w[11], dst=w[12], dst_size=w[13],
                window=w[14], window_start=w[15], window_size=w[16], model=w[17],
                flags=int(text, 16) & 15)


def bad_config(c, previous):
    s, t = c['source'], c['target']
    return (not s[0] or s[0] != t[0] or s[1] != 0 or t[1] != 1 or
            not s[3] or not t[3] or not c['window'] or not c['model'] or
            bool(previous and (s[0] != previous['source'][0] or
                 s[3] <= previous['source'][3] or t[3] <= previous['target'][3] or
                 c['window'] <= previous['window'])))


def rejection(r, c, watermark, revoked, active):
    if (r['source'] != c['source'] or r['target'] != c['target'] or
            r['identity'][:2] != (c['source'][0], 0) or
            r['window'] != c['window'] or r['model'] != c['model']):
        return 1
    if revoked or c['flags'] != 15:
        return 2
    if r['identity'][2] <= watermark:
        return 6
    a, b, size = c['src'] + r['src'], c['dst'] + r['dst'], r['size']
    if (not size or r['src'] + size > c['src_size'] or
            r['dst'] + size > c['dst_size'] or r['dst'] < c['window_start'] or
            r['dst'] + size > c['window_start'] + c['window_size'] or
            a + size > 1 << 64 or b + size > 1 << 64):
        return 3
    if r['src'] < c['published'] or r['src'] + size > c['published'] + c['published_size']:
        return 4
    if (size > 4096 or any(v % 64 for v in (size, a, b, r['src'], r['dst'])) or
            a + size > 1 << 34 or a // 4096 != (a + size - 1) // 4096):
        return 5
    if any(r['dst'] < v['r']['dst'] + v['r']['size'] and
           v['r']['dst'] < r['dst'] + size for v in active.values()):
        return 7
    return 0


def parse(text, full):
    require(not any(s in text for s in ('SG_FAIL', 'SG_WATCHDOG', 'Fatal:')), 'simulation failure')
    cases = 6 if full else 15
    require(text.count(f'SG ALL_PASS mode={"full" if full else "unit"} cases={cases}') == 1, 'missing end marker')
    passed = [int(s.split()[2]) for s in text.splitlines() if s.startswith('SG PASS ')]
    require(passed == list(range(1, cases + 1)), 'scenario coverage')
    grouped = defaultdict(list)
    for line in text.splitlines():
        if line.startswith(('SG ', 'SF ')) and not line.startswith(('SG PASS ', 'SG ALL_PASS ')):
            p = line.split()
            require(len(p) >= 5 and p[1] in {'SAMPLE', 'CTRL', 'CFG', 'REQ', 'REJECT', 'FWD', 'RETIRE',
                                            'CMD', 'AR', 'DATA', 'LOCAL', 'DMA', 'ACK'}, 'unknown trace')
            grouped[(int(p[3]), int(p[2]))].append((p[0], p[1], p[4:]))
    require({k[0] for k in grouped} == set(range(1, cases + 1)), 'epoch coverage')
    return grouped


def audit(text, full):
    groups = parse(text, full)
    stats, endings = Counter(), []
    epoch = None
    for (e, ts), rows in sorted(groups.items()):
        if e != epoch:
            if epoch is not None:
                require(not any(rejects), 'unconsumed rejection at reset')
                endings.append(dict(epoch=epoch, held=len(active), watermark=watermark, halted=halted))
            epoch = e
            active, history, rejects = {}, {}, [None, None]
            c, watermark, halted, revoked, cfg_error = None, 0, False, False, False
        sg = defaultdict(list)
        for prefix, kind, data in rows:
            if prefix == 'SG':
                sg[kind].append(data)
        if sg:
            require(len(sg['SAMPLE']) == len(sg['CTRL']) == 1, f'{e}:{ts} missing/duplicate clock observation')
            counts = [sum(k[0] == l for k in active) for l in range(2)]
            pending = sum((1 << l) for l in range(2) if any(k[0] == l and not v['forwarded'] for k, v in active.items()))
            expected = [watermark, *counts, pending, int(halted), int(not active and not halted), int(cfg_error)]
            require(list(map(int, sg['SAMPLE'][0])) == expected, f'{e}:{ts} ownership/state mismatch; expected {expected}, got {sg["SAMPLE"][0]}')
            stats['samples'] += 1
            stats['peak_contexts'] = max(stats['peak_contexts'], len(active))
            stats['pending_cycles'] += bool(pending)
            enable, stop, link, fault, revoke = map(int, sg['CTRL'][0])
            retirements = []
            invalid_retire = False
            for p in sg['RETIRE']:
                lane, status, size = map(int, p[:3]); r = request(p[3])
                key = (lane, r['identity'])
                item = active.get(key)
                invalid = (item is None or item['r']['raw'] != r['raw'] or
                           not item['forwarded'] or item['reported'] or size != (0 if status else r['size']))
                invalid_retire |= invalid
                retirements.append((key, status, size, invalid))
            cfg_error = False
            require(len(sg['CFG']) <= 1 and len(sg['REQ']) <= 1, 'non-atomic ingress/config')
            # Check the pre-edge state; all effects below become visible next sample.
            for p in sg['REJECT']:
                lane, code = map(int, p[:2]); ident = words(p[2], 3)
                require(rejects[lane] == (code, ident), 'rejection identity/code mismatch')
                rejects[lane] = None
                stats['rejected'] += 1
                stats[f'reject_code_{code}'] += 1
            for p in sg['FWD']:
                lane = int(p[0]); r = request(p[1]); key = (lane, r['identity'])
                queue = [k for k, v in active.items() if k[0] == lane and not v['forwarded']]
                require(queue and queue[0] == key and active[key]['r']['raw'] == r['raw'], 'unreserved/reordered forward')
                active[key]['forwarded'] = ts
                stats['forwarded'] += 1
            for p in sg['REQ']:
                require(c is not None and enable and not (sg['CFG'] or halted or stop or fault or invalid_retire) and link, 'admission while closed')
                lane = int(p[0]); r = request(p[1])
                require(lane in (0, 1) and rejects[lane] is None, 'invalid ingress lane/rejection capacity')
                error = rejection(r, c, watermark, revoked or revoke, active)
                stats['requests'] += 1
                if error:
                    rejects[lane] = (error, r['identity'])
                else:
                    require(counts[lane] < 2, 'reservation capacity exceeded')
                    key = (lane, r['identity'])
                    require(key not in history, 'identity reused')
                    item = dict(r=r, cfg=deepcopy(c), forwarded=0, reported=False,
                                command=0, native=0, beats=0, local=None, dma=False, ack=False)
                    active[key] = item; history[key] = item
                    watermark = r['identity'][2]
                    stats['admitted'] += 1
            if sg['CFG']:
                require(not active and not halted, 'config replaced an active lease')
                new = lease(sg['CFG'][0][0]); cfg_error = bad_config(new, c)
                if not cfg_error:
                    c = new; revoked = bool(revoke)
                stats['bad_config' if cfg_error else 'configured'] += 1
            revoked |= bool(revoke)
            halted |= bool(stop or not link or fault or invalid_retire)
        else:
            retirements = []  # Native AR edges need not coincide with user clocks.

        # Independently relate downstream reader/data/DMA/ACK evidence to ownership.
        for prefix, kind, p in rows:
            if prefix != 'SF':
                continue
            require(full, 'downstream trace in unit test')
            lane = int(p[0])
            if kind in {'DMA', 'ACK'}:
                w = words(p[1], 9, 4); key = (lane, w[:3]); item = history.get(key)
                require(item is not None, 'event has no reservation')
                r = item['r']
                require(w[3:7] == r['target'] and w[7:9] == (r['window'], r['size']) and int(p[1], 16) & 15 == 0,
                        'completion identity/generation/count mismatch')
                require(item['beats'] * 64 == r['size'] and item['local'] == 0, 'completion before complete local payload')
                require(not item[kind.lower()], 'duplicate completion')
                item[kind.lower()] = True
                stats[kind.lower()] += 1
            elif kind == 'AR':
                address, beats = map(int, p[1:])
                candidates = [v for (l, _), v in history.items() if l == lane and v['command'] and v['native'] < v['r']['size']]
                require(candidates, 'native read without command')
                item = candidates[0]
                require(1 <= beats <= 16 and address == item['cfg']['src'] + item['r']['src'] + item['native'], 'native split address/length')
                item['native'] += beats * 32
                require(item['native'] <= item['r']['size'], 'native over-read')
                stats['native_ar'] += 1
            else:
                sequence = int(p[1]); key = (lane, (epoch, 0, sequence)); item = history.get(key)
                require(item is not None and item['forwarded'], 'reader event without forwarded context')
                r = item['r']
                if kind == 'CMD':
                    address, beats = map(int, p[2:])
                    require(not item['command'] and address == item['cfg']['src'] + r['src'] and beats * 64 == r['size'], 'reader command mismatch')
                    item['command'] = ts; stats['commands'] += 1
                elif kind == 'DATA':
                    index, last = map(int, p[2:4]); data = int(p[4], 16).to_bytes(64, 'little')
                    require(item['command'] and index == item['beats'] + 1 and index * 64 <= r['size'] and last == (index * 64 == r['size']), 'payload index/last')
                    address = item['cfg']['src'] + r['src'] + item['beats'] * 64
                    expected = bytes((a ^ (a >> 8) ^ (a >> 16) ^ 0x5b) & 255 for a in range(address, address + 64))
                    require(data == expected, 'payload byte mismatch')
                    item['beats'] += 1; stats['payload_bytes'] += 64
                elif kind == 'LOCAL':
                    status, size = map(int, p[2:])
                    require(item['command'] and item['local'] is None and size == (0 if status else r['size']), 'local completion count/context')
                    if not status:
                        require(item['beats'] * 64 == item['native'] == r['size'], 'local completion before complete payload/native reads')
                    item['local'] = status
                else:
                    raise ValueError('unknown downstream event')
        for key, status, size, invalid in retirements:
            if invalid:
                stats['invalid_retire'] += 1
                continue
            item = active[key]; item['reported'] = True
            if status:
                halted = True; stats['failed_retire_retained'] += 1
            else:
                if full:
                    require(item['local'] == 0 and item['dma'] and item['ack'] and item['beats'] * 64 == size, 'release without local+DMA+ACK')
                    stats['successful_payload_bytes'] += size
                del active[key]
                stats['successful_retire'] += 1
    require(not any(rejects), 'unconsumed final rejection')
    endings.append(dict(epoch=epoch, held=len(active), watermark=watermark, halted=halted))
    require(stats['peak_contexts'] == 4 and stats['forwarded'] and stats['rejected'], 'missing exercised paths')
    if full:
        require(stats['successful_retire'] == 27 and stats['successful_payload_bytes'] == 46848, 'full success coverage')
    return dict(counters=dict(sorted(stats.items())), epoch_endings=endings)


def negative_controls(text, full):
    lines = text.splitlines()
    changes = []
    for token in ('SG REQ ', 'SG FWD ', 'SG RETIRE ', 'SG REJECT ', 'SG CFG ', 'SG CTRL '):
        index = next(i for i, s in enumerate(lines) if s.startswith(token))
        changes.append(('drop_' + token.split()[1].lower(), index, None))
    index = next(i for i, s in enumerate(lines) if s.startswith('SG SAMPLE '))
    p = lines[index].split(); p[4] = str(int(p[4]) + 1)
    changes.append(('watermark', index, ' '.join(p)))
    p = lines[index].split(); p[5] = str(int(p[5]) + 1)
    changes.append(('ownership_count', index, ' '.join(p)))
    if full:
        for token in ('SF ACK ', 'SF DMA ', 'SF LOCAL ', 'SF AR ', 'SF CMD ', 'SF DATA '):
            index = next(i for i, s in enumerate(lines) if s.startswith(token))
            changes.append(('drop_' + token.split()[1].lower(), index, None))
        index = next(i for i, s in enumerate(lines) if s.startswith('SF DATA '))
        p = lines[index].split(); p[-1] = format(int(p[-1], 16) ^ 1, '0128x')
        changes.append(('payload_bit', index, ' '.join(p)))
    rejected = []
    for name, index, value in changes:
        altered = lines[:index] + ([value] if value is not None else []) + lines[index + 1:]
        try:
            audit('\n'.join(altered), full)
        except (ValueError, KeyError, IndexError, OverflowError):
            rejected.append(name)
        else:
            raise ValueError('audit accepted negative control: ' + name)
    return rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    a = ap.parse_args(); out = a.output.resolve()
    result = json.loads((out / 'result.json').read_text())
    require(result['status'] == 'EXPORTED_PENDING_REVIEW' and result['returncode'] == 0 and
            not result['timed_out'] and result['functional_marker'], 'runner failed')
    require(result['sources'] == result['sources_after'], 'sources changed during run')
    for name, digest in result['sources'].items():
        p = (out.parent / 'repo' / name).resolve()
        require(p.is_relative_to(out.parent / 'repo') and sha(p) == digest, 'source binding: ' + name)
    for name, digest in json.loads((out / 'FILES.json').read_text()).items():
        p = (out / name).resolve()
        require(p.is_relative_to(out) and sha(p) == digest, 'artifact binding: ' + name)
    full = result['full']
    if full:
        profile = json.loads((out.parent / 'repo/NATIVE_PROFILE.json').read_text())
        for name, digest in profile['selected_rtl'].items():
            require(result['sources'].get(name) == digest, 'reference RTL changed: ' + name)
        for name, config in profile['selected_read_configs'].items():
            observed = dict(line.split('\t', 1) for line in (out / 'suite' / name).read_text().splitlines()
                            if line.startswith('CONFIG.'))
            require(observed == config, 'reference read IP changed: ' + name)
    log = out / ('suite/simulation.log' if full else 'simulation.log')
    text = log.read_text(errors='strict')
    report = audit(text, full)
    report.update(status='SHARED_TRACE_REVIEWED', mode='full_behavioral' if full else 'unit',
                  result_sha256=sha(out / 'result.json'), log_sha256=sha(log),
                  negative_controls=negative_controls(text, full))
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
