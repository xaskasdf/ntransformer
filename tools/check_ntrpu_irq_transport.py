"""Reconstruct IRQ events from external source/host pin samples, not DUT state."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

HEX = {'seq', 'epoch', 'sseq', 'sepoch', 'rseq', 'repoch'}


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def review(text, profile):
    def need(ok, why):
        if not ok:
            raise ValueError(why)

    try:
        need(profile in (0, 1, 2), 'profile')
        need(not re.search(r'(?mi)^\s*(Fatal:|Error:|FAIL\b)', text), 'tool/bank error')
        period, host_period, phase = [(5000, 4000, 700), (5000, 6200, 700), (6200, 3400, 1100)][profile]
        cases, resets, configs, passes = [], [], [], []
        counts = Counter()
        case_counts = {i: Counter() for i in range(1, 16)}
        previous = {}
        pending = None
        irq = 0
        rejected_expected = 0
        held = {}
        last_time = -1
        stopped_at = resumed_at = None
        gaps = []
        samples = Counter()
        observed = []
        for line in text.splitlines():
            if not line.startswith('NTRPU_IRQ_'):
                continue
            kind = line.split()[0].removeprefix('NTRPU_IRQ_')
            raw = fields(line)
            need(not passes, 'trace after terminal PASS')
            if kind == 'CONFIG':
                row = {k: int(v) for k, v in raw.items()}
                configs.append(row)
                need(row == dict(profile=profile, source_ps=period, host_ps=host_period, phase_ps=phase, fault=0), 'configuration')
                continue
            need(len(configs) == 1, 'configuration missing/duplicate')
            if kind == 'CASE':
                ident = int(raw['id'])
                need(ident == len(cases)+1 and pending is None and not irq, 'case order or old event')
                cases.append(ident)
                continue
            if kind == 'RESET':
                row = {k: int(v) for k, v in raw.items()}
                resets.append(row)
                need(row['case'] in (0, 9, 10, 11, 12, 13, 14, 15), 'reset case')
                if pending is not None:
                    counts['aborted'] += 1
                    case_counts[pending['case']]['aborted'] += 1
                    pending = None
                held.clear()
                irq = rejected_expected = 0
                if row['stop']:
                    need(row['case'] == 15 and stopped_at is None, 'clock stop identity')
                    stopped_at = row['ps']
                continue
            if kind == 'CLOCK_RESUME':
                need(int(raw['case']) == 15 and resumed_at is None and stopped_at is not None, 'resume identity')
                resumed_at = int(raw['ps'])
                need(resumed_at-stopped_at >= 63*period, 'clock stop duration')
                continue
            if kind == 'PASS':
                row = {k: int(v) for k, v in raw.items()}
                passes.append(row)
                need(pending is None and not irq and not held, 'pending output at PASS')
                need(row == dict(cases=15, accepted=23, retired=16, aborted=7, resets=8,
                                 rejected=2, irq_edges=counts['irq_edges']), 'PASS counts')
                continue
            need(kind in ('SRC', 'DST'), 'unknown trace marker')
            # Unknown power-up values are allowed only while the sampled domain is reset.
            need(raw['rst'] in ('0', '1'), 'unknown reset')
            active = raw['rst'] == '1'
            row = {}
            for k, v in raw.items():
                if re.search('[xz]', v, re.I):
                    need(not active, 'unknown active pin')
                    row[k] = None
                else:
                    row[k] = int(v, 16 if k in HEX else 10)
            ident, time = row['case'], row['ps']
            need(time >= last_time, 'time order')
            last_time = time
            need(ident == (cases[-1] if cases else 0), 'sample case')
            prev = previous.get(kind)
            need(row['tick'] == (prev['tick']+1 if prev else 1), 'missing/duplicate clock sample')
            if kind == 'SRC':
                need(time == period//2+(row['tick']-1)*period, 'source clock accounting')
            elif prev:
                delta = time-prev['ps']
                if delta != host_period:
                    need(stopped_at is not None and resumed_at is not None and
                         prev['ps'] <= stopped_at < resumed_at <= time and delta > host_period, 'unexplained host clock gap')
                    gaps.append(delta)
            else:
                need(time == phase+host_period//2, 'host clock phase')
            previous[kind] = row
            samples[kind] += 1
            if not active:
                if kind == 'SRC':
                    need(row['ready'] == row['srdy'] == row['ret'] == 0, 'source admission during reset')
                    rejected_expected = 0
                else:
                    need(row['req'] == 0, 'IRQ during reset')
                    irq = 0
                continue
            if kind == 'SRC':
                need(row['reject'] == rejected_expected, 'service rejection response')
                rejected_expected = 0
                for name, valid, ready, payload in [
                    ('event', 'ev', 'ready', ('seq', 'epoch', 'en')),
                    ('service', 'svc', 'srdy', ('sseq', 'sepoch')),
                    ('retirement', 'ret', 'rrdy', ('rseq', 'repoch')),
                ]:
                    data = tuple(row[k] for k in payload)
                    if name in held:
                        need(row[valid] == 1 and data == held[name], 'held '+name+' changed')
                    if row[valid] and not row[ready]:
                        held[name] = data
                    else:
                        held.pop(name, None)
                if row['ev'] and row['ready']:
                    need(pending is None, 'second event admitted')
                    pending = dict(case=ident, seq=row['seq'], epoch=row['epoch'], enabled=row['en'],
                                   service=False, ack=False, requests=0, accepted_ps=time, request_ps=None)
                    counts['accepted'] += 1
                    case_counts[ident]['accepted'] += 1
                if row['svc'] and row['srdy']:
                    need(pending is not None, 'service without event')
                    if (row['sseq'], row['sepoch']) == (pending['seq'], pending['epoch']):
                        need(not pending['service'], 'duplicate service')
                        pending['service'] = True
                        pending['service_ps'] = time
                    else:
                        rejected_expected = 1
                        counts['rejected'] += 1
                if row['ret']:
                    need(pending is not None and pending['service'], 'retire without serviced event')
                    need((row['rseq'], row['repoch']) == (pending['seq'], pending['epoch']), 'retirement identity')
                    need(not pending['enabled'] or (pending['ack'] and pending['requests'] == 1 and not irq), 'early retirement')
                    if row['rrdy']:
                        observed.append(dict(pending, retired_ps=time))
                        counts['retired'] += 1
                        case_counts[ident]['retired'] += 1
                        pending = None
            else:
                if row['req']:
                    need(pending is not None and pending['enabled'], 'unrequested/masked IRQ')
                    need((row['seq'], row['epoch']) == (pending['seq'], pending['epoch']), 'IRQ identity')
                    if not irq:
                        need(prev is None or not prev['ack'], 'new IRQ into stale ACK')
                        pending['requests'] += 1
                        need(pending['requests'] == 1, 'duplicate IRQ')
                        pending['request_ps'] = time
                        counts['irq_edges'] += 1
                    if row['ack']:
                        pending['ack'] = True
                        pending.setdefault('ack_ps', time)
                if irq and not row['req']:
                    need(pending is not None and pending['service'] and pending['ack'], 'IRQ withdrawn before both acknowledgements')
                    pending['withdrawn_ps'] = time
                irq = row['req']
        need(cases == list(range(1, 16)) and len(passes) == 1, 'case/terminal coverage')
        need([r['case'] for r in resets] == [0, 9, 10, 11, 12, 13, 14, 15], 'reset coverage')
        need([r['stop'] for r in resets] == [0]*7+[1] and len(gaps) == 1, 'stopped-clock reset coverage')
        need(all(counts[k] == v for k, v in dict(accepted=23, retired=16, aborted=7, rejected=2).items()), 'reconstructed counts')
        for ident, tally in case_counts.items():
            expected = dict(accepted=2 if ident >= 9 or ident == 7 else 1, retired=2 if ident == 7 else 1)
            if ident >= 9:
                expected['aborted'] = 1
            need(dict(tally) == expected, 'per-case coverage')
        return dict(status='PASS', profile=profile, counts=dict(counts), samples=dict(samples),
                    clock_gap_ps=gaps[0], events=observed,
                    scope='Protocol peers with three XPM CDCs; no actual XDMA, FFN, reset supervisor, physical MTBF or board timing.')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return dict(status='FAIL', errors=[str(exc)])


def mutations(text):
    result = {'tool_error': text+'\nError: injected tool error\n'}
    lines = text.splitlines()

    def change(name, predicate, key=None, value=None, remove=False):
        changed = lines.copy()
        index = next(i for i, line in enumerate(lines) if predicate(line, fields(line)))
        if remove:
            changed.pop(index)
        else:
            old = fields(changed[index])[key]
            replacement = value(old) if callable(value) else value
            changed[index] = re.sub(r'\b'+key+r'=\S+', key+'='+replacement, changed[index], count=1)
        result[name] = '\n'.join(changed)+'\n'

    change('missing_pass', lambda l, r: l.startswith('NTRPU_IRQ_PASS'), remove=True)
    change('irq_epoch', lambda l, r: l.startswith('NTRPU_IRQ_DST') and r['req']=='1', 'epoch', '00000000')
    change('retirement_epoch', lambda l, r: l.startswith('NTRPU_IRQ_SRC') and r['ret']=='1', 'repoch', '00000000')
    change('unknown_active_irq', lambda l, r: l.startswith('NTRPU_IRQ_DST') and r['rst']=='1', 'req', 'x')
    change('missing_rejection', lambda l, r: l.startswith('NTRPU_IRQ_SRC') and r['reject']=='1', 'reject', '0')
    change('missing_source_sample', lambda l, r: l.startswith('NTRPU_IRQ_SRC') and r['rst']=='1', remove=True)
    change('source_clock_time', lambda l, r: l.startswith('NTRPU_IRQ_SRC'), 'ps', lambda v: str(int(v)+1))
    change('missing_abort', lambda l, r: l.startswith('NTRPU_IRQ_RESET') and r['case']=='10', remove=True)
    change('missing_clock_resume', lambda l, r: l.startswith('NTRPU_IRQ_CLOCK_RESUME'), remove=True)
    change('duplicate_case', lambda l, r: l.startswith('NTRPU_IRQ_CASE') and r['id']=='2', 'id', '1')
    change('held_event_identity', lambda l, r: l.startswith('NTRPU_IRQ_SRC') and r['case']=='7' and r['ev']=='1' and r['ready']=='0', 'seq', '00000000')
    change('masked_notification', lambda l, r: l.startswith('NTRPU_IRQ_SRC') and r['case']=='4' and r['ev']==r['ready']=='1', 'en', '1')
    # Remove every observed transport ACK for an event while retaining the
    # published retirement, so the validator cannot accept a green marker alone.
    changed = [re.sub(r'\back=1\b', 'ack=0', l) if l.startswith('NTRPU_IRQ_DST case=1 ') else l for l in lines]
    result['missing_transport_ack'] = '\n'.join(changed)+'\n'
    changed = [re.sub(r'\bsseq=\S+', 'sseq=00000000', l) if l.startswith('NTRPU_IRQ_SRC case=1 ') and 'svc=1 ' in l else l for l in lines]
    result['wrong_service_identity'] = '\n'.join(changed)+'\n'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--profile', type=int, required=True)
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'), args.profile)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')


if __name__ == '__main__':
    main()
