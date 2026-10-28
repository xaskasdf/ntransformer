"""Reconstruct supervised IRQ traffic, actual resets, clock losses and admission from pins."""
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
        case_counts = {i: Counter() for i in range(1, 22)}
        previous = {}
        pending = None
        irq = 0
        rejected_expected = 0
        held = {}
        last_time = -1
        stops, recoveries, gaps = [], [], []
        last_pins = None
        asserted = {'rst': 0, 'hrst': 0}
        loss_delays = []
        masks = {15:1,16:1,17:2,18:3,19:1,20:2,21:3}
        releases = Counter()
        m_samples = 0
        samples = Counter()
        observed = []
        for line in text.splitlines():
            if not line.startswith('NTRPU_SIRQ_'):
                continue
            kind = line.split()[0].removeprefix('NTRPU_SIRQ_')
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
                need(row['case'] in [0,*range(9,22)], 'reset case')
                need(row['stop']==masks.get(row['case'],0),'reset stopped domains')
                if row['case']>=16:
                    delay=row['ps']-stops[-1]['ps']
                    need(0<delay<=800000,'clock-loss reset deadline')
                    loss_delays.append(delay)
                if pending is not None:
                    counts['aborted'] += 1
                    case_counts[pending['case']]['aborted'] += 1
                    pending = None
                held.clear()
                irq = rejected_expected = 0
                continue
            if kind == 'STOP':
                row={k:int(v) for k,v in raw.items()}
                need(row['case']==15+len(stops) and row['mask']==masks[row['case']], 'stop order/mask')
                need(not stops or 'resume' in stops[-1], 'overlapping stop')
                stops.append(row)
                continue
            if kind == 'CLOCK_RESUME':
                row={k:int(v) for k,v in raw.items()}
                need(stops and 'resume' not in stops[-1], 'missing stop/duplicate resume')
                stop=stops[-1]
                need((row['case'],row['mask'])==(stop['case'],stop['mask']), 'resume identity')
                need(row['ps']-stop['ps']>=64*period, 'clock stop duration')
                stop['resume']=row['ps']
                continue
            if kind == 'RECOVER':
                row={k:int(v) for k,v in raw.items()}
                need(row['case']==([0,*range(9,22)])[len(recoveries)], 'recovery order')
                need(pending is None and not irq, 'stale event at recovery')
                need(last_pins and all(last_pins[k] for k in ['run','rst','hrst','da','ha','ready']), 'recovery before admission')
                recoveries.append(row)
                continue
            if kind in {'M','OUT'}:
                need(all(re.fullmatch(r'\d+',v) for v in raw.values()), 'unknown supervision pin')
                row={k:int(v) for k,v in raw.items()}
                need(set(row)=={'case','ps','mt','dt','ht','ext','den','hen','rst','hrst','da','ha','req','can','held','ready','run'},'supervision fields')
                need(row['ps']>=last_time,'supervision time order');last_time=row['ps']
                for counter,domain in [('dt','SRC'),('ht','DST')]:
                    need(row[counter]==previous.get(domain,{}).get('tick',0),'supervision edge count')
                if kind=='M':
                    m_samples+=1
                    need(row['mt']==m_samples and row['ps']==5000+(m_samples-1)*10000+1, 'management clock')
                need(all(row[k] in (0,1) for k in ['ext','den','hen','rst','hrst','da','ha','req','can','held','ready','run']), 'nonbinary supervision')
                if row['da'] or row['ha']:
                    need(row['run'] and row['rst'] and row['hrst'] and row['ready'] and not row['req'] and not row['ext'],'unsafe supervised admission')
                if row['can']:need(row['held'] and row['req'] and not row['ext'],'unacknowledged rearm')
                if last_pins:
                    for reset,counter,domain,admit in [('rst','dt','SRC','da'),('hrst','ht','DST','ha')]:
                        if last_pins[reset] and not row[reset]:asserted[reset]=row[counter]
                        if not last_pins[reset] and row[reset]:
                            need(row[counter]-asserted[reset]>=64, 'insufficient real reset edges')
                            need(row['ps']==previous[domain]['ps']+1,'asynchronous reset release')
                            releases[reset]+=1
                        if not last_pins[admit] and row[admit]:
                            need(row['ps']==previous[domain]['ps']+1,'asynchronous admission release')
                            releases[admit]+=1
                last_pins=row
                continue
            if kind == 'PASS':
                row = {k: int(v) for k, v in raw.items()}
                passes.append(row)
                need(pending is None and not irq and not held, 'pending output at PASS')
                need(row == dict(cases=21, accepted=35, retired=22, aborted=13, resets=14,
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
            domain_bit=2 if kind=='SRC' else 1
            for stop in stops:
                if stop['mask'] & domain_bit and time>stop['ps'] and ('resume' not in stop or time<stop['resume']):
                    need(False,'clock edge while stopped')
            nominal=period if kind=='SRC' else host_period
            if prev:
                delta=time-prev['ps']
                if delta!=nominal:
                    valid=[q for q in stops if q['mask'] & domain_bit and 'resume' in q and prev['ps']<=q['ps']<q['resume']<=time]
                    need(len(valid)==1 and delta>nominal, 'unexplained clock gap')
                    gaps.append(dict(domain=kind,case=valid[0]['case'],ps=delta))
            else:
                need(time==(period//2 if kind=='SRC' else phase+host_period//2),'initial clock phase')
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
        need(cases == list(range(1, 22)) and len(passes) == 1, 'case/terminal coverage')
        need([r['case'] for r in resets] == [0,*range(9,22)], 'reset coverage')
        need(len(stops)==7 and len(gaps)==9 and len(loss_delays)==6, 'stopped-clock reset coverage')
        need(len(recoveries)==14 and releases=={'rst':14,'hrst':14,'da':14,'ha':14}, 'release coverage')
        need(all(counts[k] == v for k, v in dict(accepted=35, retired=22, aborted=13, rejected=2).items()), 'reconstructed counts')
        for ident, tally in case_counts.items():
            expected = dict(accepted=2 if ident >= 9 or ident == 7 else 1, retired=2 if ident == 7 else 1)
            if ident >= 9:
                expected['aborted'] = 1
            need(dict(tally) == expected, 'per-case coverage')
        return dict(status='PASS', profile=profile, counts=dict(counts), samples=dict(samples),
                    clock_gaps=gaps, clock_loss_reset_ps=loss_delays, recoveries=len(recoveries), releases=dict(releases), management_samples=m_samples, events=observed,
                    scope='Actual supervisor plus IRQ XPM transport; external service/ACK peers remain protocol models. No AXI converters, XDMA, FFN, DDR or physical timing qualification.')
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

    change('missing_pass', lambda l, r: l.startswith('NTRPU_SIRQ_PASS'), remove=True)
    change('irq_epoch', lambda l, r: l.startswith('NTRPU_SIRQ_DST') and r['req']=='1', 'epoch', '00000000')
    change('retirement_epoch', lambda l, r: l.startswith('NTRPU_SIRQ_SRC') and r['ret']=='1', 'repoch', '00000000')
    change('unknown_active_irq', lambda l, r: l.startswith('NTRPU_SIRQ_DST') and r['rst']=='1', 'req', 'x')
    change('missing_rejection', lambda l, r: l.startswith('NTRPU_SIRQ_SRC') and r['reject']=='1', 'reject', '0')
    change('missing_source_sample', lambda l, r: l.startswith('NTRPU_SIRQ_SRC') and r['rst']=='1', remove=True)
    change('source_clock_time', lambda l, r: l.startswith('NTRPU_SIRQ_SRC'), 'ps', lambda v: str(int(v)+1))
    change('missing_abort', lambda l, r: l.startswith('NTRPU_SIRQ_RESET') and r['case']=='10', remove=True)
    change('missing_clock_resume', lambda l, r: l.startswith('NTRPU_SIRQ_CLOCK_RESUME'), remove=True)
    change('duplicate_case', lambda l, r: l.startswith('NTRPU_SIRQ_CASE') and r['id']=='2', 'id', '1')
    change('held_event_identity', lambda l, r: l.startswith('NTRPU_SIRQ_SRC') and r['case']=='7' and r['ev']=='1' and r['ready']=='0', 'seq', '00000000')
    change('masked_notification', lambda l, r: l.startswith('NTRPU_SIRQ_SRC') and r['case']=='4' and r['ev']==r['ready']=='1', 'en', '1')
    # Remove every observed transport ACK for an event while retaining the
    # published retirement, so the validator cannot accept a green marker alone.
    changed = [re.sub(r'\back=1\b', 'ack=0', l) if l.startswith('NTRPU_SIRQ_DST case=1 ') else l for l in lines]
    result['missing_transport_ack'] = '\n'.join(changed)+'\n'
    changed = [re.sub(r'\bsseq=\S+', 'sseq=00000000', l) if l.startswith('NTRPU_SIRQ_SRC case=1 ') and 'svc=1 ' in l else l for l in lines]
    result['wrong_service_identity'] = '\n'.join(changed)+'\n'
    change('missing_management_sample',lambda l,r:l.startswith('NTRPU_SIRQ_M') and r['case']=='16',remove=True)
    change('false_admission',lambda l,r:l.startswith('NTRPU_SIRQ_M') and r['run']=='0','da','1')
    change('missing_auto_stop',lambda l,r:l.startswith('NTRPU_SIRQ_STOP') and r['case']=='17',remove=True)
    change('wrong_clock_mask',lambda l,r:l.startswith('NTRPU_SIRQ_STOP') and r['case']=='18','mask','1')
    change('missing_recovery',lambda l,r:l.startswith('NTRPU_SIRQ_RECOVER') and r['case']=='19',remove=True)
    change('external_count_corrupt',lambda l,r:l.startswith('NTRPU_SIRQ_OUT') and r['rst']=='1','dt',lambda v:str(int(v)+1))
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
