"""Qualify real FFN/ABI2/CDC results, host transactions and IRQ lifecycle.

Numerical bytes use the existing independent memory/fixture references. IRQ
ownership and timing come from observed boundary handshakes, not FSM state.
"""
import argparse
import json
from pathlib import Path
import re
from check_ntrpu_ffn_host import CASES, records, expected_writes
from check_ntrpu_ffn_memory import review as memory_review, RECOVERY_CASES
from check_ntrpu_ffn_memory_fixture import review_log as fixture_review
from check_ntrpu_ffn_transfer import review as transfer_review, mutations as transfer_mutations


def expected_abi2_writes(ident, dim, hidden, scenario, tiles):
    mode = (ident-1) % 3
    rows = expected_writes(ident, dim, hidden, scenario, True, tiles)
    rows[0] = (0x14, int(mode != 2), 0)
    if scenario == 4:
        return rows
    rows = rows[:-2]  # Replace the ABI1 START-rejection/ACK with the ABI2 lifecycle.
    if mode == 0:
        rows += [(0x14, 0, 0), (0x38, 0xffffffff, 0), (0x3c, ident+600, 0), (0x10, 4, 2)]
    elif mode == 2:
        rows += [(0x14, 1, 0)]
    rows += [(0x10, 1, 2), (0x38, 0xfffffffe, 0), (0x3c, ident+600, 0), (0x10, 4, 0)]
    if mode == 1:
        rows += [(0x10, 1, 2), (0x10, 4, 2)]
    return rows


def review_control(text, seed, fp32=True, fixture=None, host_period_ps=4000, *, delayed_reset=False, recovery_each=False):
    if not fp32:
        raise ValueError('Only FP32 is qualified by this integration')
    result = memory_review(text, seed, True, delayed_reset=delayed_reset, recovery_each=recovery_each) if fixture is None else fixture_review(text, fixture, seed, True)
    errors = result['errors']
    def need(ok, why):
        if not ok:
            errors.append(why)
    need(not re.search(r'(?mi)^\s*(Error:|Fatal:|FAIL\b|CRITICAL WARNING:)', text), 'tool/test error')
    need(host_period_ps in (4000, 6200), 'clock profile')
    specs = (RECOVERY_CASES if recovery_each else CASES) if fixture is None else [(256, 720, 0, 0)]*3
    tiles = (2, 2, 2) if fixture is None else (4, 4, 8)
    n = len(specs)
    ids = list(range(1, n+1))
    finished = [i for i, spec in enumerate(specs, 1) if spec[2] != 4]
    aborted = [i for i, spec in enumerate(specs, 1) if spec[2] == 4]
    host = records(text, 'NTRPU_FFN_HOST_CASE')
    writes = records(text, 'NTRPU_FFN_HOST_WRITE')
    reads = records(text, 'NTRPU_FFN_IRQ_READ')
    memory = {c['id']: c for c in result['cases']}
    events = {name: records(text, 'NTRPU_FFN_IRQ_'+name) for name in
              ['BEGIN', 'JOB', 'DONE', 'EVENT', 'EDGE', 'ACK', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']}
    for name in ['BEGIN', 'JOB']:
        need([r.get('id') for r in events[name]] == ids, 'one '+name+' per admitted job')
    for name in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']:
        need([r.get('id') for r in events[name]] == finished, 'one '+name+' per non-aborted job')
    need([r.get('id') for r in host] == ids, 'host completion/reset order')
    for rows in [reads, writes, events['EDGE'], events['ACK']]:
        need(all(r.get('id') in ids for r in rows), 'unknown case identity')
    need([w.get('order') for w in writes] == [i % 3 for i in range(len(writes))], 'AW/W skew schedule')
    reset_assert = records(text, 'NTRPU_FFN_IRQ_RESET_ASSERT')
    reset_ready = records(text, 'NTRPU_FFN_IRQ_RESET_READY')
    if fixture is None:
        need(records(text, 'NTRPU_FFN_IRQ_CONFIG') == [dict(host_period_ps=host_period_ps,
             device_period_ps=5000, host_phase_ps=700, reset_edges=64)], 'error-bank clock/reset contract')
        need(len(reset_assert) == len(reset_ready) == 5, 'initial and four in-flight resets')
        for a, b in zip(reset_assert, reset_ready):
            need(b.get('time_ps', 0)-a.get('time_ps', 0) >= 64*max(5000, host_period_ps), 'reset overlap/drain interval')
    else:
        need(not reset_assert and not reset_ready, 'fixture uses separate initial reset only')
    for name in ['JOB', 'DONE', 'EVENT', 'SERVICE', 'RETIRE']:
        for r in events[name]:
            need(r.get('time_ps', 0) % 5000 == 2500, 'source edge '+name)
    for name in ['EDGE', 'ACK']:
        for r in events[name]:
            need((r.get('time_ps', 0)-700) % host_period_ps == host_period_ps//2, 'host edge '+name)
    accepted = completed = 0
    timings = []
    for ident, spec in enumerate(specs, 1):
        dim, hidden, scenario, target = spec
        mode = (ident-1) % 3
        ws = [w for w in writes if w.get('id') == ident]
        rs = [r for r in reads if r.get('id') == ident]
        er = {name: [r for r in rows if r.get('id') == ident] for name, rows in events.items()}
        hs = [r for r in host if r.get('id') == ident]
        need(len(hs) == 1, 'one host outcome '+str(ident))
        expected = expected_abi2_writes(ident, dim, hidden, scenario, tiles)
        actual = [(r.get('addr'), r.get('data'), r.get('resp')) for r in ws]
        if scenario == 4:
            # A real in-flight reset can interrupt post-START shadow poisoning.
            # It must preserve an exact ordered prefix, including all descriptor
            # fields and the accepted START (first64 writes), with no later ACK.
            need(64 <= len(actual) <= len(expected) and actual == expected[:len(actual)],
                 'reset-aborted MMIO prefix '+str(ident))
        else:
            need(actual == expected, 'complete ABI2 MMIO write trace '+str(ident))
        need(len(rs) >= 6, 'initial MMIO read count')
        init = [(r.get('addr'), r.get('data')) for r in rs[:6]]
        need(init[:5] == [(0, 0x4652544e), (4, 0x20000), (8, 15), (0x18, accepted), (0x1c, completed)], 'ABI2 identity/acceptance counters')
        if len(init) == 6:
            need(init[5][0] == 0xc and init[5][1] & 0x187 == 0x100, 'ready for START')
        for name in ['BEGIN', 'JOB', 'DONE', 'EVENT', 'SERVICE', 'RETIRE']:
            for r in er[name]:
                need((r.get('sequence'), r.get('epoch')) == (0xfffffffe, ident+600), 'owner identity '+name)
        for r in er['BEGIN']:
            need(r.get('mode') == mode and r.get('fault') == 0, 'IRQ mode or injected defect')
        if len(er['BEGIN']) != 1 or len(er['JOB']) != 1 or len(hs) != 1:
            continue
        begin, job, h = er['BEGIN'][0], er['JOB'][0], hs[0]
        need(job['time_ps'] > begin['time_ps'], 'START/compute acceptance order')
        if fixture is None:
            preceding = [r['time_ps'] for r in reset_ready if r['time_ps'] < job['time_ps']]
            need(bool(preceding) and begin['time_ps'] >= max(preceding), 'job follows coordinated reset drain')
        if scenario == 4:
            need(h == dict(id=ident, reset=1), 'reset abort cannot publish a completion')
            need(all(not er[name] for name in ['DONE', 'EVENT', 'EDGE', 'ACK', 'SERVICE', 'RETIRE', 'END']), 'aborted computation has no notification/retirement')
            later = [a['time_ps'] for a in reset_assert if a['time_ps'] > job['time_ps']]
            need(bool(later), 'in-flight reset after acceptance')
            accepted = completed = 0
            continue
        accepted += 1
        completed += 1
        if any(len(er[name]) != 1 for name in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']):
            continue
        done, event, request, service, retire, end = [er[k][0] for k in ['DONE', 'EVENT', 'SERVICE_REQUEST', 'SERVICE', 'RETIRE', 'END']]
        mem = memory.get(ident, {})
        status = 6 if scenario in (1, 2) else 7 if scenario == 3 else 1 if scenario == 5 else 0
        stages = 4 if scenario == 0 else 0 if scenario == 5 else target
        need(h.get('reset') == 0 and h.get('status') == status and h.get('stages') == stages and h.get('flags') == 0, 'host result classification')
        need((h.get('sequence'), h.get('epoch')) == (0xfffffffe, ident+600), 'host completion identity')
        need(done.get('status') == status and done.get('stages') == stages, 'actual compute result classification')
        need(h.get('bytes') == done.get('bytes') == mem.get('down_bytes') and
             h.get('stage_bytes') == done.get('stage_bytes') == mem.get('bytes'), 'physical/compute/CSR byte agreement')
        need(h.get('cancel') == int(scenario == 3), 'cancel coverage')
        need(done['time_ps']-job['time_ps'] == h.get('compute_cycles', -1)*5000, 'hardware elapsed count vs compute handshakes')
        need(job['time_ps'] < done['time_ps'] < event['time_ps'] < request['time_ps'] <= service['time_ps'] < retire['time_ps'] < end['time_ps'], 'completion/event/service/retirement ordering')
        need(event.get('enable') == int(mode != 2) and end.get('mode') == mode, 'captured IRQ mode')
        need(0 < end.get('retire_polls', 0) < 128, 'bounded observed retirement polls')
        need(h.get('writes') == len(ws), 'MMIO write count')
        need(h.get('reads') == len(rs) == h.get('polls', -999)+19+end['retire_polls']+int(mode == 1), 'MMIO read count')
        need(h.get('polls', 0) > 0, 'completion polling occurred')
        need(0 < h.get('compute_cycles', 0) < h.get('total_cycles', 0) < (250000 if fixture is None else 1000000), 'device elapsed range')
        need(0 < h.get('program_cycles', 0) and h['program_cycles']+h.get('poll_cycles', 0) < h['total_cycles'], 'host lifecycle duration')
        need(h['total_cycles'] > mem.get('cycles', 0), 'consumer finishes before retirement')
        p = h.get('polls', 0)
        middle = rs[6+p:6+p+13]
        addresses = list(range(0x40, 0x64, 4))+[0x64, 0x68, 0x18, 0x1c]
        need([r.get('addr') for r in middle] == addresses, 'metadata/owner/counter read order')
        if len(middle) == 13:
            values = {r['addr']: r['data'] for r in middle}
            need(values[0x40] & 15 == status and (values[0x40] >> 8) & 7 == stages, 'completion status register')
            need(values[0x44] == h['bytes'] and values[0x48] == h['stage_bytes'], 'byte register values')
            need(values[0x4c] == values[0x64] == 0xfffffffe and values[0x50] == values[0x68] == ident+600, 'returned and admitted identities')
            need(values[0x54] == values[0x60] == 0 and values[0x58]+(values[0x5c] << 32) == h['compute_cycles'], 'metadata diagnostic/elapsed register')
            need(values[0x18] == accepted and values[0x1c] == completed, 'exactly-once counters')
        polls = rs[6:6+p]
        need(len(polls) == p and all(r.get('addr') == 0xc for r in polls), 'poll addresses')
        if polls:
            need(polls[-1]['data'] & 4 and all(not r['data'] & 4 for r in polls[:-1]), 'first observed completion')
        trailing = rs[6+p+13:]
        need(all(r.get('addr') == 0xc for r in trailing), 'retirement status reads')
        if trailing:
            need(trailing[-1]['data'] & 0x187 == 0x100, 'idle after retirement')
            if mode == 1:
                need(trailing[0]['data'] & 0x24 == 0x24, 'software ACK retained completion')
        for r in rs:
            if r.get('addr') == 0xc:
                value = r['data']
                need(value & ~0x1ff == 0 and not value & 0x80, 'status reserved/fault bits')
                need(not (value & 4 and value & 3), 'completion cannot overlap next job')
        if mode == 2:
            need(not er['EDGE'] and not er['ACK'], 'polling requires no interrupt or transport ACK')
        else:
            need([r.get('level') for r in er['EDGE']] == [1, 0] and len(er['ACK']) == 1, 'one held IRQ and transport ACK')
            if len(er['EDGE']) == 2 and len(er['ACK']) == 1:
                rise, fall = [r['time_ps'] for r in er['EDGE']]
                ack = er['ACK'][0]['time_ps']
                need(event['time_ps'] < rise <= ack < fall < retire['time_ps'], 'host IRQ/ACK/drain order')
                need(fall > service['time_ps'], 'IRQ cannot fall before software service')
                need((ack < request['time_ps']) if mode == 0 else (ack > service['time_ps']), 'two acknowledgement orders')
        timings.append(dict(id=ident,mode=mode,compute_cycles=h['compute_cycles'],
                            event_to_retire_ps=retire['time_ps']-event['time_ps'],
                            service_to_retire_ps=retire['time_ps']-service['time_ps'],
                            event_to_irq_ps=(er['EDGE'][0]['time_ps']-event['time_ps']) if er['EDGE'] else None))
    for i in range(1, n):
        prev = [r for r in events['END'] if r.get('id') == i]
        nex = [r for r in events['BEGIN'] if r.get('id') == i+1]
        if prev and nex:
            need(prev[0]['time_ps'] < nex[0]['time_ps'], 'new job only after retirement')
    result.update(status='FAIL' if errors else 'PASS', host_cases=host, mmio_writes=len(writes),
                  mmio_reads=len(reads), irq_timings=timings, completed_ids=finished, aborted_ids=aborted,
                  host_period_ps=host_period_ps,
                  scope='Real FFN/HLS -> ABI2 CSR -> XPM IRQ transport with separate MMIO service and protocol host ACK; '+
                        'AMD asynchronous AXI4/AXI4-Lite and behavioral RAM. No real XDMA, DDR, clock-loss supervisor or full inference.')
    return result


def review(text, seed, host_period_ps, fixture=None, *, delayed_reset=False, recovery_each=False):
    if fixture is None:
        return review_control(text, seed, True, None, host_period_ps, delayed_reset=delayed_reset, recovery_each=recovery_each)
    if delayed_reset or recovery_each:
        raise ValueError('Delayed reset contract is only defined for the synthetic error bank')
    fixture = Path(fixture)
    result = transfer_review(text, fixture, seed,
                            host_reviewer=lambda t,s,f,p: review_control(t,s,f,p,host_period_ps))
    errors = result['errors']
    def need(ok, why):
        if not ok:
            errors.append(why)
    config = records(text, 'NTRPU_ASYNC_CONFIG')
    need(len(config) == 1, 'one transfer clock configuration')
    if len(config) != 1:
        result['status'] = 'FAIL'
        return result
    conf = config[0]
    need(conf == dict(host_period_ps=host_period_ps,host_phase_ps=700,device_period_ps=5000,
                      reset_edges=64,time_ps=conf.get('time_ps')), 'transfer clock/reset contract')
    need(conf.get('time_ps',0) >= 64*max(host_period_ps,5000), 'initial reset before traffic')
    metrics = records(text, 'NTRPU_ASYNC_METRICS')
    ends = records(text, 'NTRPU_FFN_IRQ_END')
    requests = records(text, 'NTRPU_FFN_IRQ_SERVICE_REQUEST')
    need([r.get('id') for r in metrics] == [1,2,3], 'three timed transfers')
    def edges(ps, period, phase):
        return max(0,(ps-phase-period//2)//period+1)
    for row, end, request, legacy in zip(metrics,ends,requests,result.get('transfer_cases', [])):
        times = [row.get('start_ps',0)]
        for stage in ['load','compute','read']:
            need(row.get(stage+'_ps',0)>0, 'positive transfer duration')
            times.append(times[-1]+row.get(stage+'_ps',0))
        need(times[0]>=conf['time_ps'] and row.get('end_ps')==times[-1], 'transfer absolute interval')
        for stage,a,b,old in zip(['load','compute','read'],times,times[1:],['load_cycles','control_compute_cycles','result_cycles']):
            need(row.get('host_'+stage+'_edges')==edges(b,host_period_ps,700)-edges(a,host_period_ps,700), 'host edges '+stage)
            need(legacy.get(old)==edges(b,5000,0)-edges(a,5000,0), 'device edges '+stage)
        need(end.get('time_ps',0)>request.get('time_ps',0)>times[-1], 'software service after output readback')
    result.update(status='FAIL' if errors else 'PASS',async_metrics=metrics,clock_config=conf,
                  scope='Host AXI -> CDC -> shared RAM -> real FFN/HLS -> host readback; ABI2 IRQ retirement. '+
                        'Pinned Brandon weights and synthetic inputs. Protocol transport ACK, not XDMA/PCIe or physical bandwidth.')
    return result


def mutations(text, fixture=False):
    def change(prefix, field, value):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)\w+', lambda m:m[1]+value, text,count=1,flags=re.M)
    result = {
        'fatal':text+'\nError: injected\n',
        'missing_pass':text.replace('NTRPU_FFN_MEMORY_PASS ','REMOVED_PASS ',1),
        'core_output':change('NTRPU_FFN_MEMORY_W','data','0'),
        'owner':change('NTRPU_FFN_IRQ_EVENT','sequence','ffffffff'),
        'event_mode':change('NTRPU_FFN_IRQ_EVENT','enable','0'),
        'job_clock':change('NTRPU_FFN_IRQ_JOB','time_ps','1'),
        'compute_cycles':change('NTRPU_FFN_HOST_CASE','compute_cycles','1'),
        'missing_retire':re.sub(r'^NTRPU_FFN_IRQ_RETIRE id=1 .+\n','',text,count=1,flags=re.M),
        'missing_service':re.sub(r'^NTRPU_FFN_IRQ_SERVICE id=1 .+\n','',text,count=1,flags=re.M),
        'missing_transport_ack':re.sub(r'^NTRPU_FFN_IRQ_ACK id=1 .+\n','',text,count=1,flags=re.M),
        'irq_early_drop':change('NTRPU_FFN_IRQ_EDGE id=1 level=0','time_ps','2700'),
        'stale_ack_accepted':text.replace('data=00000004 resp=2','data=00000004 resp=0',1),
        'admission_counter':change('NTRPU_FFN_IRQ_READ[^\n]*addr=00000018','data','00000001'),
        'abi_version':change('NTRPU_FFN_IRQ_READ[^\n]*addr=00000004','data','00010000'),
        'end_before_retire':change('NTRPU_FFN_IRQ_END','time_ps','1'),
    }
    if fixture:
        result.update(transfer_mutations(text))
        result['host_clock'] = change('NTRPU_ASYNC_CONFIG','host_period_ps','4001')
        result['elapsed'] = change('NTRPU_ASYNC_METRICS','end_ps','1')
        result['host_edges'] = change('NTRPU_ASYNC_METRICS','host_read_edges','1')
    else:
        result['reset_duration'] = change('NTRPU_FFN_IRQ_RESET_READY','time_ps','1')
        result['reset_missing_start'] = re.sub(r'^NTRPU_FFN_HOST_WRITE id=16 addr=00000010 data=00000001[^\n]+\n','',text,count=1,flags=re.M)
        result['reset_poison_value'] = text.replace('NTRPU_FFN_HOST_WRITE id=16 addr=00000020 data=00000000','NTRPU_FFN_HOST_WRITE id=16 addr=00000020 data=ffffffff',1)
        result['reset_late_ack'] = text+'\nNTRPU_FFN_HOST_WRITE id=16 addr=00000010 data=00000004 resp=0 order=0\n'

    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('log',type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--host-period-ps',type=int,choices=[4000,6200],required=True)
    p.add_argument('--fixture',type=Path)
    a=p.parse_args()
    r=review(a.log.read_text(),a.seed,a.host_period_ps,a.fixture)
    print(json.dumps(r,indent=2))
    raise SystemExit(r['status']!='PASS')
