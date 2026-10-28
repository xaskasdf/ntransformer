"""Reconstruct ABI2 MMIO and completion ownership from sampled interface pins.

The compute/XDMA peers are protocol models. Internal FSM state is never logged;
the CSR-to-notifier boundary is sampled explicitly alongside external ports.
"""
import argparse
import collections
import json
from pathlib import Path
import re

CASES = ['abi2_decode_and_strobes', 'service_before_transport',
         'transport_before_service_stale_ack', 'captured_polling_mode',
         'service_before_event_acceptance', 'malformed_returned_identity',
         'cancelled_completion', 'immediate_error_completion', 'stopped_host_clock',
         'held_read_across_retirement'] + [f'reset_phase_{i}' for i in range(6)] + [
         'wrong_retirement_fault_and_reset', 'service_rejection_fault_and_reset']
HEX = set('awa wd ws ara rd js je ds de ts te seq epoch'.split())


def review(text, profile):
    def require(ok, why):
        if not ok:
            raise ValueError(why)

    require(not re.search(r'(?mi)^\s*(Error:|Fatal:|FAIL\b|CRITICAL WARNING:)|NTRPU_CSR_FAIL', text), 'tool/test failure')
    config = re.findall(r'^NTRPU_CIRQ_CONFIG profile=(\d+) source_half_ps=(\d+) host_half_ps=(\d+) fault=(\d+)$', text, re.M)
    halves = [(2500, 2000), (2500, 3100), (3100, 1700)][profile]
    require(config == [(str(profile), str(halves[0]), str(halves[1]), '0')], 'clock/fault configuration')
    require(re.findall(r'^NTRPU_CSR_CASE id=(\d+) name=(\w+)$', text, re.M) ==
            [(str(i+1), n) for i, n in enumerate(CASES)], 'case coverage/order')
    terminals = re.findall(r'^NTRPU_CIRQ_PASS (.+)$', text, re.M)
    require(len(terminals) == 1, 'single terminal PASS')
    summary = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', terminals[0])}
    expected = dict(cases=18, reads=96, writes=266, resets=9, jobs=26, done=25, cancel=1)
    require(all(summary.get(k) == v for k, v in expected.items()), 'terminal coverage')
    # The manually held read in case10 is sampled, but has no task-level READ line.
    task_reads = re.findall(r'^NTRPU_CSR_READ id=(\d+) addr=([a-f0-9]+) data=([a-f0-9]+) expected=([a-f0-9]+) resp=(\d+) stall=(\d+)$', text, re.M)
    task_writes = re.findall(r'^NTRPU_CSR_WRITE id=(\d+) addr=([a-f0-9]+) data=([a-f0-9]+) strb=([a-f0-9]+) resp=(\d+) order=(\d+) stall=(\d+)$', text, re.M)
    require(len(task_reads) == 96 and len(task_writes) == 266, 'task traffic counts')
    require(all(int(r[0]) == i and r[2] == r[3] for i, r in enumerate(task_reads)), 'task reads')
    require(all(int(w[0]) == i for i, w in enumerate(task_writes)), 'task write numbering')
    require({int(w[3], 16) for w in task_writes} == set(range(16)), 'byte strobes')
    require({int(w[5]) for w in task_writes} == {0, 1, 2}, 'independent AW/W ordering')
    samples = []
    for line in text.splitlines():
        if line.startswith(('NTRPU_CIRQ_S ', 'NTRPU_CIRQ_H ')):
            fields = dict(re.findall(r'(\w+)=(\w+)', line))
            try:
                row = {k: (0 if fields.get('reset') == '0' and re.search('[xz]', v) else int(v, 16 if k in HEX else 10)) for k, v in fields.items()}
            except ValueError as e:
                raise ValueError('unknown interface value') from e
            samples.append((line[11], row))
    require(samples, 'interface samples missing')
    require(all(samples[i][1]['ps'] <= samples[i+1][1]['ps'] for i in range(len(samples)-1)), 'sample time order')
    counts = collections.Counter()
    events, current = [], None
    prior_irq = 0
    prior = {}
    last_ps = {}
    gaps = []
    seen_reset = False
    pending = running = complete = sent = serviced = delivered = fault = cancel = 0
    irq_enable = captured_irq = accepted = completed = 0
    owner = (0, 0)
    regs = collections.defaultdict(int)
    aw = w = b = read = None
    accepted_writes, accepted_reads = [], []
    job_ps = 0

    def reset_model():
        nonlocal pending, running, complete, sent, serviced, delivered, fault, cancel
        nonlocal irq_enable, captured_irq, accepted, completed, owner, regs, aw, w, b, read
        pending = running = complete = sent = serviced = delivered = fault = cancel = 0
        irq_enable = captured_irq = accepted = completed = 0
        owner = (0, 0)
        regs = collections.defaultdict(int)
        aw = w = b = read = None

    for domain, s in samples:
        ps = s['ps']
        if domain in last_ps:
            delta = ps-last_ps[domain]
            if domain == 'S':
                require(delta == 2*halves[0], 'missing/changed source clock sample')
            else:
                require(delta >= 2*halves[1], 'invalid host clock step')
                if delta > 2*halves[1]:
                    gaps.append(delta)
        last_ps[domain] = ps
        counts[domain+'_samples'] += 1
        if domain == 'H':
            if not s['reset']:
                require(s['irq'] == 0, 'IRQ during reset')
                prior_irq = 0
                continue
            if s['irq']:
                require(current is not None and current['irq'], 'unsolicited/polling IRQ')
                require((s['seq'], s['epoch']) == current['owner'], 'IRQ owner changed')
                if not prior_irq:
                    counts['irq_rises'] += 1
                    require('irq_ps' not in current, 'repeated IRQ assertion')
                    current['irq_ps'] = ps
                if s['ack']:
                    current['ack_ps'] = current.get('ack_ps', ps)
            elif prior_irq:
                require(current is not None and 'service_ps' in current and 'ack_ps' in current,
                        'IRQ removed before both confirmations')
                current['irq_low_ps'] = ps
            prior_irq = s['irq']
            continue
        if not s['reset']:
            if current is not None and not current.get('retired'):
                current['aborted'] = True
                current = None
            reset_model()
            prior = {}
            seen_reset = True
            continue
        if seen_reset:
            counts['resets'] += 1
            seen_reset = False
        # Ready/valid stability is checked at each boundary, independently of tasks.
        for valid, ready, fields in [('awv', 'awr', ['awa']), ('wv', 'wr', ['wd', 'ws']),
                                     ('bv', 'br', ['bresp']), ('arv', 'arr', ['ara']),
                                     ('rv', 'rr', ['rd', 'rresp']), ('jv', 'jr', ['js', 'je']),
                                     ('ev', 'er', ['ei', 'js', 'je']), ('sv', 'sr', ['js', 'je'])]:
            if prior.get(valid) and not prior[ready]:
                require(s[valid] and all(s[f] == prior[f] for f in fields), 'stalled '+valid+' changed')
        require(s['jv'] == pending and s['dr'] == int(running and not complete), 'job/completion admission')
        require((s['js'], s['je']) == owner, 'descriptor owner changed')
        require(s['ev'] == int(complete and not sent and not fault), 'event retention')
        require(s['ei'] == captured_irq, 'captured IRQ mask changed')
        require(s['sv'] == int(sent and serviced and not delivered and not fault), 'service retention/order')
        retire_ready = int(complete and sent and delivered and not fault and (s['ts'], s['te']) == owner)
        require(s['tr'] == retire_ready, 'retirement guard')
        require(s['bv'] == int(b is not None), 'write response VALID')
        if b is not None:
            require(s['bresp'] == b, 'write response value')
        require(s['rv'] == int(read is not None), 'read response VALID')
        if read is not None:
            require((s['rd'], s['rresp']) == read, 'read snapshot/value')
        # Compute read and write expectations from state BEFORE this sampled edge.
        status = (s['er'] << 8) | (fault << 7) | (sent << 6) | (serviced << 5) | (cancel << 4) | (complete << 2) | (running << 1) | pending
        def valid_address(a):
            return a % 4 == 0 and (a <= 0x68 or (0x100 <= a < 0x200 and a % 32 != 28))
        def read_value(a):
            if not valid_address(a):
                return (0, 3)
            constants = {0: 0x4652544e, 4: 0x20000, 8: 15, 12: status, 16: 0,
                         20: irq_enable, 24: accepted, 28: completed, 0x64: owner[0], 0x68: owner[1]}
            return (constants[a] if a in constants else regs[a], 0)
        new_read = read_value(s['ara']) if s['arv'] and s['arr'] else None
        commit = aw is not None and w is not None and b is None
        response = command = merged = 0
        if commit:
            data, strobe = w
            mask = sum(255 << (8*i) for i in range(4) if strobe & (1 << i))
            command = data & mask
            if not valid_address(aw):
                response = 3
            elif aw == 0x10:
                if command == 1:
                    response = 2 if pending or running or complete or not s['er'] or fault else 0
                elif command == 2:
                    response = 2 if not (pending or running) or cancel or (s['dv'] and s['dr']) else 0
                elif command == 4:
                    response = 2 if not complete or serviced or fault or (regs[0x38], regs[0x3c]) != owner else 0
                elif command:
                    response = 2
            elif aw == 0x14 or 0x20 <= aw <= 0x3c or 0x100 <= aw < 0x200:
                old = irq_enable if aw == 0x14 else regs[aw]
                merged = (old & ~mask) | command
                allowed = {0x14: 1, 0x2c: 0x3fff3fff, 0x30: 3, 0x34: 0x3f3f3f}.get(aw, 0xffffffff)
                response = 2 if merged & ~allowed else 0
            else:
                response = 2
        if s['br'] and s['bv']:
            b = None
        if s['rr'] and s['rv']:
            read = None
        if new_read is not None:
            read = new_read
            accepted_reads.append((s['ara'], *new_read))
        if s['jv'] and s['jr']:
            counts['jobs'] += 1
            pending, running = 0, 1
            accepted += 1
            job_ps = ps
        if s['ev'] and s['er']:
            require(current is None or current.get('retired'), 'event overwritten')
            current = dict(owner=owner, irq=captured_irq, event_ps=ps)
            events.append(current)
        if s['sv'] and s['sr']:
            require(current is not None and 'service_ps' not in current, 'duplicate/unowned service')
            current['service_ps'] = ps
            delivered = 1
        if s['tv'] and s['tr'] and not s['reject']:
            require(current is not None and 'service_ps' in current, 'retirement without service')
            require(not current['irq'] or 'ack_ps' in current, 'retirement before host ACK')
            require((s['ts'], s['te']) == current['owner'], 'retirement identity')
            require(not current.get('retired'), 'duplicate retirement')
            current.update(retired=True, retire_ps=ps)
            complete = sent = serviced = delivered = 0
        if s['reject'] or (s['tv'] and not s['tr']):
            fault = 1
        if s['ev'] and s['er']:
            sent = 1
        if s['dv'] and s['dr']:
            counts['done'] += 1
            running, complete = 0, 1
            completed += 1
            captured_irq = irq_enable
            regs[0x40] = 0x1c00 | s['status']
            regs[0x44], regs[0x48] = 8192, 155648
            regs[0x4c], regs[0x50] = s['ds'], s['de']
            regs[0x58] = (ps-job_ps)//(2*halves[0])
            regs[0x60] = int(s['ds'] != owner[0]) | (int(s['de'] != owner[1]) << 1)
        if commit:
            accepted_writes.append((aw, *w, response))
            b = response
            if response == 0:
                if aw == 0x10:
                    if command == 1:
                        pending, cancel = 1, 0
                        owner = (regs[0x20], regs[0x24])
                    elif command == 2:
                        cancel = 1
                    elif command == 4:
                        serviced = 1
                elif aw == 0x14:
                    irq_enable = merged
                else:
                    regs[aw] = merged
            aw = w = None
        if s['awv'] and s['awr']:
            require(aw is None, 'AW overwritten')
            aw = s['awa']
        if s['wv'] and s['wr']:
            require(w is None, 'W overwritten')
            w = (s['wd'], s['ws'])
        prior = s
    expected_writes = [(int(a, 16), int(d, 16), int(st, 16), int(r)) for _, a, d, st, r, _, _ in task_writes]
    require(accepted_writes == expected_writes, 'sampled writes vs task records')
    expected_reads = [(int(a, 16), int(d, 16), int(r)) for _, a, d, _, r, _ in task_reads]
    require(len(accepted_reads) == 97, 'sampled read count')
    require(accepted_reads.count((0x64, 0x1020304a, 0)) == 1, 'held owner read coverage')
    require([r for r in accepted_reads if r != (0x64, 0x1020304a, 0)] == expected_reads, 'sampled reads vs task records')
    for k in ['resets', 'jobs', 'done']:
        require(counts[k] == expected[k], 'sample count '+k)
    retired = [e for e in events if e.get('retired')]
    aborted = [e for e in events if e.get('aborted')]
    require(len(events) == 25 and len(retired) == 19 and len(aborted) == 6, 'event outcomes')
    require(len(gaps) == 2, 'stopped-host coverage')
    latencies = [e['retire_ps']-e['event_ps'] for e in retired]
    return dict(status='PASS', profile=profile, counts=dict(counts), cases=18, reads=97, writes=266,
                events=len(events), retired=len(retired), aborted=len(aborted),
                host_clock_gaps_ps=gaps, retirement_latency_ps=[min(latencies), max(latencies)], events_detail=events)


def mutations(text):
    def field(prefix, field, value, predicate=lambda row: True, nth=1):
        hits = 0
        lines = text.splitlines(True)
        for i, line in enumerate(lines):
            if line.startswith(prefix) and predicate(dict(re.findall(r'(\w+)=(\w+)', line))):
                hits += 1
                if hits < nth:
                    continue
                lines[i] = re.sub(r'\b'+field+r'=\w+', field+'='+value, line, count=1)
                return ''.join(lines)
        raise ValueError('mutation target absent: '+field)
    return {
        'missing_pass': re.sub(r'^NTRPU_CIRQ_PASS .*\n', '', text, flags=re.M),
        'missing_case': re.sub(r'^NTRPU_CSR_CASE id=5 .*\n', '', text, flags=re.M),
        'fatal_after_pass': text+'\nError: injected evidence error\n',
        'source_clock': field('NTRPU_CIRQ_S ', 'ps', '0', lambda r:int(r['ps'])>1000000),
        'read_value': field('NTRPU_CIRQ_S ', 'rd', '4652544f', lambda r:r['rv']=='1'),
        'write_response': field('NTRPU_CIRQ_S ', 'bresp', '2', lambda r:r['bv']=='1' and r['bresp']=='0'),
        'owner_sequence': field('NTRPU_CIRQ_S ', 'js', 'deadbeef', lambda r:r['jv']=='1'),
        'completion_ready': field('NTRPU_CIRQ_S ', 'dr', '1', lambda r:r['reset']=='1' and r['dr']=='0'),
        'event_mask': field('NTRPU_CIRQ_S ', 'ei', '0', lambda r:r['ev']=='1' and r['ei']=='1'),
        'event_retention': field('NTRPU_CIRQ_S ', 'ev', '0', lambda r:r['ev']=='1'),
        'service_order': field('NTRPU_CIRQ_S ', 'sv', '1', lambda r:r['reset']=='1' and r['ev']=='1' and r['sv']=='0'),
        'retire_guard': field('NTRPU_CIRQ_S ', 'tr', '1', lambda r:r['reset']=='1' and r['tr']=='0'),
        'irq_owner': field('NTRPU_CIRQ_H ', 'seq', 'deadbeef', lambda r:r['irq']=='1'),
        'irq_missing_ack': re.sub(r'(^NTRPU_CIRQ_H .*\back=)1', r'\g<1>0', text, flags=re.M),
        'irq_drop_early': field('NTRPU_CIRQ_H ', 'irq', '0', lambda r:r['irq']=='1', nth=2),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--profile', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(review(args.log.read_text(), args.profile), indent=2))
