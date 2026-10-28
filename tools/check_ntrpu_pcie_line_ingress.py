"""Independent byte/transaction replay of the bounded ingress fixture.

This complements pin-level stability assertions; it is not a hardware proof
of external isolation. The fixture supplies the trusted barrier inputs.
"""
import argparse
import json
from pathlib import Path
import re

CASES = [
    ('complete', [0]*16, 0),
    ('sparse_and_identical_duplicate', [0]*33, 0),
    ('conflicting_duplicate', [0, 2], 4),
    ('high_address_alias', [3, 2], 3),
    ('illegal_mask', [2], 2), ('unsupported_size', [2], 2),
    ('unsupported_burst', [2], 2), ('early_last', [2], 2),
    ('late_last', [2], 2), ('stop_after_aw', [2], 1),
    ('stop_after_w', [2], 1), ('stop_exposed_line', [0]*16, 1),
    ('stop_waiting_destination', [0]*16, 1), ('destination_error', [0]*16, 5),
    ('foreign_destination_response', [0]*16, 5),
    ('write_after_publication', [0]*16+[2], 6),
    ('holes_then_stop', [0]*16, 1), ('queue_pressure', [0, 0], 1),
    ('missing_last_long_drain', [2], 2),
]
LINE_JOBS = {1, 2, 12, 13, 14, 15, 16}


def request_fixture(job, index):
    base = (0x1000fffc0 if job % 2 else 0x100000000)
    offset, mask = 0, 15
    if job in {1, 12, 13, 14, 15}:
        offset = index*4
    elif job == 2:
        if index < 2:
            offset, mask = 0, 5
        else:
            offset, mask = ((index-1)//2)*4, (5 if (index-1)%2 == 0 else 10)
    elif job == 4 and index == 0:
        base ^= 1 << 32
    elif job == 5:
        mask = 16  # deliberately outside the addressed DWORD
    elif job == 16:
        offset = index*4 if index < 16 else 0
    elif job == 17:
        offset, mask = index*4, 1
    elif job == 18:
        offset = index*4
    first = (mask & -mask).bit_length()-1 if job != 5 else 0
    return dict(addr=base+offset+first, len=1 if job in {7, 8} else 0,
                size=3 if job == 6 else 2, strb=mask << offset)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def review(text, seed):
    need(not re.search(r'^(Fatal:|ERROR:|WARNING:|CRITICAL WARNING:)', text, re.M), 'positive log diagnostic')
    events = []
    for line in text.splitlines():
        if not line.startswith('NTRPU_INGRESS_'):
            continue
        label, *values = line.split()
        fields = dict(value.split('=', 1) for value in values)
        need(len(fields) == len(values), 'duplicate field')
        events.append((label.removeprefix('NTRPU_INGRESS_'), fields))
    need(events and events[-1][0] == 'PASS', 'missing terminal PASS')
    need(sum(k == 'PASS' for k, _ in events) == 1, 'duplicate PASS')
    job = 0
    armed = False
    address = None
    beats = []
    bank = [None]*64
    responses = 0
    line_cycle = None
    last_b_cycle = None
    done_cycle = None
    foreign = 0
    result = False
    counts = dict(aw=0, w=0, b=0, lines=0, results=0)
    byte_writes = 0
    max_frame_beats = 0
    last_cycle = 0
    measurements = []
    passes = None
    for kind, f in events:
        if 'cycle' in f:
            cycle = int(f['cycle'])
            need(cycle >= last_cycle, 'event chronology')
            last_cycle = cycle
        if kind == 'CASE':
            if job:
                need(result and responses == len(CASES[job-1][1]), 'previous job incomplete')
            else:
                need(counts == dict(aw=1, w=1, b=1, lines=0, results=0), 'unarmed negative write')
            need(address is None and not beats, 'case changed with pending transport')
            job += 1
            need(job <= 19 and int(f['job']) == job and f['name'] == CASES[job-1][0], 'fixture case order')
            armed = False
            bank = [None]*64
            responses = 0
            line_cycle = done_cycle = last_b_cycle = None
            foreign = 0
            result = False
        elif kind == 'ARM':
            need(job > 0 and not armed and not result, 'duplicate/unexpected ARM')
            need(int(f['epoch'], 16) == job and int(f['tag'], 16) == 0x1000+job, 'ARM identity')
            window = 0x1000fffc0 if job % 2 else 0x100000000
            need(int(f['window'], 16) == window and int(f['dest'], 16) == 0x1fffffffc0, 'ARM full address')
            armed = True
        elif kind == 'AW':
            need(address is None and (armed or job == 0), 'AW overwritten/unarmed context')
            address = {k: int(v, 16 if k in {'addr', 'id'} else 10) for k, v in f.items()}
            expected = request_fixture(job, responses)
            need(all(address[k] == expected[k] for k in ['addr','len','size']), 'AW fixture intent')
            counts['aw'] += 1
        elif kind == 'W':
            need(not beats or beats[-1]['last'] == 0, 'new frame before prior B')
            beat = {k: int(v, 16 if k in {'data', 'strb'} else 10) for k, v in f.items()}
            expected = request_fixture(job, responses)
            payload = int.from_bytes(bytes((job*37+j*19+seed*7)&255 for j in range(64)), 'little')
            if job == 3 and responses == 1:
                payload ^= 1
            last = int(len(beats) == (260 if job == 19 else 1)) if job in {7,9,19} else 1
            need(beat['strb'] == expected['strb'] and beat['data'] == payload and beat['last'] == last, 'W fixture intent')
            beats.append(beat)
            counts['w'] += 1
        elif kind == 'B':
            need(address is not None and beats and beats[-1]['last'] == 1, 'B without AW/WLAST')
            need(cycle > address['cycle'] and cycle > beats[-1]['cycle'], 'B before captured channels')
            need(int(f['id'], 16) == address['id'], 'B ID mismatch')
            resp = int(f['resp'])
            need(responses < len(CASES[job-1][1]) if job else responses == 0, 'extra B')
            need(resp == (CASES[job-1][1][responses] if job else 3), 'unexpected response')
            max_frame_beats = max(max_frame_beats, len(beats))
            if resp == 0:
                need(len(beats) == 1 and address['len'] == 0 and address['size'] == 2, 'successful write framing')
                beat = beats[0]
                low = address['addr']
                limit = low//4*4+4
                base = low//64*64
                need(base == window and beat['strb'] != 0, 'write reservation')
                raw = beat['data'].to_bytes(64, 'little')
                for lane in range(64):
                    if beat['strb'] >> lane & 1:
                        need(low <= base+lane < limit, 'write outside DWORD')
                        expected = (job*37+lane*19+seed*7) & 255
                        need(raw[lane] == expected, 'fixture byte pattern')
                        need(bank[lane] is None or bank[lane] == raw[lane], 'conflicting duplicate accepted')
                        bank[lane] = raw[lane]
                        byte_writes += 1
            if job == 19:
                need(len(beats) == 261 and all(b['last'] == 0 for b in beats[:-1]), 'long drain coverage')
            address = None
            beats = []
            responses += 1
            counts['b'] += 1
            last_b_cycle = cycle
        elif kind == 'LINE':
            need(job in LINE_JOBS and line_cycle is None and all(b is not None for b in bank), 'partial/duplicate line')
            need(address is None and not beats and cycle > last_b_cycle, 'line before ingress drain')
            need(int(f['epoch'], 16) == job and int(f['tag'], 16) == 0x1000+job, 'line identity')
            need(int(f['addr'], 16) == 0x1fffffffc0, 'line destination')
            need(int(f['data'], 16).to_bytes(64, 'little') == bytes(bank), 'line byte reference')
            line_cycle = cycle
            counts['lines'] += 1
        elif kind == 'DONE':
            need(line_cycle is not None and cycle > line_cycle and done_cycle is None, 'unexpected destination response')
            need(int(f['epoch'], 16) == job, 'destination epoch')
            if int(f['tag'], 16) != 0x1000+job:
                need(job == 15 and not foreign and int(f['tag'], 16) == ((0x1000+job)^0x10000), 'foreign completion fixture')
                foreign += 1
            else:
                need(int(f['error']) == (1 if job == 14 else 0), 'destination error fixture')
                done_cycle = cycle
        elif kind == 'RESULT':
            need(not result and address is None and not beats and responses == len(CASES[job-1][1]), 'result before ingress drain')
            need(int(f['epoch'], 16) == job and int(f['tag'], 16) == 0x1000+job, 'result identity')
            need(int(f['status']) == CASES[job-1][2], 'result status')
            need((done_cycle is not None) == (job in LINE_JOBS), 'missing destination completion')
            need(foreign == (1 if job == 15 else 0), 'foreign response coverage')
            if done_cycle is not None:
                need(cycle > done_cycle, 'result before destination')
                measurements.append(dict(job=job,line_cycle=line_cycle,destination_cycle=done_cycle,result_cycle=cycle))
            result = True
            counts['results'] += 1
        elif kind == 'PASS':
            passes = {k: int(v) for k, v in f.items()}
        else:
            raise ValueError('unknown trace event '+kind)
    need(job == 19 and result and address is None and not beats, 'terminal state')
    need(counts == dict(aw=161, w=423, b=161, lines=7, results=19), 'exact transaction counts')
    need(passes['jobs'] == 19 and passes['seed'] == seed, 'summary fixture identity')
    need(all(passes[k] == v for k, v in counts.items()), 'summary/event disagreement')
    need(all(passes[k] > 0 for k in ['aw_stall','w_stall','b_stall','line_stall','result_stall']), 'backpressure coverage')
    finish = re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns|us)\s*:', text, re.M)
    need(len(finish) == 1, 'one simulator finish')
    final_ps = int(finish[0][0])*{'ps':1,'ns':1000,'us':1000000}[finish[0][1]]
    need(final_ps == (200+passes['cycles']*5)*1000, 'observed cycle/time accounting')
    return dict(status='PASS_LINE_INGRESS_TRACE',seed=seed,counts=counts,accepted_byte_writes=byte_writes,
                published_bytes=448,max_frame_beats=max_frame_beats,summary=passes,
                final_ps=final_ps,measurements=measurements)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    print(json.dumps(review(args.log.read_text(), args.seed), indent=2))
