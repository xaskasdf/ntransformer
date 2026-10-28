"""Check every proposal, retained candidate event and unchanged public R7 row.

Expected metadata/reference events are observations of the unchanged R7 DUT,
not a separate protocol model. Invalid differences are returned explicitly.
"""
import re
from check_ntrpu_pcie_rc_proposal import calculate
from check_ntrpu_pcie_request_inventory_r5 import review as protocol_review


def parse(line, prefix, keys):
    m = re.fullmatch(prefix + r' index=(\d+) ' +
                     ' '.join(k + r'=([0-9a-fA-FxXzZ]+)' for k in keys), line)
    if not m:
        raise ValueError('Malformed row: ' + line[:150])
    return int(m[1]), dict(zip(keys, m.groups()[1:]))


def bits(word, start, width):
    padded = word.lower().zfill(512)
    return padded[len(padded)-start-width:len(padded)-start] if start else padded[-width:]


def hex_bits(word):
    values = []
    for i in range(0, len(word), 4):
        nibble = word[i:i+4]
        chars = set(nibble)
        if chars <= {'0','1'}:
            values.append(format(int(nibble,2),'x'))
        elif chars == {'x'}:
            values.append('x')
        elif chars == {'z'}:
            values.append('z')
        else:
            values.append('X' if 'x' in chars else 'Z')
    return ''.join(values)


def field(word, start, width):
    return int(bits(word, start, width), 2)


def review(log, vectors, manifest, baseline_log):
    protocol = protocol_review(log, vectors, manifest)
    if log.splitlines().count('RC_PROP_SHADOW_SUITE_PASS rows=22330 scenarios=67') != 1:
        raise ValueError('Incomplete shadow suite')
    inputs = vectors.splitlines()
    public = [s for s in log.splitlines() if s.startswith('PCIE_INV_ROW ')]
    baseline = [s for s in baseline_log.splitlines() if s.startswith('PCIE_INV_ROW ')]
    if public != baseline:
        raise ValueError('R7 public outputs differ, including invalid traffic')
    rows = [s for s in log.splitlines() if s.startswith('RC_PROP_ROW ')]
    contexts = [s for s in log.splitlines() if s.startswith('RC_PROP_CONTEXT ')]
    events = [s for s in log.splitlines() if s.startswith('RC_PROP_EVENTS ')]
    if any(len(x) != len(inputs) for x in [rows, contexts, events]) or len(inputs) != 22330:
        raise ValueError('Missing/duplicate shadow rows')
    event_keys = ['valid','complete','tags','remaining','address','removed']
    proposal_keys = ['scenario','seen','valid','headers','delivered','remaining','address','length','terminal']
    prior_observing = prior_bitmap = 0
    context = None
    differences, metadata_differences = [], []
    normal_headers = non_normal_headers = healthy_rows = faulty_rows = unresolved_rows = 0
    known_x_headers = four_header_cycles = partial_updates = completions = 0
    for i, (raw, row, ctxrow, evrow, pub) in enumerate(zip(inputs, rows, contexts, events, public)):
        v = raw.split()
        scenario, rst = int(v[0]), int(v[2])
        fault = int(re.search(r' fault=([0-9a-f]+)', pub)[1], 16)
        observing = int(re.search(r' observing=(\d+)', pub)[1])
        bitmap = int(re.search(r' bitmap=([0-9a-f]+)', pub)[1], 16)
        ri, r = parse(row, 'RC_PROP_ROW', proposal_keys)
        ci, c = parse(ctxrow, 'RC_PROP_CONTEXT', ['owned','original','address'])
        ei, e = parse(evrow, 'RC_PROP_EVENTS', event_keys + ['unresolved'] + ['ref_'+k for k in event_keys])
        if (ri, ci, ei) != (i, i, i) or int(r['scenario']) != scenario:
            raise ValueError('Wrong shadow sequence')
        accepted = bool(rst and prior_observing and v[12] == '1' and v[13] == '1')
        u = int(v[15], 16)
        starts = {0:0,1:1,3:2,7:3,15:4}.get((u>>64)&15, -1)
        positions = {((u>>(68+2*s))&3) for s in range(max(starts,0))} if accepted else set()
        expected_seen = sum(1<<q for q in positions)
        expected = dict.fromkeys(['seen','valid','delivered','remaining','address','length','terminal'],0)
        expected['seen'] = expected_seen
        hs = ['0'*96 for _ in range(4)]
        proposals = {}
        owned = 0
        released = set()
        for q in sorted(positions):
            h = bits(v[14], q*128, 96)
            hs[q] = h
            tag = field(h,64,8)
            own = bool((prior_bitmap>>tag)&1) and tag not in released
            owned |= int(own)<<q
            if bits(h,30,1) == '1' and own:
                released.add(tag)
            normal = (bits(h,12,4)=='0000' and bits(h,43,3)=='000' and bits(h,46,1)=='0')
            if normal:
                # Unknown unused fields remain captured; arithmetic needs only these bits.
                compact = field(h,16,13)<<16 | field(h,0,12) | field(h,32,11)<<32 | field(h,30,1)<<30
                p = calculate(compact)
            else:
                p = dict(normal=0,delivered=0,remaining=0,address=0,length=0,terminal=0)
            proposals[q] = (h, p, own)
            normal_headers += int(normal); non_normal_headers += int(not normal)
            known_x_headers += int(bool(set(h)-{'0','1'}))
            expected['valid'] |= int(normal)<<q
            for k,w in [('delivered',13),('remaining',13),('address',12),('length',1),('terminal',1)]:
                expected[k] |= p[k]<<(w*q)
            if normal and own:
                original = (int(c['original'],16)>>(13*q))&8191
                address = (int(c['address'],16)>>(12*q))&4095
                if (field(h,16,13),field(h,0,12)) != (original,address):
                    if not fault:
                        raise ValueError(f'Healthy metadata mismatch at {i}:{q}')
                    metadata_differences.append(dict(row=i,scenario=scenario,lane=q,tag=tag,
                        fault=hex(fault),declared_bytes=field(h,16,13),expected_bytes=original,
                        declared_address=field(h,0,12),expected_address=address))
        if r['headers'] != ''.join(hex_bits(h) for h in reversed(hs)):
            raise ValueError(f'Captured header mismatch at {i}')
        if int(c['owned'],16) != owned:
            raise ValueError(f'Ownership prefix mismatch at {i}')
        for k,n in expected.items():
            if int(r[k],16) != n:
                raise ValueError(f'Arithmetic/capture mismatch at {i}: {k}')
        four_header_cycles += int(expected_seen == 15)
        candidate = dict.fromkeys(event_keys + ['unresolved'],0)
        if not rst:
            context = None
        elif accepted:
            if context is not None and (u>>96)&1:
                context['normal'] = False
            ends = {0:0,1:1,3:2,7:3,15:4}.get((u>>76)&15,-1)
            end_map = {((u>>(80+4*s))&15):s for s in range(max(ends,0))}
            for lane in range(16):
                if lane%4 == 0 and lane//4 in proposals:
                    h,p,own = proposals[lane//4]
                    context = dict(owned=own, tag=field(h,64,8), terminal=bits(h,30,1)=='1',
                        normal=bool(p['normal']),ready=bool(p['normal']),
                        original=field(h,16,13) if p['normal'] else None, **{k:p[k] for k in ['delivered','remaining','address']})
                if lane in end_map:
                    s = end_map[lane]
                    if context is not None and context['owned']:
                        if not context['ready']:
                            candidate['unresolved'] |= 1<<s
                        elif context['terminal']:
                            candidate['complete'] |= 1<<s
                            candidate['removed'] |= context['original']<<(13*s)
                        elif context['normal']:
                            candidate['valid'] |= 1<<s
                            candidate['tags'] |= context['tag']<<(8*s)
                            candidate['remaining'] |= context['remaining']<<(13*s)
                            candidate['address'] |= context['address']<<(12*s)
                            candidate['removed'] |= context['delivered']<<(13*s)
                    context = None
        for k,n in candidate.items():
            if int(e[k],16) != n:
                raise ValueError(f'Retained candidate event mismatch at {i}: {k}')
        delta = [k for k in event_keys if e[k] != e['ref_'+k]]
        if candidate['unresolved']:
            unresolved_rows += 1
        if delta or candidate['unresolved']:
            if not fault:
                raise ValueError(f'Healthy transition mismatch at {i}: {delta}')
            differences.append(dict(row=i,scenario=scenario,fault=hex(fault),fields=delta,
                unresolved=candidate['unresolved'],proposed={k:e[k] for k in event_keys},
                reference={k:e['ref_'+k] for k in event_keys}))
        partial_updates += candidate['valid'].bit_count()
        completions += candidate['complete'].bit_count()
        healthy_rows += int(not fault); faulty_rows += int(bool(fault))
        prior_observing, prior_bitmap = observing, bitmap
    return dict(status='REVIEWED_PCIE_RC_PROPOSAL_SHADOW_WITH_INVALID_DIFFERENCES',
        rows=len(rows),scenarios=67,protocol=protocol,normal_headers=normal_headers,
        non_normal_headers=non_normal_headers,headers_with_unknown_bits=known_x_headers,
        four_header_cycles=four_header_cycles,healthy_rows=healthy_rows,faulty_rows=faulty_rows,
        healthy_event_differences=0,all_r7_public_rows_identical=True,
        partial_updates=partial_updates,completions=completions,unresolved_rows=unresolved_rows,
        invalid_event_differences=differences,invalid_metadata_differences=metadata_differences,
        expected_metadata_source='Unchanged R7 table with earlier same-beat updates',
        candidate_context='Simulation-only; no integrated production parser',
        full_protocol_equivalence=False,active_pipeline_integrated=False)
