"""Replay transport ownership, actual bytes and externally observed completion."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pattern(tag, beat):
    return sum((((tag*0x9e3779b9) ^ (beat*0x10203) ^ (lane*0x01010101)) & 0xffffffff)
               << (32*lane) for lane in range(16))


def workload(case):
    if case == 1:
        return [(j % 2, j+1, (1, 8, 64)[j % 3]) for j in range(24)]
    if case == 6:
        return [(0, 500+j, 64) for j in range(8)] + [(1, 520+j, 64) for j in range(8)]
    if case == 7:
        return [(0, 600+j, 1) for j in range(28)] + [(1, 630+j, 8) for j in range(8)]
    return {
        2: [(0, 100, 64)], 3: [(0, 200, 1), (1, 201, 8)],
        4: [(0, 300, 64), (1, 301, 8)],
        5: [(0, 400, 1), (1, 401, 64), (0, 402, 8)],
        8: [(0, 700, 1), (0, 701, 1)], 9: [(0, 750, 8)],
        10: [(0, 800, 8)], 11: [(0, 850, 1)],
        12: [(0, 900, 64)], 13: [(0, 950, 8)],
    }[case]


def replay(lines):
    cases = []
    jobs = {}
    case = 0
    suite = False
    samples = total_bytes = 0
    cycle = -1
    for number, line in enumerate(lines):
        fields = line.split()
        require(fields and fields[0] == 'TE', f'Bad event at {number}')
        kind, a = fields[1], fields[2:]
        if kind == 'CASE':
            require(case == len(cases), 'Previous case incomplete')
            case = int(a[1])
            require(case == len(cases)+1, 'Case sequence')
            jobs = {}
            ingress = [deque(), deque()]
            waiting = [deque(), deque()]
            reserved = [deque(), deque()]
            completions = [deque(), deque()]
            ordered = [[], []]
            seen = []
            aw_count = [0, 0]
            w_count = [0, 0]
            b_count = [0, 0]
            c_count = [0, 0]
            active = None
            budget = 0
            barrier = None
            fenced = []
            stopped = faulted = False
            sample_fault = False
            continue
        if kind == 'SUITE_PASS':
            require(case == 13 and len(cases) == 13 and a[0] == 'cases=13', 'Incomplete suite')
            suite = True
            continue
        require(case and not suite, 'Event outside a case')
        if kind == 'CASE_PASS':
            require(int(a[0]) == case and seen == workload(case), 'Workload changed or missing')
            require(list(map(int, a[1:])) == [len(ordered[0]), len(ordered[1]), *c_count], 'Incorrect case totals')
            quarantined = sorted(tag for tag, job in jobs.items() if not job['completed'])
            require(quarantined == {4: [301], 5: [402], 13: [950]}.get(case, []), 'Unexpected quarantine')
            require(all(not jobs[tag]['reserved'] for tag in quarantined), 'Committed work did not drain')
            require(barrier == (100 if case == 10 else None), 'Fence termination')
            require(fenced == {3: [33], 11: [110], 12: [120]}.get(case, []), 'Fence history')
            require(faulted == (case in (4, 5, 10, 11, 13)), 'Fault scenario missing')
            require(stopped == (case in (2, 6, 7, 9, 12)), 'Stop scenario missing')
            require(budget == 0 and active is None, 'Outstanding payload')
            cases.append(dict(case=case, submitted=len(jobs), completed=sum(c_count), quarantined=quarantined,
                              bytes=sum(j['out']*64 for j in jobs.values()), fenced=fenced, pending_fence=barrier))
            continue
        if kind == 'DRAIN':
            require(all(j['completed'] for j in jobs.values()) and barrier is None and budget == 0, 'False drain marker')
            continue
        if kind == 'S':
            values = [int(a[i], 16 if i in (5, 6) else 2 if i in (2, 3, 4, 10, 11, 13, 14) else 10)
                      for i in range(15)]
            cyc, sc, stop, fault, drained, queued, banks, state, ing0, ing1, empty, core_drain, bgt, ready, done = values
            require(cyc > cycle and sc == case, 'Sample cycle/scenario')
            cycle = cyc
            samples += 1
            sample_fault = bool(fault)
            require(not faulted or fault, 'Fault lost without reset')
            faulted |= bool(fault)
            stopped |= bool(stop)
            require(queued == sum((len(ingress[ch])+len(waiting[ch])) << (4*ch) for ch in range(2)), 'Queued ownership')
            require([ing0, ing1] == list(map(len, ingress)), 'Ingress ownership')
            require(empty == sum((not completions[ch]) << ch for ch in range(2)), 'Completion buffer ownership')
            require(bgt == budget, 'Payload budget')
            if stopped or fault or barrier is not None:
                require(ready == 0, 'Admission after stop/fault/fence')
            if done:
                require(barrier is not None and all(j['completed'] for j in jobs.values()), 'Premature fence result')
            if drained:
                require(all(j['completed'] for j in jobs.values()) and barrier is None and budget == 0
                        and not banks and core_drain, 'Premature drained')
            continue
        require(int(a[0]) == cycle, 'Event without same-cycle sample')
        if kind == 'FENCE':
            require(barrier is None and not sample_fault, 'Invalid fence admission')
            barrier = int(a[1])
            continue
        if kind == 'FENCED':
            require(barrier == int(a[1]) and all(j['completed'] for j in jobs.values()), 'Fence overtook work')
            fenced.append(barrier)
            barrier = None
            continue
        if kind == 'IN':
            tag, beat, data = int(a[1]), int(a[2]), int(a[3], 16)
            require(active == tag and budget > 0, 'Unsolicited input')
            require(beat == jobs[tag]['in'] and data == pattern(tag, beat), 'Input data/sequence')
            jobs[tag]['in'] += 1
            budget -= 1
            if budget == 0:
                active = None
            continue
        ch, tag = int(a[1]), int(a[2])
        require(ch in (0, 1), 'Bad lane')
        if kind == 'SUB':
            desc = int(a[3], 16)
            beats = (desc & 255)+1
            expected = (0x76543210 << 200) | (tag << 136) | ((0x100000000+tag*4096) << 72) | ((0x800000000+tag*4096) << 8) | (beats-1)
            require(tag not in jobs and desc == expected and not stopped and not sample_fault and barrier is None, 'Invalid publication')
            jobs[tag] = dict(channel=ch, desc=desc, beats=beats, reserved=False, fetched=False,
                             aw=False, status=None, completed=False, **{'in': 0, 'out': 0})
            seen.append((ch, tag, beats))
            ordered[ch].append(tag)
            ingress[ch].append(tag)
        else:
            require(tag in jobs and jobs[tag]['channel'] == ch, 'Unknown tag/lane')
            job = jobs[tag]
            if kind == 'ADMIT':
                require(ingress[ch] and ingress[ch].popleft() == tag and not sample_fault, 'Ingress FIFO admission')
                waiting[ch].append(tag)
            elif kind == 'RES':
                require(waiting[ch] and waiting[ch].popleft() == tag and not sample_fault and not job['reserved'], 'Reservation ownership')
                job['reserved'] = True
                reserved[ch].append(tag)
            elif kind == 'FETCH':
                require(reserved[ch] and reserved[ch].popleft() == tag and not job['fetched'], 'Fetch reservation')
                require(int(a[3], 16) == job['desc'] and active is None and budget == 0, 'Fetch metadata/budget')
                job['fetched'] = True
                active, budget = tag, job['beats']
            elif kind == 'AW':
                require(ordered[ch][aw_count[ch]] == tag and job['fetched'] and not job['aw'], 'Address ownership/order')
                require(int(a[3], 16) == 0x100000000+tag*4096 and int(a[4]) == job['beats'], 'Address metadata')
                aw_count[ch] += 1
                job['aw'] = True
            elif kind == 'W':
                beat, last, data = int(a[3]), int(a[4]), int(a[5], 16)
                require(ordered[ch][w_count[ch]] == tag and job['fetched'] and beat == job['out'] < job['in'], 'Write ownership/sequence')
                require(last == (beat == job['beats']-1) and data == pattern(tag, beat), 'Write bytes/WLAST')
                job['out'] += 1
                total_bytes += 64
                w_count[ch] += last
            elif kind == 'B':
                require(ordered[ch][b_count[ch]] == tag and job['aw'] and job['out'] == job['beats'] and job['status'] is None, 'Response causality/order')
                job['status'] = int(a[3])
                require(job['status'] == (2 if tag == 400 else 0), 'Response status')
                b_count[ch] += 1
            elif kind == 'CORE_CPL':
                require(job['status'] is not None and tag not in completions[ch] and not job['completed'], 'Buffered completion causality')
                completions[ch].append(tag)
            elif kind == 'CPL':
                require(completions[ch] and completions[ch].popleft() == tag and ordered[ch][c_count[ch]] == tag, 'External completion ownership/order')
                require(job['status'] == int(a[3]) and not job['completed'], 'Completion status/duplicate')
                job['completed'] = True
                c_count[ch] += 1
            else:
                raise ValueError('Unknown event: '+kind)
    require(suite, 'Missing suite pass')
    return dict(cases=cases, samples=samples, bytes=total_bytes,
                submitted=sum(c['submitted'] for c in cases), completed=sum(c['completed'] for c in cases),
                quarantined=sum(len(c['quarantined']) for c in cases))


def negative_controls(lines):
    mutations = []
    for event, index, replacement in [('W', -1, '0'), ('AW', 5, '0'), ('CPL', 5, '3'), ('FETCH', 4, '999')]:
        at = next(i for i, s in enumerate(lines) if s.startswith('TE '+event+' '))
        changed = lines.copy()
        f = changed[at].split();f[index] = replacement;changed[at] = ' '.join(f)
        mutations.append((event+'_corruption', changed))
    for event in ('IN', 'ADMIT', 'RES', 'CORE_CPL', 'CASE_PASS'):
        at = next(i for i, s in enumerate(lines) if s.startswith('TE '+event+' '))
        mutations.append(('missing_'+event, lines[:at]+lines[at+1:]))
    at = next(i for i, s in enumerate(lines) if s.startswith('TE SUB '))
    mutations.append(('duplicate_SUB', lines[:at]+[lines[at]]+lines[at:]))
    next_sample = next(i for i in range(at+1, len(lines)) if lines[i].startswith('TE S '))
    for name, index, replacement in [('false_drain', 6, '1'), ('queued_loss', 7, 'ff')]:
        changed = lines.copy();f = changed[next_sample].split();f[index] = replacement
        changed[next_sample] = ' '.join(f);mutations.append((name, changed))
    at = next(i for i, s in enumerate(lines) if s.startswith('TE FENCE '))
    next_sample = next(i for i in range(at+1, len(lines)) if lines[i].startswith('TE S '))
    changed = lines.copy();f = changed[next_sample].split();f[-1] = '1';changed[next_sample] = ' '.join(f)
    mutations.append(('premature_fence', changed))
    rejected = []
    for name, changed in mutations:
        try:
            replay(changed)
        except (ValueError, IndexError, KeyError):
            rejected.append(name)
        else:
            raise ValueError('Mutation not detected: '+name)
    return rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path, help='Runner output directory')
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--synthesis', type=Path, help='Prior output with identical RTL, for simulation-only runs')
    args = ap.parse_args()
    result = json.loads((args.output/'result.json').read_text())
    require(result['status'] == 'EXPORTED_PENDING_REVIEW' and result['sources'] == result['sources_after'], 'Run/provenance failed')
    files = json.loads((args.output/'FILES.json').read_text())
    require(all(hashlib.sha256((args.output/n).read_bytes()).hexdigest() == h for n, h in files.items()), 'Evidence hash mismatch')
    log = (args.output/'suite/simulation.log').read_text()
    require(not re.search(r'^\s*(?:Fatal:|ERROR:|FAIL )', log, re.M), 'Simulation error')
    lines = [s for s in log.splitlines() if s.startswith('TE ')]
    review = replay(lines)
    review['negative_controls'] = negative_controls(lines)
    synthesis = args.synthesis or args.output
    syn_result = json.loads((synthesis/'result.json').read_text())
    require(syn_result['status'] == 'EXPORTED_PENDING_REVIEW' and syn_result['sources'] == syn_result['sources_after']
            and not syn_result.get('simulation_only'), 'Synthesis provenance failed')
    logic = [n for n in result['sources'] if n.startswith('fpga/rtl/') or n.startswith('tools/prepare_')]
    require(all(result['sources'][n] == syn_result['sources'][n] for n in logic), 'Synthesis logic differs')
    syn_files = json.loads((synthesis/'FILES.json').read_text())
    require(all(hashlib.sha256((synthesis/n).read_bytes()).hexdigest() == h for n, h in syn_files.items()), 'Synthesis evidence hashes')
    structure = (synthesis/'suite/structure.txt').read_text()
    require(structure == 'LATCHES 0\nBLACKBOXES 0\n', 'Synthesis structural failure')
    require('NTRPU_CANDIDATE_TRANSPORT_COMPLETE' in (synthesis/'console.log').read_text(), 'Incomplete synthesis')
    review.update(status='TRANSPORT_RTL_AND_SYNTH_PASS', seconds=result['seconds'], synthesis_structure=structure,
                  post_route_timing=False, real_pcie_hbm=False,
                  synthesis_output=str(synthesis), synthesis_sources=syn_result['sources'],
                  simulation_sha256=hashlib.sha256(log.encode()).hexdigest())
    args.review.write_text(json.dumps(review, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: review[k] for k in ('status', 'samples', 'bytes', 'submitted', 'completed', 'quarantined', 'seconds')}))


if __name__ == '__main__':
    main()
