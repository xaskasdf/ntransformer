"""Independently replay candidate preparation, atomic allocation, and payload."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import check_ntrpu_registered_dispatch as reference

need = reference.need


def samples(text):
    current = None
    for line in text.splitlines():
        f = line.split()
        if line.startswith('RS S '):
            if current is not None:
                yield current
            current = dict(cycle=int(f[2]), case=int(f[3]), fault=int(f[6], 2),
                           banks=int(f[8], 16), queued=int(f[9], 16), can=int(f[10], 2),
                           take=int(f[11], 2), lease=int(f[13], 2))
        elif line.startswith('CP CAND '):
            need(current is not None and current['cycle'] == int(f[2]) and 'candidate' not in current,
                 'one candidate sample per cycle')
            current.update(candidate=int(f[3], 2), channel=int(f[4], 2), desc=int(f[5], 16),
                           state=int(f[6]), preferred=int(f[7], 2), eligible=int(f[8], 2))
        elif line.startswith('RH HEAD '):
            need(current is not None and current['cycle'] == int(f[2]) and 'heads' not in current,
                 'one head sample per cycle')
            current.update(heads=int(f[3], 2), good=int(f[4], 2), words=[int(f[5], 16), int(f[6], 16)])
    if current is not None:
        yield current


def candidates(text):
    prior = None
    count = prepared = committed = cancellations = 0
    for s in samples(text):
        need('candidate' in s and 'heads' in s, 'complete candidate instrumentation')
        if prior is None or prior['case'] != s['case']:
            prior = None
            need(s['candidate'] == 0, 'reset starts without candidate')
        need(s['eligible'] == s['heads'] & s['can'], 'live eligible offers')
        want_take = (1 << s['channel']) if (s['candidate'] and s['state'] == 0 and not s['fault']
                     and s['eligible'] >> s['channel'] & 1) else 0
        need(s['take'] == want_take, 'commit requires prepared candidate and current offer')
        if prior is not None:
            prepare = prior['state'] == 0 and not prior['fault'] and not prior['take'] and prior['eligible'] != 0
            need(bool(s['candidate']) == prepare, 'candidate appears only after preparation edge')
            if prepare:
                ch = prior['preferred'] if (prior['eligible'] >> prior['preferred']) & 1 else 1-prior['preferred']
                need(s['channel'] == ch and s['desc'] == prior['words'][ch], 'prepared metadata and arbitration')
                prepared += 1
            else:
                need(s['channel'] == prior['channel'] and s['desc'] == prior['desc'], 'inactive metadata retained')
        if s['candidate']:
            need(s['heads'] >> s['channel'] & 1 and s['desc'] == s['words'][s['channel']], 'candidate still owns FIFO head only')
        if s['take']:
            need(not s['lease'], 'no allocation behind a live payload lease')
            committed += 1
        if prior and prior['candidate'] and prior['fault'] and not s['candidate']:
            cancellations += 1
        prior = s
        count += 1
    need(count > 0, 'candidate trace present')
    return dict(samples=count, prepared=prepared, committed=committed, fault_cancellations=cancellations)


def candidate_negatives(text):
    lines = text.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith('CP CAND ') and line.split()[3] == '1')
    output = []
    for name in ('candidate_data_corruption', 'candidate_withdrawn', 'candidate_wrong_channel'):
        changed = list(lines)
        fields = changed[i].split()
        if name == 'candidate_data_corruption':
            fields[5] = format(int(fields[5], 16) ^ 256, '058x')
        elif name == 'candidate_withdrawn':
            fields[3] = '0'
        else:
            fields[4] = str(1-int(fields[4]))
        changed[i] = ' '.join(fields)
        try:
            candidates('\n'.join(changed))
        except ValueError as exc:
            output.append(dict(mutation=name, rejection=str(exc)))
        else:
            raise ValueError('Missed negative control ' + name)
    return output


def extra(text):
    need('CP EXTRA_PASS cases=2 ' in text, 'extra scenarios complete')
    review = candidates(text)
    need(review['committed'] == 1 and review['fault_cancellations'] == 1, 'one cancelled candidate and one committed')
    events = [line.split() for line in text.splitlines() if line.startswith('RS ')]
    by_kind = lambda kind: [f for f in events if f[1] == kind]
    need([(int(f[3]), int(f[4])) for f in by_kind('SUB')] == [(0, 500), (1, 501)], 'extra workload')
    need([(int(f[3]), int(f[4])) for f in by_kind('RES')] == [(1, 501)], 'fault before commit consumes no bank')
    for kind in ('ECHO', 'FETCH', 'AW', 'B', 'CPL'):
        need(len(by_kind(kind)) == 1 and int(by_kind(kind)[0][4]) == 501, 'extra exactly one ' + kind)
    ins, outs = by_kind('IN'), by_kind('W')
    need(len(ins) == len(outs) == 64, 'extra complete payload')
    for beat, (i, o) in enumerate(zip(ins, outs)):
        need(int(i[3]) == int(o[4]) == 501 and int(i[4]) == int(o[5]) == beat and
             int(i[5], 16) == int(o[7], 16) == reference.pattern(501, beat), 'extra exact data and order')
    case9 = [s for s in samples(text) if s['case'] == 9]
    need(all(s['banks'] == 0 and s['lease'] == 0 and s['take'] == 0 for s in case9), 'no allocation in fault case')
    need(case9[-1]['fault'] and case9[-1]['queued'] == 1, 'unreserved entry quarantined')
    return dict(**review, bytes=4096, completed=1, quarantined=[500])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    folder = args.run
    record = json.loads((folder/'output/result.json').read_text())
    need(record['status'] == 'EXPORTED_PENDING_REVIEW' and not record['timed_out'] and record['returncode'] == 0,
         'completed evaluation')
    need(record['sources'] == record['sources_after'], 'sources unchanged')
    text = (folder/'output/suite/simulation.log').read_text()
    extra_text = (folder/'output/suite/extra.log').read_text()
    need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL )', text+'\n'+extra_text), 'simulation diagnostics clean')
    result = reference.replay(text)
    result.update(registered_heads=reference.check_heads(text), candidates=candidates(text),
                  extra=extra(extra_text), negative_controls=reference.negatives(text)+reference.head_negatives(text)+candidate_negatives(text))
    need(result['candidates']['committed'] == 85, 'all prior allocations retained')
    result.update(status='CANDIDATE_PROTOCOL_PASS_PHYSICAL_REVIEW_SEPARATE',
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_checker_sha256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest())
    with (folder/'REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
