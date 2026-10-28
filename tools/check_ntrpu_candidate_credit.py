"""Review live-credit withdrawal and destination reselection trace."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_registered_dispatch import need


def review(text):
    need('CU SUITE_PASS cases=2' in text and not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL )', text), 'complete clean credit test')
    commits = {1: [], 2: []}
    withdrawn = {1: 0, 2: 0}
    paused = 0
    previous = None
    for line in text.splitlines():
        if not line.startswith('CU S '):
            continue
        f = line.split()
        now, case = int(f[2]), int(f[3])
        can, valid, ch, take = (int(x, 2) for x in f[4:8])
        queued, desc = (int(x, 16) for x in f[8:10])
        need(case in commits and take in (0, 1, 2), 'one valid destination')
        if previous is not None and previous['case'] == case and previous['started']:
            want = previous['queued'] - (1 if previous['take'] == 1 else 16 if previous['take'] == 2 else 0)
            need(queued == want, 'no FIFO pop without atomic acceptance')
        started = bool(can or valid or (previous and previous['case'] == case and previous['started']))
        if valid:
            tag = 10 + ch
            expected = (0x12345678 << 200) | (tag << 136) | ((0x100000000+tag*4096) << 72) | (0x800000000 << 8) | 7
            need(desc == expected, 'candidate retains exact descriptor')
            if not (can >> ch & 1):
                need(take == 0, 'withdrawn offer cannot commit')
                withdrawn[case] += 1
        if take:
            need(valid and take == 1 << ch and can & take, 'live credit and prepared identity required')
            commits[case].append(ch)
        if case == 2 and started and not can:
            need(take == 0 and queued == 0x11, 'pause keeps both entries')
            paused += 1
        previous = dict(case=case, started=started, queued=queued, take=take)
    need(commits == {1: [1, 0], 2: [0, 1]}, 'peer reselection and round-robin order')
    need(withdrawn[1] >= 1 and withdrawn[2] >= 1 and paused >= 8, 'withdrawal cases covered')
    return dict(commits=commits, withdrawn_candidates=withdrawn, fully_paused_cycles=paused)


def negatives(text):
    lines = text.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith('CU S ') and int(line.split()[7], 2))
    results = []
    for name in ('stale_credit_commit', 'wrong_destination', 'changed_candidate'):
        changed = list(lines)
        f = changed[i].split()
        if name == 'stale_credit_commit':
            f[4] = '00'
        elif name == 'wrong_destination':
            f[7] = format(int(f[7], 2) ^ 3, '02b')
        else:
            f[9] = format(int(f[9], 16) ^ 256, '058x')
        changed[i] = ' '.join(f)
        try:
            review('\n'.join(changed))
        except ValueError as exc:
            results.append(dict(mutation=name, rejection=str(exc)))
        else:
            raise ValueError('Missed credit negative ' + name)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    folder = ap.parse_args().run
    record = json.loads((folder/'output/result.json').read_text())
    need(record['status'] == 'EXPORTED_PENDING_REVIEW' and record['returncode'] == 0 and not record['timed_out'], 'runner completion')
    need(record['sources'] == record['sources_after'], 'sources preserved')
    text = (folder/'output/suite/credit.log').read_text()
    result = dict(status='CREDIT_WITHDRAWAL_CONTRACT_PASS', **review(text), negative_controls=negatives(text),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (folder/'REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
