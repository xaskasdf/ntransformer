"""Independent payload ledger and diagnostic checks for the retained-channel bank."""
import collections
import json
import re
import sys
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(line):
    result = {}
    for key, value in re.findall(r'(\w+)=(\S+)', line):
        result[key] = int(value, 16 if key in {'sd', 'md'} else 10)
    return result


def review(text):
    require(not re.search(r'^(Fatal:|ERROR:|FAIL )', text, re.M), 'failed simulation')
    require(text.splitlines().count('FAULT_SUITE_PASS cases=7') == 1, 'suite marker')
    lines = [line for line in text.splitlines() if line.startswith('FAULT_')]
    pairs, summaries = collections.defaultdict(list), {}
    awaiting = None
    for line in lines:
        if line.startswith('FAULT_PRE '):
            require(awaiting is None, 'missing post observation')
            awaiting = fields(line)
        elif line.startswith('FAULT_POST '):
            post = fields(line)
            require(awaiting is not None, 'missing pre observation')
            require(post['case'] == awaiting['case'] and post['dir'] == awaiting['dir'] and
                    post['t'] == awaiting['t'] + 1, 'paired cycle identity')
            pairs[post['case']].append((awaiting, post))
            awaiting = None
        elif line.startswith('FAULT_CASE_PASS '):
            f = fields(line)
            require(f['case'] not in summaries, 'duplicate case')
            summaries[f['case']] = f
    require(awaiting is None and set(pairs) == set(summaries) == set(range(7)), 'complete cases')
    # Physical event counts, independently fixed by the test fixture. Saturated
    # hardware counters do not retain conservation or exact totals after fault.
    expected = [(2, 2, 0, 1), (2, 0, 2, 1), (4, 4, 0, 2),
                (4, 4, 0, 2), (4, 0, 4, 2), (3, 0, 3, 0), (1, 1, 0, 0)]
    result = {}
    for case, cycles in pairs.items():
        direction = int(case not in {0, 2})
        queue, captures, receipts, retired = collections.deque(), [], [], []
        seen_up = sticky = fence = False
        faults = 0
        previous_t = None
        cold_cycles = initial_low = live_loss = blocked = 0
        overflow_causes = set()
        for pre, post in cycles:
            require(pre['dir'] == direction, 'fixture direction')
            if previous_t is not None:
                require(pre['t'] - previous_t == 5000, 'missing source cycle')
            previous_t = pre['t']
            if not pre['cold']:
                cold_cycles += 1
                require(not pre['sv'], 'cold reset must not inject traffic')
                queue.clear(); captures.clear(); receipts.clear(); retired.clear()
                seen_up = sticky = fence = False
                faults = 0
            else:
                if not seen_up and not pre['bound']:
                    initial_low += 1
                require(pre['invalid'] == int(sticky or bool(pre['loss'])), 'causal invalidation')
                require(pre['fence'] == int(fence) and pre['fault'] == faults, 'retained fault state')
                require(pre['pending'] == len(queue), 'pre inventory')
                for key, ledger in [('accepted', captures), ('delivered', receipts), ('abandoned', retired)]:
                    require(pre[key] == min(3, len(ledger)), 'pre counter ' + key)
                visible = bool(queue) and (not direction or (pre['bound'] and not fence))
                require(pre['mv'] == int(visible), 'visible retained offer')
                if queue:
                    require(pre['md'] == queue[0], 'ordered payload remains stable')
                if not direction and (pre['invalid'] or not pre['bound']):
                    require(pre['sr'] == 0, 'request admission after invalidation/reset')
                    blocked += pre['sv']
                discard = direction and (fence or (pre['invalid'] and not pre['bound']))
                if direction and pre['invalid'] and pre['bound'] and queue and not fence:
                    live_loss += 1
                pushed = bool(pre['sv'] and pre['sr'])
                popped = bool(pre['mv'] and pre['mr'])
                if popped:
                    require(not discard and bool(queue), 'receipt requires a retained item')
                    receipts.append(queue.popleft())
                if pushed:
                    require(discard or len(queue) < 2, 'bounded capacity')
                    captures.append(pre['sd'])
                    queue.append(pre['sd'])
                if discard:
                    retired.extend(queue)
                    queue.clear()
                    fence = True
                if seen_up and not pre['bound'] and not pre['invalid']:
                    faults |= 1
                for key, ledger in [('accepted', captures), ('delivered', receipts), ('abandoned', retired)]:
                    if len(ledger) > 3:
                        faults |= 2
                        overflow_causes.add(key)
                seen_up |= bool(pre['bound'])
                sticky |= bool(pre['invalid'] or faults)
            require(post['pending'] == len(queue), 'post inventory')
            for key, ledger in [('accepted', captures), ('delivered', receipts), ('abandoned', retired)]:
                require(post[key] == min(3, len(ledger)), 'counter saturation ' + key)
            require(post['fault'] == faults and post['invalid'] == int(sticky), 'diagnostic causes/lifetime')
            require(post['fence'] == int(fence), 'response reset fence')
            if queue:
                require(post['md'] == queue[0], 'post ordered payload')
        a, d, b, f = expected[case]
        require((len(captures), len(receipts), len(retired), faults) == (a, d, b, f), 'fixture event totals')
        require(not queue and cold_cycles >= 3 and initial_low >= 3, 'fixture lifecycle')
        require(collections.Counter(captures) == collections.Counter(receipts + retired), 'all captured identities accounted')
        fixture = ([0x12345, 0x0abcd] if case in {0, 1} else
                   list(range(0x10010, 0x10014)) if case in {2, 3} else
                   list(range(0x10100, 0x10104)) if case == 4 else
                   [0x12345, 0x0abcd, 0x055aa] if case == 5 else [0x12345])
        require(captures == fixture, 'fixture identities')
        summary = summaries[case]
        require([summary[k] for k in ['accepted', 'delivered', 'abandoned', 'fault']] ==
                [min(a, 3), min(d, 3), min(b, 3), f], 'summary evidence')
        if case in {0, 2}:
            require(blocked >= 3, 'post-fault request block exercised')
        if case == 6:
            require(live_loss >= 8, 'held response through live loss exercised')
        result[case] = dict(samples=len(cycles), captures=a, output_handshakes=d,
                            local_retirements=b, faults=f, live_loss_samples=live_loss,
                            overflow_causes=sorted(overflow_causes))
    return dict(cases=result, captures=sum(x[0] for x in expected),
                output_handshakes=sum(x[1] for x in expected), local_retirements=sum(x[2] for x in expected))


if __name__ == '__main__':
    print(json.dumps(review(Path(sys.argv[1]).read_text(encoding='utf-8')), indent=2))
