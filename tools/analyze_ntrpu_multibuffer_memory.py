"""Audit memory-suite warnings, trace sensitivity and functional bulk rates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from check_ntrpu_multibuffer_memory import replay, require, review


def corruptions(text):
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('MEM W '))
    changed = lines.copy()
    fields = changed[index].split()
    fields[-1] = format(int(fields[-1], 16) ^ 1, '0128x')
    changed[index] = ' '.join(fields)
    yield 'payload_bit', '\n'.join(changed), 'payload mismatch'

    index = next(i for i, line in enumerate(lines) if line.startswith('MEM AW '))
    changed = lines.copy()
    fields = changed[index].split()
    fields[5] = format(int(fields[5], 16) & 0xffffffff, '010x')
    changed[index] = ' '.join(fields)
    yield 'address_truncated_to_32_bits', '\n'.join(changed), 'AW alias/length'

    index = next(i for i, line in enumerate(lines) if line.startswith('MEM B '))
    yield 'missing_memory_response', '\n'.join(lines[:index] + lines[index + 1:]), 'premature completion'

    index = next(i for i, line in enumerate(lines) if line.startswith('MEM CPL '))
    changed = lines.copy()
    fields = changed[index].split()
    fields[-1] = '2'
    changed[index] = ' '.join(fields)
    yield 'completion_status', '\n'.join(changed), 'completion status/input'
    yield 'duplicate_completion', '\n'.join(lines[:index] + [lines[index]] + lines[index:]), 'completion order'

    index = max(i for i, line in enumerate(lines) if line.startswith('MEM RESET '))
    tag = next(line.split()[4] for line in reversed(lines[:index]) if line.startswith('MEM CMD '))
    stale = 'MEM CPL ' + lines[index].split()[2] + ' 0 ' + tag + ' 0'
    yield 'stale_completion_after_reset', '\n'.join(lines[:index + 1] + [stale] + lines[index + 1:]), 'completion order'


def bulk(text):
    active = False
    jobs = {}
    last_aw = [None, None]
    last_b = [None, None]
    max_pending = [0, 0]
    pending = [0, 0]
    for line in text.splitlines():
        if line == 'MEM CASE bulk':
            active = True
            continue
        if active and line.startswith('MEM RATE '):
            break
        if not active or not line.startswith('MEM '):
            continue
        fields = line.split()
        kind = fields[1]
        if kind not in ('AW', 'W', 'B'):
            continue
        now, port, tag = map(int, fields[2:5])
        job = jobs.setdefault(tag, {'port': port})
        if kind == 'AW':
            if last_aw[port] is not None:
                require(last_b[port] is not None and last_b[port] < now, 'guard serialization changed')
            job['aw'] = now
            last_aw[port] = now
            last_b[port] = None
            pending[port] += 1
            max_pending[port] = max(max_pending[port], pending[port])
        elif kind == 'W':
            job.setdefault('first_w', now)
            if fields[6] == '1':
                job['last_w'] = now
        else:
            job['b'] = now
            last_b[port] = now
            pending[port] -= 1
    require(len(jobs) == 32 and pending == [0, 0], 'bulk accounting')
    ports = []
    for port in range(2):
        selected = sorted((j for j in jobs.values() if j['port'] == port), key=lambda j: j['aw'])
        require(len(selected) == 16 and max_pending[port] == 1, 'bulk outstanding coverage')
        ports.append(dict(port=port, memory_aw_pending_peak=max_pending[port],
                          mean_last_w_to_b_ns=mean(j['b'] - j['last_w'] for j in selected) / 1000,
                          mean_last_w_to_next_first_w_ns=mean(b['first_w'] - a['last_w']
                                                             for a, b in zip(selected, selected[1:])) / 1000))
    return ports


def analyze(root):
    checked = review(root)
    tests = []
    for profile in range(3):
        text = (root / f'output/suite/profile-{profile}-negative-0.log').read_text()
        replay(text, profile)
        for name, corrupted, reason in corruptions(text):
            try:
                replay(corrupted, profile)
            except RuntimeError as error:
                require(str(error) == reason, 'unexpected rejection: ' + str(error))
                tests.append(dict(profile=profile, mutation=name, rejected=str(error)))
            else:
                raise RuntimeError('Trace corruption escaped: ' + name)
        checked['profiles'][profile]['bulk']['ports'] = bulk(text)
    console = (root / 'output/console.log').read_text()
    warnings = [line for line in console.splitlines() if line.startswith('WARNING:')]
    fifo_warning = 'Behavioral models for independent clock FIFO configurations do not model synchronization delays.'
    unknown = [line for line in warnings if not any(reason in line for reason in (
        '[IP_Flow 19-4994]', '[XSIM 43-3431]', fifo_warning))]
    require(not unknown, 'unreviewed warnings: ' + repr(unknown))
    checked.update(trace_negative_controls=tests,
                   warning_review=dict(total=len(warnings),
                       independent_fifo_model=sum(fifo_warning in line for line in warnings),
                       meanings={'IP_Flow 19-4994': 'Regenerates the CDC OOC XDC inside this fresh project only',
                                 'XSIM 43-3431': 'Vivado environment LIBRARY_PATH affects the compiler',
                                 'independent_fifo_model': 'Functional model omits physical synchronization delay; rates are not hardware estimates'}),
                   rate_limit='Functional model and behavioral target only; shared 512-bit 200 MHz input caps raw ingress at 12.8 GB/s',
                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    with (args.run / 'ANALYSIS.json').open('x') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result, indent=2))
