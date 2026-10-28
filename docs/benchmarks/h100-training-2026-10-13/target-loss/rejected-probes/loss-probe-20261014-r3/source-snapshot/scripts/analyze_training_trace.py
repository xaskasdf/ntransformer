#!/usr/bin/env python3
"""Summarize a Kineto training trace without confusing copy_ with host traffic.

CUDA duration sums overlap across streams. Activity union and its complement
are measured only within the profiled device activity envelope, not a utilization
counter or an estimate of CPU-removal speedup. Copy tensor traffic is logical,
not measured HBM traffic. Profiler timings are diagnostic, not benchmark rates.
"""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path


def union_duration(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def analyze(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        events = json.load(handle)['traceEvents']
    duration = [e for e in events if e.get('ph') == 'X']
    cpu = [e for e in duration if e.get('cat') == 'cpu_op']
    gpu = [e for e in duration if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    runtime = [e for e in duration if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    op_ids = {e.get('args', {}).get('External id'): e['name'] for e in cpu}
    by_device = {}
    for device in sorted(set(e.get('args', {}).get('device') for e in gpu)):
        rows = [e for e in gpu if e.get('args', {}).get('device') == device]
        first, last = min(e['ts'] for e in rows), max(e['ts'] + e['dur'] for e in rows)
        active = union_duration([(e['ts'], e['ts'] + e['dur']) for e in rows])
        by_device[str(device)] = dict(envelope_ms=(last - first) / 1000,
                                     active_union_ms=active / 1000,
                                     uncovered_ms=(last - first - active) / 1000,
                                     event_count=len(rows))
    transfers = {}
    for event in gpu:
        if event['cat'] == 'gpu_memcpy':
            row = transfers.setdefault(event['name'], dict(calls=0, bytes=0, duration_sum_ms=0))
            row['calls'] += 1
            row['bytes'] += event.get('args', {}).get('bytes', 0)
            row['duration_sum_ms'] += event['dur'] / 1000
    copies = collections.defaultdict(lambda: dict(calls=0, logical_read_write_bytes=0))
    sizes = {'float': 4, 'double': 8, 'c10::BFloat16': 2, 'c10::Half': 2,
             'long int': 8, 'long': 8, 'int': 4, 'bool': 1, 'c10::complex<float>': 8}
    for event in cpu:
        if event['name'] != 'aten::copy_':
            continue
        args = event.get('args', {})
        types, dims = args.get('Input type', [])[:2], args.get('Input Dims', [])[:2]
        label = json.dumps(dict(types=types, shapes=dims), sort_keys=True)
        row = copies[label]
        row['calls'] += 1
        if len(types) == 2 and len(dims) == 2 and all(t in sizes for t in types):
            row['logical_read_write_bytes'] += sum(math.prod(s) * sizes[t] for s, t in zip(dims, types))
        else:
            row['unknown_type'] = True
    operations = collections.defaultdict(lambda: dict(cpu_calls=0, cpu_inclusive_ms=0,
                                                     cuda_calls=0, cuda_duration_sum_ms=0))
    for event in cpu:
        row = operations[event['name']]
        row['cpu_calls'] += 1
        row['cpu_inclusive_ms'] += event['dur'] / 1000
    for event in gpu:
        name = op_ids.get(event.get('args', {}).get('External id'), '<unattributed>')
        row = operations[name]
        row['cuda_calls'] += 1
        row['cuda_duration_sum_ms'] += event['dur'] / 1000
    syncs = collections.defaultdict(lambda: dict(calls=0, cpu_duration_ms=0))
    for event in runtime:
        if any(word in event['name'] for word in ('Synchronize', 'Launch', 'Malloc', 'Free')):
            row = syncs[event['name']]
            row['calls'] += 1
            row['cpu_duration_ms'] += event['dur'] / 1000
    annotations = collections.defaultdict(lambda: dict(calls=0, duration_ms=0))
    for event in duration:
        if event.get('cat') == 'user_annotation':
            row = annotations[event['name']]
            row['calls'] += 1
            row['duration_ms'] += event['dur'] / 1000
    fills = collections.defaultdict(lambda: dict(calls=0, logical_write_bytes=0))
    for event in cpu:
        if event['name'] != 'aten::fill_':
            continue
        args = event.get('args', {})
        dims, types = args.get('Input Dims', []), args.get('Input type', [])
        values = args.get('Concrete Inputs', [])
        key = str(values[1]) if len(values) > 1 else '<unknown>'
        row = fills[key]
        row['calls'] += 1
        if dims and types and types[0] in sizes:
            row['logical_write_bytes'] += math.prod(dims[0]) * sizes[types[0]]
        else:
            row['unknown_type'] = True
    return dict(trace=str(path), scope='Profiled events only; CPU/GPU sums overlap and include profiler overhead',
                categories=dict(collections.Counter(e.get('cat', '<none>') for e in events)),
                device_activity=by_device, transfers=transfers, runtime=syncs, annotations=annotations,
                fills=fills,
                operations=dict(sorted(operations.items(), key=lambda pair: -pair[1]['cuda_duration_sum_ms'])),
                copies=[dict(json.loads(key), **value) for key, value in
                        sorted(copies.items(), key=lambda pair: -pair[1]['logical_read_write_bytes'])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('device_activity', 'transfers', 'runtime', 'annotations')}, indent=2))
