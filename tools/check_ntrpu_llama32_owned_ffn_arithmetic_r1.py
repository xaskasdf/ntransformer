"""Review full real-weight Llama FFN writes and the complete accepted AXI read plan."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

from check_ntrpu_ffn_memory import reference as closed_form
from check_ntrpu_ffn_memory_fixture import activation, f32, gemv
from ntrpu_matrix_layout import pack_matrix

MASK = (1 << 64) - 1
FNV_OFFSET = 0xcbf29ce484222325


def reference(fixture):
    manifest_path=fixture/'manifest.json'
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!='bf36c6afbe46df663aa9823769f6c6c74e1608826fce7182e21ffe42646e0194':
        raise ValueError('Pinned full Llama fixture')
    manifest=json.loads(manifest_path.read_text())
    if (manifest['dim'],manifest['hidden'],manifest['samples'],manifest['source_sample_id'])!=(2048,8192,1,1):
        raise ValueError('Complete shape/source sample')
    for name,sha in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((fixture/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Fixture file identity '+name)
    def data(name):return bytes.fromhex((fixture/name).read_text())
    from check_ntrpu_ffn_memory_fixture import decode
    dim,hidden=2048,8192
    input_image=data('job-001-input.hex')
    if input_image[:4032]!=bytes([0xa5])*4032:raise ValueError('Input guard')
    buffers={0:input_image[4032:]};dense=[]
    for b,rows,blocks in [(1,8192,64),(2,8192,64),(3,2048,256)]:
        image=data(f'weight-{b}.hex');raw=decode(image,rows,blocks,32)
        if hashlib.sha256(raw).hexdigest()!=manifest['weights'][b-1]['dense_sha256']:
            raise ValueError('Original GGUF block bytes')
        dense.append(raw);buffers[b]=image[4032:]
    for b in (4,5,6):buffers[b]=data(f'job-001-buffer-{b}.hex')
    buffers[7]=data('job-001-final-fp32.hex')
    expected={0:4096,1:17825792,2:17825792,3:17825792,4:65536,5:65536,6:16384,7:8192}
    if {b:len(raw) for b,raw in buffers.items()}!=expected:raise ValueError('Full buffer lengths')
    reads = []
    def read(phase, b, offset, size):
        while size:
            count = min(size // 64, 64, (4096 - (4032 + offset) % 4096) // 64)
            reads.append(dict(id=1, phase=phase, buffer=b, offset=offset, beats=count))
            offset += count * 64
            size -= count * 64

    for phase, rows, columns, b in [(0, hidden, dim, 1), (1, hidden, dim, 2)]:
        offset = 0
        for first in range(0, rows, 32):
            read(phase, 0, 0, columns // 32 * 64)
            for kb in range(0, columns // 32, 32):
                size = 32 * min(32, columns // 32 - kb) * 34
                read(phase, b, offset, size)
                offset += size
    for first in range(0, hidden, 32):
        read(2, 4, first * 8, 256)
        read(2, 5, first * 8, 256)
    offset = 0
    for first in range(0, dim, 32):
        read(3, 6, 0, hidden // 32 * 64)
        for kb in range(0, hidden // 32, 32):
            size = 32 * min(32, hidden // 32 - kb) * 34
            read(3, 3, offset, size)
            offset += size

    hashes = {b: FNV_OFFSET for b in range(8)}
    beats = dict.fromkeys(range(8), 0)
    for r in reads:
        b, off, size = r['buffer'], r['offset'], r['beats'] * 64
        payload = buffers[b][off:off + size]
        assert len(payload) == size
        for word, in struct.iter_unpack('<Q', payload):
            hashes[b] = ((hashes[b] ^ word) * 0x100000001b3) & MASK
        beats[b] += r['beats']
    return dict(dim=dim, hidden=hidden, buffers=buffers, reads=reads,
                read_hashes=hashes, read_beats=beats,
                dense_sha256=[hashlib.sha256(w).hexdigest() for w in dense])


def metadata(ref):
    return dict(dim=ref['dim'], hidden=ref['hidden'], tile_k=32,
                dense_sha256=ref['dense_sha256'],
                buffer_sha256={b: hashlib.sha256(v).hexdigest() for b, v in ref['buffers'].items()},
                buffer_bytes={b: len(v) for b, v in ref['buffers'].items()},
                read_hashes={b: f'{v:016x}' for b, v in ref['read_hashes'].items()},
                read_beats=ref['read_beats'], read_bursts=len(ref['reads']),
                arithmetic='Full real-weight NumPy reference previously matched byte-exactly by native stdlib and AMD-math kernel execution; complete packed images independently decoded',
                scope='Complete Unsloth Q8_0 layer0 FFN; exact post-O ABI1 sample1/token0; no full-model inference, physical throughput or DDR qualification')


def review(text, ref, seed=1):
    errors = []
    def need(ok, why):
        if not ok:
            errors.append(why)
    def records(prefix):
        return [{k: int(v, 16) if k in {'data', 'strb', 'hash'} else int(v)
                 for k, v in re.findall(r'(\w+)=([0-9a-f]+)', line)}
                for line in text.splitlines() if line.startswith(prefix + ' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)', text, re.M | re.I), 'Executed failure')
    need(records('NTRPU_FFN_MEMORY_SEED') == [dict(seed=seed)], 'Seed')
    need(records('NTRPU_FFN_MEMORY_FORMAT') == [dict(fp32=1)], 'Output format')
    need(records('NTRPU_LLAMA32_REAL_WEIGHTS') == [dict(layer=0,source_sample=1,bytes=53477376)], 'Real full-weight workload')
    need(records('NTRPU_FFN_LARGE_CONFIG') == [dict(dim=ref['dim'], hidden=ref['hidden'], tile_k=32)], 'Shape/config')
    need(records('NTRPU_FFN_LARGE_AR') == ref['reads'], 'Complete AR plan including phases, offsets and page splits')
    need(records('NTRPU_FFN_LARGE_READ') == [dict(id=1, buffer=b, beats=ref['read_beats'][b], hash=ref['read_hashes'][b])
                                          for b in range(8)], 'Every accepted input beat checksum')
    writes = records('NTRPU_FFN_MEMORY_W')
    expected = [dict(id=1, buffer=b, offset=off, strb=MASK,
                     data=int.from_bytes(raw[off:off+64], 'little'))
                for b in range(4, 8) for raw in [ref['buffers'][b]] for off in range(0, len(raw), 64)]
    need(writes == expected, 'Complete physical output bytes, order and masks')
    cases = records('NTRPU_FFN_MEMORY_CASE')
    need(len(cases) == 1, 'One completed root job')
    c = cases[0] if cases else {}
    need(tuple(c.get(k) for k in ['id', 'dim', 'hidden', 'mode', 'target', 'reset']) ==
         (1, ref['dim'], ref['hidden'], 0, 0, 0), 'Case contract')
    need(0 < c.get('cycles', 0) < 30000000, 'Cycle bound')
    need(sum(c.get('p'+str(p), 0) for p in range(4)) == c.get('cycles'), 'Phase cycles')
    total = sum(len(ref['buffers'][b]) for b in range(4, 8))
    # AW splits are independent of input reads: vectors commit 32 rows at once;
    # activation commits one ABI1 record per 32 hidden elements.
    aw = 0
    for b, rows, width in [(4, ref['hidden'], 8), (5, ref['hidden'], 8), (7, ref['dim'], 4)]:
        for row in range(0, rows, 32):
            off = (4032 + row * width) % 4096
            aw += 1 + int(off + 32 * width > 4096)
    aw += ref['hidden'] // 32
    need(tuple(c.get(k) for k in ['ar', 'r', 'aw', 'w', 'b', 'bytes', 'down_bytes']) ==
         (len(ref['reads']), sum(ref['read_beats'].values()), aw, total // 64, aw, total, ref['dim'] * 4), 'AXI/completion accounting')
    summaries = records('NTRPU_FFN_MEMORY_PASS')
    need(len(summaries) == 1, 'Final PASS count')
    s = summaries[0] if summaries else {}
    need(tuple(s.get(k) for k in ['cases', 'cycles', 'bytes', 'resets', 'cancels']) ==
         (1, c.get('cycles'), total, 0, 0), 'Suite accounting')
    for name in ['ar_stalls', 'aw_stalls', 'w_stalls', 'done_stalls']:
        need(s.get(name, 0) > 0, 'Backpressure ' + name)
    progress = records('NTRPU_FFN_LARGE_PROGRESS')
    need([p.get('cycles') for p in progress] == list(range(100000, c.get('cycles', 0)+1, 100000)), 'Progress continuity')
    for p in progress:
        phase, count = p.get('phase', -1), p.get('bytes', -1)
        need(phase in range(4) and count % 64 == 0 and
             sum(len(ref['buffers'][b]) for b in range(4, 4+phase)) <= count <=
             sum(len(ref['buffers'][b]) for b in range(4, 5+phase)), 'Progress phase/bytes')
    return dict(status='FAIL' if errors else 'PASS', errors=errors, cases=cases, summary=s,
                verified_output_bytes=total, verified_input_bytes=sum(ref['read_beats'].values())*64,
                reference=metadata(ref))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--log', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    ref = reference(a.fixture)
    result = review(a.log.read_text(), ref) if a.log else dict(status='REFERENCE_READY', **metadata(ref))
    if a.output:
        with a.output.open('x', encoding='utf-8') as out:
            out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] == 'FAIL')
