"""Synthetic causal controls for the full-size producer transcript reviewer."""
import argparse
import json
from pathlib import Path
import ntrpu_input_attention_fixture_r1 as layout
import ntrpu_input_attention_source_review_r1 as review


def synthetic_source(fixture, spec, ident, ordinal, first_cycle=10):
    rows, memory = [], {}
    cycle = first_cycle
    sample = fixture['samples'][spec['sample']]
    seq, epoch = 0x1234567800000000 + spec['group'], 0xfedcba9800000000 + spec['group']
    writes = 0

    def emit(event, **fields):
        nonlocal cycle
        cycle += 1
        row = dict(cycle=cycle, job=ident, **fields)
        if event in review.SCHEMA:
            row = {k: row[k] for k in review.SCHEMA[event].split()}
        rows.append((event, row))

    emit('ADMIT')
    emit('PSTART', sequence=seq, epoch=epoch)

    def read_burst(kind, phase, address, beats):
        emit('PAR', kind=kind, phase=phase, address=address, beats=beats)
        for beat in range(beats):
            a = address + beat * 64
            if not kind:
                plane = int(a >= layout.address(1))
                offset = spec['sample']*4096 + plane*2048 + (a-layout.address(plane))//4
                data = sum(fixture['norm']['source32.hex'][offset+i] << (32*i) for i in range(16))
            elif layout.address(2) <= a < layout.address(2)+4096:
                data = memory[a]
            else:
                data = fixture['weights'][phase][(a-((phase+4)<<40))//64]
            emit('PR', kind=kind, phase=phase, address=a, beat=beat, last=int(beat==beats-1), data=data)

    def write_burst(kind, phase, address, beats, frames):
        nonlocal writes
        emit('PAW', kind=kind, phase=phase, address=address, beats=beats)
        for i, value in enumerate(frames):
            emit('PW', kind=kind, phase=phase, index=writes, last=int(i==beats-1), data=value)
            writes += 1
        emit('PB', kind=kind, phase=phase, address=address, beats=beats, response=0)
        for i, value in enumerate(frames):
            memory[address+i*64] = value

    for kind, phase, address, beats, _ in review.READ_PLAN:
        if kind == 0:
            read_burst(kind, phase, address, beats)
    emit('PNSTART')
    for i, value in enumerate(fixture['norm']['abi64.hex'][spec['sample']*512:(spec['sample']+1)*512]):
        emit('PNWRITE', index=i, data=value)
    emit('PNDONE', status=0, words=512)
    for i, frame in enumerate(sample['activation']):
        write_burst(0, 0, layout.address(2)+i*64, 1, [frame])
    for phase, count in enumerate((2048, 512, 512)):
        emit('PMATRIX', phase=phase, tag=ordinal*192+phase*64, rows=count)
        for group in range(count//32):
            for kind, p, a, n, g in review.READ_PLAN:
                if kind and p == phase and g == group:
                    read_burst(kind, p, a, n)
            for i in range(group*32, (group+1)*32):
                emit('PMAC', phase=phase, index=i, data=sample['fp64'][phase][i])
            offset = group*2
            for address, beats in review.bursts(layout.address(6+phase)+group*128, 128):
                write_burst(1, phase, address, beats, sample['fp32'][phase][offset:offset+beats])
                offset += beats
        if phase < 2:
            emit('PMACK', phase=phase, tag=ordinal*192+phase*64, status=0, rows=count, bytes=count*4)
    emit('PDONE', status=0, norm=0, qkv=0, matrix=0, hls=0, reads=256, records=64, phase=2, matrices=3, bytes=12288)
    emit('CSTART', sequence=seq, epoch=epoch, layer=0, position=spec['sample'])
    emit('ROPE_START')
    emit('COMPLETE')
    emit('ACK')
    emit('CACK')
    emit('PACK')
    emit('PMACK', phase=2, tag=ordinal*192+128, status=0, rows=512, bytes=2048)
    emit('RELEASE')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest, fixture, _, _ = layout.load(args.fixture)
    summaries, rejected = [], []
    ordinal = 0
    baseline = None
    for ident, spec in enumerate(manifest['plan'], 1):
        if spec['mode'] == 1:
            result = review.review_source([], fixture, spec, ident, -1)
        else:
            events = synthetic_source(fixture, spec, ident, ordinal)
            result = review.review_source(events, fixture, spec, ident, ordinal)
            if ident == 1:
                baseline = events
            ordinal += 1
        summaries.append(dict(job=ident, **result['summary']))
    spec = manifest['plan'][0]

    def reject(name, events):
        try:
            review.review_source(events, fixture, spec, 1, 0)
        except (ValueError, KeyError, IndexError):
            rejected.append(name)
        else:
            raise AssertionError('Accepted producer corruption: '+name)

    for tag, field in [('PSTART', 'sequence'), ('PSTART', 'epoch'), ('PNWRITE', 'data'),
        ('PNWRITE', 'index'), ('PNDONE', 'words'), ('PNDONE', 'status'), ('PMATRIX', 'tag'),
        ('PMATRIX', 'rows'), ('PMAC', 'data'), ('PMAC', 'index'), ('PAR', 'address'),
        ('PAR', 'beats'), ('PR', 'address'), ('PR', 'data'), ('PR', 'beat'), ('PR', 'last'),
        ('PAW', 'address'), ('PAW', 'beats'), ('PW', 'data'), ('PW', 'last'), ('PW', 'index'),
        ('PB', 'address'), ('PB', 'response'), ('PB', 'beats'), ('PMACK', 'tag'), ('PMACK', 'bytes'),
        ('PDONE', 'reads'), ('PDONE', 'records'), ('PDONE', 'bytes'), ('CSTART', 'position'),
        ('CSTART', 'epoch')]:
        index = next(i for i, (kind, _) in enumerate(baseline) if kind == tag)
        changed = baseline.copy()
        row = dict(changed[index][1])
        row[field] ^= 1
        changed[index] = (tag, row)
        reject(tag+'.'+field, changed)
    for tag in ('PSTART', 'PNSTART', 'PNDONE', 'PDONE', 'CSTART', 'CACK', 'PACK'):
        index = next(i for i, (kind, _) in enumerate(baseline) if kind == tag)
        reject(tag+'.missing', baseline[:index]+baseline[index+1:])
        reject(tag+'.duplicate', baseline[:index]+[baseline[index]]+baseline[index:])
    for label, target, before in [('producer_ack_before_public', 'PACK', 'ACK'),
                                  ('consumer_before_producer_publication', 'CSTART', 'PDONE'),
                                  ('norm_before_input_reads', 'PNSTART', 'PAR')]:
        changed = baseline.copy()
        index = next(i for i, (tag, _) in enumerate(changed) if tag == target)
        cycle = next(row['cycle'] for tag, row in changed if tag == before)-1
        changed[index] = (target, dict(changed[index][1], cycle=cycle))
        changed.sort(key=lambda item: item[1]['cycle'])
        reject(label, changed)
    for index, (tag, row) in enumerate(baseline):
        if tag == 'PR' and row['kind'] and layout.address(2) <= row['address'] < layout.address(2)+4096:
            altered = baseline.copy()
            altered[index] = (tag, dict(row, data=row['data'] ^ 1))
            reject('actual_abi_read_changed', altered)
            break
    result = dict(status='PASS_SYNTHETIC_INPUT_ATTENTION_SOURCE_CONTROLS', synthetic=True,
        rtl_executed=False, arithmetic_qualified=False, plan=summaries, rejected=rejected,
        fixture_manifest_sha256=layout.sha(args.fixture / 'manifest.json'),
        checker_sha256=layout.sha(Path(review.__file__)), controls_sha256=layout.sha(Path(__file__)))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], jobs=len(summaries), controls=len(rejected))))


if __name__ == '__main__':
    main()
