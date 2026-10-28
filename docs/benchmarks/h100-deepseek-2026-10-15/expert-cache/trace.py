import json
import prepare as p
b, ROOT = p.b, p.ROOT

def main():
    with b.admitted():
        b.ENV = p.DEFAULT | dict(DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf',
            DS41_PEER_MODE='0', DS41_VRAM_GB='2', DS41_PEER_VRAM_GB='2')
        b.command('ctest', [p.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
            'test_ds41_(cache|q8|mxfp4|precision|peer|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
        assert not any(word in (ROOT / 'ctest.log').read_text() for word in ['FAIL', 'SKIP', 'Not Run'])
        b.ENV = p.DEFAULT | dict(DS41_CACHE_POLICY='lru', DS41_CACHE_TRACE='1')
        b.command('trace-lru-p512', [ROOT / 'bench', b.MODEL, ROOT / 'corpus.txt', 512, 64, 3,
            ROOT / 'trace-lru-p512', 'trace-lru-p512'])
    b.ENV = p.DEFAULT
    b.command('compile-replay', [p.PREFIX / 'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O3',
        '-I' + str(ROOT / 'source/src'), ROOT / 'replay_ds41_cache.cpp', '-o', ROOT / 'replay'])
    results = {}
    for percent in [0, 50, 80, 90]:
        results[percent] = []
        for gpu in [0, 1]:
            files = [ROOT / 'trace-lru-p512' / f'cache-{round}-{phase}-gpu{gpu}.u64'
                for round in range(4) for phase in ['prefill', 'decode']]
            label = f'replay-{percent}-gpu{gpu}'
            b.command(label, [ROOT / 'replay', percent, *files])
            rows = [json.loads(line) for line in (ROOT / (label + '.log')).read_text().splitlines()]
            if percent == 0: assert all(row['hit_mismatches'] == 0 for row in rows)
            for row in rows: row['gpu'] = gpu
            results[percent].extend(rows)
    b.save('replay-summary.json', results)
    for percent, rows in results.items():
        warm = [r for r in rows if '-decode-' in r['file'] and 'cache-0-' not in r['file']]
        print(percent, dict(warm_decode_mib_per_token=sum(r['misses']*r['slice_bytes'] for r in warm)/192/2**20,
              warm_first_use_misses=sum(r['first_use'] for r in warm)), flush=True)
    b.status('trace-complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise
