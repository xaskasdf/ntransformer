import json
import subprocess
import prepare as p
b, ROOT = p.b, p.ROOT

def main():
    with b.admitted():
        b.ENV = p.DEFAULT | dict(DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf',
            DS41_VRAM_GB='2', DS41_PEER_VRAM_GB='2')
        b.command('ctest', [p.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
            'test_ds41_(q8|mxfp4|precision|peer|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
        assert not any(word in (ROOT / 'ctest.log').read_text() for word in ['FAIL', 'SKIP', 'Not Run'])
        b.command('q8-microbenchmark', [ROOT / 'build/test_ds41_q8', 'bench'])
        assert 'BENCH rows=5120' in (ROOT / 'q8-microbenchmark.log').read_text()
        comparisons = []
        for mode in ['prologue', 'tail']:
            for grouped in ['0', '1']:
                reference = None
                for coop in ['0', '1']:
                    name = 'fixture-' + mode + '-grouped' + grouped + '-coop' + coop
                    b.ENV = p.DEFAULT | dict(DS41_Q8_COOPERATIVE=coop, DS41_GROUPED_TAIL=grouped)
                    b.command(name, [ROOT / 'build/test_ds41_backbone', mode, ROOT / (name + '.f32')])
                    digest = b.sha(ROOT / (name + '.f32'))
                    if reference is None: reference = digest
                    assert reference == digest
                    comparisons.append(dict(case=name, sha256=digest, bitwise_equal=True))
        b.save('numerical-tests.json', comparisons)
        with (ROOT / 'telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for n in [32, 128, 512]:
                    variants = [('legacy', '0'), ('cooperative', '1')]
                    if n == 128: variants.reverse()
                    for label, flag in variants:
                        name = label + '-p' + str(n)
                        b.ENV = p.DEFAULT | dict(DS41_PEER_MODE='execute', DS41_MXFP4_COMPUTE='fp32', DS41_Q8_COOPERATIVE=flag)
                        b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
                            b.MODEL, ROOT / 'corpus.txt', n, 64, 3, ROOT / name, name])
                        rows = [json.loads(line) for line in (ROOT / name / 'measurements.jsonl').read_text().splitlines()]
                        for row in rows:
                            if row.get('phase') == 'decode':
                                assert row['q8_projection_launches'] > 0
                                assert (row['q8_cooperative_launches'] > 0) == (flag == '1')
                                assert row['peer_fallback_calls'] == 0
                        assert all(row['repeat_bitwise_equal'] for row in rows if row['kind'] == 'completion')
                    assert b.sha(ROOT / ('legacy-p' + str(n)) / 'all-logits.f32') == b.sha(ROOT / ('cooperative-p' + str(n)) / 'all-logits.f32')
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise
