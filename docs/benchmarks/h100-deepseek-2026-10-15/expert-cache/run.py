import json
import subprocess
import prepare as p
b, ROOT = p.b, p.ROOT

def main():
    with b.admitted():
        with (ROOT / 'telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                workloads = [('p32', 'corpus.txt', 32), ('p128', 'corpus.txt', 128),
                    ('p512', 'corpus.txt', 512), ('heldout-p512', 'corpus-heldout.txt', 512)]
                for index, (workload, corpus, n) in enumerate(workloads):
                    policies = ['lru', 'slru'] if index % 2 == 0 else ['slru', 'lru']
                    for policy in policies:
                        label = policy + '-' + workload
                        b.ENV = p.DEFAULT | dict(DS41_CACHE_POLICY=policy, DS41_CACHE_PROTECTED_PERCENT='80', DS41_CACHE_TRACE='0')
                        b.command(label, ['/usr/bin/time', '-v', '-o', ROOT / (label + '-resources.txt'), ROOT / 'bench',
                            b.MODEL, ROOT / corpus, n, 64, 3, ROOT / label, label])
                        rows = [json.loads(line) for line in (ROOT / label / 'measurements.jsonl').read_text().splitlines()]
                        for row in rows:
                            if row.get('phase') == 'decode':
                                assert row['routed_cpu_calls'] == 0 and row['peer_fallback_calls'] == 0
                                assert row['routed_gpu_calls'] == 40 * row['tokens']
                                assert row['q8_cooperative_launches'] == row['q8_projection_launches'] > 0
                                promotions = row['cache0_promotions'] + row['cache1_promotions']
                                assert promotions >= 0 if policy == 'slru' else promotions == 0
                        all_promotions = sum(r['cache0_promotions'] + r['cache1_promotions'] for r in rows if r['kind'] == 'phase')
                        assert (all_promotions > 0) == (policy == 'slru')
                        assert all(r['repeat_bitwise_equal'] for r in rows if r['kind'] == 'completion')
                    a, c = ROOT / ('lru-' + workload), ROOT / ('slru-' + workload)
                    for filename in ['all-logits.f32', 'prompt-ids.json', 'target-ids.json']:
                        assert b.sha(a / filename) == b.sha(c / filename), (workload, filename)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('measurements-complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise
