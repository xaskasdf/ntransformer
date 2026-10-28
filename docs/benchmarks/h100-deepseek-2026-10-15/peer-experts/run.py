import subprocess
import prepare as p
b, ROOT = p.b, p.ROOT

def main():
    with b.admitted():
        b.ENV = p.DEFAULT | dict(DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf', DS41_VRAM_GB='2', DS41_PEER_VRAM_GB='2')
        b.command('ctest', [p.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
            'test_ds41_(mxfp4|precision|peer|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
        assert not any(word in (ROOT / 'ctest.log').read_text() for word in ['FAIL', 'SKIP', 'Not Run'])
        comparisons = []
        for compute in ['fp32', 'f16', 'q8']:
            for shape in ['small', 'real', 'max']:
                reference = None
                for peer in ['0', 'copy', 'execute']:
                    name = 'moe-' + compute + '-' + shape + '-' + peer
                    b.ENV = p.DEFAULT | dict(DS41_VRAM_GB='2', DS41_PEER_VRAM_GB='2', DS41_PEER_MODE=peer, DS41_MXFP4_COMPUTE=compute, DS41_MOE_TRACE='1')
                    b.command(name, [ROOT / 'build/test_ds41_moe_overlap', shape, ROOT / (name + '.f32')])
                    assert 'PASS scoped overlap' in (ROOT / (name + '.log')).read_text()
                    digest = b.sha(ROOT / (name + '.f32'))
                    if reference is None: reference = digest
                    assert digest == reference, name
                    comparisons.append(dict(case=name, sha256=digest, matches_single_gpu=True))
        b.save('numerical-tests.json', comparisons)
        with (ROOT / 'telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                cases = [('single-fp32', '0', 'fp32'), ('copy-fp32', 'copy', 'fp32'), ('execute-fp32', 'execute', 'fp32'),
                         ('execute-f16', 'execute', 'f16'), ('execute-q8', 'execute', 'q8')]
                for n in [32, 128]:
                    for label, peer, compute in cases:
                        name = label + '-p' + str(n)
                        b.ENV = p.DEFAULT | dict(DS41_PEER_MODE=peer, DS41_MXFP4_COMPUTE=compute)
                        b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
                            b.MODEL, ROOT / 'corpus.txt', n, 64, 3, ROOT / name, name])
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error: b.status('failed', error=repr(error)); raise
