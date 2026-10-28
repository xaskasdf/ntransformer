import subprocess
import pilot as p
b, ROOT = p.b, p.ROOT

def tests():
    b.ENV = p.DEFAULT | dict(DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf')
    b.command('ctest', [p.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
        'test_ds41_(mxfp4|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
    assert not any(word in (ROOT / 'ctest.log').read_text() for word in ['FAIL', 'SKIP', 'Not Run'])
    comparisons = []
    for binary, shapes in [('test_ds41_mxfp4', ['small', 'unaligned', 'wide']), ('test_ds41_moe_overlap', ['small', 'real', 'max'])]:
        for shape in shapes:
            reference = None
            for label, flags in [('control', {}), ('cooperative', {'DS41_MXFP4_COOPERATIVE': '1'}),
                                 ('cooperative-register', {'DS41_MXFP4_COOPERATIVE': '1', 'DS41_MXFP4_REGISTER_LUT': '1'})]:
                name = binary + '-' + shape + '-' + label
                b.ENV = p.DEFAULT | dict(DS41_VRAM_GB='2', DS41_MOE_TRACE='1') | flags
                b.command(name, [ROOT / 'build' / binary, shape, ROOT / (name + '.f32')])
                assert not any(word in (ROOT / (name + '.log')).read_text() for word in ['FAIL', 'SKIP'])
                digest = b.sha(ROOT / (name + '.f32'))
                if reference is None: reference = digest
                assert digest == reference, name
                comparisons.append(dict(case=name, sha256=digest, matches_control=True))
    b.save('numerical-tests.json', comparisons)

def case(name, flags, n=32):
    b.ENV = p.DEFAULT | flags
    b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
        b.MODEL, ROOT / 'corpus.txt', n, 64, 3, ROOT / name, name])

def main():
    with b.admitted():
        with (ROOT / 'candidate-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                tests()
                for name, flags in [('cooperative-p32', {'DS41_MXFP4_COOPERATIVE': '1'}),
                                    ('cooperative-trace', {'DS41_MXFP4_COOPERATIVE': '1', 'DS41_MOE_TRACE': '1'}),
                                    ('cooperative-register-p32', {'DS41_MXFP4_COOPERATIVE': '1', 'DS41_MXFP4_REGISTER_LUT': '1'}),
                                    ('cooperative-register-trace', {'DS41_MXFP4_COOPERATIVE': '1', 'DS41_MXFP4_REGISTER_LUT': '1', 'DS41_MOE_TRACE': '1'})]:
                    case(name, flags)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('candidates-complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        b.status('failed', error=repr(error)); raise
