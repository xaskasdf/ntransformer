import json
import shutil
import subprocess
import sys
import run as r

b = r.b
ROOT = r.ROOT

def prepare():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'cooperative-complete'
    shutil.copy2(ROOT / 'environment.json', ROOT / 'cooperative-environment.json')
    shutil.copytree(ROOT / 'source', ROOT / 'cooperative-source')
    # The local file has been reviewed and the new default explicitly selected.
    shutil.copy2(ROOT / 'ds41_forward-final.cpp', ROOT / 'source/src/model/ds41/ds41_forward.cpp')
    shutil.copy2(ROOT / 'ds41_shared-final.cu', ROOT / 'source/src/model/ds41/ds41_shared.cu')
    b.command('build-final', [r.PREFIX / 'bin/cmake', '--build', ROOT / 'build', '-j8', '--target',
                           'ntransformer', 'test_ds41_router', 'test_ds41_backbone', 'test_ds41_shared',
                           'test_ds41_moe_overlap', 'test_ds41_prefill', 'test_ds41_hc'])
    command = json.loads((ROOT / 'compile-harness-command.json').read_text())['command']
    b.command('compile-harness-final', command)
    env = json.loads((ROOT / 'environment.json').read_text())
    env.update(source_sha256={str(p.relative_to(ROOT / 'source')): b.sha(p) for p in (ROOT / 'source').rglob('*') if p.is_file()},
               binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a'))
    b.save('environment.json', env)
    b.status('final-prepared')

def default_case(name, n):
    b.ENV = r.DEFAULT.copy()
    b.ENV['DS41_BENCH_ALL_LOGITS'] = '1'
    b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
                     b.MODEL, ROOT / 'corpus.txt', n, 64, 3, ROOT / name, name])

def main():
    if '--prepare' in sys.argv:
        prepare()
        return
    with b.admitted():
        with (ROOT / 'final-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                b.ENV = r.DEFAULT.copy()
                b.ENV['DS41_TEST_MODEL'] = '/home/ubuntu/tiny-deepseek41.gguf'
                b.command('ctest-final', [r.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V',
                                        '-R', 'test_ds41_(router|shared|backbone|moe_overlap|prefill|hc)'])
                assert not any(t in (ROOT / 'ctest-final.log').read_text() for t in ['FAIL', 'SKIP', 'Not Run'])
                default_case('defaults-p32', 32)
                r.case('control-p128', 0, n=128)
                default_case('defaults-p128', 128)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
    b.status('complete')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        b.status('failed', error=repr(e))
        raise
