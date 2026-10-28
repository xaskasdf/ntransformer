import json
import shutil
import subprocess
import sys
import run as r

b, ROOT = r.b, r.ROOT

def prepare():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'contexts-complete'
    shutil.copy2(ROOT / 'environment.json', ROOT / 'initial-environment.json')
    shutil.copytree(ROOT / 'source', ROOT / 'initial-source')
    for name in ['ds41_gpu.cu', 'ds41_gpu.h', 'ds41_forward.cpp']:
        shutil.copy2(ROOT / (name + '.final'), ROOT / 'source/src/model/ds41' / name)
    shutil.copy2(ROOT / 'test_ds41_attention.cpp.final', ROOT / 'source/tests/test_ds41_attention.cpp')
    b.command('build-final', json.loads((ROOT / 'build-command.json').read_text())['command'])
    b.command('compile-harness-final', json.loads((ROOT / 'compile-harness-command.json').read_text())['command'])
    env = json.loads((ROOT / 'environment.json').read_text())
    env.update(source_sha256={str(p.relative_to(ROOT / 'source')): b.sha(p) for p in (ROOT / 'source').rglob('*') if p.is_file()},
               binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a'))
    b.save('environment.json', env)
    b.status('final-prepared')

def main():
    if '--prepare' in sys.argv:
        prepare(); return
    with b.admitted():
        with (ROOT / 'final-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                b.ENV = r.DEFAULT.copy()
                b.ENV['DS41_TEST_MODEL'] = '/home/ubuntu/tiny-deepseek41.gguf'
                b.command('ctest-final', [r.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V', '-R',
                                        'test_ds41_(attention|router|shared|backbone|moe_overlap|prefill|hc)'])
                assert not any(t in (ROOT / 'ctest-final.log').read_text() for t in ['FAIL', 'SKIP', 'Not Run'])
                b.ENV = r.DEFAULT.copy(); b.ENV['DS41_BENCH_ALL_LOGITS'] = '1'
                name = 'defaults-p32'
                b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench',
                                 b.MODEL, ROOT / 'corpus.txt', 32, 64, 3, ROOT / name, name])
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('complete')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        b.status('failed', error=repr(e)); raise
