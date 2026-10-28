import json
import shutil
import subprocess
import run as r

b, ROOT = r.b, r.ROOT

def main():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'initial-complete'
    shutil.copy2(ROOT / 'environment.json', ROOT / 'initial-environment.json')
    shutil.copytree(ROOT / 'source', ROOT / 'initial-source')
    shutil.copy2(ROOT / 'bench', ROOT / 'bench-initial')
    shutil.copy2(ROOT / 'ds41_shared-cooperative.cu', ROOT / 'source/src/model/ds41/ds41_shared.cu')
    shutil.copy2(ROOT / 'test_ds41_router-cooperative.cpp', ROOT / 'source/tests/test_ds41_router.cpp')
    command = json.loads((ROOT / 'build-command.json').read_text())['command']
    b.command('build-cooperative', command)
    b.command('compile-harness-cooperative', json.loads((ROOT / 'compile-harness-command.json').read_text())['command'])
    env = json.loads((ROOT / 'environment.json').read_text())
    env.update(source_sha256={str(p.relative_to(ROOT / 'source')): b.sha(p) for p in (ROOT / 'source').rglob('*') if p.is_file()},
               binary_sha256=b.sha(ROOT / 'bench'), library_sha256=b.sha(ROOT / 'build/libntransformer_lib.a'))
    b.save('environment.json', env)
    r.DEFAULT['DS41_HC_COOPERATIVE'] = '1'
    with b.admitted():
        with (ROOT / 'cooperative-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                b.ENV = r.DEFAULT.copy()
                b.ENV.update(DS41_HC_PROJECTION_GPU='1', DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf')
                b.command('ctest-cooperative', [r.PREFIX / 'bin/ctest', '--test-dir', ROOT / 'build', '-V',
                                              '-R', 'test_ds41_(router|shared|backbone|moe_overlap|prefill|hc)'])
                assert not any(t in (ROOT / 'ctest-cooperative.log').read_text() for t in ['FAIL', 'SKIP', 'Not Run'])
                b.ENV['DS41_HC_COOPERATIVE'] = '0'
                b.command('router-direct-expanded', [ROOT / 'build/test_ds41_router'])
                assert 'PASS router/HC: 72' in (ROOT / 'router-direct-expanded.log').read_text()
                r.case('cooperative-p32', 1)
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
    b.status('cooperative-complete')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        b.status('failed', error=repr(e))
        raise
