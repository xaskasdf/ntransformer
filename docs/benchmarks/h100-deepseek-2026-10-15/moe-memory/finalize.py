import json
import shutil
import subprocess
import sys
import analyze
import run as r
p, b, ROOT = r.p, r.b, r.ROOT

def prepare():
    state = json.loads((ROOT/'status.json').read_text())['state']
    if '--resume-prepare' in sys.argv:
        assert state == 'failed' and (ROOT/'candidate-source').is_dir()
        saved = json.loads((ROOT/'candidate-environment.json').read_text())
        for name, digest in saved['source_sha256'].items():
            assert b.sha(ROOT/'candidate-source'/name) == digest, name
    else:
        assert state == 'contexts-complete'
    analyze.main()
    measured = json.loads((ROOT/'summary.json').read_text())['cases']
    for n in [32, 128, 512]:
        d = measured['combined-p'+str(n)]['comparison']
        assert d['all_logits_bitwise_equal'] and d['expert_upload_bytes_equal'] and d['consumed_ids_equal']
        assert d['decode_speedup'] > 1.05
    env = json.loads((ROOT/'environment.json').read_text())
    env.update(harness_sha256=b.sha(ROOT/'benchmark_ds41.cpp'), harness_path='benchmark_ds41.cpp')
    if '--resume-prepare' not in sys.argv:
        b.save('candidate-environment.json', env)
        shutil.copytree(ROOT/'source', ROOT/'candidate-source')
    for name, destination in [('ds41_gpu.cu', 'src/model/ds41/ds41_gpu.cu'),
                              ('test_ds41_moe_overlap.cpp', 'tests/test_ds41_moe_overlap.cpp'),
                              ('test_ds41_mxfp4.cpp', 'tests/test_ds41_mxfp4.cpp'),
                              ('CMakeLists.txt', 'CMakeLists.txt')]:
        shutil.copy2(ROOT/(name+'.final'), ROOT/'source'/destination)
    (ROOT/'source/tools').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'benchmark_ds41.cpp', ROOT/'source/tools/benchmark_ds41.cpp')
    for label in ['configure', 'build', 'compile-harness']:
        b.command(label+'-final', json.loads((ROOT/(label+'-command.json')).read_text())['command'])
    env.update(source_sha256={str(f.relative_to(ROOT/'source')): b.sha(f) for f in (ROOT/'source').rglob('*') if f.is_file()},
               binary_sha256=b.sha(ROOT/'bench'), library_sha256=b.sha(ROOT/'build/libntransformer_lib.a'))
    b.save('environment.json', env)
    b.status('final-prepared')

def main():
    prepare()
    with b.admitted():
        with (ROOT/'final-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                b.ENV = p.DEFAULT | dict(DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf')
                b.command('ctest-final', [p.PREFIX/'bin/ctest', '--test-dir', ROOT/'build', '-V', '-R',
                                         'test_ds41_(mxfp4|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
                assert not any(word in (ROOT/'ctest-final.log').read_text() for word in ['FAIL','SKIP','Not Run'])
                comparisons = []
                for binary, shapes in [('test_ds41_mxfp4', ['small','unaligned','wide']), ('test_ds41_moe_overlap', ['small','real','max'])]:
                    for shape in shapes:
                        reference = None
                        for label, flags in [('control', dict(DS41_MXFP4_COOPERATIVE='0', DS41_MXFP4_REGISTER_LUT='0')),
                                             ('cooperative', dict(DS41_MXFP4_COOPERATIVE='1', DS41_MXFP4_REGISTER_LUT='0')),
                                             ('combined', dict(DS41_MXFP4_COOPERATIVE='1', DS41_MXFP4_REGISTER_LUT='1'))]:
                            name = binary+'-'+shape+'-'+label+'-final'
                            b.ENV = p.DEFAULT | dict(DS41_VRAM_GB='2', DS41_MOE_TRACE='1') | flags
                            b.command(name, [ROOT/'build'/binary, shape, ROOT/(name+'.f32')])
                            assert not any(word in (ROOT/(name+'.log')).read_text() for word in ['FAIL','SKIP'])
                            digest = b.sha(ROOT/(name+'.f32'))
                            if reference is None: reference = digest
                            assert digest == reference, name
                            comparisons.append(dict(case=name, sha256=digest, matches_control=True))
                b.save('numerical-tests-final.json', comparisons)
                r.case('defaults-p32', {})
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    analyze.main()
    measured = json.loads((ROOT/'summary.json').read_text())['cases']
    assert measured['defaults-p32']['comparison']['all_logits_bitwise_equal']
    b.status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        b.status('failed', error=repr(error)); raise
