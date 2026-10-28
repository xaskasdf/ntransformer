"""Bounded correctness checks during Brandon; no performance measurements."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import run as r
p, b, ROOT = r.p, r.b, r.ROOT

def memory_available():
    free = int(b.capture(['nvidia-smi', '-i', '1', '--query-gpu=memory.free', '--format=csv,noheader,nounits']).strip())
    assert free >= 8192, 'less than 8 GiB free for bounded correctness tests'

def main():
    state = json.loads((ROOT/'status.json').read_text())
    if state['state'] == 'waiting':
        pid = state['pid']
        cmd = Path('/proc')/str(pid)/'cmdline'
        assert str(ROOT/'finalize.py').encode() in cmd.read_bytes()
        b.save('final-admission-wait.json', state | dict(resolution='Earlier user authorization permits small correctness runs alongside training. Final speed rerun omitted; completed isolated paired measurements retained.'))
        os.kill(pid, signal.SIGTERM)  # Only our idle admission waiter; no GPU child exists.
        time.sleep(1)
    else:
        assert state['state'] == 'failed' and (ROOT/'final-admission-wait.json').exists()
        previous = json.loads((ROOT/'final-admission-wait.json').read_text())['pid']
        assert not (Path('/proc')/str(previous)).exists()
    before = (ROOT/'candidate-source/src/model/ds41/ds41_gpu.cu').read_text()
    expected = before.replace('6.27 MiB each', '5.98 MiB each', 1).replace('return value && atoi(value) != 0;', 'return !value || atoi(value) != 0;', 1).replace('return flag && atoi(flag) != 0;', 'return !flag || atoi(flag) != 0;', 1)
    assert expected == (ROOT/'source/src/model/ds41/ds41_gpu.cu').read_text()
    memory_available()
    with (ROOT/'small-tests-telemetry.csv').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                    '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
        try:
            b.ENV = p.DEFAULT | dict(DS41_VRAM_GB='2', DS41_TEST_MODEL='/home/ubuntu/tiny-deepseek41.gguf')
            b.command('ctest-final', [p.PREFIX/'bin/ctest', '--test-dir', ROOT/'build', '-V', '-R',
                                     'test_ds41_(mxfp4|attention|router|shared|backbone|moe_overlap|prefill|hc)'])
            assert not any(word in (ROOT/'ctest-final.log').read_text() for word in ['FAIL','SKIP','Not Run'])
            comparisons = []
            for binary, shapes in [('test_ds41_mxfp4', ['small','unaligned','wide']), ('test_ds41_moe_overlap', ['small','real','max'])]:
                for shape in shapes:
                    reference = None
                    for label, flags in [('control', dict(DS41_MXFP4_COOPERATIVE='0', DS41_MXFP4_REGISTER_LUT='0')),
                                         ('cooperative', dict(DS41_MXFP4_COOPERATIVE='1', DS41_MXFP4_REGISTER_LUT='0')),
                                         ('defaults', {})]:
                        memory_available()
                        name = binary+'-'+shape+'-'+label+'-final'
                        b.ENV = p.DEFAULT | dict(DS41_VRAM_GB='2', DS41_MOE_TRACE='1') | flags
                        b.command(name, [ROOT/'build'/binary, shape, ROOT/(name+'.f32')])
                        assert not any(word in (ROOT/(name+'.log')).read_text() for word in ['FAIL','SKIP'])
                        digest = b.sha(ROOT/(name+'.f32'))
                        if reference is None: reference = digest
                        assert digest == reference, name
                        comparisons.append(dict(case=name, sha256=digest, matches_control=True))
            b.save('numerical-tests-final.json', comparisons)
        finally:
            monitor.terminate(); monitor.wait(timeout=10)
    b.save('validation-final.json', dict(ctest_cases=28, numerical_processes=18,
        default_flags_unset_checked=True, kernel_source_unchanged=True, expert_cache_budget_gib=2,
        concurrent_training=True, throughput_remeasured=False,
        performance_evidence='Completed isolated candidate-source controls and combined P32/P128/P512 cases; final changes only flag defaults and a size comment.'))
    b.status('complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        b.status('failed', error=repr(error)); raise
