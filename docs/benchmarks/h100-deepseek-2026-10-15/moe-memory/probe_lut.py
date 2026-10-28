import subprocess
import pilot as p
b, ROOT = p.b, p.ROOT
try:
    with b.admitted():
        with (ROOT / 'lut-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for name, trace in [('lut-control', '0'), ('lut-trace', '1')]:
                    b.ENV = p.DEFAULT | dict(DS41_MXFP4_REGISTER_LUT='1', DS41_MOE_TRACE=trace)
                    b.command(name, ['/usr/bin/time', '-v', '-o', ROOT / (name + '-resources.txt'), ROOT / 'bench-pilot',
                        b.MODEL, ROOT / 'corpus.txt', 32, 64, 3, ROOT / name, name])
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('lut-complete')
except BaseException as error:
    b.status('failed', error=repr(error)); raise
