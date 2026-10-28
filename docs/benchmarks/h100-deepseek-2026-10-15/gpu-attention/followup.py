import subprocess
import run as r

b, ROOT = r.b, r.ROOT
try:
    with b.admitted():
        with (ROOT / 'followup-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for n in [128, 512]:
                    r.case('control-p' + str(n), 0, n)
                    r.case('resident-p' + str(n), 1, n)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('contexts-complete')
except BaseException as e:
    b.status('failed', error=repr(e)); raise
