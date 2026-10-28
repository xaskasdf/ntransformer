import json
import subprocess
import run as r
p, b, ROOT = r.p, r.b, r.ROOT

def main():
    assert json.loads((ROOT/'status.json').read_text())['state'] == 'candidates-complete'
    with b.admitted():
        with (ROOT/'contexts-telemetry.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu',
                                        '--format=csv', '--loop-ms=1000'], stdout=log, stderr=subprocess.STDOUT)
            try:
                for n in [32, 128, 512]:
                    for label, flags in [('control', dict(DS41_MXFP4_COOPERATIVE='0', DS41_MXFP4_REGISTER_LUT='0')),
                                         ('combined', dict(DS41_MXFP4_COOPERATIVE='1', DS41_MXFP4_REGISTER_LUT='1'))]:
                        r.case(label+'-p'+str(n), flags, n)
            finally:
                monitor.terminate(); monitor.wait(timeout=10)
    b.status('contexts-complete')

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        b.status('failed', error=repr(error)); raise
