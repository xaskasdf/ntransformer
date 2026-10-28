from common import *
import runpy
for _ in range(120):
    state=json.loads((root/'status.json').read_text())['state']
    if state=='complete':break
    if state=='failed':raise RuntimeError('initial experiment failed')
    time.sleep(5)
else:raise TimeoutError('initial experiment did not finish')
runpy.run_path(str(root/'run_probes.py'),run_name='__main__')
