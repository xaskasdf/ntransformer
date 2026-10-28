"""CPU-only checks using the successful pilot whose log caused decoding failure."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import common as b

spec=importlib.util.spec_from_file_location('candidate',b.ROOT/'run-next.py')
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
label='pilot-cpu20-t26-offload1'
raw=(b.ROOT/(label+'.log')).read_bytes()
try:raw.decode('utf-8')
except UnicodeDecodeError:pass
else:raise AssertionError('real failure fixture is not present')
workload=json.loads((b.ROOT/'workload-pilot.json').read_text())
before=b.sha(b.ROOT/(label+'.log'))
data=candidate.validate(label,workload,'llama')
assert data and b.sha(b.ROOT/(label+'.log'))==before
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)
    with patch.object(candidate,'ROOT',root),patch.object(b,'command',return_value=1):
        config=dict(op_offload=True,threads=26,cpu_moe=20)
        path=root/'fixture.log'
        path.write_bytes(b'invalid metadata: \xe2\x80\nCUDA error: out of memory\n')
        assert candidate.llama('fixture','pilot',config,2,required=False) is None
        path.write_bytes(b'invalid metadata: \xe2\x80\nunrelated error\n')
        try:candidate.llama('fixture','pilot',config,2,required=False)
        except RuntimeError as error:assert 'non-OOM' in str(error)
        else:raise AssertionError('non-OOM failure was accepted')
print('PASS 3 CPU log checks: actual pilot validated with original bytes preserved; OOM recognized; non-OOM failure rejected',flush=True)
