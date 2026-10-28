"""Select whole binary scenarios from the exact fourteen reviewed numeric banks."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3 import select

root = Path(sys.argv[1]).resolve()
manifest_path = root / 'pcie_symbolic_numeric_ab_binary_r1.json'
assert not manifest_path.exists()
raw = dict(main='pcie_request_inventory_r7.txt',ownership='pcie_rc_ownership_integration.txt',
    rq='pcie_rq_admission_integration.txt',prefix='pcie_rc_prefix_integration.txt',
    lanes='pcie_rc_lane_integration_rtl.txt',context='pcie_symbolic_value_context.txt')
specs = {**{'raw_'+k:v for k,v in raw.items()},
    **{'drained_'+k:'pcie_symbolic_numeric_ab_drained_'+k+'_r1.txt' for k in raw},
    'directed':'pcie_symbolic_ced_directed_r1.txt',
    'controls':'pcie_symbolic_numeric_ab_controls_r1.txt'}
prior = json.loads((root / 'pcie_symbolic_ced_binary_r1.json').read_text())
banks = {}
for name,filename in specs.items():
    source = root / filename
    original = source.read_bytes()
    lines = original.decode().splitlines()
    selected,excluded = select(lines)
    data = ('\n'.join(selected)+'\n').encode()
    if not excluded:
        assert selected == lines
        target = source
        data = original
    elif name in prior['banks'] and (root / prior['banks'][name]['path']).read_bytes() == data:
        target = root / prior['banks'][name]['path']
    else:
        target = root / ('pcie_symbolic_numeric_ab_binary_'+name+'_r1.txt')
        with target.open('xb') as f:
            f.write(data)
    assert target.read_bytes() == data
    banks[name] = dict(source=source.name,source_sha256=hashlib.sha256(original).hexdigest(),
        path=target.name,sha256=hashlib.sha256(data).hexdigest(),source_rows=len(lines),rows=len(selected),
        scenarios=len({v.split()[0] for v in selected}),excluded_whole_scenarios=excluded,unchanged=not excluded)
record = dict(banks=banks,total_rows=sum(v['rows'] for v in banks.values()),
    total_source_rows=sum(v['source_rows'] for v in banks.values()),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    selector=dict(path='tools/generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3.py',
        sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3.py').read_bytes()).hexdigest()),
    scope='All fourteen numeric raw/drained/directed/control banks. Exclude complete cold-isolated scenarios with X/Z inputs only; no filtering by fault or malformed binary traffic. Four-state proof retained separately; no supplemental historical main corpus.')
assert record['total_source_rows'] == 178185
with manifest_path.open('x') as f:
    json.dump(record,f,indent=2); f.write('\n')
print(json.dumps(dict(total_rows=record['total_rows'],banks={k:dict(rows=v['rows'],excluded_scenarios=len(v['excluded_whole_scenarios'])) for k,v in banks.items()})))
