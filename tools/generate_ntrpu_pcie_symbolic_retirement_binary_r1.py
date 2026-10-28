"""Select whole binary scenarios from all fifteen R1 banks; keep malformed binary traffic."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3 import select

root=Path(sys.argv[1]).resolve()
out=root/'pcie_symbolic_retirement_binary_r1.json'
assert not out.exists()
raw=dict(main='pcie_request_inventory_r7.txt',ownership='pcie_rc_ownership_integration.txt',
    rq='pcie_rq_admission_integration.txt',prefix='pcie_rc_prefix_integration.txt',
    lanes='pcie_rc_lane_integration_rtl.txt',context='pcie_symbolic_value_context.txt')
specs={**raw,**{'drained_'+n:'pcie_symbolic_retirement_drained_'+n+'_r1.txt' for n in raw},
    'directed':'pcie_symbolic_ced_directed_r1.txt','controls':'pcie_symbolic_numeric_ab_controls_r1.txt',
    'range_return':'pcie_symbolic_retirement_range_r1.txt'}
prior=json.loads((root/'pcie_symbolic_numeric_ab_binary_r1.json').read_text())
banks={}
for name,filename in specs.items():
    source=root/filename;original=source.read_bytes();lines=original.decode().splitlines()
    selected,excluded=select(lines);data=('\n'.join(selected)+'\n').encode()
    previous='raw_'+name if name in raw else name
    if not excluded:
        assert selected==lines;target=source;data=original
    elif previous in prior['banks'] and (root/prior['banks'][previous]['path']).read_bytes()==data:
        target=root/prior['banks'][previous]['path']
    else:
        target=root/('pcie_symbolic_retirement_binary_'+name+'_r1.txt')
        with target.open('xb') as f:f.write(data)
    assert target.read_bytes()==data
    banks[name]=dict(source=source.name,source_sha256=hashlib.sha256(original).hexdigest(),
        path=target.name,sha256=hashlib.sha256(data).hexdigest(),source_rows=len(lines),rows=len(selected),
        scenarios=len({v.split()[0] for v in selected}),excluded_whole_scenarios=excluded,unchanged=not excluded)
record=dict(banks=banks,total_rows=sum(v['rows'] for v in banks.values()),
    total_source_rows=sum(v['source_rows'] for v in banks.values()),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    selector=dict(path='tools/generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3.py',
        sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_symbolic_ced_binary_vectors_r3.py').read_bytes()).hexdigest()),
    scope='All fifteen original/full R1 banks; exclude complete cold-isolated X/Z-input scenarios only. No filter by fault or malformed binary traffic. RTL four-state evidence remains required.')
assert record['total_source_rows']==182806 and record['total_rows']==174184
assert banks['range_return']['rows']==277 and banks['range_return']['unchanged']
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(total_rows=record['total_rows'],banks={k:dict(rows=v['rows'],excluded_scenarios=len(v['excluded_whole_scenarios'])) for k,v in banks.items()})))
