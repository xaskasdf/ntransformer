"""Exercise malformed RQ pointer order and aliases around allocation dependencies."""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r6 import generate as baseline
from generate_ntrpu_pcie_inventory_vectors_r2 import place,rq_header


def generate(native):
    v=baseline(native)
    # With no preceding partial packet, exactly (0,2) is a legal two-SOP map.
    # Retain duplicates, reversed pointers and disallowed DWORD positions.
    for first in range(4):
        for second in range(4):
            if (first,second)==(0,2):continue
            v.case_begin(f'fault_rq_pointer_{first}_{second}')
            data=place((16*first,rq_header(81,0x100,1)),
                       (16*second,rq_header(82,0x200,1)))
            user=(0xff | 3<<20 | first<<22 | second<<24 | 3<<26 |
                  (4*first+3)<<28 | (4*second+3)<<32)
            v.step(rq_beat=(data,user),expect_error=True)
            v.end()
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native_evidence',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();raw=a.native_evidence.read_bytes();native=json.loads(raw);v=generate(native)
    with a.output.open('x') as f:f.write('\n'.join(v.rows)+'\n')
    manifest=dict(schema=1,rows=len(v.rows),scenarios=v.scenarios,
        vectors_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_dependencies={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in [f'generate_ntrpu_pcie_inventory_vectors_r{n}.py' for n in range(2,7)]},
        native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with a.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
