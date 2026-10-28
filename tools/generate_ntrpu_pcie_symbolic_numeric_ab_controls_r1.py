"""Raw packets separating disabled delivery from zero bytes, plus all numeric cold distances."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors,place,rc_header
from generate_ntrpu_pcie_symbolic_value_vectors import request
from generate_ntrpu_pcie_symbolic_ced_vectors import close_without_drain


def generate():
    v = Vectors()
    for offset in (1,0,2,3):
        for count in (1,0,7):
            for error in (9,2):
                v.case_begin(f'disabled_delivery_offset_{offset}_count_{count}_error_{error}')
                v.step(rq_beat=request(80,count=8,first=(15<<offset)&15))
                data = place((0,rc_header(80,count,32-offset,0x100+offset,True,error)+bytes(count*4)))
                user = 1<<64 | 1<<76 | (2+count)<<80
                v.step(rc_beat=(data,user),expect_error=True)
                for _ in range(4):
                    v.step()
                v.end()
    for distance in range(5):
        v.case_begin(f'cold_cancels_numeric_distance_{distance}')
        v.step(rq_beat=request(81,count=8))
        for _ in range(distance):
            v.step()
        close_without_drain(v)
    v.case_begin('final_cold_rearm'); v.end()
    assert len(v.rows)==352 and len(v.scenarios)==30
    return v


if __name__=='__main__':
    out = Path(sys.argv[1]); v = generate(); text = '\n'.join(v.rows)+'\n'
    with out.open('x',encoding='utf-8',newline='\n') as f:
        f.write(text)
    manifest = dict(rows=len(v.rows),scenarios=v.scenarios,vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_rows=0,
                    first_distinguishing_origin=5,
                    first_disabled_delivery=dict(tag=80,remaining=31,address=257,words=1,enabled=0,
                                                  expected_delivery=0,mutant_delivery=3,terminal_removed=31),
                    scope='24 raw error-completion cases with owned nonzero metadata and disabled delivery; five cold distances0..4 and rearm. R7/probe/model/S1/CED comparisons remain active on every row, including after fault.')
    with out.with_suffix('.json').open('x') as f:
        json.dump(manifest,f,indent=2); f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
