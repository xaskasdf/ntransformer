"""Retain the full bank and exercise malformed descriptors at arithmetic bounds."""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r5 import generate as baseline
from generate_ntrpu_pcie_inventory_vectors_r2 import place,rq,rc_header


def generate(native):
    v=baseline(native)
    # Illegal counts still reach the observer's diagnostic arithmetic. Their
    # bounds must not overflow narrower guards or make an invalid inventory empty.
    for count,first,last in [(0,1,0),(0,8,1),(1025,15,15),(2047,15,15)]:
        v.case_begin('fault_width_rq_'+str(count)+'_'+str(first))
        v.step(rq_beat=rq(91,0,count=count,first=first,last=last),expect_error=True)
        v.end()
    for count in (0,1025,2047):
        v.case_begin('fault_width_rc_'+str(count))
        v.step(rq_beat=rq(93))
        v.step(rc_beat=(place((0,rc_header(93,count,4,0x100)+b'abcd')),
                        15<<12 | 1<<64 | 1<<76 | 3<<80),expect_error=True)
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
            for name in ['generate_ntrpu_pcie_inventory_vectors_r2.py','generate_ntrpu_pcie_inventory_vectors_r3.py','generate_ntrpu_pcie_inventory_vectors_r4.py','generate_ntrpu_pcie_inventory_vectors_r5.py']},
        native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with a.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
