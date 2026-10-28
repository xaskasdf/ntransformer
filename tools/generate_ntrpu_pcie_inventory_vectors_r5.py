"""Extend the full PCIe bank with simultaneous tag-release/allocation hazards."""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r4 import generate as baseline, rq_pair, rc_four
from generate_ntrpu_pcie_inventory_vectors_r2 import rq, rc, rc_header


def generate(native):
    v=baseline(native)
    v.case_begin('full_256_RC4_release_RQ2_cross_order_reuse')
    for i in range(0,256,2):
        v.step(rq_beat=rq_pair(i,4*i,i+1,4*(i+1)))
    addresses={i:4*i for i in range(256)}
    for tags,reuse,new_addresses in [([0,1,2,3],[3,0],[0xc00,0xc40]),
                                      ([4,5,6,7],[1,2],[0xc80,0xcc0])]:
        v.step(rc_beat=rc_four([rc_header(i,1,4,addresses[i])+i.to_bytes(4,'little') for i in tags]),
               rq_beat=rq_pair(reuse[0],new_addresses[0],reuse[1],new_addresses[1]))
        for i in tags:del addresses[i]
        addresses.update(zip(reuse,new_addresses))
        assert len(v.model.tags)==len(addresses)
        assert set(tag for function,tag in v.model.tags)==set(addresses)
    assert len(addresses)==252
    tags=sorted(addresses,reverse=True)
    for i in range(0,len(tags),4):
        v.step(rc_beat=rc_four([rc_header(tag,1,4,addresses[tag])+tag.to_bytes(4,'little') for tag in tags[i:i+4]]))
    v.end()

    v.case_begin('fault_released_tag_duplicate_RQ2')
    v.step(rq_beat=rq(42))
    v.step(rc_beat=rc(42),rq_beat=rq_pair(42,0x200,42,0x300),expect_error=True)
    v.end()

    v.case_begin('fault_duplicate_terminal_same_beat')
    for tag in (5,6,7):v.step(rq_beat=rq(tag))
    v.step(rc_beat=rc_four([rc_header(tag,1,4,0x100)+b'abcd' for tag in (5,5,6,7)]),expect_error=True)
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
            for name in ['generate_ntrpu_pcie_inventory_vectors_r2.py','generate_ntrpu_pcie_inventory_vectors_r3.py','generate_ntrpu_pcie_inventory_vectors_r4.py']},
        native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with a.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
