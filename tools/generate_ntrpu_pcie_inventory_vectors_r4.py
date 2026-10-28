"""Exercise all four RC metadata writes and forwarding into later RC/RQ packets."""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r3 import generate as baseline
from generate_ntrpu_pcie_inventory_vectors_r2 import place,rq_header,rc_header,rq,rc


def rq_pair(a,addr_a,b,addr_b):
    return (place((0,rq_header(a,addr_a,1)),(32,rq_header(b,addr_b,1))),
            0xff | 3<<20 | 2<<24 | 3<<26 | 3<<28 | 11<<32)


def rc_four(packets):
    assert len(packets)==4
    return (place(*[(16*i,p) for i,p in enumerate(packets)]),
            sum(15<<(16*i+12) for i in range(4)) | 15<<64 | 0xe4<<68 | 15<<76 | 0xfb73<<80)


def generate(native):
    v=baseline(native)
    for terminal in (False,True):
        v.case_begin('four_same_tag_splits_'+('terminal_reuse' if terminal else 'all_nonterminal_commit'))
        total=16 if terminal else 20
        v.step(rq_beat=rq(31,count=total//4))
        packets=[rc_header(31,1,total-4*i,0x100+4*i,terminal and i==3)+bytes([i])*4 for i in range(4)]
        v.step(rc_beat=rc_four(packets),rq_beat=rq(31,0x200) if terminal else None)
        if terminal:v.step(rc_beat=rc(31,address=0x200))
        else:v.step(rc_beat=rc(31,address=0x110))
        v.end()
    v.case_begin('four_distinct_nonterminal_updates_plus_RQ2')
    for tag in range(4):v.step(rq_beat=rq(tag,0x100+tag*0x10,count=2))
    v.step(rc_beat=rc_four([rc_header(i,1,8,0x100+i*0x10,False)+bytes([i])*4 for i in range(4)]),
           rq_beat=rq_pair(100,0x200,101,0x300))
    v.step(rc_beat=rc_four([rc_header(i,1,4,0x104+i*0x10)+bytes([i+4])*4 for i in range(4)]))
    v.step(rc_beat=rc(101,address=0x300));v.step(rc_beat=rc(100,address=0x200));v.end()
    v.case_begin('continuation_three_same_tag_splits_and_RQ2_reuse')
    v.step(rq_beat=rq(255,count=17))
    v.step(rc_beat=(place((0,rc_header(255,14,68,0x100,False)),(12,bytes(range(52)))),
                    ((1<<52)-1)<<12 | 1<<64))
    pieces=[(0,bytes(range(52,56)))]
    for i in range(3):pieces.append((16*(i+1),rc_header(255,1,12-4*i,0x138+4*i,i==2)+bytes([i])*4))
    user=15 | sum(15<<(16*(i+1)+12) for i in range(3)) | 7<<64 | 0x39<<68 | 15<<76 | 0xfb70<<80
    v.step(rc_beat=(place(*pieces),user),rq_beat=rq_pair(255,0x200,0,0x300))
    v.step(rc_beat=rc(0,address=0x300));v.step(rc_beat=rc(255,address=0x200));v.end()
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native_evidence',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();raw=a.native_evidence.read_bytes();native=json.loads(raw);v=generate(native)
    with a.output.open('x') as f:f.write('\n'.join(v.rows)+'\n')
    manifest=dict(schema=1,rows=len(v.rows),scenarios=v.scenarios,
        vectors_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_dependencies={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ['generate_ntrpu_pcie_inventory_vectors_r2.py','generate_ntrpu_pcie_inventory_vectors_r3.py']},
        native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with a.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
