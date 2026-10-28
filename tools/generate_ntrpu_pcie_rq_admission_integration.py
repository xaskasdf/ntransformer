"""Raw-input RQ ordering, duplicate-pointer, capacity and RC-before-RQ cases."""
import argparse, hashlib, json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq, rc, rq_header, rc_header


def pair(pointers=(0,2),tags=(34,35),writes=(False,False),addresses=(0x800,0x804),mask=3):
    data=bytearray(64)
    # Higher declared slot wins when both pointers target the same physical header.
    for p,tag,write,address in zip(pointers,tags,writes,addresses):
        if write and p<3:data[16*p+16:16*p+20]=b'abcd'
        data[16*p:16*p+16]=rq_header(tag,address,1,write)
    user=0xff | mask<<20 | pointers[0]<<22 | pointers[1]<<24 | mask<<26
    for slot,(p,write) in enumerate(zip(pointers,writes)):
        user|=min(15,4*p+3+int(write))<<(28+4*slot)
    return int.from_bytes(data,'little'),user


def completions(items):
    pieces=[];user=0
    for slot,(tag,address) in enumerate(items):
        pieces.append((16*slot,rc_header(tag,1,4,address)+b'abcd'))
        user|=15<<(16*slot+12) | 1<<(64+slot) | slot<<(68+2*slot)
        user|=1<<(76+slot) | (4*slot+3)<<(80+4*slot)
    return place(*pieces),user


def finish(v):
    if not v.fault:
        entries=[(tag,entry['next_address']) for (_,tag),entry in v.model.tags.items()]
        for start in range(0,len(entries),4):v.step(rc_beat=completions(entries[start:start+4]))
    else:
        # Continue genuine traffic after the malformed beat; R7, not Python's
        # partially rejected state, remains the complete post-fault reference.
        v.step(rq_beat=rq(180,0x300))
        v.step(rc_beat=rc(180,address=0x300))
    v.end()


def generate():
    v=Vectors()
    for p0 in range(4):
        for p1 in range(4):
            for same in (False,True):
                for kinds in range(4):
                    writes=(bool(kinds&1),bool(kinds&2))
                    tags=(34,34 if same else 35)
                    v.case_begin(f'rq_order_{p0}_{p1}_{int(same)}_{kinds}')
                    bad=(p0,p1)!=(0,2) or (same and kinds==0)
                    v.step(rq_beat=pair((p0,p1),tags,writes),expect_error=bad)
                    finish(v)
    for stored in (254,255,256):
        for pointers in ((0,2),(2,0),(1,3),(1,1)):
            for profile in range(4):
                v.case_begin(f'rq_capacity_{stored}_{pointers[0]}_{pointers[1]}_{profile}')
                for tag in range(0,stored-1,2):
                    v.step(rq_beat=pair(tags=(tag,tag+1),addresses=(0x100+4*tag,0x104+4*tag)))
                if stored%2:v.step(rq_beat=rq(stored-1,0x100+4*(stored-1)))
                if profile==0:tags=(254,255);writes=(False,False);releases=None;bad=stored!=254
                elif profile==1:tags=(255,255);writes=(False,False);releases=None;bad=True
                elif profile==2:tags=(0,1);writes=(False,False);releases=completions([(0,0x100),(1,0x104)]);bad=False
                else:tags=(0,255);writes=(True,False);releases=None;bad=stored==256
                v.step(rq_beat=pair(pointers,tags,writes),rc_beat=releases,expect_error=bad or pointers!=(0,2))
                finish(v)
    for mask in range(4):
        for p0 in range(4):
            for p1 in range(4):
                v.case_begin(f'rq_mask_{mask}_{p0}_{p1}')
                valid=(mask==1 and p0==0) or (mask==3 and (p0,p1)==(0,2))
                v.step(rq_beat=pair((p0,p1),mask=mask),expect_error=not valid)
                finish(v)
    return v


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();v=generate();text='\n'.join(v.rows)+'\n'
    manifest=dict(rows=len(v.rows),scenarios=v.scenarios,vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_rows=0,
        scope='Raw RQ admission supplement; independent Python expectations before malformed traffic, immutable R7 RTL and full internal table comparison after faults. No internal DUT state injection.')
    with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    with args.output.with_suffix('.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))


if __name__=='__main__':main()
