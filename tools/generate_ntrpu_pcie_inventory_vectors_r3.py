"""Extend the reviewed RTL bank to its maximum 257 * 4096 pending-byte bound."""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import generate as baseline,place,rq_header,rc_header,rq


def generate(native):
    v=baseline(native)
    v.case_begin('maximum_pending_bytes_257x4KiB')
    for pair in range(128):
        data=place((0,rq_header(2*pair,2*pair*4096,1024)),
                   (32,rq_header(2*pair+1,(2*pair+1)*4096,1024)))
        v.step(rq_beat=(data,0xffff | 3<<20 | 2<<24 | 3<<26 | 3<<28 | 11<<32))
    pattern=bytes(n%251 for n in range(4096))
    v.step(rc_beat=(place((0,rc_header(0,1024,4096,0)),(12,pattern[:52])),((1<<52)-1)<<12 | 1<<64),
           rq_beat=rq(0,1<<32,count=1024))
    assert len(v.model.live)==257 and sum(e['remaining'] for e in v.model.live.values())==1052672
    for offset in range(52,len(pattern),64):
        chunk=pattern[offset:offset+64];user=(1<<len(chunk))-1
        if offset+64>=len(pattern):user|=1<<76 | (len(chunk)//4-1)<<80
        v.step(rc_beat=(place((0,chunk)),user))
    for tag in reversed(range(256)):
        raw=rc_header(tag,1024,4096,0)+pattern
        for index,offset in enumerate(range(0,len(raw),64)):
            chunk=raw[offset:offset+64]
            user=sum(1<<n for n in range(len(chunk)) if offset+n>=12) | (1<<64 if index==0 else 0)
            if offset+64>=len(raw):user|=1<<76 | (len(chunk)//4-1)<<80
            v.step(rc_beat=(place((0,chunk)),user))
    v.end()
    return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native_evidence',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();raw=a.native_evidence.read_bytes();native=json.loads(raw);v=generate(native)
    with a.output.open('x') as f:f.write('\n'.join(v.rows)+'\n')
    manifest=dict(schema=1,rows=len(v.rows),scenarios=v.scenarios,
        vectors_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_dependency_sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_inventory_vectors_r2.py').read_bytes()).hexdigest(),
        native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with a.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
