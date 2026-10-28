"""Malformed-but-retained raw traffic: byte overflow then in-range sticky fault."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors,rq,rc

v=Vectors();v.case_begin('byte_range_return_without_tag_read_range')
overrides=[]

def oracle(tags,reads,byte_count,requests,completed,bitmap):
    # Closed-form packet counts and spans, independent of the RTL candidate.
    row=v.rows[-1].split();assert len(row)==28 and row[17]=='1'
    before=row[18:]
    row[18:] = list(map(str,[1,tags,reads,byte_count,0,0,requests,completed,0]))+[format(bitmap,'x')]
    v.rows[-1]=' '.join(row)
    overrides.append(dict(row=len(v.rows)-1,prior_tail=before,expected_tail=row[18:]))

byte_count=0
for tag in range(256):
    count=2047 if tag<2 else 1024
    v.step(rq_beat=rq(tag,0,count=count),expect_error=True)
    byte_count+=8188 if tag<2 else 4096
    oracle(tag+1,tag+1,byte_count,tag+1,0,(1<<(tag+1))-1)
peak=len(v.rows)-1
assert byte_count==1056760 and peak==259
v.step(rc_beat=rc(0,address=0,remaining=8188,terminal=True),expect_error=True)
returned=len(v.rows)-1;byte_count-=8188
assert byte_count==1048572 and returned==260
oracle(255,255,byte_count,256,1,((1<<256)-1)^1)
for _ in range(8):
    v.step();oracle(255,255,byte_count,256,1,((1<<256)-1)^1)
v.end()
cold=len(v.rows);assert cold==272
v.case_begin('cold_clear_after_persistent_byte_range');v.end()
assert len(v.rows)==277

root=Path(sys.argv[1]);target=root/'pcie_symbolic_retirement_range_r1.txt'
with target.open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(v.rows)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(path=target.name,sha256=sha(target),rows=len(v.rows),scenarios=v.scenarios,
    witnesses=dict(peak=peak,returns=[returned,returned+1,returned+2],cold=cold),
    closed_form=dict(large_span=8188,large_reads=2,normal_span=4096,normal_reads=254,
        peak_bytes=1056760,returned_bytes=1048572,maximum_bytes=1052672),
    oracle_overrides=overrides,generator_sha256=sha(Path(__file__)),
    dependencies={'generate_ntrpu_pcie_inventory_vectors_r2.py':sha(Path(__file__).with_name('generate_ntrpu_pcie_inventory_vectors_r2.py'))},
    scope='Raw RQ/RC; malformed count diagnostics retained. Closed-form counter/bitmap oracles override Python untrusted counters after the first malformed request. R7/S1/CED/A/B/late/R comparisons remain independent.')
with target.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(target),rows=277,sha256=sha(target),witnesses=manifest['witnesses'])))
