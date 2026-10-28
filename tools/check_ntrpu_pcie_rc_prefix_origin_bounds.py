"""Exhaust decoded SOP/event placements; this is not universal RTL equivalence."""
import argparse
import hashlib
import json
import re
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(model):
    tested=0
    mask_union=0
    for sop_mask in range(16):
        for event_lane in range(16):
            origin=model(sop_mask,event_lane)
            assert -1<=origin<=3
            if origin>=0:
                assert sop_mask&(1<<origin)
                assert 4*origin<=event_lane
            for header in range(4):
                # Write eligibility, first-EOP and tag matches only remove events.
                if event_lane<4*header:
                    assert origin<header
                    if origin>=0:mask_union|=1<<(4*header+origin)
                tested+=1
    return tested,mask_union

def latest_source(sop_mask,event_lane):
    candidates=[h for h in range(4) if sop_mask&(1<<h) and 4*h<=event_lane]
    return max(candidates,default=-1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('selector',type=Path)
    p.add_argument('vectors',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    source=a.selector.read_text()
    required=[
        'event_lane[slot]=0; event_origin[slot]=-1;',
        'for(lane=0;lane<16;lane=lane+1)',
        'event_lane[end_slot[lane]]=lane;',
        'for(source_header=0;source_header<4;source_header=source_header+1)',
        'if(sop[source_header*4] && source_header*4<=event_lane[slot])',
        'event_origin[slot]=source_header;',
        'if(event_write[scan] && event_lane[scan]<header_index*4 &&',
        'else header_prefix_mask[header_index*4+event_origin[scan]]=1;',
    ]
    assert all(piece in source for piece in required)
    origin_assignments=re.findall(r'event_origin\[[^\]]+\]\s*=\s*([^;]+);',source)
    assert origin_assignments==['-1','source_header']
    tested,allowed=check(latest_source)
    assert tested==1024 and allowed==0x7310
    reference_union=0;rows=0
    for line in a.vectors.read_text().splitlines():
        mask=(int(line.split()[-1],16)>>780)&65535
        assert mask&~allowed==0
        reference_union|=mask;rows+=1
    assert rows==155168 and reference_union==allowed
    controls=[]
    for label,bad in [
        ('future_origin',lambda m,l: min(3,latest_source(m,l)+1)),
        ('ignore_sop_presence',lambda m,l:l//4),
    ]:
        try:check(bad)
        except AssertionError:controls.append(dict(name=label,rejected=True))
        else:raise AssertionError('Bad origin model accepted: '+label)
    result=dict(status='PASS_DECODED_RC_ORIGIN_BOUND',selector=str(a.selector),
        selector_sha256=sha(a.selector),checker_sha256=sha(Path(__file__)),
        decoded_sop_maps=16,event_lanes=16,header_positions=4,comparisons=tested,
        allowed_prefix_mask=f'{allowed:04x}',reference_rows=rows,reference_vectors_sha256=sha(a.vectors),
        controls=controls,rtl_input_binding=required,
        scope='Exhaustive finite placement model, bound to the existing selector source and binary reference outputs. Eligibility can only suppress an event. Does not prove arbitrary RTL rewrites or replace four-state RTL and own-netlist replay.',
        new_rtl_qualified=False)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
