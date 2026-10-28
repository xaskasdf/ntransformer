"""Directed C/E/D lifetime, cancellation and adjacent-origin raw traffic."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, binary, place, rc_header
from generate_ntrpu_pcie_symbolic_value_vectors import request, completion, start, tail, change_unknown_to_z
from ntrpu_pcie_straddle import LogicWord


def close_without_drain(v):
    v.case.update(last_vector=len(v.rows)-1,reference_fault=v.fault,
        peak_tags=v.model.peak_tags,peak_reads=v.model.peak_reads,peak_span_bytes=v.model.peak_span_bytes,
        intentionally_cold_cancelled=True)


def generate():
    v=Vectors()
    for distance in (0,1,2):
        v.case_begin('cold_after_request_distance_'+str(distance))
        v.step(rq_beat=request(41,count=16,tc=3,attr=2))
        for _ in range(distance): v.step()
        close_without_drain(v)
        first=len(v.rows)
        v.case_begin('rearm_after_request_distance_'+str(distance))
        # Current raw input activity during cold still revokes quiet, although
        # the request itself is cancelled and must never update any table.
        fields=v.rows[first].split();data,user=request(99,0x700)
        fields[8]='1';fields[10]=binary(data,512);fields[11]=format(user,'x')
        v.rows[first]=' '.join(fields)
        v.step(rq_beat=request(41,0x300,tc=6,attr=1))
        v.step(rc_beat=completion(41,address=0x300,tc=6,attr=1))
        v.end()
    for distance in (0,1,2):
        v.case_begin('cold_after_detached_context_distance_'+str(distance))
        v.step(rq_beat=request(42,count=16,tc=4,attr=3))
        v.step(rc_beat=start(42,64,True,4,3),rq_beat=request(42,0x200,tc=2,attr=1))
        for _ in range(distance): v.step()
        close_without_drain(v)
        v.case_begin('rearm_after_context_distance_'+str(distance))
        v.step(rq_beat=request(42,0x400,tc=1,attr=2))
        v.step(rc_beat=completion(42,address=0x400,tc=1,attr=2))
        v.end()
    v.case_begin('warm_reset_and_link_loss_do_not_cancel_origins')
    v.step(rq_beat=request(43,count=16,tc=5,attr=1))
    v.step(rc_beat=start(43,64,True,5,1),rq_beat=request(43,0x200,tc=6,attr=3),reset=1,expect_error=True)
    v.step(rc_beat=tail(),link=0,expect_error=True)
    v.step(rc_beat=completion(43,address=0x200,tc=6,attr=3),expect_error=True)
    v.end()
    v.case_begin('adjacent_tag_reuse_crosses_sequence_wrap')
    # Keep wrap inside an uninterrupted busy stream at raw row 128.
    while len(v.rows)<124: v.step()
    v.step(rq_beat=request(44,0x100,tc=0,attr=0))
    previous_address=0x100;previous_tc=0;previous_attr=0
    for i in range(1,65):
        address=0x100+4*i;tc=i&7;attr=i&3
        v.step(rc_beat=completion(44,address=previous_address,tc=previous_tc,attr=previous_attr),
               rq_beat=request(44,address,tc=tc,attr=attr))
        previous_address,previous_tc,previous_attr=address,tc,attr
    v.step(rc_beat=completion(44,address=previous_address,tc=previous_tc,attr=previous_attr))
    v.end()
    for unknown in ('x','z'):
        v.case_begin('unknown_tag_retained_context_'+unknown)
        v.step(rq_beat=request(45,count=8))
        data=place((0,rc_header(45,1,32,0x100,False)+b'abcd'))
        user=15<<12 | 1<<64 | 1<<76 | 3<<80
        v.step(rc_beat=(LogicWord(data,((1<<512)-1) ^ (1<<64),512),user),expect_error=True)
        if unknown=='z': change_unknown_to_z(v,14,64)
        v.step();v.step()
        v.step(rc_beat=(place((0,rc_header(45,1,32,0x100,False)+b'abcd')),user),expect_error=True)
        v.step(rc_beat=completion(45,bytes(28),0x104,28),expect_error=True)
        v.end()
    return v


if __name__=='__main__':
    output=Path(sys.argv[1]);v=generate();text='\n'.join(v.rows)+'\n'
    with output.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    manifest=dict(rows=len(v.rows),scenarios=v.scenarios,vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_rows=0,
        scope='Directed C/E/D raw stage lifetime; 0/1/2-cycle cold distances, cold raw activity, warm reset/link preservation, adjacent reuse/wrap and unknown tag context. Independent R7 remains oracle after fault.')
    with output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=manifest['vectors_sha256'])))
