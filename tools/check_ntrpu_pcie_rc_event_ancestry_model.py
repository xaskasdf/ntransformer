"""Preflight a parallel EOP-event selector against the frozen sequential event model."""
import argparse
import hashlib
import json
from pathlib import Path
from generate_ntrpu_pcie_rc_ownership_vectors import vectors, expected


def direct(flags, bitmap, retained, data, user):
    if flags != 7:
        return 0
    thermometers = {0: 0, 1: 1, 3: 2, 7: 3, 15: 4}
    starts = {4*((user >> (68+2*i)) & 3) for i in range(thermometers.get((user >> 64) & 15, 0))}
    ends = {(user >> (80+4*i)) & 15: i for i in range(thermometers.get((user >> 76) & 15, 0))}
    headers = {}
    for lane in starts:
        bits = data >> (32*lane)
        tag = bits >> 64 & 255
        headers[lane] = dict(tag=tag, terminal=bool(bits >> 30 & 1),
                             normal=not (bits >> 12 & 15 or bits >> 43 & 15),
                             base=bool(bitmap >> tag & 1), origin=lane//4)
    # Parallel ownership already measured in R7; no release-result feedback.
    for lane,h in headers.items():
        h['owned'] = h['base'] and not any(p < lane and q['base'] and q['terminal'] and q['tag']==h['tag']
                                         for p,q in headers.items())
    initial = dict(tag=retained & 255, owned=bool(retained >> 8 & 1),
                   terminal=bool(retained >> 9 & 1), normal=bool(retained >> 10 & 1) and not (user >> 96 & 1), origin=-1)
    events = []
    for lane,slot in ends.items():
        source = max((p for p in starts if p <= lane), default=-1)
        context = headers[source] if source >= 0 else initial
        first = not any(source <= p < lane for p in ends)
        if first and context['owned'] and not context['terminal'] and context['normal']:
            events.append(dict(context, lane=lane, slot=slot))
    result = 0
    for lane,h in headers.items():
        prior = [e for e in events if e['lane'] < lane and e['tag']==h['tag']]
        if prior:
            newest = max(prior, key=lambda e:e['lane'])
            result |= 1 << (lane//4)
            result |= newest['slot'] << (4+2*(lane//4))
            mask = sum(1 << origin for origin in {e['origin'] for e in prior} if origin >= 0)
            result |= mask << (780+4*(lane//4))
            result |= int(any(e['origin']==-1 for e in prior)) << (796+lane//4)
        result |= int(h['owned']) << (800+lane//4)
        if h['owned'] and h['terminal']:
            result |= 1 << (804+lane//4)
            result |= h['tag'] << (808+8*(lane//4))
    for tag in {e['tag'] for e in events}:
        last = max((e for e in events if e['tag']==tag), key=lambda e:e['lane'])
        result |= 1 << (12+tag)
        result |= last['slot'] << (268+2*tag)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    checked=0
    for row in vectors():
        actual,reference=direct(*row),expected(*row)
        assert actual==reference, dict(index=checked,input=row,actual=hex(actual),reference=hex(reference))
        checked+=1
    assert checked==155168
    result=dict(status='PASS_BINARY_EOP_EVENT_ANCESTRY_MODEL',rows=checked,output_bits=840,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_rc_ownership_vectors.py').read_bytes()).hexdigest(),
                scope='Independent direct EOP events vs sequential event list, binary inputs only. No candidate RTL, four-state proof, synthesis or own-netlist qualification.')
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
