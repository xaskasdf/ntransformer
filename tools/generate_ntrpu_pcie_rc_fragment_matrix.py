"""All physical SOP subsets and up-to-four EOP positions, with/without context.

Inputs deliberately include malformed packet arrangements. Expected outputs
come from the separate sequential packet replay and unchanged R1 reference.
"""
import hashlib,json,itertools
from pathlib import Path
from generate_ntrpu_pcie_rc_event_boundaries import header,packet


def generate():
    rows=[];phases=[];cases=0
    eops=[positions for count in range(5) for positions in itertools.combinations(range(16),count)]
    assert len(eops)==2517
    counts=[0,1,4,13,32,1024,1025,2047]
    def emit(phase,beat=None,rst=1):
        b=dict(data=0,user=0,owned=0,metadata=0,expected_valid=0)
        if beat:b.update(beat)
        rows.append(f'{phase} {rst} 1 {int(beat is not None)} 1 {b["owned"]:x} 01a0 {len(rows):08x} {b["expected_valid"]:x} {b["metadata"]:030x} {b["data"]:0128x} {b["user"]:041x}')
    for retained in [0,1]:
        start=len(rows)
        for mask in range(16):
            starts=[q for q in range(4) if mask&(1<<q)]
            for ends in eops:
                emit(retained,rst=0)
                if retained:emit(retained,packet(9,32)[0])
                data=user=metadata=0
                user|=((1<<len(starts))-1)<<64 | ((1<<len(ends))-1)<<76
                for slot,q in enumerate(starts):
                    count=counts[(cases+q)%len(counts)];address=0x100+4*q+(cases+q)%4
                    data|=header(20+q,count,4*count,address)<<(128*q)
                    metadata|=(4*count | address<<13)<<(30*q)
                    user|=q<<(68+2*slot)
                for slot,lane in enumerate(ends):user|=lane<<(80+4*slot)
                # Alternating enable patterns exercise both active/inactive bytes.
                for lane in range(16):
                    if lane%4==3 or cases%8==7:user|=15<<(4*lane)
                emit(retained,dict(data=data,user=user,owned=mask,expected_valid=mask,metadata=metadata))
                emit(retained);cases+=1
        phases.append(dict(id=retained,name='all_sop_eop_retained_'+str(retained),start=start,rows=len(rows)-start))
    text='\n'.join(rows)+'\n'
    return text,dict(rows=len(rows),phases=phases,cases=cases,sop_subsets=16,eop_position_sets=2517,
        retained_context_modes=2,count_samples=counts,all_physical_sop_eop_combinations=True,
        binary_inputs_only=True,latency_cycles=2,vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
        scope='Ordered pointer encodings, all physical SOP subsets and EOP sets of cardinality0..4. Includes invalid packet layouts. Count/enable samples are distributed, not an exhaustive cross product; no allocation bank or native endpoint.')


if __name__=='__main__':
    text,manifest=generate();base=Path(__file__).resolve().parents[1]/'fpga/tb/data/pcie_rc_fragment_matrix'
    with base.with_suffix('.txt').open('x',newline='\n') as f:f.write(text)
    with base.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=manifest['rows'],cases=manifest['cases'],sha256=manifest['vectors_sha256'])))
