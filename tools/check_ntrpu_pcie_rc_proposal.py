"""Independent integer arithmetic model for each accepted descriptor slot."""
import re


def calculate(header):
    bc=(header>>16)&8191; address=header&4095; count=(header>>32)&2047
    normal=((header>>12)&15)==0 and ((header>>43)&7)==0 and ((header>>46)&1)==0
    if not normal:return dict(normal=0,delivered=0,remaining=0,address=0,length=0,terminal=0)
    # Model the reference's unsigned 15-bit comparison explicitly.
    available=(4*count-(address%4))%32768
    delivered=min(available,bc)
    length=int(count==0 or count>1024 or delivered==0 or count!=(address%4+delivered+3)//4)
    terminal=int(((header>>30)&1)!=int(delivered==bc))
    return dict(normal=1,delivered=delivered,remaining=bc-delivered,address=(address+delivered)%4096,length=length,terminal=terminal)


def review(log,vectors,manifest):
    if re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|FAIL )',log,re.M):raise ValueError('Simulation diagnostic')
    if log.splitlines().count('RC_PROPOSAL_SCOPE lanes=4 latency=1 native=0')!=1:raise ValueError('Wrong scope')
    lines=[s for s in log.splitlines() if s.startswith('RC_PROPOSAL_ROW ')]
    inputs=vectors.splitlines()
    if len(lines)!=len(inputs) or len(lines)!=manifest['rows']:raise ValueError('Missing/duplicate rows')
    if log.splitlines().count(f'RC_PROPOSAL_SUITE_PASS rows={len(inputs)}')!=1:raise ValueError('Missing suite completion')
    phases={p['id']:0 for p in manifest['phases']};counts=set();offsets=set();all_four=resets=normal_total=non_normal=0
    keys=['seen','valid','headers','delivered','remaining','address','length','terminal']
    for index,(line,vector) in enumerate(zip(lines,inputs)):
        phase,rst,valid,data=vector.split();phase=int(phase);rst=int(rst,16);valid=int(valid,16);data=int(data,16)
        phases[phase]+=1;expected=dict.fromkeys(keys,0)
        if not rst:resets+=1
        else:
            all_four+=int(valid==15)
            for q in range(4):
                if valid&(1<<q):
                    h=(data>>(128*q))&((1<<96)-1);r=calculate(h)
                    expected['seen']|=1<<q;expected['headers']|=h<<(96*q)
                    expected['valid']|=r['normal']<<q
                    for k,width in [('delivered',13),('remaining',13),('address',12),('length',1),('terminal',1)]:expected[k]|=r[k]<<(width*q)
                    counts.add((h>>32)&2047);offsets.add(h&3)
                    normal_total+=r['normal'];non_normal+=1-r['normal']
        match=re.fullmatch(r'RC_PROPOSAL_ROW row=(\d+) phase=(\d+) '+r' '.join(k+r'=([0-9a-f]+)' for k in keys),line)
        if not match:raise ValueError('Malformed/unknown output')
        values=match.groups()
        if int(values[0])!=index or int(values[1])!=phase:raise ValueError('Sequence mismatch')
        actual={k:int(v,16) for k,v in zip(keys,values[2:])}
        if actual!=expected:raise ValueError(f'Proposal mismatch row {index}: {actual} != {expected}')
    if phases!={p['id']:p['rows'] for p in manifest['phases']}:raise ValueError('Phase coverage missing')
    if counts!=set(range(2048)) or offsets!=set(range(4)) or all_four<4096 or not non_normal or resets<2:raise ValueError('Boundary coverage missing')
    return dict(status='PASS_PCIE_RC_PROPOSAL_RTL_REFERENCE',rows=len(inputs),phases=phases,
        normal_headers=normal_total,non_normal_headers=non_normal,all_four_cycles=all_four,reset_cycles=resets,
        all_count_values=2048,all_low_address_offsets=4,latency_cycles=1,known_binary_inputs_only=True)
