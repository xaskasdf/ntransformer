"""Review actual wrapper control and symbolic AXI peers, never numerical inference."""
import itertools,re
PLAN=[0,0]+[m for i in list(range(1,61))+list(range(100,556)) for m in (i,0)]
PAIRS=list(itertools.combinations(range(16),2))
ABORT={15,37,38,39,40,41}
REVOKE=set(range(30,37))|set(range(46,50))|{60}
STIM={12,13,14,26,27,28,29,42,43,44,45,50,51,52,53,54,55}
SCHEMA={'BEGIN':'cycle job mode sequence epoch','REGION':'cycle job index address base bytes','ADMIT':'cycle job',
 'START':'cycle job owner sequence epoch','SWITCH':'cycle job owner','AR':'cycle job owner address',
 'R':'cycle job owner address data','AW':'cycle job owner address','W':'cycle job owner data','B':'cycle job owner address',
 'RELEASE':'cycle job owner','COMPLETE':'cycle job status valid sequence epoch address bytes attention ffn',
 'STIM':'cycle job mode','RESTORE':'cycle job mode','REVOKE':'cycle job mode','ABORT':'cycle job mode',
 'RESET':'cycle job','ACK':'cycle job source child','IDLE':'cycle job',
 'PASS':'seed jobs publics offered resets reads writes source_acks child_acks cycles'}
HEX={'sequence','epoch','address','base','data'}
SPANS=[65536,4456448,4096,8192,8192,8192,8192,4096,17825792,17825792,17825792,65536,65536,16384,8192,8192]
def early(m):return 1<=m<=14 or 100<=m<=219 or 340<=m<=555
def source(m):return not early(m) and m not in (15,40)
def child(m):return source(m) and m not in (16,18,20,21,22,23,26,27,37,41,42,43,50,52,53,54,55,58)
def status(m):
    if 100<=m<=219 or 340<=m<=555:return 1
    for value,modes in [(2,{1,2,3,5,13,27,29,43,45,53}),(7,{4,12,26,28,42,44,52}),
        (11,{6,14,18,19,50,51,54,55}),(1,{7,8,9,10,11}),(3,{16,17}),(5,{20,21,22,23,24,25,58,59})]:
        if m in modes:return value
    return 0
def regions(j,m):
    root=0xffff000000000000 if m==56 else 0x100000000000
    rows=[dict(address=root+i*0x10000000+j*0x100000+4032,base=root+i*0x10000000+j*0x100000,bytes=span+4032) for i,span in enumerate(SPANS)]
    if 100<=m<=339 or 436<=m<=555:
        i,k=PAIRS[m-436 if m>=436 else m-220 if m>=220 else m-100]
        rows[k]['base']=rows[i]['base']+(rows[i]['bytes'] if 220<=m<=339 else rows[i]['bytes']-64 if m>=436 else 0)
        rows[k]['address']=rows[k]['base']+4032
    if 340<=m<=435:
        i,v=divmod(m-340,6);r=rows[i]
        if v==0:r['address']+=1
        elif v==1:r['bytes']=0
        elif v==2:r.update(base=r['address'],bytes=SPANS[i]-1 if i else 0)
        elif v==3:r['base']=r['address']+64
        elif v==4:r.update(base=0xfffffffffffff000,bytes=8192)
        else:r.update(address=0xffffffffffffffc0,base=0,bytes=0xffffffffffffffff)
    return rows
def pattern(addr):return sum((addr^(0x13579bdf2468ace0+i))<<(64*i) for i in range(8))
def tag(v):return sum(v<<(32*i) for i in range(16))
def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|AFC_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('AFC_'):continue
        fields=line.split();kind=fields[0][4:];assert kind in SCHEMA
        fields=[p.split('=') for p in fields[1:]]
        assert all(len(p)==2 for p in fields) and [p[0] for p in fields]==SCHEMA[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields};assert all(v>=0 for v in row.values());rows.append((kind,row))
    return rows
def review_events(rows,seed):
    assert rows[0]==('RESET',dict(cycle=6,job=0)) and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1 and sum(k=='RESET' and r['job']==0 for k,r in rows)==1
    summary=rows[-1][1];expected=dict(seed=seed,jobs=1034,publics=1029,offered=653,resets=7,reads=2688,writes=1344,source_acks=678,child_acks=662)
    assert all(summary[k]==v for k,v in expected.items()),summary
    buckets={j:[] for j in range(1,1035)};job=0;last_cycle=0
    for kind,r in rows:
        if 'cycle' in r:assert r['cycle']>=last_cycle;last_cycle=r['cycle']
        if kind=='BEGIN':assert r['job']==job+1;job=r['job']
        if r.get('job',0):assert r['job']==job;buckets[job].append((kind,r))
    assert job==1034 and summary['cycles']==last_cycle
    for j,m in enumerate(PLAN,1):
        events=buckets[j];by={k:[r for kind,r in events if kind==k] for k in SCHEMA}
        seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j;reg=regions(j,m)
        assert len(by['BEGIN'])==len(by['ADMIT'])==1
        begin=by['BEGIN'][0];admit=by['ADMIT'][0]
        assert begin==dict(cycle=begin['cycle'],job=j,mode=m,sequence=seq,epoch=epoch) and admit['cycle']==begin['cycle']+3
        assert [r['index'] for r in by['REGION']]==list(range(16))
        for r,want in zip(by['REGION'],reg):assert r==dict(cycle=begin['cycle'],job=j,index=r['index'],**want)
        owners=[o for o,yes in enumerate((source(m),child(m))) if yes]
        assert [r['owner'] for r in by['START']]==owners,(j,m,owners,by['START'])
        for r in by['START']:assert r['sequence']==seq and r['epoch']==epoch and r['cycle']>=admit['cycle']+137
        has_ffn_bus=child(m) or m==41
        switches=[1]+([] if m in ABORT else [0]) if has_ffn_bus else []
        assert [r['owner'] for r in by['SWITCH']]==switches
        pending=None;aw=None;w=None;committed={};read_indices=[0,0];reads=[0,0];writes=[0,0]
        for k,r in events:
            if k=='AR':
                o=r['owner'];assert o in owners and pending is None and read_indices[o]<2
                index=((4,1),(7,6))[o][read_indices[o]];assert r['address']==reg[index]['address']
                assert by['START'][owners.index(o)]['cycle']<r['cycle']
                if o==1:assert 0 in committed and by['SWITCH'][0]['cycle']<=r['cycle']
                pending=(o,r['address'],r['cycle']);read_indices[o]+=1
            elif k=='R':
                assert pending and (r['owner'],r['address'])==pending[:2] and r['cycle']>pending[2]
                expected_data=committed[0] if r['owner']==1 and reads[1]==0 else pattern(r['address'])
                assert r['data']==expected_data
                reads[r['owner']]+=1;pending=None
            elif k=='AW':
                assert aw is None and r['owner'] in owners
                assert r['address']==reg[7 if r['owner']==0 else 15]['address']
                aw=(r['owner'],r['address'],r['cycle'])
            elif k=='W':
                o=r['owner'];assert w is None and o in owners and reads[o]==2
                want=pattern(reg[4]['address'])^pattern(reg[1]['address'])^tag(0x46) if o==0 else committed[0]^pattern(reg[6]['address'])^tag(0x52)
                assert r['data']==want;w=(o,r['data'],r['cycle'])
            elif k=='B':
                assert aw and w and aw[0]==w[0]==r['owner'] and aw[1]==r['address'] and r['cycle']>max(aw[2],w[2])
                assert r['owner'] not in committed;committed[r['owner']]=w[1];writes[r['owner']]+=1;aw=None;w=None
        assert aw is None and w is None
        for o in (0,1):
            partial=m==(37 if o==0 else 38)
            assert read_indices[o]==(1 if partial else 2 if o in owners else 0)
            assert reads[o]==(0 if partial else 2 if o in owners else 0)
            assert writes[o]==int(o in owners and not partial)
        assert bool(pending)==(m in (37,38))
        terminal=m not in (15,37,38,40,41)
        assert len(by['COMPLETE'])==int(terminal)
        if terminal:
            c=by['COMPLETE'][0];ok=status(m)==0
            att=3 if m==16 else 5 if m==58 else 7 if m in (26,27,42,43,50) else 0
            ffn=3 if m==17 else 5 if m==59 else 7 if m in (28,29,44,45,51) else 0
            want=dict(cycle=c['cycle'],job=j,status=status(m),valid=int(ok),sequence=seq^((1<<40) if m==1 else 0),
                epoch=epoch^((1<<44) if m==2 else 0),address=reg[15]['address'] if ok else 0,bytes=8192 if ok else 0,attention=att,ffn=ffn)
            assert c==want,(j,m,c,want)
            assert c['cycle']>max([admit['cycle']]+[r['cycle'] for r in by['B']])
            if ok:assert 1 in committed
        for k,modes in [('STIM',STIM),('RESTORE',{18,19}),('REVOKE',REVOKE),('ABORT',ABORT),('RESET',ABORT)]:
            assert len(by[k])==int(m in modes),(j,k)
            if by[k]:
                assert by[k][0]['cycle']>admit['cycle']
                if k!='RESET':assert by[k][0]['mode']==m
        if m in (18,19):assert by['RESTORE'][0]['cycle']<c['cycle']
        if m in ABORT:
            assert not by['ACK'] and not by['RELEASE'] and not by['IDLE']
            assert by['ABORT'][0]['cycle']<by['RESET'][0]['cycle']
            if m==39:assert by['ABORT'][0]['cycle']>=c['cycle']+5
        else:
            assert len(by['ACK'])==len(by['IDLE'])==1
            ack=by['ACK'][0];assert ack['source']==int(source(m)) and ack['child']==int(child(m))
            assert ack['cycle']>=c['cycle']+6 and by['IDLE'][0]['cycle']>=ack['cycle']
            assert [r['owner'] for r in by['RELEASE']]==owners
            assert all(r['cycle']==ack['cycle'] for r in by['RELEASE'])
            if has_ffn_bus:assert by['SWITCH'][-1]['cycle']>ack['cycle']
            if m in REVOKE:assert c['cycle']+5<=by['REVOKE'][0]['cycle']<ack['cycle']
    return dict(seed=seed,jobs=1034,completions=1029,offered_descriptors=653,accepted_unrevoked_descriptors=640,
        source_acks=678,child_acks=662,read_beats=2688,write_beats=1344,cycles=summary['cycles'],
        overlap_cases=240,adjacency_cases=120,span_faults=96,arithmetic_qualified=False,physical_timing_qualified=False,
        scope='Actual attention/FFN wrapper and handoff; controlled peers perform two single-beat reads and one write each. Captured allocations, pre-write exclusion, symbolic B-committed ABI1 handoff, ownership, final ACK and reset. No model arithmetic or complete Q/K/V processing.')
def review(text,seed):return review_events(parse(text),seed)
def controls(text,seed):
    rows=parse(text);rejected=[]
    def reject(name,bad):
        try:review_events(bad,seed)
        except (AssertionError,KeyError,IndexError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    for kind,keys in [('BEGIN','mode sequence epoch'),('REGION','index address base bytes'),('ADMIT','cycle'),('START','owner sequence epoch'),
        ('SWITCH','owner'),('AR','owner address'),('R','owner address data'),('AW','owner address'),('W','owner data'),('B','owner address'),
        ('RELEASE','owner cycle'),('COMPLETE','status valid sequence epoch address bytes attention ffn'),('ACK','source child'),('PASS','reads writes offered')]:
        i=next(i for i,(k,r) in enumerate(rows) if k==kind)
        for key in keys.split():
            bad=rows.copy();r=dict(rows[i][1]);r[key]^=(1<<40 if key in ('sequence','epoch') else 64 if key in ('address','base') else 1)
            bad[i]=(kind,r);reject(kind+'.'+key,bad)
    for kind in ('REGION','ADMIT','START','SWITCH','AR','R','AW','W','B','RELEASE','COMPLETE','ACK','IDLE','STIM','RESTORE','REVOKE','ABORT','RESET'):
        i=next(i for i,(k,r) in enumerate(rows) if k==kind)
        reject('missing-'+kind,rows[:i]+rows[i+1:]);reject('duplicate-'+kind,rows[:i]+[rows[i]]+rows[i:])
    return rejected
