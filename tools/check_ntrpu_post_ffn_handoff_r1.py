"""Independent protocol trace review. No arithmetic/AXI throughput qualification."""
import re

PLAN=[0,0]+[m for i in range(1,261) for m in (i,0)]
META='ffn_status ffn_child residual_status ffn_hls stages phase ffn_bytes stage_bytes'
GOOD=dict(zip(META.split(),(0,0,0,0,4,3,8192,155648)))
SCHEMA={
    'BEGIN':'cycle job mode sequence epoch residual ffn output',
    'FFN':'cycle job index addr base bytes','PROTECT':'cycle job index base bytes',
    'SOURCE':'cycle job status valid residual ffn resbytes ffnbytes root_hls post_hls reads residual_beats ffn_beats',
    'PORT':'cycle job index addr base bytes',
    'OFFER':'cycle job sequence epoch residual ffn output base bytes',
    'CHILD':'cycle job status sequence epoch valid read write hls address bytes '+META,
    'COMPLETE':'cycle job status source child hls valid read write '+META+' root_hls post_hls',
    'STIM':'cycle job mode','RESTORE':'cycle job mode','REVOKE':'cycle job mode','ABORT':'cycle job mode',
    'RESET':'cycle job','SOURCE_ACK':'cycle job','CHILD_ACK':'cycle job','ACK':'cycle job source child',
    'PASS':'seed jobs publics offered resets admitted responses source_acks child_acks cycles',
}
HEX={'sequence','epoch','residual','ffn','output','addr','base','address'}
STATUS={**{m:1 for m in range(6,22)},**{m:2 for m in (1,2,3,25,53,259)},**{m:3 for m in (4,30)},
    **{m:11 for m in (5,26,27,28,29,31,32,260)},**{m:7 for m in (22,23,24,52,256,257)},
    **{m:5 for m in (33,34,35,36,37,38,39,55,56)},**{m:5 for m in range(57,70)},
    **{m:11 for m in range(78,83)},**{m:1 for m in range(88,216)}}
REVOKES=set(range(40,47))|set(range(70,78))|set(range(83,88))
ABORTS={47,48,49,54,258}
STIMS=set(range(24,30))|set(range(78,83))|{256,257,259,260}

def model(j,mode):
    seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j
    residual=0x700000000fc0+j*65536;ffn=0x800000000fc0+j*65536;output=0x990000000fc0+j*65536
    if mode==50:residual,ffn,output=0xfffffffffffe0000,0xfffffffffffc0000,0xfffffffffffa0000
    if mode==51:output=residual+8192
    alloc={i:dict(addr=0x200000000fc0+i*0x10000000+j*65536,
        base=0x200000000000+i*0x10000000+j*65536,
        bytes=(17825792 if i<4 else 65536 if i<6 else 16384 if i==6 else 8192)+4032) for i in range(1,8)}
    protected={i:dict(base=0x100000000000+i*0x10000000+j*65536,bytes=16384) for i in range(8)}
    protected[6]=dict(base=residual,bytes=8192);protected[7]=dict(base=ffn,bytes=4096)
    if mode in (88,89):protected[mode-82]['base']+=64
    if 90<=mode<=105:
        i=(mode-90)//2
        protected[i]=dict(base=0xfffffffffffff000,bytes=8192) if (mode-90)%2 else dict(protected[i],bytes=0)
    if 106<=mode<=185:
        d=(mode-106)//16;i=((mode-106)%16)//2;span=(65536,65536,16384,8192,8192)[d]
        addr=protected[i]['base']+((protected[i]['bytes']-64) if (mode-106)%2 else 0)
        if d<4:alloc[d+4]=dict(addr=addr,base=addr,bytes=span)
    if 186<=mode<=215:
        d=(mode-186)//6;v=(mode-186)%6;span=(65536,65536,16384,8192,8192)[d]
        addr=output if d==4 else alloc[d+4]['addr']
        values=[(addr+1,addr,span+64),(addr,addr,span-1),(addr,addr+64,span),
            (addr,0xfffffffffffff000,8192),(0xffffffffffffffc0,0,0xffffffffffffffff),(addr,addr,0)]
        if d<4:alloc[d+4]=dict(zip(('addr','base','bytes'),values[v]))
    if 216<=mode<=255:
        d=(mode-216)//8;i=(mode-216)%8;span=(65536,65536,16384,8192,8192)[d]
        addr=protected[i]['base']+protected[i]['bytes']
        if d<4:alloc[d+4]=dict(addr=addr,base=addr-4032,bytes=span+4032)
        else:output=addr
    source=dict(status=3 if mode==4 else 0,valid=int(mode not in (4,5)),residual=residual,ffn=ffn,
        resbytes=8191 if mode==6 else 8192,ffnbytes=4095 if mode==7 else 4096,
        root_hls=0,post_hls=0,reads=384,residual_beats=128,ffn_beats=64)
    if mode==8:source['residual']+=1
    if mode==9:source['ffn']+=1
    if mode==10:source['residual']=0xffffffffffffffc0
    if mode==11:source['ffn']=0xffffffffffffffc0
    if mode==21:source['ffn']=residual+8128
    if 57<=mode<=61:source[('root_hls','post_hls','reads','residual_beats','ffn_beats')[mode-57]]=(1,1,383,127,63)[mode-57]
    return seq,epoch,residual,ffn,output,alloc,protected,source

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|PFHAND_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('PFHAND_'):continue
        parts=line.split();tag=parts[0][7:];assert tag in SCHEMA
        fields=[p.split('=') for p in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields};assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows

def review_events(rows,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1 and sum(k=='RESET' and r['job']==0 for k,r in rows)==1
    summary=rows[-1][1]
    expected=dict(seed=seed,jobs=522,publics=518,offered=325,resets=6,admitted=360,responses=358,source_acks=515,child_acks=357)
    assert all(summary[k]==v for k,v in expected.items()),summary
    last_job=0;last_cycle=-1;buckets={j:{k:[] for k in SCHEMA} for j in range(1,523)}
    for k,r in rows:
        if 'cycle' in r:assert r['cycle']>=last_cycle;last_cycle=r['cycle']
        if k=='BEGIN':assert r['job']==last_job+1;last_job=r['job']
        if r.get('job',0):assert r['job']==last_job;buckets[last_job][k].append(r)
    assert last_job==522 and summary['cycles']==last_cycle+1
    for j,mode in enumerate(PLAN,1):
        e=buckets[j];seq,epoch,residual,ffn,output,alloc,protected,source=model(j,mode)
        assert len(e['BEGIN'])==1
        begin=e['BEGIN'][0]
        assert {k:begin[k] for k in ('mode','sequence','epoch','residual','ffn','output')}==dict(mode=mode,sequence=seq,epoch=epoch,residual=residual,ffn=ffn,output=output)
        for tag,data in [('FFN',alloc),('PROTECT',protected)]:
            assert [r['index'] for r in e[tag]]==list(data)
            for r in e[tag]:
                want=data[r['index']];assert r['cycle']==begin['cycle'] and {k:r[k] for k in want}==want,(j,tag,r,want)
        assert len(e['SOURCE'])==1
        assert {k:e['SOURCE'][0][k] for k in source}==source and e['SOURCE'][0]['cycle']==begin['cycle']
        child=mode==0 or 24<=mode<=56 and mode!=47 or 62<=mode<=87 or 216<=mode<=255
        assert len(e['OFFER'])==int(child)
        assert len(e['PORT'])==8*int(child)
        if child:
            offer=e['OFFER'][0]
            want=dict(sequence=seq,epoch=epoch,residual=residual,ffn=ffn,output=output,base=output-4032,bytes=12224)
            assert {k:offer[k] for k in want}==want and offer['cycle']>begin['cycle']+45
            assert [r['index'] for r in e['PORT']]==list(range(8))
            for r in e['PORT']:
                want=alloc[r['index']] if r['index'] else dict(addr=ffn,base=ffn,bytes=4096)
                assert {k:r[k] for k in want}==want and r['cycle']==offer['cycle']
        respond=child and mode not in (48,54)
        assert len(e['CHILD'])==int(respond)
        if respond:
            response=e['CHILD'][0]
            child_status=7 if 24<=mode<=29 or 78<=mode<=82 else 3 if mode in (30,39,55,56) else 0
            child_valid=int(child_status==0 or mode==39)
            if mode==37:child_valid=0
            want=dict(status=child_status,sequence=seq^((1<<40) if mode==31 else 0),epoch=epoch^((1<<44) if mode==32 else 0),
                valid=child_valid,read=255 if mode==34 else 257 if mode==55 else 256,
                write=127 if mode==35 else 129 if mode==56 else 128,hls=3 if mode==36 else 0,
                address=output+64 if mode==33 else output,bytes=8191 if mode==38 else 8192,**GOOD)
            if 62<=mode<=69:want[META.split()[mode-62]]=(1,1,1,1,3,2,8191,155647)[mode-62]
            assert {k:response[k] for k in want}==want and offer['cycle']<response['cycle']
        terminal=mode not in (47,48,54,258)
        assert len(e['COMPLETE'])==int(terminal)
        if terminal:
            c=e['COMPLETE'][0];status=STATUS.get(mode,0)
            assert c['status']==status and c['source']==(3 if mode==4 else 0) and c['valid']==int(status==0),(j,mode,c,status)
            for key,peer in [('child','status'),('hls','hls'),('read','read'),('write','write')]+[(k,k) for k in META.split()]:
                assert c[key]==(response[peer] if respond else 0),(j,key)
            assert (c['root_hls'],c['post_hls'])==(source['root_hls'],source['post_hls'])
            assert c['cycle']>=response['cycle'] if respond else c['cycle']>begin['cycle']
        for tag,modes in [('STIM',STIMS),('RESTORE',{31,32}),('REVOKE',REVOKES),('ABORT',ABORTS),('RESET',ABORTS)]:
            assert len(e[tag])==int(mode in modes),(j,tag)
            if e[tag]:
                if tag!='RESET':assert e[tag][0]['mode']==mode
                assert e[tag][0]['cycle']>begin['cycle']
        if mode in (31,32):assert response['cycle']<e['RESTORE'][0]['cycle']<c['cycle']
        if mode in REVOKES:assert c['cycle']+5<=e['REVOKE'][0]['cycle']
        if mode in ABORTS:
            assert not e['ACK'] and not e['SOURCE_ACK'] and not e['CHILD_ACK']
            assert e['ABORT'][0]['cycle']<e['RESET'][0]['cycle']
            if mode==49:assert c['cycle']+5<=e['ABORT'][0]['cycle']
        else:
            assert len(e['ACK'])==1;ack=e['ACK'][0]
            assert ack['source']==int(mode not in (1,2)) and ack['child']==int(child)
            assert len(e['SOURCE_ACK'])==ack['source'] and len(e['CHILD_ACK'])==ack['child']
            assert ack['cycle']>c['cycle']+5
            for tag in ('SOURCE_ACK','CHILD_ACK'):
                if e[tag]:assert e[tag][0]['cycle']==ack['cycle']
            if mode in REVOKES:assert e['REVOKE'][0]['cycle']<ack['cycle']
    return dict(seed=seed,jobs=522,completions=518,offered_descriptors=325,acknowledged_unrevoked_descriptors=304,
        source_acks=515,child_acks=357,overlap_cases=80,adjacency_cases=40,cycles=summary['cycles'],arithmetic_qualified=False,
        scope='Actual handoff RTL, controlled completion peers, eight captured protected allocations and five write destinations. No live producer, arithmetic, AXI transport or physical timing qualification.')

def review(text,seed):return review_events(parse(text),seed)

def controls(text,seed):
    rows=parse(text);rejected=[]
    def reject(name,bad):
        try:review_events(bad,seed)
        except (AssertionError,KeyError,IndexError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    for tag,keys in [('BEGIN','sequence epoch output mode'),('FFN','index addr base bytes'),('PROTECT','index base bytes'),
        ('SOURCE','status valid residual ffn resbytes ffnbytes root_hls post_hls reads residual_beats ffn_beats'),
        ('PORT','index addr base bytes'),('OFFER','sequence epoch residual ffn output base bytes'),
        ('CHILD','status sequence epoch valid read write hls address bytes '+META),
        ('COMPLETE','status source child hls valid read write '+META+' root_hls post_hls'),
        ('ACK','source child'),('SOURCE_ACK','cycle'),('CHILD_ACK','cycle'),('REVOKE','mode'),('PASS','offered')]:
        for key in keys.split():
            i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1])
            r[key]^=1<<40 if key in ('sequence','epoch') else 64 if key in HEX else 1
            bad[i]=(tag,r);reject(tag+'.'+key,bad)
    for tag in ('FFN','PROTECT','SOURCE','PORT','OFFER','CHILD','COMPLETE','ACK','SOURCE_ACK','CHILD_ACK','STIM','RESTORE','REVOKE','ABORT','RESET'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    return rejected
