"""Review held source/child ownership and atomic release; protocol peers only."""
import re
PLAN=[0]*2+[m for i in range(1,57) for m in (i,0)]
SCHEMA={
    'BEGIN':'cycle job mode sequence epoch residual ffn output',
    'OFFER':'cycle job sequence epoch residual ffn output base bytes',
    'CHILD':'cycle job status sequence epoch valid read write hls address bytes',
    'COMPLETE':'cycle job status source child hls valid read write',
    'STIM':'cycle job mode','RESTORE':'cycle job mode','REVOKE':'cycle job mode','ABORT':'cycle job mode',
    'RESET':'cycle job','SOURCE_ACK':'cycle job','CHILD_ACK':'cycle job','ACK':'cycle job source child',
    'PASS':'seed jobs publics offered resets admitted responses source_acks child_acks cycles',
}
HEX={'sequence','epoch','residual','ffn','output','base','address'}
STATUS={**{m:1 for m in range(6,22)},**{m:2 for m in (1,2,3,25,53)},**{m:3 for m in (4,30)},
    **{m:11 for m in (5,26,27,28,29,31,32)},**{m:7 for m in (22,23,24,52)},**{m:5 for m in (33,34,35,36,37,38,39,55,56)}}

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|FRHAND_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('FRHAND_'):continue
        parts=line.split();tag=parts[0][7:];assert tag in SCHEMA
        fields=[p.split('=') for p in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields};assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows

def review_events(rows,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1 and sum(k=='RESET' and r['job']==0 for k,r in rows)==1
    summary=rows[-1][1];expected=dict(seed=seed,jobs=114,publics=111,offered=68,resets=5,admitted=90,responses=88,source_acks=108,child_acks=87)
    assert all(summary[k]==v for k,v in expected.items())
    last_job=0;last_cycle=-1;buckets={j:{k:[] for k in SCHEMA} for j in range(1,115)}
    for k,r in rows:
        if 'cycle' in r:assert r['cycle']>=last_cycle;last_cycle=r['cycle']
        if k=='BEGIN':assert r['job']==last_job+1;last_job=r['job']
        if r.get('job',0):assert r['job']==last_job;buckets[last_job][k].append(r)
    assert last_job==114 and summary['cycles']==last_cycle+1
    for j,mode in enumerate(PLAN,1):
        e=buckets[j];seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j
        residual=0x700000000fc0+j*65536;ffn=0x800000000fc0+j*65536;output=0x990000000fc0+j*65536
        if mode==50:residual,ffn,output=0xfffffffffffe0000,0xfffffffffffc0000,0xfffffffffffa0000
        if mode==51:output=residual+8192
        assert len(e['BEGIN'])==1
        begin=e['BEGIN'][0]
        assert {k:begin[k] for k in ('mode','sequence','epoch','residual','ffn','output')}==dict(mode=mode,sequence=seq,epoch=epoch,residual=residual,ffn=ffn,output=output)
        child=mode==0 or (mode>=24 and mode!=47)
        assert len(e['OFFER'])==int(child)
        if child:
            offer=e['OFFER'][0]
            assert {k:offer[k] for k in ('sequence','epoch','residual','ffn','output','base','bytes')}==dict(sequence=seq,epoch=epoch,residual=residual,ffn=ffn,output=output,base=output-4032,bytes=12224)
            assert begin['cycle']<offer['cycle']
        respond=child and mode not in (48,54)
        assert len(e['CHILD'])==int(respond)
        if respond:
            response=e['CHILD'][0]
            child_status=7 if 24<=mode<=29 else 3 if mode in (30,39,55,56) else 0
            child_valid=int(child_status==0 or mode==39)
            if mode==37:child_valid=0
            want=dict(status=child_status,sequence=seq^((1<<40) if mode==31 else 0),epoch=epoch^((1<<44) if mode==32 else 0),
                valid=child_valid,read=255 if mode==34 else 257 if mode==55 else 256,
                write=127 if mode==35 else 129 if mode==56 else 128,hls=3 if mode==36 else 0,
                address=output+64 if mode==33 else output,bytes=8191 if mode==38 else 8192)
            assert {k:response[k] for k in want}==want and offer['cycle']<response['cycle']
        terminal=mode not in (47,48,54)
        assert len(e['COMPLETE'])==int(terminal)
        if terminal:
            c=e['COMPLETE'][0];status=STATUS.get(mode,0)
            assert c['status']==status and c['source']==(3 if mode==4 else 0) and c['valid']==int(status==0)
            assert (c['child'],c['hls'],c['read'],c['write'])==tuple(response[k] for k in ('status','hls','read','write')) if respond else (c['child'],c['hls'],c['read'],c['write'])==(0,0,0,0)
            assert c['cycle']>=response['cycle'] if respond else c['cycle']>begin['cycle']
        for tag,modes in [('STIM',range(24,30)),('RESTORE',(31,32)),('REVOKE',range(40,47)),('ABORT',(47,48,49,54)),('RESET',(47,48,49,54))]:
            assert len(e[tag])==int(mode in modes)
            if e[tag]:
                if tag!='RESET':assert e[tag][0]['mode']==mode
                assert e[tag][0]['cycle']>begin['cycle']
        if mode in (31,32):assert response['cycle']<e['RESTORE'][0]['cycle']<c['cycle']
        if 40<=mode<=46:assert c['cycle']+5<=e['REVOKE'][0]['cycle']
        if mode in (47,48,49,54):
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
            if 40<=mode<=46:assert e['REVOKE'][0]['cycle']<ack['cycle']
    counts={tag:sum(len(e[tag]) for e in buckets.values()) for tag in SCHEMA}
    assert counts['COMPLETE']==111 and counts['OFFER']==90 and counts['CHILD']==88 and counts['SOURCE_ACK']==108 and counts['CHILD_ACK']==87
    return dict(seed=seed,jobs=114,offered_descriptors=68,acknowledged_unrevoked_descriptors=60,source_acks=108,child_acks=87,
        cycles=summary['cycles'],cases=[dict(job=j,mode=m) for j,m in enumerate(PLAN,1)],arithmetic_qualified=False,
        scope='Controlled FFN/residual completion peers. 64-bit ownership, bounded disjoint vectors, immutable offers, simultaneous final ACK, faults/revocation/reset. No arithmetic, memory transport or live FFN integration.')

def review(text,seed):return review_events(parse(text),seed)

def controls(text,seed):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<44),('BEGIN','output',64),('BEGIN','mode',1),
        ('OFFER','sequence',1<<40),('OFFER','epoch',1<<44),('OFFER','residual',64),('OFFER','ffn',64),('OFFER','output',64),('OFFER','base',64),('OFFER','bytes',1),
        ('CHILD','status',1),('CHILD','sequence',1<<40),('CHILD','epoch',1<<44),('CHILD','valid',1),('CHILD','read',1),('CHILD','write',1),('CHILD','hls',1),('CHILD','address',64),('CHILD','bytes',1),
        ('COMPLETE','status',1),('COMPLETE','source',1),('COMPLETE','child',1),('COMPLETE','hls',1),('COMPLETE','valid',1),('COMPLETE','read',1),('COMPLETE','write',1),
        ('ACK','source',1),('ACK','child',1),('SOURCE_ACK','cycle',1),('CHILD_ACK','cycle',1),('REVOKE','mode',1),('PASS','offered',1)]
    def reject(name,bad):
        try:review_events(bad,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    for tag,key,mask in changes:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r);reject(tag+'.'+key,bad)
    for tag in ('OFFER','CHILD','COMPLETE','ACK','SOURCE_ACK','CHILD_ACK','STIM','RESTORE','REVOKE','ABORT','RESET'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    assert len(rejected)==55
    return rejected
