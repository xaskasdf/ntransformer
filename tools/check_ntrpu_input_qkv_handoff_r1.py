"""Review the full norm-to-QKV ownership trace; peers perform no arithmetic."""
import re

FIELDS={
    'BEGIN':'cycle job mode sequence epoch','REGION':'cycle job index address base bytes',
    'NSTART':'cycle job address','NDONE':'cycle job status','QSTART':'cycle job address','QDONE':'cycle job status',
    'DIAGNOSTIC':'cycle job norm_status norm_hls norm_reads norm_records qkv_status matrix_status phase matrices bytes',
    'COMPLETE':'cycle job status valid norm_status qkv_status matrices bytes',
    'PUBLIC':'cycle job status valid revoked','NACK':'cycle job','QACK':'cycle job',
    'REVOKE':'cycle job mode','RESTORE':'cycle job mode','RETIRED':'cycle job norm qkv',
    'ABORT':'cycle job mode','RESET':'cycle job',
    'PASS':'seed jobs publics accepted norm_starts qkv_starts norm_acks qkv_acks resets cycles'}
WRITERS=(2,6,7,8)
PLAN=[0]*4+[m for fault in range(1,41) for m in (fault,0)]
PLAN += [m for fault in range(100,163) for m in (fault,0)]
PLAN += [m for wi,w in enumerate(WRITERS) for other in range(9) if other!=w for m in (200+wi*9+other,0)]
SPANS=(8192,8192,4096,4456448,1114112,1114112,8192,2048,2048)
LATE={25,26,34,35,39,40}
FOREIGN={17,24,39,40}

def status(mode):
    if mode>=100 or mode==36:return 1
    if mode in (1,2,3,9,21):return 2
    if mode in (4,17,18,24,31):return 11
    if mode in (5,6,7,8,19,20):return 7
    if mode in (10,22):return 3
    if mode in (11,12,13,14,15,16,23,32,33,37,38):return 5
    if mode in (27,28):return 8
    return 0

def regions(job,mode):
    addresses=[0x100000000+i*0x10000000+job*0x1000000000 for i in range(9)]
    bases=addresses.copy();sizes=list(SPANS)
    if 100<=mode<200:
        region,kind=divmod(mode-100,7)
        if kind==0:sizes[region]-=64
        elif kind==1:addresses[region]+=1;sizes[region]+=64
        elif kind==2:bases[region]=addresses[region]=0xfffffffffffff000;sizes[region]=8192
        elif kind==3:bases[region]=addresses[region]+64
        elif kind==4:bases[region]-=1;sizes[region]+=64
        elif kind==5:sizes[region]+=1
        elif kind==6:sizes[region]=0
    elif mode>=200:
        wi,other=divmod(mode-200,9);writer=WRITERS[wi]
        bases[writer]=addresses[writer]=bases[other];sizes[writer]=max(SPANS[writer],SPANS[other])
    if mode==36:sizes[0]=16384;bases[2]=bases[0];addresses[2]=bases[0]+8192;sizes[2]=12288
    if job==2:addresses[1]=addresses[0];bases[1]=bases[0]
    if job==3:addresses[4]=addresses[3];bases[4]=bases[3]
    return list(zip(addresses,bases,sizes))

def parse(text):
    assert not re.search(r'(?im)^\s*(?:FATAL|Fatal:|ERROR|IQH_FAIL)',text),'Diagnostic'
    result=[]
    for line in text.splitlines():
        if not line.startswith('IQH_'):continue
        fields=line.split();kind=fields[0].removeprefix('IQH_');assert kind in FIELDS,'Unknown event'
        pairs=[f.split('=') for f in fields[1:]]
        assert all(len(p)==2 for p in pairs) and [k for k,v in pairs]==FIELDS[kind].split(),'Fields'
        row={}
        for key,value in pairs:
            hexval=key in ('sequence','epoch','address','base') or kind=='REGION' and key=='bytes'
            assert re.fullmatch('[0-9a-f]{16}' if hexval else '[0-9]+',value),'Value'
            row[key]=int(value,16 if hexval else 10)
        result.append((kind,row))
    return result

def review(text,seed):
    events=parse(text);assert events and events[-1][0]=='PASS' and sum(k=='PASS' for k,r in events)==1,'Terminal'
    summary=events[-1][1]
    expected=dict(seed=seed,jobs=274,publics=272,accepted=139,norm_starts=172,qkv_starts=159,norm_acks=170,qkv_acks=158,resets=3)
    assert all(summary[k]==v for k,v in expected.items()),'Coverage'
    rows=events[:-1];assert rows[0][0]=='RESET' and rows[0][1]['job']==0
    assert [r['cycle'] for k,r in rows]==sorted(r['cycle'] for k,r in rows),'Cycle order'
    assert summary['cycles']==rows[-1][1]['cycle'] and 15000<summary['cycles']<18000
    starts=[i for i,(k,r) in enumerate(rows) if k=='BEGIN'];assert len(starts)==len(PLAN)==274 and starts[0]==1
    counts={k:0 for k in FIELDS};accepted=0;metrics=[]
    for job,(offset,mode) in enumerate(zip(starts,PLAN),1):
        batch=rows[offset:starts[job] if job<len(starts) else len(rows)]
        assert all(r['job']==job for k,r in batch),'Foreign job'
        begin=batch[0][1];assert (begin['mode'],begin['sequence'],begin['epoch'])==(mode,0x1234567800000000+job,0xfedcba9800000000+job),'Identity'
        norm=not(mode>=100 or mode in (1,2,3,4,5,6,36))
        qkv=norm and mode not in (7,8,9,10,11,12,13,14,15,16,17,27,29)
        want=['BEGIN']+['REGION']*9
        if norm:want+=['NSTART']
        if norm and mode!=29:want+=['NDONE']
        if qkv:want+=['QSTART']
        if qkv and mode!=30:want+=['QDONE']
        if mode in (29,30):want+=['ABORT','RESET']
        else:
            want+=['DIAGNOSTIC','COMPLETE']
            if mode in LATE:want+=['REVOKE']
            want+=['PUBLIC']
            if mode==40:want+=['QACK']
            if mode in FOREIGN:want+=['RESTORE']
            if norm:want+=['NACK']
            if qkv and mode!=40:want+=['QACK']
            want+=['RETIRED']
        assert [k for k,r in batch]==want,('Event order',job,mode)
        descriptor=regions(job,mode)
        for index,(kind,row) in enumerate(batch[1:10]):
            assert (row['index'],row['address'],row['base'],row['bytes'])==(index,*descriptor[index]),'Region stimulus'
            assert row['cycle']==begin['cycle']
        tagged={k:r for k,r in batch if k!='REGION'}
        for k,r in batch:
            counts[k]+=1
            if 'mode' in r:assert r['mode']==mode
        nstatus=7 if mode in (7,8,9,27) else 3 if mode==10 else 5 if mode==11 else 0
        qstatus=7 if mode in (18,19,20,21,28,31) else 3 if mode==22 else 0
        for kind in ('NSTART','QSTART'):
            if kind in tagged:
                assert tagged[kind]['address']==descriptor[2][0],'Lease address'
                assert tagged[kind]['cycle']>begin['cycle']
        for start,done,childstatus,timeout in [('NSTART','NDONE',nstatus,mode==27),('QSTART','QDONE',qstatus,mode==28)]:
            if done in tagged:
                assert tagged[done]['status']==childstatus,'Child status'
                assert tagged[done]['cycle']-tagged[start]['cycle']==(72 if timeout else 12),'Child duration'
        if qkv:assert tagged['QSTART']['cycle']>tagged['NDONE']['cycle'],'Norm-before-QKV'
        if mode in (29,30):
            assert tagged['RESET']['cycle']>tagged['ABORT']['cycle']>tagged['QSTART' if qkv else 'NSTART']['cycle']
            continue
        diag=tagged['DIAGNOSTIC'];complete=tagged['COMPLETE'];public=tagged['PUBLIC'];retired=tagged['RETIRED']
        expected_diag=dict(norm_status=nstatus if norm else 0,norm_hls=int(mode==11)*4,
            norm_reads=(255 if mode==15 else 256) if norm else 0,norm_records=(63 if mode==16 else 64) if norm else 0,
            qkv_status=qstatus if qkv else 0,matrix_status=3 if mode in (22,38) else 0,
            phase=2 if qkv else 0,matrices=(2 if mode==23 else 3) if qkv else 0,bytes=12288 if qkv else 0)
        assert {k:diag[k] for k in expected_diag}==expected_diag,'Diagnostic snapshot'
        assert all(complete[k]==diag[k] for k in ('norm_status','qkv_status','matrices','bytes'))
        assert complete['cycle']==diag['cycle'] and complete['cycle']>tagged.get('QDONE',tagged.get('NDONE',begin))['cycle']
        assert (complete['status'],complete['valid'])==(status(mode),int(status(mode)==0)),'Completion'
        accepted_here=int(mode==0);accepted+=accepted_here
        assert (public['status'],public['valid'],public['revoked'])==(status(mode),accepted_here,1-accepted_here),'Public result'
        assert public['cycle']>=complete['cycle']+6,'Retained public completion'
        if mode in LATE:assert complete['cycle']<tagged['REVOKE']['cycle']<public['cycle']
        if mode in FOREIGN:assert tagged['RESTORE']['cycle']>=public['cycle']+3
        for kind in ('NACK','QACK'):
            if kind in tagged:
                ack=tagged[kind]['cycle']
                if mode in FOREIGN and not(kind=='QACK' and mode==40):assert ack>tagged['RESTORE']['cycle'],'Foreign release'
                else:assert ack==public['cycle'],'Public/source release'
                assert retired['cycle']>ack,'Drain before reuse'
        assert retired['cycle']>public['cycle'] and (retired['norm'],retired['qkv'])==(int(norm),int(qkv))
        metrics.append(dict(job=job,mode=mode,status=status(mode),cycles=retired['cycle']-begin['cycle']))
    for kind,key in [('PUBLIC','publics'),('NSTART','norm_starts'),('QSTART','qkv_starts'),('NACK','norm_acks'),('QACK','qkv_acks')]:assert counts[kind]==summary[key]
    assert counts['NDONE']==171 and counts['QDONE']==158 and counts['RESET']+1==3 and accepted==139
    return dict(seed=seed,jobs=274,accepted=accepted,summary=summary,job_metrics=metrics,
        arithmetic_qualified=False,scope='Real handoff controller, synthetic norm/QKV peers; no AXI or arithmetic integration.')

def controls(text,seed):
    variants={f'missing_{k.lower()}':re.sub(r'^IQH_'+k+r' [^\n]*\n','',text,count=1,flags=re.M) for k in FIELDS if k!='PASS'}
    for event,keys in FIELDS.items():
        for key in keys.split():
            if key in ('job','cycle'):continue
            if event not in ('BEGIN','REGION','DIAGNOSTIC','COMPLETE','PUBLIC','RETIRED','PASS'):continue
            pattern=r'(IQH_'+event+r'[^\n]*\b'+key+r'=)([0-9a-f]+)'
            def change(m):
                v=m.group(2);base=16 if event=='REGION' and key in ('address','base','bytes') or key in ('sequence','epoch') else 10
                n=int(v,base)^1;return m.group(1)+(f'{n:016x}' if base==16 else str(n))
            variants[event.lower()+'_'+key]=re.sub(pattern,change,text,count=1)
    variants.update(unknown=text+'\nIQH_UNKNOWN cycle=0 job=0\n',diagnostic=text+'\nIQH_FAIL injected\n',
        duplicate_pass=text+next(s for s in text.splitlines() if s.startswith('IQH_PASS'))+'\n',
        early_ack=re.sub(r'(IQH_NACK cycle=)\d+',r'\g<1>0',text,count=1),
        foreign_job=re.sub(r'(IQH_QDONE[^\n]*job=)\d+',r'\g<1>0',text,count=1))
    rejected=[]
    for name,changed in variants.items():
        assert changed!=text,name
        try:review(changed,seed)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
