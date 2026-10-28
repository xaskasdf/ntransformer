"""Independently replay shared AXI ownership with explicit non-arithmetic peers."""
import re

PLAN=[0,0]+[m for i in range(1,48) for m in (i,0)]
SCHEMA={
    'BEGIN':'cycle job mode sequence epoch output',
    'START':'cycle job owner sequence epoch','FFN_DONE':'cycle job status',
    'SWITCH':'cycle job owner','AR':'cycle job owner index address',
    'R':'cycle job owner index data','AW':'cycle job owner address',
    'W':'cycle job owner data','B':'cycle job owner address data',
    'RELEASE':'cycle job ffn residual','COMPLETE':'cycle job status ffn residual valid reads writes',
    'ABORT':'cycle job mode','REVOKE':'cycle job mode','RESET':'cycle job',
    'BUSY_OFFER':'cycle job','ACK':'cycle job',
    'PASS':'seed jobs publics offered resets reads writes cycles',
}
HEX={'sequence','epoch','output','address','data'}

def status(mode):
    if 1<=mode<=18 or 20<=mode<=24:return 1
    if 25<=mode<=28 or mode in (33,37):return 2
    if mode in (29,30,31,32,35,36):return 7
    if mode in (34,38):return 3
    return 0

def pattern(address):
    return sum((address^(0x13579bdf2468ace0+lane))<<(64*lane) for lane in range(8))

def repeat32(value):return sum(value<<(32*lane) for lane in range(16))

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|FFRCTRL_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('FFRCTRL_'):continue
        parts=line.split();tag=parts[0][8:];assert tag in SCHEMA,tag
        fields=[p.split('=') for p in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split(),line
        row={k:int(v,16 if k in HEX else 10) for k,v in fields}
        assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows

def review_events(rows,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1 and sum(k=='RESET' and r['job']==0 for k,r in rows)==1
    summary=rows[-1][1]
    assert all(summary[k]==v for k,v in dict(seed=seed,jobs=96,publics=92,offered=55,resets=6).items())
    last_job=0;last_cycle=-1;buckets={j:[] for j in range(1,97)}
    for tag,row in rows:
        if 'cycle' in row:assert row['cycle']>=last_cycle;last_cycle=row['cycle']
        if tag=='BEGIN':assert row['job']==last_job+1;last_job=row['job']
        if row.get('job',0):assert row['job']==last_job;buckets[last_job].append((tag,row))
    assert last_job==96 and summary['cycles']==last_cycle
    reads=writes=0
    for job,mode in enumerate(PLAN,1):
        events=buckets[job];by={tag:[r for k,r in events if k==tag] for tag in SCHEMA}
        seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        base=(0xff00000000000000 if mode==46 else 0x100000000)+job*0x1000000000
        addresses=[base+i*0x20000000+4032 for i in range(8)]
        residual=(0xff10000000000000 if mode==46 else 0x700000000000)+job*65536+4032
        output=(0xff20000000000fc0 if mode==46 else 0x710000000fc0)+job*65536
        if mode==19:output=base+4032+4096
        assert len(by['BEGIN'])==1
        begin=by['BEGIN'][0]
        assert {k:begin[k] for k in ('mode','sequence','epoch','output')}==dict(mode=mode,sequence=seq,epoch=epoch,output=output)
        ffn=mode in (0,19) or mode>=31
        res=ffn and mode not in (31,32,33,34,39,40)
        assert [r['owner'] for r in by['START']]==([0,1] if res else [0] if ffn else [])
        for r in by['START']:assert r['sequence']==seq and r['epoch']==epoch and r['cycle']>begin['cycle']
        ffn_status=7 if mode in (31,32,33) else 3 if mode==34 else 0
        res_status=7 if mode in (35,36,37) else 3 if mode==38 else 0
        assert len(by['FFN_DONE'])==int(ffn and mode not in (39,40))
        if by['FFN_DONE']:assert by['FFN_DONE'][0]['status']==ffn_status
        reset=mode in (39,40,41,42,45)
        assert [r['owner'] for r in by['SWITCH']]==([1] if res and reset else [1,0] if res else [])
        payload0=pattern(addresses[0])^pattern(addresses[1])^repeat32(0x46)
        payload1=pattern(residual)^payload0^repeat32(0x52)
        expected_reads=[[addresses[0],addresses[1]],[residual,addresses[7]]]
        expected_writes=[addresses[7],output];expected_payloads=[payload0,payload1]
        selected=0;pending_read=None;pending_aw=None;pending_w=None
        committed={};start_cycles={};ar=[0,0];rr=[0,0];aw=[0,0];ww=[0,0];bb=[0,0]
        for tag,r in events:
            owner=r.get('owner');cycle=r['cycle']
            if tag=='START':start_cycles[owner]=cycle
            elif tag=='SWITCH':
                assert pending_read is None and pending_aw is None and pending_w is None
                if owner==1:assert 0 in committed and by['FFN_DONE'][0]['cycle']<=cycle
                else:assert by['RELEASE'] and by['RELEASE'][0]['cycle']<cycle
                selected=owner
            elif tag in ('AR','R','AW','W','B'):
                assert owner==selected and owner in start_cycles and start_cycles[owner]<cycle
                if tag=='AR':
                    assert pending_read is None and ar[owner]<2 and r['index']==ar[owner]
                    address=expected_reads[owner][ar[owner]];assert r['address']==address
                    if owner==1 and ar[owner]==1:assert 0 in committed and committed[0][0]<cycle
                    pending_read=(owner,address,cycle);ar[owner]+=1
                elif tag=='R':
                    assert pending_read is not None and pending_read[0]==owner and pending_read[2]<cycle and r['index']==rr[owner]
                    want=committed[0][1] if owner==1 and rr[owner]==1 else pattern(pending_read[1])
                    assert r['data']==want
                    pending_read=None;rr[owner]+=1;reads+=1
                elif tag=='AW':
                    assert pending_aw is None and aw[owner]==0 and r['address']==expected_writes[owner] and rr[owner]==2
                    pending_aw=(owner,r['address'],cycle);aw[owner]+=1
                elif tag=='W':
                    assert pending_w is None and ww[owner]==0 and r['data']==expected_payloads[owner] and rr[owner]==2
                    pending_w=(owner,r['data'],cycle);ww[owner]+=1
                else:
                    assert pending_aw is not None and pending_w is not None
                    assert pending_aw[0]==pending_w[0]==owner and pending_aw[1]==r['address'] and pending_w[1]==r['data']
                    assert max(pending_aw[2],pending_w[2])<cycle and owner not in committed
                    committed[owner]=(cycle,r['data']);pending_aw=pending_w=None;bb[owner]+=1;writes+=1
            elif tag=='FFN_DONE':assert 0 in committed and committed[0][0]<cycle
            elif tag=='COMPLETE':assert pending_read is None and pending_aw is None and pending_w is None
        for owner,launched in enumerate((ffn,res)):
            if not launched:assert (ar[owner],rr[owner],aw[owner],ww[owner],bb[owner])==(0,0,0,0,0)
            elif mode==39 and owner==0 or mode==41 and owner==1:
                assert ar[owner]==1 and rr[owner] in (0,1) and (aw[owner],ww[owner],bb[owner])==(0,0,0)
            elif mode==40 and owner==0 or mode==42 and owner==1:
                assert (ar[owner],rr[owner],aw[owner],ww[owner],bb[owner])==(2,2,1,0,0)
            else:assert (ar[owner],rr[owner],aw[owner],ww[owner],bb[owner])==(2,2,1,1,1)
        complete=mode not in (39,40,41,42)
        assert len(by['COMPLETE'])==int(complete)
        if complete:
            c=by['COMPLETE'][0]
            assert {k:c[k] for k in ('status','ffn','residual','valid','reads','writes')}==dict(status=status(mode),ffn=ffn_status if ffn else 0,residual=res_status if res else 0,valid=int(status(mode)==0),reads=sum(rr),writes=sum(bb))
            assert c['cycle']>begin['cycle']
            if ffn:assert c['cycle']>by['FFN_DONE'][0]['cycle']
            if res:assert c['cycle']>committed[1][0]
        assert len(by['RESET'])==len(by['ABORT'])==int(reset)
        if reset:
            assert by['ABORT'][0]['mode']==mode and by['ABORT'][0]['cycle']<by['RESET'][0]['cycle']
            assert not by['ACK'] and not by['RELEASE']
            if mode==45:assert by['ABORT'][0]['cycle']>=c['cycle']+5
        else:
            assert len(by['ACK'])==1 and by['ACK'][0]['cycle']>=c['cycle']+8
            assert len(by['RELEASE'])==int(ffn)
            if ffn:
                release=by['RELEASE'][0]
                assert (release['ffn'],release['residual'])==(1,int(res))
                assert c['cycle']+5<release['cycle']<by['ACK'][0]['cycle']
        assert len(by['REVOKE'])==int(mode in (43,44))
        if by['REVOKE']:assert by['REVOKE'][0]['mode']==mode and c['cycle']+5<by['REVOKE'][0]['cycle']<by['RELEASE'][0]['cycle']
        assert len(by['BUSY_OFFER'])==int(mode==47)
        if mode==47:assert start_cycles[0]<=by['BUSY_OFFER'][0]['cycle']<start_cycles[1]
    assert summary['reads']==reads and summary['writes']==writes
    return dict(seed=seed,jobs=96,completions=92,offered_descriptors=55,acknowledged_unrevoked_descriptors=52,
        resets=6,read_beats=reads,write_beats=writes,cycles=summary['cycles'],arithmetic_qualified=False,
        scope='Actual full-wrapper and handoff RTL, substituted two-read/one-write protocol peers. B-committed byte causality and shared AXI ownership only.')

def review(text,seed):return review_events(parse(text),seed)

def controls(text,seed):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<44),('BEGIN','output',64),('BEGIN','mode',1),
        ('START','sequence',1<<40),('START','epoch',1<<44),('START','owner',1),('FFN_DONE','status',1),
        ('SWITCH','owner',1),('AR','address',64),('AR','index',1),('AR','owner',1),('R','data',1),('R','index',1),
        ('AW','address',64),('W','data',1),('B','address',64),('B','data',1),('B','owner',1),
        ('COMPLETE','status',1),('COMPLETE','ffn',1),('COMPLETE','residual',1),('COMPLETE','valid',1),('COMPLETE','reads',1),('COMPLETE','writes',1),
        ('RELEASE','ffn',1),('RELEASE','residual',1),('REVOKE','mode',1),('ABORT','mode',1),('PASS','reads',1),('PASS','writes',1),('PASS','offered',1)]
    def reject(name,bad):
        try:review_events(bad,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    for tag,key,mask in changes:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r);reject(tag+'.'+key,bad)
    for tag in ('START','FFN_DONE','SWITCH','AR','R','AW','W','B','COMPLETE','RELEASE','REVOKE','ABORT','RESET','ACK','BUSY_OFFER'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    return rejected
