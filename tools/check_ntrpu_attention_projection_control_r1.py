"""Audit parent identity, reserved internal ranges and ownership from protocol traces."""
import re

FIELDS={
    'CONFIG':'seed watchdog exhaustion initial arithmetic_peer',
    'RESET':'cycle job count','ORPHAN':'cycle job',
    'BEGIN':'cycle job mode sequence epoch','OFFER':'cycle job sequence epoch weight activation output',
    'CHILD':'cycle job sequence epoch','ACK':'cycle job status bytes rows sequence epoch',
    'CANCEL':'cycle job','ABORT_RESET':'cycle job mode','REVOKED':'cycle job mode',
    'PUBLIC':'cycle job mode status sequence epoch valid address bytes',
    'PASS':'seed exhaustion jobs children replies cancels publics spans resets cycles',
}
STATUSES={1:2,2:7,3:2,4:7,5:2,6:11,7:11,8:3,9:5,10:5,11:5,12:5,13:8,14:8,20:7,21:2,22:7,24:9}
CANCELLED_MODES={2,3,4,5,6,7,13,14,20,21,22,23}


def review(text,seed,exhaustion=0):
    assert exhaustion in (0,1)
    assert not re.search(r'(?mi)^\s*(?:FATAL|Fatal:|ERROR|APROJ_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('APROJ_'):continue
        tokens=line.split();tag=tokens.pop(0).removeprefix('APROJ_')
        assert tag in FIELDS,'Unknown event'
        pairs=[token.split('=') for token in tokens]
        assert all(len(pair)==2 for pair in pairs)
        assert [pair[0] for pair in pairs]==FIELDS[tag].split(),'Event schema'
        values={}
        for key,value in pairs:
            width=0
            if key in ('sequence','epoch'):width=8 if tag in ('OFFER','CHILD','ACK') else 16
            elif key in ('initial','weight','activation','output','address'):width=16
            assert re.fullmatch('[0-9a-f]{'+str(width)+'}' if width else '[0-9]+',value)
            values[key]=int(value,16 if width else 10)
        rows.append((tag,values))
    initial=0xffffffffffffffc0 if exhaustion else 0x00000000ffffff80
    assert rows[0]==('CONFIG',dict(seed=seed,watchdog=24,exhaustion=exhaustion,initial=initial,arithmetic_peer=0))
    assert rows[-1][0]=='PASS' and sum(tag=='PASS' for tag,_ in rows)==1
    summary=rows[-1][1]
    expected=dict(seed=seed,exhaustion=exhaustion,jobs=4 if exhaustion else 47,
        children=2 if exhaustion else 45,replies=2 if exhaustion else 43,
        cancels=0 if exhaustion else 12,publics=4 if exhaustion else 43,
        spans=2 if exhaustion else 26,resets=2 if exhaustion else 6)
    assert {k:summary[k] for k in expected}==expected
    assert (50<summary['cycles']<150) if exhaustion else (950<summary['cycles']<1400)
    events=rows[1:-1];times=[v['cycle'] for _,v in events]
    assert times==sorted(times) and summary['cycles']>times[-1],'Cycle ordering'
    starts=[i for i,(tag,_) in enumerate(events) if tag=='BEGIN']
    plan=[0,24,24,0] if exhaustion else [0]+[mode for fault in range(1,24) for mode in (fault,0)]
    assert len(starts)==len(plan)
    preamble=events[:starts[0]]
    assert [tag for tag,_ in preamble]==(['RESET'] if exhaustion else ['RESET','ORPHAN','RESET'])
    reset_count=0
    for tag,value in preamble:
        assert value['job']==0
        if tag=='RESET':reset_count+=1;assert value['count']==reset_count
    next_tag=initial;counts=dict(children=0,replies=0,cancels=0,publics=0,spans=0)
    for job,(offset,mode) in enumerate(zip(starts,plan),1):
        batch=events[offset:starts[job] if job<len(plan) else len(events)]
        assert all(v['job']==job for _,v in batch),'Cross-owner event'
        begin=batch[0][1];parent_seq=0x13579bdf00000000+job;parent_epoch=0x2468ace000000000+job
        assert begin['mode']==mode and begin['sequence']==parent_seq and begin['epoch']==parent_epoch
        want=['BEGIN']
        if mode not in (1,24):
            assert next_tag+63<1<<64,'Internal namespace exhausted'
            allocated=next_tag;next_tag+=64;want+=['OFFER']
            if mode!=17:want+=['CHILD']
            if mode not in (17,18,23):want+=['ACK']
        else:
            if mode==24:assert next_tag>=1<<64
            allocated=None
        if mode in (17,18,19,23):want+=['ABORT_RESET','RESET']
        else:
            if mode in (15,16):want+=['REVOKED']
            want+=['PUBLIC']
        if exhaustion and job==3:want+=['RESET']
        assert [tag for tag,_ in batch if tag!='CANCEL']==want,f'Job {job}: protocol order'
        cancels=[v for tag,v in batch if tag=='CANCEL']
        assert len(cancels)==int(mode in CANCELLED_MODES),'Cancellation coverage'
        named={tag:v for tag,v in batch}
        if cancels:
            assert cancels[0]['cycle']>named['CHILD']['cycle']
            assert cancels[0]['cycle']<=named.get('ACK',named['RESET'] if 'RESET' in named else named['PUBLIC'])['cycle']
        for tag,v in batch:
            if 'mode' in v:assert v['mode']==mode
            if tag in ('OFFER','CHILD','ACK'):
                assert (v['epoch']<<32)|v['sequence']==allocated,'Reserved child identity'
                assert v['sequence']%64==0 and v['sequence']+63<=0xffffffff
            if tag=='OFFER':
                assert v['weight']==0x1100000000+job*0x1000000
                assert v['activation']==0x2200000000+job*0x10000
                assert v['output']==0x3300000000+job*0x10000
            elif tag=='CHILD':counts['children']+=1
            elif tag=='ACK':
                counts['replies']+=1
                assert v['cycle']>named['CHILD']['cycle'],'Early child completion'
                assert v['status']==(3 if mode in (8,11) else 0)
                assert v['bytes']==(64 if mode==8 else 8191 if mode==9 else 8193 if mode==11 else 8192)
                assert v['rows']==(16 if mode==8 else 2047 if mode==10 else 2049 if mode==12 else 2048)
            elif tag=='CANCEL':counts['cancels']+=1
            elif tag=='RESET':
                reset_count+=1;assert v['count']==reset_count;next_tag=initial
            elif tag=='PUBLIC':
                counts['publics']+=1;status=STATUSES.get(mode,0)
                assert v['status']==status and v['sequence']==parent_seq and v['epoch']==parent_epoch,'Public identity/status'
                valid=int(status==0 and mode not in (15,16));assert v['valid']==valid
                assert v['bytes']==(8192 if valid else 0)
                assert v['address']==(0x3300000000+job*0x10000 if valid else 0),'Published span'
                counts['spans']+=int(status==0)
                if 'ACK' in named:assert v['cycle']>named['ACK']['cycle']
    assert all(summary[k]==v for k,v in counts.items()) and reset_count==summary['resets']
    return dict(seed=seed,exhaustion=exhaustion,summary=summary,initial_tag=f'{initial:016x}',
        jobs=summary['jobs'],full_parent_identity=True,child_group_reservation=64,
        arithmetic_qualified=False,scope='Protocol peer and address/identity publication only. No actual GEMV, AXI traffic or arithmetic.')


def controls(text,seed,exhaustion=0):
    def remove(tag):return re.sub(r'^APROJ_'+tag+r' [^\n]*\n','',text,count=1,flags=re.M)
    def bit(tag,key):
        return re.sub(r'('+tag+r'[^\n]*\b'+key+r'=)([0-9a-f]+)',
            lambda m:m[1]+f'{int(m[2],16)^(1<<(len(m[2])*4-1)):0{len(m[2])}x}',text,count=1)
    variants={
        'missing_begin':remove('BEGIN'),'missing_offer':remove('OFFER'),'missing_child':remove('CHILD'),
        'missing_ack':remove('ACK'),'missing_public':remove('PUBLIC'),'missing_reset':remove('RESET'),
        'parent_sequence_high':bit('APROJ_PUBLIC','sequence'),'parent_epoch_high':bit('APROJ_PUBLIC','epoch'),
        'child_sequence':bit('APROJ_OFFER','sequence'),'child_epoch':bit('APROJ_OFFER','epoch'),
        'output_address':bit('APROJ_PUBLIC','address'),
        'status':re.sub(r'(APROJ_PUBLIC[^\n]*status=)0',r'\g<1>1',text,count=1),
        'seed':text.replace(f'PASS seed={seed} ',f'PASS seed={seed+1} ',1),
        'summary':re.sub(r'(APROJ_PASS[^\n]*children=)\d+',r'\g<1>999',text,count=1),
        'unknown_event':text+'\nAPROJ_UNKNOWN job=0\n','diagnostic':text+'\nAPROJ_FAIL injected\n',
        'cycle':re.sub(r'(APROJ_ACK cycle=)\d+',r'\g<1>0',text,count=1),
    }
    if not exhaustion:
        variants.update(missing_cancel=remove('CANCEL'),missing_revoked=remove('REVOKED'),
            missing_abort=remove('ABORT_RESET'),missing_orphan=remove('ORPHAN'),
            late_authority=re.sub(r'(APROJ_PUBLIC[^\n]*mode=15[^\n]*valid=)0',r'\g<1>1',text,count=1))
    rejected=[]
    for name,changed in variants.items():
        assert changed!=text
        try:review(changed,seed,exhaustion)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
