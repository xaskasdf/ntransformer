"""Independently review two-input RAM ownership; HLS arithmetic peers are synthetic."""
import re

FIELDS={
    'BEGIN':'cycle job mode sequence epoch position','START':'cycle job position','DONE':'cycle job status',
    'ACK':'cycle job','RESULT':'cycle job index address data','REVOKED':'cycle job mode',
    'INVALID_READ':'cycle job address','COMPLETE':'cycle job mode status sequence epoch position','ABORT_RESET':'cycle job mode','RESET':'cycle job count',
    'PASS':'seed watchdog jobs completions results starts dones acks resets reads writes held cycles'}
SKIP={31,32,33,34,36,39,40}
PLAN=[0]+[m for fault in range(1,57) if fault not in SKIP for m in (fault,0)]
STATUS={1:3,2:2,3:2,4:2,5:11,6:7,7:7,8:2,9:11,10:11,11:11,12:4,13:4,14:4,15:4,16:4,17:4,
    18:5,19:11,20:4,21:8,22:7,23:2,35:4,37:11,38:11,41:5,42:4,43:4,44:4,45:11,46:11,48:1,49:2,50:2,51:11,52:11,53:4,54:4,55:4}
RESET_MODES={26,27,28,29,30,47}

def record_word(job,address,seed):
    return 0x3f000000 | ((job*4096+address+seed)&0x7fffff)


def parse(text):
    assert not re.search(r'(?im)^\s*(?:FATAL|Fatal:|ERROR|ROPEOWNER_FAIL)',text),'Diagnostic'
    events=[]
    for line in text.splitlines():
        if not line.startswith('ROPEOWNER_'):continue
        fields=line.split();kind=fields[0].removeprefix('ROPEOWNER_');assert kind in FIELDS
        pairs=[f.split('=') for f in fields[1:]]
        assert all(len(p)==2 for p in pairs) and [k for k,v in pairs]==FIELDS[kind].split()
        row={}
        for key,value in pairs:
            hexval=key in ('sequence','epoch','data')
            assert re.fullmatch('[0-9a-f]{8}' if key=='data' else '[0-9a-f]{16}' if hexval else '[0-9]+',value)
            row[key]=int(value,16 if hexval else 10)
        events.append((kind,row))
    return events

def review(text,seed):
    events=parse(text);assert events and events[-1][0]=='PASS' and sum(k=='PASS' for k,r in events)==1
    summary=events[-1][1];expected=dict(seed=seed,watchdog=64,jobs=99,completions=93,results=135680,starts=89,dones=87,acks=92,resets=7)
    assert all(summary[k]==v for k,v in expected.items()),'Coverage'
    rows=events[:-1];assert rows[0]==('RESET',dict(cycle=1,job=0,count=1))
    cycles=[r['cycle'] for k,r in rows];assert cycles==sorted(cycles) and summary['cycles']>cycles[-1]
    begins=[i for i,(k,r) in enumerate(rows) if k=='BEGIN'];assert len(begins)==len(PLAN)==99 and begins[0]==1
    reset_count=1;metrics=[];counts={k:0 for k in ('START','DONE','ACK','RESULT','COMPLETE')}
    for job,(offset,mode) in enumerate(zip(begins,PLAN),1):
        batch=rows[offset:begins[job] if job<len(PLAN) else len(rows)]
        assert all(r['job']==job for k,r in batch),'Foreign owner'
        sequence=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        position=131072 if mode==48 else 0 if job==1 else 131071 if job==2 else job*1031%131072
        begin=batch[0][1];assert begin['position']==position
        assert (begin['mode'],begin['sequence'],begin['epoch'])==(mode,sequence,epoch)
        want=['BEGIN']
        if mode not in (1,2,3,4,5,6,30,48,49,50):want.append('START')
        if mode==28:
            middle=[k for k,r in batch[2:-1]];assert sorted(middle)==['ABORT_RESET','DONE'];want.extend(middle)
        elif mode not in (1,2,3,4,5,6,26,27,30,48,49,50):want.append('DONE')
        if mode not in (2,3,26,27,28,30,47):want.append('ACK')
        if mode in RESET_MODES:
            if mode!=28:want.append('ABORT_RESET')
            want.append('RESET');reset_count+=1
        else:
            if STATUS.get(mode,0)==0:want+=['RESULT']*2560
            if mode==56:want+=['INVALID_READ']*2
            if mode in (24,25):want.append('REVOKED')
            want.append('COMPLETE')
        assert [k for k,r in batch]==want,('Ownership/order',job,mode)
        index=0;tagged={};invalid_addresses=[]
        for kind,row in batch:
            if kind in counts:counts[kind]+=1
            if 'mode' in row:assert row['mode']==mode
            if 'position' in row:assert row['position']==position
            if kind=='INVALID_READ':invalid_addresses.append(row['address'])
            if kind=='RESULT':
                address=(index*3+seed)%2560
                assert (row['index'],row['address'],row['data'])==(index,address,record_word(job,address,seed)),'RAM bytes/address'
                index+=1
            else:tagged[kind]=row
            if kind=='RESET':assert row['count']==reset_count
            if kind=='DONE':assert row['status']==(2 if mode==18 else 6 if mode==41 else 0)
            if kind=='COMPLETE':assert (row['status'],row['sequence'],row['epoch'])==(STATUS.get(mode,0),sequence,epoch)
        assert invalid_addresses==([2560,4095] if mode==56 else [])
        if 'START' in tagged:assert tagged['START']['cycle']>begin['cycle']
        if 'ACK' in tagged and 'DONE' in tagged:
            assert tagged['ACK']['cycle']>tagged['DONE']['cycle'],'Early source release'
            if mode in (45,46):assert tagged['ACK']['cycle']-tagged['DONE']['cycle']>=72,'Foreign response not held'
        if mode==47:assert tagged['RESET']['cycle']-tagged['DONE']['cycle']>=72
        if 'COMPLETE' in tagged:
            assert tagged['COMPLETE']['cycle']>tagged.get('ACK',begin)['cycle']
            metrics.append(dict(job=job,mode=mode,status=tagged['COMPLETE']['status'],fp32_words=index,
                cycles_including_public_readback=tagged['COMPLETE']['cycle']-begin['cycle']))
    assert reset_count==7
    for kind,key in [('START','starts'),('DONE','dones'),('ACK','acks'),('RESULT','results'),('COMPLETE','completions')]:assert counts[kind]==summary[key]
    # Input request/write counters are bank diagnostics, not independent memory traces.
    assert 300000<summary['reads']<400000 and summary['writes']==(87-2)*2560+64
    assert summary['held']>summary['results'] and 600000<summary['cycles']<800000
    return dict(seed=seed,jobs=99,completions=93,fp32_words=135680,resets=7,summary=summary,jobs_metrics=metrics,
        arithmetic_qualified=False,scope='Actual Q/K and coefficient RAM owner with synthetic HLS/source peers; no RoPE arithmetic or AXI integration.')

def controls(text,seed):
    variants={f'missing_{kind.lower()}':re.sub(r'^ROPEOWNER_'+kind+r' [^\n]*\n','',text,count=1,flags=re.M)
        for kind in ('BEGIN','START','DONE','ACK','RESULT','COMPLETE','REVOKED','RESET','ABORT_RESET')}
    variants.update({
        'output_data':re.sub(r'(ROPEOWNER_RESULT[^\n]*data=)[0-9a-f]+',r'\g<1>00000001',text,count=1),
        'output_address':re.sub(r'(ROPEOWNER_RESULT[^\n]*address=)\d+',r'\g<1>2560',text,count=1),
        'output_order':re.sub(r'(ROPEOWNER_RESULT[^\n]*index=)0',r'\g<1>1',text,count=1),
        'sequence_high':text.replace('sequence=12345678','sequence=02345678',1),
        'epoch_high':text.replace('epoch=fedcba98','epoch=eedcba98',1),
        'status':re.sub(r'(ROPEOWNER_COMPLETE[^\n]*status=)0',r'\g<1>1',text,count=1),
        'early_ack':re.sub(r'(ROPEOWNER_ACK cycle=)\d+',r'\g<1>0',text,count=1),
        'seed':text.replace(f'PASS seed={seed} ',f'PASS seed={seed+1} ',1),
        'coverage':text.replace('jobs=99 ','jobs=80 ',1),
        'unknown':text+'\nROPEOWNER_UNKNOWN cycle=0 job=0\n','diagnostic':text+'\nROPEOWNER_FAIL injected\n',
        'position':re.sub(r'(ROPEOWNER_START[^\n]*position=)\d+',r'\g<1>131072',text,count=1),
        'consumer_range':re.sub(r'(ROPEOWNER_INVALID_READ[^\n]*address=)\d+',r'\g<1>0',text,count=1),
        'duplicate_pass':text+next(line for line in text.splitlines() if line.startswith('ROPEOWNER_PASS'))+'\n'})
    rejected=[]
    for name,changed in variants.items():
        assert changed!=text,name
        try:review(changed,seed)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
