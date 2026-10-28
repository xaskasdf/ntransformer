"""Independent QKV lifecycle trace review; no matrix arithmetic qualification."""
from collections import Counter
import hashlib
import re


HEX = {'sequence', 'epoch', 'tag', 'weight', 'activation', 'output', 'q', 'k', 'v'}
FIELDS = {
    'RESET': 'cycle job', 'BEGIN': 'cycle job mode target variant expect sequence epoch tag',
    'HOLD': 'cycle job phase tag rows weight activation output',
    'OFFER': 'cycle job phase tag rows weight activation output',
    'CANCEL': 'cycle job phase', 'DONE': 'cycle job phase status rows bytes tag',
    'PUBLIC': 'cycle job status revoked valid matrices bytes sequence epoch q k v qbytes kbytes vbytes',
    'CHILD_ACK': 'cycle job phase public', 'SOURCE_ACK': 'cycle job', 'REVOKE': 'cycle job mode',
    'END': 'cycle job mode status source_acks', 'RESET_JOB': 'job mode',
    'PASS': 'seed tag_test jobs offers child_acks source_acks publics cancels resets drains revoked reserved held cycles',
}


def scenarios(mode):
    if mode == 1:
        return [(0,0,0),(90,0,0),(90,0,0)]
    if mode == 2:
        return [(90,0,0),(90,0,0)]
    return ([(0,0,0)] + [(m,0,0) for m in range(1,8)] +
        [(10,i,v) for i in range(7) for v in range(8)] +
        [(m,i,j) for i in range(4,7) for j in range(7) if i != j for m in (20,21)] +
        [(m,0,0) for m in range(30,44) if m != 39] +
        [(m,i,0) for m in range(50,59) for i in range(3)] + [(59,0,0)] +
        [(m,0,0) for m in range(60,71)] + [(71,i,0) for i in range(3)] +
        [(72,0,0),(73,0,0),(0,0,0)])


def status(mode):
    if mode in (1,2,3,31,41,56): return 2
    if mode in (4,5,6,7,51,52,53): return 5
    if mode in (10,20,21): return 1
    if mode in (30,40,55): return 7
    if mode in (*range(32,39),43,54,57): return 11
    if mode in (42,58,59): return 8
    if mode == 50: return 3
    if mode == 90: return 10
    return 0


def review(text, seed, tag_test=0, initial_tag=0xffffffc0):
    assert not re.search(r'(?im)^(?:FATAL:|ERROR:|QKV_FAIL)', text), 'Native failure'
    events=[];jobs={};summary=None;last_cycle=0;next_tag=initial_tag;reserved=set()
    for line in text.splitlines():
        if not line.startswith('QKV_'): continue
        parts=line.split();kind=parts[0].removeprefix('QKV_')
        assert kind in FIELDS, ('Unknown event',kind)
        assert len(parts)-1 == len(FIELDS[kind].split())
        pairs=[p.split('=',1) for p in parts[1:]]
        assert [p[0] for p in pairs] == FIELDS[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in pairs}
        assert summary is None, 'Event after PASS'
        if kind == 'PASS':
            summary=row;continue
        if 'cycle' in row:
            assert row['cycle'] >= last_cycle
            last_cycle=row['cycle']
        if kind == 'RESET':
            next_tag=initial_tag
        elif kind == 'BEGIN':
            assert row['job'] == len(jobs)+1
            assert row['tag'] == next_tag and row['expect'] == status(row['mode'])
            assert row['sequence'] == 0x13579bdf00000000 + row['job']
            assert row['epoch'] == 0xfedcba9800000000 + row['job']
            jobs[row['job']]={'begin':row,'events':[]}
        else:
            assert row['job'] in jobs and row['job'] == len(jobs)
            jobs[row['job']]['events'].append((kind,row))
            if kind == 'HOLD' and row['phase'] == 0 and row['job'] not in reserved:
                reserved.add(row['job']);next_tag=(next_tag+192) & ((1<<64)-1)
        events.append((kind,row))
    assert summary is not None and summary['seed']==seed and summary['tag_test']==tag_test
    assert [(r['begin']['mode'],r['begin']['target'],r['begin']['variant']) for r in jobs.values()] == scenarios(tag_test)
    counts=Counter(k for k,_ in events)
    expected_summary=dict(jobs=len(jobs), offers=counts['OFFER'], child_acks=counts['CHILD_ACK'],
        source_acks=counts['SOURCE_ACK'],publics=counts['PUBLIC'],cancels=counts['CANCEL'],resets=counts['RESET'],
        drains=sum(r['begin']['mode'] in (62,63,64,65,68,73) for r in jobs.values()),
        revoked=counts['REVOKE'],reserved=len(reserved),held=counts['HOLD'],cycles=last_cycle)
    assert all(summary[k]==v for k,v in expected_summary.items()), ('Summary/trace disagreement',expected_summary,summary)
    for job,record in jobs.items():
        b=record['begin'];mode=b['mode'];target=b['target'];ev=record['events']
        group={k:[r for kind,r in ev if kind==k] for k in FIELDS}
        early=mode in (*range(1,8),10,20,21,*range(30,39),90)
        reset=mode in (70,71,72,73)
        n=0 if early or mode==70 else (target+1 if mode==71 or 50<=mode<=58 else 1 if mode==59 or 40<=mode<=43 else 3)
        assert len(group['OFFER'])==n and [r['phase'] for r in group['OFFER']]==list(range(n))
        assert bool(group['RESET_JOB'])==reset and len(group['RESET_JOB'])==int(reset)
        assert len(group['END'])==int(not reset)
        assert len(group['PUBLIC'])==int(not reset or mode==73)
        assert len(group['REVOKE'])==int(60<=mode<=69 or mode==73)
        assert len(group['CANCEL'])==int(40<=mode<=43 or 54<=mode<=58)
        assert len(group['SOURCE_ACK'])==int(not reset and mode not in (1,2))
        assert len(group['CHILD_ACK'])==(target if mode==71 else 2 if mode in (72,73) else n)
        assert len(group['DONE'])==(target if mode==71 else n)
        assert [r['phase'] for r in group['DONE']]==list(range(len(group['DONE'])))
        assert [r['phase'] for r in group['CHILD_ACK']]==list(range(len(group['CHILD_ACK'])))
        base=0x100000000+job*0x10000
        for kind,row in ev:
            if kind in ('HOLD','OFFER','DONE'):
                phase=row['phase'];assert 0<=phase<3
                assert row['tag']==b['tag']+phase*64
            if kind in ('HOLD','OFFER'):
                assert row['rows']==(2048 if row['phase']==0 else 512)
                assert row['activation']==base and row['weight']==base+(row['phase']+1)*0x10000000
                assert row['output']==base+(row['phase']+4)*0x10000000
        if n:
            for offer in group['OFFER']:
                held=[r for r in group['HOLD'] if r['phase']==offer['phase']]
                assert held and all(r['cycle']<offer['cycle'] for r in held)
        else:
            assert bool(group['HOLD'])==(mode==70)
        for row in group['DONE']:
            phase=row['phase'];fault_phase=40<=mode<=43 or 50<=mode<=58 and phase==target
            expected=(3 if mode==50 else 7,0,0) if fault_phase else (0,2048 if phase==0 else 512,8192 if phase==0 else 2048)
            if 51<=mode<=54 and phase==target:
                expected=(0,(2048 if phase==0 else 512)-(mode==52),
                          (8192 if phase==0 else 2048)+(-64 if mode==51 else 64 if mode==53 else 0))
            assert (row['status'],row['rows'],row['bytes'])==expected
            assert row['cycle']>group['OFFER'][phase]['cycle']
        for cancel in group['CANCEL']:
            assert cancel['phase']==(0 if 40<=mode<=43 else target)
            assert group['OFFER'][cancel['phase']]['cycle']<cancel['cycle']<=group['DONE'][cancel['phase']]['cycle']
        public=group['PUBLIC'][0] if group['PUBLIC'] else None
        if public:
            assert public['sequence']==b['sequence'] and public['epoch']==b['epoch']
            assert public['status']==status(mode)
            revoked=mode in (1,2,3,*range(30,39),*range(40,44),*range(54,70),73)
            assert public['revoked']==revoked and public['valid']==(mode==0)
            matrices=0 if early or 40<=mode<=43 else target if 50<=mode<=58 else 1 if mode==59 else 3
            total=[0,8192,10240,12288][matrices]
            assert public['matrices']==matrices and public['bytes']==total
            assert [public[k] for k in ('q','k','v','qbytes','kbytes','vbytes')]==(
                [base+0x40000000,base+0x50000000,base+0x60000000,8192,2048,2048] if mode==0 else [0]*6)
            if group['DONE']:assert public['cycle']>group['DONE'][-1]['cycle']
        for ack in group['CHILD_ACK']:
            assert ack['cycle']>group['DONE'][ack['phase']]['cycle']
            terminal=public is not None and ack['phase']==n-1 and mode!=59
            assert ack['public']==int(terminal)
            if terminal:
                assert ack['cycle']>=public['cycle']
                assert (ack['cycle']>public['cycle'])==(mode in (62,63,64,65,68))
            else:
                if public:assert ack['cycle']<public['cycle']
        for ack in group['SOURCE_ACK']:
            assert public is not None and ack['cycle']>=public['cycle']
            expected_cycle=group['CHILD_ACK'][-1]['cycle'] if n and mode!=59 else public['cycle']
            assert ack['cycle']==expected_cycle
        if group['END']:
            end=group['END'][0]
            assert end['mode']==mode and end['status']==status(mode)
            assert end['source_acks']==len(group['SOURCE_ACK']) and end['cycle']>public['cycle']
        if group['RESET_JOB']:assert group['RESET_JOB'][0]['mode']==mode
    return dict(seed=seed,tag_test=tag_test,initial_tag=f'{initial_tag:016x}',**expected_summary,
                log_sha256=hashlib.sha256(text.encode()).hexdigest(),protocol_only=True,arithmetic_qualified=False)


def controls(text, seed, tag_test=0, initial_tag=0xffffffc0):
    lines=text.splitlines();cases={}
    def change(label,kind,key,value):
        index=next(i for i,l in enumerate(lines) if l.startswith('QKV_'+kind+' '))
        copy=lines.copy();copy[index]=re.sub(r'\b'+key+r'=\S+',key+'='+str(value),copy[index]);cases[label]='\n'.join(copy)+'\n'
    change('summary_jobs','PASS','jobs',999)
    change('public_identity_high','PUBLIC','sequence','0000000000000001')
    change('public_result_leak','PUBLIC','q','deadbeef')
    change('public_bytes','PUBLIC','bytes',1)
    change('source_ack_order','SOURCE_ACK','cycle',0)
    change('begin_tag','BEGIN','tag','deadbeef')
    change('wrong_status','PUBLIC','status',15)
    if tag_test in (0,1):
        change('phase_tag','OFFER','tag','deadbeef')
        change('high_activation_address','OFFER','activation','0000000000010000')
        change('weight_address','HOLD','weight','0000000000000000')
        change('rows','OFFER','rows',17)
        change('done_bytes','DONE','bytes',64)
        change('child_release_before_public','CHILD_ACK','public',1)
        index=next(i for i,l in enumerate(lines) if l.startswith('QKV_CHILD_ACK '))
        cases['duplicate_child_ack']='\n'.join(lines[:index]+[lines[index]]+lines[index:])+'\n'
    if tag_test==0:
        change('cancel_phase','CANCEL','phase',2)
        change('summary_cancels','PASS','cancels',0)
        cases['missing_revocation']='\n'.join(l for l in lines if not l.startswith('QKV_REVOKE '))+'\n'
    cases['missing_pass']='\n'.join(l for l in lines if not l.startswith('QKV_PASS '))+'\n'
    cases['fatal_after_pass']=text+'\nFATAL: QKV_FAIL injected\n'
    rejected=[]
    for name,corrupt in cases.items():
        assert corrupt!=text
        try:review(corrupt,seed,tag_test,initial_tag)
        except (AssertionError,KeyError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted trace corruption: '+name)
    return rejected
