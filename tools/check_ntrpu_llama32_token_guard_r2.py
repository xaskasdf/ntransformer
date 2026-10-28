"""Independent full-token protocol coverage; peers are synthetic, not arithmetic."""
from collections import defaultdict
import re

FIELDS={
    'BIND':'cycle bind bad_prefix poison','BEGIN':'cycle job mode target position committed',
    'LAYER':'cycle job layer position input output lease','LDONE':'cycle job layer status',
    'LACK':'cycle job layer public','HEAD':'cycle job input output','HDONE':'cycle job status',
    'HACK':'cycle job','PUBLIC':'cycle job status valid revoked layers position',
    'END':'cycle job layers heads lacks hacks poison committed',
    'PASS':'seed jobs binds publics layers lacks heads hacks successes poisoned cycles'}
HEX={'input','output','lease'}
LATE={22,23,24,34,35}

def plan():
    rows=[(0,0,0),(0,0,1)]
    faults=[(m,0) for m in range(1,6)]
    for n in range(16):faults += [(100+n,n),(200+n,n)]+[(m,n) for m in (7,8,9,10,14,20,25,28)]
    faults += [(m,0) for m in (11,12,13,21,22,23,24,26,27,29)]+[(30,7),(31,0),(34,0),(35,0)]
    faults += [(36,n) for n in range(15)]+[(37,0),(38,0)]
    for mode,target in faults:rows += [(mode,target,0),(0,0,0)]
    assert len(rows)==394
    return rows

def expected(mode,target,position):
    no_work=mode in (1,2,3,4,5,7,37,38) or mode>=100
    layer_fault=mode in (8,9,10,20,25,30)
    layers=0 if no_work else target if mode==28 else target+1 if layer_fault or mode==36 else 16
    heads=int(not no_work and not layer_fault and mode not in (28,29,36))
    completed=target if layer_fault or mode==28 else layers
    status=0 if mode==0 or mode in LATE else 2 if mode==2 else 1 if mode in (1,3,4,5) or mode>=100 else 3 if mode in (8,11) else 5 if mode in (9,10,12,13) else 7 if mode in (20,21,36,37) else 8 if mode in (28,29,30,31) else 11
    poison=int(mode!=0 and not (mode in (1,2,3,4,5,37) or mode>=100 or mode==28 and target==0))
    return dict(layers=layers,heads=heads,completed=completed,status=status,poison=poison,valid=int(mode==0),committed=position+int(mode==0))

def parse(text):
    assert not re.search(r'(?im)^\s*(?:FATAL|Fatal:|ERROR:|TOK_FAIL)',text),'Simulation diagnostic'
    events=[];previous=0
    for line in text.splitlines():
        if not line.startswith('TOK_'):continue
        parts=line.split();kind=parts[0][4:];assert kind in FIELDS,kind
        fields=[x.split('=') for x in parts[1:]]
        assert all(len(x)==2 for x in fields) and [k for k,v in fields]==FIELDS[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields}
        assert all(v>=0 for k,v in row.items() if k!='bad_prefix')
        assert row['cycle']>=previous if 'cycle' in row else True
        previous=row.get('cycle',previous);events.append((kind,row))
    return events

def review(text,seed):
    events=parse(text);assert events and events[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in events)==1
    binds=[r for k,r in events if k=='BIND']
    assert [r['bind'] for r in binds]==list(range(1,len(binds)+1))
    assert [r['bad_prefix'] for r in binds[:24]]==list(range(16))+list(range(-2,-10,-1))
    assert all(r['poison']==int(r['bad_prefix']!=-1) for r in binds)
    assert all(r['bad_prefix']==-1 for r in binds[24:])
    begins=[r for k,r in events if k=='BEGIN'];planned=plan()
    assert [r['job'] for r in begins]==list(range(1,len(planned)+1))
    assert [(r['mode'],r['target'],r['position']) for r in begins]==planned
    by_job=defaultdict(list)
    for kind,row in events:
        if 'job' in row:
            assert 1<=row['job']<=len(planned)
            by_job[row['job']].append((kind,row))
    metrics=[];last_end=None;committed=0;needs_bind=True
    for job,(mode,target,position) in enumerate(planned,1):
        part=by_job[job];assert part[0][0]=='BEGIN' and part[-1][0]=='END'
        e=defaultdict(list)
        for k,r in part:e[k].append(r)
        for k in ('BEGIN','PUBLIC','END'):assert len(e[k])==1
        begin,public,end=(e[k][0] for k in ('BEGIN','PUBLIC','END'))
        bound=[r for r in binds if (last_end or 0)<r['cycle']<=begin['cycle']]
        if bound:assert bound[-1]['bad_prefix']==-1;committed=0;needs_bind=False
        assert not needs_bind and begin['committed']==committed==position
        ex=expected(mode,target,position)
        assert {k:end[k] for k in ('layers','heads','poison','committed')}=={k:ex[k] for k in ('layers','heads','poison','committed')}
        assert (public['status'],public['valid'],public['revoked'],public['layers'])==(ex['status'],ex['valid'],int(mode!=0),ex['completed'])
        assert public['position']==(position+1 if mode==3 else position)
        assert begin['cycle']<public['cycle']<end['cycle']
        assert len(e['LAYER'])==len(e['LDONE'])==len(e['LACK'])==end['lacks']==ex['layers']
        assert len(e['HEAD'])==len(e['HDONE'])==len(e['HACK'])==end['hacks']==ex['heads']
        for n,(launch,done,ack) in enumerate(zip(e['LAYER'],e['LDONE'],e['LACK'])):
            assert launch['layer']==done['layer']==ack['layer']==n
            output=0x3000010000+(n%2)*0x10000
            input_=0x3000000000 if n==0 else 0x3000010000+((n-1)%2)*0x10000
            assert (launch['position'],launch['input'],launch['output'],launch['lease'])==(position,input_,output,0xfffffffffffffc00+n*32)
            assert begin['cycle']<launch['cycle']<done['cycle']<ack['cycle']<end['cycle']
            if n:assert e['LACK'][n-1]['cycle']<launch['cycle']
            if n==15:
                assert ack['public']==1 and ack['cycle']>public['cycle']
                if ex['heads']:assert ack['cycle']>e['HACK'][0]['cycle']
            elif n<ex['layers']-1 or mode in (28,36) or mode==0 or n<15 and ex['heads']:
                assert ack['public']==0 and ack['cycle']<public['cycle']
            else:assert ack['public']==1 and ack['cycle']>public['cycle']
            expected_status=3 if mode==8 and n==target else 7 if mode in (20,30) and n==target else 0
            assert done['status']==expected_status
        if ex['heads']:
            start,done,ack=(e[k][0] for k in ('HEAD','HDONE','HACK'))
            assert start['input']==0x3000020000 and start['output']==0x4000000000
            assert e['LDONE'][15]['cycle']<start['cycle']<done['cycle']<public['cycle']<ack['cycle']<end['cycle']
            assert done['status']==(3 if mode==11 else 7 if mode in (21,27,31) else 0)
        committed=ex['committed'];needs_bind=bool(ex['poison']);last_end=end['cycle']
        metrics.append(dict(job=job,mode=mode,target=target,position=position,**ex))
    total=events[-1][1]
    wanted=dict(seed=seed,jobs=394,binds=len(binds),publics=394,
        layers=sum(r['layers'] for r in metrics),lacks=sum(r['layers'] for r in metrics),
        heads=sum(r['heads'] for r in metrics),hacks=sum(r['heads'] for r in metrics),
        successes=sum(r['mode']==0 for r in metrics),poisoned=sum(r['poison'] for r in metrics))
    assert all(total[k]==v for k,v in wanted.items()),(total,wanted)
    assert total['cycles']>=last_end and not needs_bind
    return dict(status='PASS_TOKEN_GUARD_R1_PROTOCOL',total=total,all_sixteen_layers=True,
        retained_final_residual=True,atomic_token_commit=True,rebind_after_partial_failure=True,
        arithmetic_executed=False,physical_timing_qualified=False,
        scope='Controller with synthetic layer/head peers. Global memory allocator and real arithmetic wiring are separate pending work.')

def controls(text,seed):
    variants=[('missing_pass',re.sub(r'^TOK_PASS.*\n?','',text,flags=re.M)),
        ('missing_layer',re.sub(r'^TOK_LAYER .*\n?','',text,count=1,flags=re.M)),
        ('wrong_residual',text.replace('input=0000003000010000','input=0000003000000000',1)),
        ('wrong_lease',text.replace('lease=fffffffffffffc00','lease=00000000fffffc00',1)),
        ('lost_bind',re.sub(r'^TOK_BIND .*\n?','',text,count=1,flags=re.M)),
        ('bad_bind_accepted',text.replace('bad_prefix=0 poison=1','bad_prefix=0 poison=0',1)),
        ('early_last_ack',text.replace('layer=15 public=1','layer=15 public=0',1)),
        ('lost_head_ack',re.sub(r'^TOK_HACK .*\n?','',text,count=1,flags=re.M)),
        ('partial_result',text.replace('valid=1 revoked=0 layers=16','valid=1 revoked=0 layers=15',1)),
        ('poison_bypassed',re.sub(r'(TOK_END .*?poison=)1',r'\g<1>0',text,count=1)),
        ('early_token_commit',text.replace('heads=0 lacks=1 hacks=0 poison=1 committed=0','heads=0 lacks=1 hacks=0 poison=1 committed=1',1)),
        ('late_result_revival',text.replace('status=0 valid=0 revoked=1','status=0 valid=1 revoked=0',1)),
        ('unknown_event',text+'TOK_UNKNOWN cycle=1\n'),
        ('native_error',text+'\nFatal: injected failure\n')]
    rejected=[]
    for label,bad in variants:
        assert bad!=text,('Inactive control',label)
        try:review(bad,seed)
        except (AssertionError,ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Corrupt transcript accepted: '+label)
    return rejected
