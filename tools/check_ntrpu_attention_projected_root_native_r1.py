"""Check projected-root protocol traces independently; no arithmetic qualification."""
import re

SCHEMA={
 'CONFIG':'seed','BEGIN':'cycle job mode sequence epoch','BIND':'cycle job','APPEND':'cycle job',
 'ROOT':'cycle job sequence epoch','OUTPUT':'cycle job','READ':'cycle job index',
 'ROOT_ACK':'cycle job','OUTPUT_ACK':'cycle job','AW':'cycle job address','W':'cycle job data',
 'B':'cycle job address','AR':'cycle job address','R':'cycle job address data','RESET':'cycle job phase',
 'COMPLETE':'cycle job mode status valid sequence epoch root_acks outputs',
 'RELEASE':'cycle job root_acks output_acks',
 'PASS':'seed jobs publics roots outputs root_acks output_acks resets binds appends aw w b ar r reads cycles'}
HEX={'sequence','epoch','address','data'}
PLAN=[0]+[mode for m in range(1,16) for mode in (m,0)]
STATUS={0:0,1:3,2:5,3:7,4:2,5:7,6:0,7:0,8:11,9:11,10:3,11:5,12:5,13:7}
K=0x4000000fc0;A=0x8000000fc0;O=0xa000000fc0

def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|PROOT_FAIL)',text),'Simulator failure'
    events=[]
    for line in text.splitlines():
        if not line.startswith('PROOT_'):continue
        kind,*parts=line.split();kind=kind[6:]
        assert kind in SCHEMA
        fields=[p.split('=') for p in parts]
        assert all(len(p)==2 for p in fields)
        assert [p[0] for p in fields]==SCHEMA[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields}
        assert all(v>=0 for v in row.values())
        events.append((kind,row))
    return events

def review_events(events,seed):
    assert events[0]==('CONFIG',{'seed':seed}) and events[-1][0]=='PASS'
    buckets={j:{k:[] for k in SCHEMA} for j in range(1,32)}
    global_rows={k:[] for k in SCHEMA};active=None;last_job=0;cycle=-1
    for kind,row in events:
        assert set(row)==set(SCHEMA[kind].split())
        if 'cycle' in row:assert row['cycle']>=cycle;cycle=row['cycle']
        if 'job' not in row:global_rows[kind].append(row);continue
        j=row['job'];assert j in buckets
        if kind=='BEGIN':assert active is None and j==last_job+1;active=j;last_job=j
        assert j==active
        buckets[j][kind].append(row)
        if kind in ('RELEASE','RESET'):active=None
    assert active is None and last_job==31 and len(global_rows['CONFIG'])==len(global_rows['PASS'])==1
    summary=global_rows['PASS'][0]
    expected=dict(seed=seed,jobs=31,publics=29,roots=31,outputs=25,root_acks=30,output_acks=24,resets=3,
                  binds=31,appends=31,aw=78,w=77,b=77,ar=54,r=53,reads=24)
    assert all(summary[k]==v for k,v in expected.items())
    assert summary['cycles']==buckets[31]['RELEASE'][0]['cycle']
    metrics=[]
    for j,mode in enumerate(PLAN,1):
        e=buckets[j];seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j
        for kind in ('BEGIN','BIND','APPEND','ROOT'):assert len(e[kind])==1,(j,kind)
        begin=e['BEGIN'][0];root=e['ROOT'][0]
        assert (begin['mode'],begin['sequence'],begin['epoch'])==(mode,seq,epoch)
        assert (root['sequence'],root['epoch'])==(seq,epoch)
        assert begin['cycle']<e['BIND'][0]['cycle']<e['APPEND'][0]['cycle']<root['cycle']
        launch=mode not in (1,2,3,4,8,14)
        complete=mode not in (14,15)
        full_output=launch and mode not in (13,15)
        assert len(e['OUTPUT'])==int(launch)
        assert len(e['READ'])==int(launch and mode!=13)
        assert len(e['ROOT_ACK'])==int(mode!=14)
        assert len(e['OUTPUT_ACK'])==int(launch and complete)
        if launch:assert root['cycle']<e['OUTPUT'][0]['cycle']
        if e['READ']:
            assert e['READ'][0]['index']==seq&2047
            assert e['OUTPUT'][0]['cycle']<e['READ'][0]['cycle']<e['ROOT_ACK'][0]['cycle']
        if mode==13:assert e['OUTPUT'][0]['cycle']<e['ROOT_ACK'][0]['cycle']
        expected_aw=[K]+([A,O] if full_output else [A] if mode==15 else [])
        expected_b=[K]+([A,O] if full_output else [])
        assert [r['address'] for r in e['AW']]==expected_aw
        assert [r['address'] for r in e['B']]==expected_b
        assert len(e['W'])==len(expected_b)
        kv=int.from_bytes((0x1122334455667788^seq).to_bytes(8,'little')*8,'little')
        activation=int.from_bytes((0x3f000000|(seq&2047)).to_bytes(4,'little')*16,'little')
        values={K:kv,A:activation,O:activation^((1<<512)-1)}
        for i,address in enumerate(expected_b):
            aw=e['AW'][i];w=e['W'][i];reply=e['B'][i]
            assert w['data']==values[address]
            assert reply['cycle']>max(aw['cycle'],w['cycle'])
            if i:assert min(aw['cycle'],w['cycle'])>e['B'][i-1]['cycle']
            else:assert e['APPEND'][0]['cycle']<min(aw['cycle'],w['cycle']) and reply['cycle']<root['cycle']
        if launch and mode!=13:assert e['AW'][1]['cycle']>e['ROOT_ACK'][0]['cycle']
        expected_ar=[K]+([A] if full_output else [])
        expected_r=[] if mode==14 else expected_ar
        assert [r['address'] for r in e['AR']]==expected_ar
        assert [r['address'] for r in e['R']]==expected_r
        assert root['cycle']<e['AR'][0]['cycle']
        for i,address in enumerate(expected_r):
            assert e['R'][i]['data']==values[address]
            assert e['AR'][i]['cycle']<e['R'][i]['cycle']
            if i:assert e['B'][1]['cycle']<e['AR'][i]['cycle']<e['R'][i]['cycle']<min(e['AW'][2]['cycle'],e['W'][2]['cycle'])
        if launch:assert e['R'][0]['cycle']<e['OUTPUT'][0]['cycle']
        if complete:
            assert len(e['COMPLETE'])==len(e['RELEASE'])==1 and not e['RESET']
            end=e['COMPLETE'][0];release=e['RELEASE'][0]
            assert (end['mode'],end['status'],end['valid'],end['sequence'],end['epoch'],end['root_acks'],end['outputs'])==(mode,STATUS[mode],int(mode==0),seq,epoch,int(launch),int(launch))
            assert release['root_acks']==1 and release['output_acks']==int(launch)
            assert release['cycle']>end['cycle']>root['cycle']
            assert end['cycle']>max(r['cycle'] for k in ('AR','R','AW','W','B') for r in e[k])
            if launch:
                assert e['ROOT_ACK'][0]['cycle']<end['cycle']
                assert e['OUTPUT_ACK'][0]['cycle']==release['cycle']
            else:assert e['ROOT_ACK'][0]['cycle']==release['cycle']
            metrics.append(dict(job=j,mode=mode,status=end['status'],valid=bool(end['valid']),cycles=end['cycle']-root['cycle']))
        else:
            assert not e['COMPLETE'] and not e['RELEASE'] and len(e['RESET'])==1
            reset=e['RESET'][0];assert reset['phase']==(1 if mode==14 else 2)
            if mode==14:assert reset['cycle']>=e['AR'][0]['cycle']
            else:assert reset['cycle']>=e['AW'][-1]['cycle']>e['ROOT_ACK'][0]['cycle']
    for key,kind in [('publics','COMPLETE'),('roots','ROOT'),('outputs','OUTPUT'),('root_acks','ROOT_ACK'),('output_acks','OUTPUT_ACK'),
                     ('binds','BIND'),('appends','APPEND'),('aw','AW'),('w','W'),('b','B'),('ar','AR'),('r','R'),('reads','READ')]:
        assert summary[key]==sum(len(e[kind]) for e in buckets.values()),key
    return dict(status='PASS_PROJECTED_ROOT_NATIVE_PROTOCOL',summary=summary,completions=metrics,
                arithmetic_qualified=False,full_tensor_transfers_qualified=False)

def review(text,seed):return review_events(parse(text),seed)

CHANGES=[('BEGIN','sequence',1<<40,1),('ROOT','epoch',1<<40,1),('READ','index',1,1),('AW','address',64,1),
 ('W','data',1,1),('B','address',64,1),('AR','address',64,1),('R','data',1,1),('COMPLETE','status',1,2),
 ('COMPLETE','valid',1,12),('COMPLETE','valid',1,14),('COMPLETE','sequence',1<<40,1),
 ('COMPLETE','root_acks',1,2),('COMPLETE','outputs',1,1),('RELEASE','output_acks',1,1),
 ('ROOT_ACK','cycle',1,2),('OUTPUT_ACK','cycle',1,1),('RESET','phase',1,28),('PASS','reads',1,None)]
MISSING=[('BIND',1),('OUTPUT',1),('ROOT_ACK',1),('OUTPUT_ACK',1)]
CONTROL_COUNT=26
def controls(text,seed):
    events=parse(text);review_events(events,seed);rejected=[]
    for kind,key,mask,job in CHANGES:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and (job is None or r.get('job')==job))
        altered=events.copy();row=dict(events[index][1]);row[key]^=mask;altered[index]=(kind,row)
        try:review_events(altered,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'{kind}.{key}.job{job}')
        else:raise AssertionError('Mutation accepted: '+kind+'.'+key)
    for kind,job in MISSING:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and r.get('job')==job)
        try:review_events(events[:index]+events[index+1:],seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'missing-{kind}.job{job}')
        else:raise AssertionError('Missing event accepted: '+kind)
    for label,bad in [('fatal','FATAL: injected\n'),('unknown','PROOT_UNKNOWN field=1\n'),('xz','PROOT_ROOT cycle=1 job=1 sequence=xx epoch=0\n')]:
        try:parse(bad)
        except (AssertionError,ValueError):rejected.append('parser-'+label)
        else:raise AssertionError('Bad parser input accepted')
    assert len(rejected)==CONTROL_COUNT==len(CHANGES)+len(MISSING)+3
    return rejected
