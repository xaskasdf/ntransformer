"""Review composition protocol traces; peers model extents, not arithmetic."""
import re

PLAN=[0]+[value for mode in range(1,18) for value in (mode,0)]
FIELDS={
 'BEGIN':'cycle job mode sequence epoch','ROOT_ACK':'cycle job',
 'ABORT':'cycle job mode',
 'COMPLETE':'cycle job mode status hls sequence epoch projection valid bytes',
 'PASS':'seed jobs publics qacks packs roots resets qstarts pstarts aw w b ar r cycles'}

def review(text,seed):
    assert not re.search(r'(?im)^\s*(FATAL:|Fatal:|ERROR:|PATH_FAIL)',text)
    rows=[];last_cycle=-1;active=None;last_job=0
    for line in text.splitlines():
        if not line.startswith('PATH_'):continue
        tokens=line.split();kind=tokens[0][5:];assert kind in FIELDS
        pairs=[part.split('=') for part in tokens[1:]]
        assert all(len(pair)==2 for pair in pairs) and [pair[0] for pair in pairs]==FIELDS[kind].split()
        row={key:int(value,16 if key in ('sequence','epoch') else 10) for key,value in pairs}
        if 'cycle' in row:assert row['cycle']>=last_cycle;last_cycle=row['cycle']
        if 'job' in row:
            if kind=='BEGIN':assert active is None and row['job']==last_job+1;active=row['job'];last_job=active
            assert row['job']==active
            if kind in ('ABORT','COMPLETE'):active=None
        rows.append((kind,row))
    def of(kind,job=None):return [row for k,row in rows if k==kind and (job is None or row.get('job')==job)]
    assert active is None and last_job==35 and rows[-1][0]=='PASS'
    assert len(of('PASS'))==1 and len(of('BEGIN'))==35 and len(of('COMPLETE'))==33 and len(of('ABORT'))==2
    summary=of('PASS')[0]
    expected=dict(seed=seed,jobs=35,publics=33,qacks=33,packs=28,roots=34,resets=3,qstarts=35,pstarts=29,aw=62,w=63,b=62,ar=29,r=28)
    for key,value in expected.items():assert summary[key]==value,key
    completions=[]
    for j,mode in enumerate(PLAN,1):
        begin=of('BEGIN',j)[0];seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j
        assert (begin['mode'],begin['sequence'],begin['epoch'])==(mode,seq,epoch)
        acks=of('ROOT_ACK',j);assert len(acks)==(0 if mode==16 else 1)
        if acks:assert acks[0]['cycle']>begin['cycle']
        if mode in (16,17):
            assert not of('COMPLETE',j) and len(of('ABORT',j))==1
            abort=of('ABORT',j)[0];assert abort['mode']==mode and abort['cycle']>begin['cycle']
            if acks:assert abort['cycle']>acks[0]['cycle']
            continue
        assert not of('ABORT',j) and len(of('COMPLETE',j))==1
        row=of('COMPLETE',j)[0]
        status=3 if mode in (1,7) else 5 if mode in (2,8,14,15) else 7 if mode in (3,4,12) else 2 if mode==5 else 11 if mode in (9,10) else 0
        projection=int(mode not in (1,2,3,9,14));valid=int(status==0 and mode not in (6,13))
        assert (row['mode'],row['status'],row['hls'],row['sequence'],row['epoch'],row['projection'],row['valid'],row['bytes'])==(mode,status,2 if mode==1 else 0,seq,epoch,projection,valid,8192 if valid else 0)
        assert row['cycle']>=acks[0]['cycle']+7
        completions.append(dict(job=j,mode=mode,cycles=row['cycle']-begin['cycle'],valid=bool(valid)))
    assert summary['cycles']>max(row.get('cycle',0) for _,row in rows)
    assert sum(row['valid'] for row in completions)==19
    return dict(status='PASS_OUTPUT_PATH_NATIVE_PROTOCOL',summary=summary,completions=completions,
                arithmetic_qualified=False,full_tensor_transfers_qualified=False,
                scope='Actual composition RTL with one-beat protocol peers; synthetic memory data and modeled region sizes.')

def controls(text,seed):
    variants={}
    for kind in FIELDS:variants['missing_'+kind.lower()]=re.sub(r'(?m)^PATH_'+kind+r' [^\n]*\n','',text,count=1)
    for kind,key,hexadecimal in [('BEGIN','sequence',True),('BEGIN','epoch',True),('COMPLETE','status',False),('COMPLETE','sequence',True),('COMPLETE','epoch',True),('COMPLETE','projection',False),('COMPLETE','valid',False),('COMPLETE','bytes',False),('PASS','qacks',False),('PASS','packs',False),('PASS','b',False),('PASS','r',False)]:
        pattern=r'(?m)^(PATH_'+kind+r' [^\n]*?\b'+key+r'=)([0-9a-f]+)'
        def replace(m,hexadecimal=hexadecimal):
            value=int(m[2],16 if hexadecimal else 10)^(1<<40 if hexadecimal else 1)
            return m[1]+(f'{value:016x}' if hexadecimal else str(value))
        variants[kind+'.'+key]=re.sub(pattern,replace,text,count=1)
    variants['fatal']=text+'\nFATAL: injected\n'
    variants['unknown']=text+'\nPATH_UNKNOWN x=1\n'
    rejected=[]
    for name,bad in variants.items():
        assert bad!=text
        try:review(bad,seed)
        except (AssertionError,ValueError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted corruption '+name)
    assert len(rejected)==19
    return rejected
