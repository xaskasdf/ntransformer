"""Independently reconcile actual residual arithmetic and AXI publication traces."""
import hashlib,json,re,struct
from pathlib import Path
from check_ntrpu_residual_r1 import exact_add

PLAN=[0]*4+[m for i in range(1,23) for m in (i,0)]
SCHEMA={
    'BEGIN':'cycle job mode sample sequence epoch',
    'AR':'cycle job address beats','R':'cycle job index response data',
    'AW':'cycle job index address','W':'cycle job index data',
    'B':'cycle job index address response data','START':'cycle job','DONE':'cycle job return',
    'WRITE':'cycle job index bits',
    'COMPLETE':'cycle job status hls valid read residual ar r b max_quiet',
    'ABORT':'cycle job mode','RESET':'cycle job','REVOKE':'cycle job mode','ACK':'cycle job',
    'PASS':'seed jobs publics pairs resets starts dones ar r b cycles',
}
HEX={'sequence','epoch','address','data','bits'}
STATUS={1:2,2:2,3:11,4:1,5:3,6:5,7:1,8:3,9:7,10:7,11:2,12:7,13:7,20:7,21:7,22:8}

def fixture(folder):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    assert m['format']=='NTRRESIDUALREAL01'
    assert m['model_sha256']=='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
    assert m['parent_manifest_sha256']=='f2e3edda6ce1e3b5a6fca72886501e118584cd6004412772c279b7c615df5d09'
    raw={n:(folder/n).read_bytes() for n in m['files']}
    for n,h in m['files'].items():assert hashlib.sha256(raw[n]).hexdigest()==h
    assert hashlib.sha256(raw['parent-manifest.json']).hexdigest()==m['parent_manifest_sha256']
    parent=json.loads(raw['parent-manifest.json']);inputs=[];outputs=[]
    for i in range(1,5):
        words=[]
        for suffix in ('residual-fp32','final-fp32','layer-output-fp32'):
            name=f'job-{i:03d}-{suffix}.bin'
            assert hashlib.sha256(raw[name]).hexdigest()==parent['files'][name]
            words.append(struct.unpack('<2048I',raw[name]))
        x,y,z=words
        assert all(w&0x7f800000!=0x7f800000 for w in x+y+z)
        assert [exact_add(a,b) for a,b in zip(x,y)]==list(z)
        inputs.extend(x+y);outputs.extend(z)
    assert [int(w,16) for w in raw['source32.hex'].split()]==inputs
    assert [int(w,16) for w in raw['residual32.hex'].split()]==outputs
    return dict(inputs=inputs,outputs=outputs)

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|RESREAL_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('RESREAL_'):continue
        parts=line.split();tag=parts[0][8:];assert tag in SCHEMA
        fields=[s.split('=') for s in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields}
        assert all(v>=0 for v in row.values());rows.append((tag,row))
    return rows

def review_events(rows,values,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1 and sum(k=='RESET' and r['job']==0 for k,r in rows)==1
    summary=rows[-1][1]
    assert (summary['seed'],summary['jobs'],summary['publics'],summary['pairs'],summary['resets'])==(seed,48,45,29,5)
    previous_cycle=-1;last_job=0;buckets={j:{k:[] for k in SCHEMA} for j in range(1,49)}
    for k,r in rows:
        if 'cycle' in r:assert r['cycle']>=previous_cycle;previous_cycle=r['cycle']
        if k=='BEGIN':assert r['job']==last_job+1;last_job=r['job']
        if r.get('job',0):assert r['job']==last_job;buckets[last_job][k].append(r)
    assert last_job==48 and summary['cycles']==previous_cycle
    totals={k:0 for k in ('ar','r','b','starts','dones')};offered=0;released=0;metrics=[]
    for j,mode in enumerate(PLAN,1):
        e=buckets[j];assert len(e['BEGIN'])==1
        begin=e['BEGIN'][0];sample=(j-1)%4
        assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'])==(mode,sample,0x1234567800000000+j,0xfedcba9800000000+j)
        source=[0x8000000fc0+p*0x100000000+j*16384 for p in range(2)]
        dest=0x2200000fc0+j*16384;cursor=0
        for ar in e['AR']:
            assert cursor<256 and ar['address']==source[cursor//128]+(cursor%128)*64
            beats=min(128-cursor%128,(4096-ar['address']%4096)//64)
            assert ar['beats']==beats
            actual=e['R'][cursor:cursor+beats]
            assert len(actual)==beats or mode==17
            for r in actual:
                assert r['index']==cursor and ar['cycle']<r['cycle']
                words=[values['inputs'][sample*4096+cursor*16+lane] for lane in range(16)]
                if mode==6:
                    for lane in range(16):
                        if (cursor*16+lane)%2048==17:words[lane]=0x7f7fffff
                assert r['data']==sum(w<<(lane*32) for lane,w in enumerate(words))
                assert r['response']==(2 if mode==5 and cursor==3 else 0)
                cursor+=1
        assert cursor==len(e['R'])
        if e['START']:
            assert len(e['START'])==1 and len(e['R'])==256 and e['R'][-1]['cycle']<e['START'][0]['cycle']
        if e['DONE']:
            assert len(e['DONE'])==len(e['START'])==1
            assert e['START'][0]['cycle']<e['DONE'][0]['cycle']
            assert e['DONE'][0]['return']==(3 if mode==6 else 0)
            latency=e['DONE'][0]['cycle']-e['START'][0]['cycle']
            assert latency==(2062 if mode==6 else 4110)
        assert len(e['WRITE'])<=2048
        for i,w in enumerate(e['WRITE']):
            assert w['index']==i and e['START'][0]['cycle']<w['cycle']
            if e['DONE']:assert w['cycle']<=e['DONE'][0]['cycle']
            if i:assert e['WRITE'][i-1]['cycle']<w['cycle']
            if mode not in (10,11):assert w['bits']==values['outputs'][sample*2048+i]
        if e['DONE']:assert len(e['WRITE'])==(0 if mode==6 else 2048)
        for tag in ('AW','W','B'):
            for i,r in enumerate(e[tag]):
                assert i<128 and r['index']==i
                assert len(e['DONE'])==1 and e['DONE'][0]['return']==0 and e['DONE'][0]['cycle']<r['cycle']
                if i:assert e['B'][i-1]['cycle']<r['cycle']
                if tag in ('AW','B'):assert r['address']==dest+i*64
                if tag in ('W','B'):
                    want=sum(values['outputs'][sample*2048+i*16+lane]<<(lane*32) for lane in range(16))
                    assert r['data']==want
                if tag=='B':
                    assert e['AW'][i]['cycle']<r['cycle'] and e['W'][i]['cycle']<r['cycle']
                    assert r['data']==e['W'][i]['data'] and r['address']==e['AW'][i]['address']
                    assert r['response']==(2 if mode==8 and i==127 else 0)
        if mode==19:assert len(e['AW'])==1 and not e['W'] and not e['B']
        else:assert len(e['AW'])==len(e['W'])==len(e['B'])
        if mode in (16,17,18,19):
            assert len(e['ABORT'])==len(e['RESET'])==1 and not e['ACK']
            assert e['ABORT'][0]['mode']==mode and e['ABORT'][0]['cycle']<e['RESET'][0]['cycle']
        else:assert not e['ABORT'] and not e['RESET'] and len(e['ACK'])==1
        if mode in (17,18,19):assert not e['COMPLETE']
        else:
            assert len(e['COMPLETE'])==1;c=e['COMPLETE'][0];status=STATUS.get(mode,0)
            hls=e['DONE'][0]['return'] if e['DONE'] else 0
            assert (c['status'],c['hls'],c['valid'])==(status,hls,int(status==0))
            assert 0<=c['max_quiet']<262144
            delivered=next((r['index'] for r in e['R'] if r['response']!=0),len(e['R']))
            assert (c['read'],c['ar'],c['r'],c['b'])==(delivered,len(e['AR']),len(e['R']),len(e['B']))
            assert c['residual']==sum(r['response']==0 for r in e['B'])
            for tag in ('R','B','DONE'):
                if e[tag]:assert e[tag][-1]['cycle']<c['cycle']
            if e['ACK']:assert c['cycle']+5<e['ACK'][0]['cycle']
            if mode==16:assert c['cycle']+5<=e['ABORT'][0]['cycle']
            if status==0:
                assert len(e['R'])==256 and len(e['B'])==128 and len(e['DONE'])==1 and len(e['WRITE'])==2048
                offered+=8192
                if mode==0:released+=8192
                metrics.append(dict(job=j,sample=sample,hls_cycles=4110,begin_to_completion_cycles=c['cycle']-begin['cycle'],
                    first_ar_to_last_b_cycles=e['B'][-1]['cycle']-e['AR'][0]['cycle']+1))
        if mode in (14,15):
            assert len(e['REVOKE'])==1 and e['REVOKE'][0]['mode']==mode
            assert e['COMPLETE'][0]['cycle']<e['REVOKE'][0]['cycle']<e['ACK'][0]['cycle']
        else:assert not e['REVOKE']
        if mode in (1,2,3,4,5,9,20,21):assert not e['START'] and not e['WRITE'] and not e['AW']
        if mode in (1,2,3,4,20):assert not e['AR'] and not e['R']
        if mode==6:assert not e['AW']
        if mode==13:assert 32<=len(e['B'])<128
        if mode==22:assert e['B'][-1]['cycle']-max(e['AW'][-1]['cycle'],e['W'][-1]['cycle'])>=1024
        for key,tag in [('ar','AR'),('r','R'),('b','B'),('starts','START'),('dones','DONE')]:totals[key]+=len(e[tag])
    assert all(summary[k]==v for k,v in totals.items())
    assert offered==29*8192 and released==26*8192
    return dict(seed=seed,jobs=48,completions=45,offered_vectors=29,acknowledged_unrevoked_vectors=26,
        offered_bytes=offered,acknowledged_unrevoked_bytes=released,cycles=summary['cycles'],totals=totals,jobs_metrics=metrics,
        exact_fp32_words=29*2048,compiled_source_attestation_required=True,
        scope='Actual generated residual HLS, exact integer reference, R/AW/W/B causality and held completion. Four retained FFN operand pairs; no live FFN producer or physical memory.')

def review(text,values,seed):return review_events(parse(text),values,seed)

def controls(text,values,seed):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<40),('BEGIN','sample',1),('AR','address',64),('AR','beats',1),
        ('R','data',1),('R','response',2),('R','index',1),('AW','address',64),('AW','index',1),('W','data',1),('W','index',1),
        ('B','data',1),('B','response',2),('B','cycle',1<<40),('WRITE','bits',1),('WRITE','index',1),('DONE','return',1),
        ('DONE','cycle',1),('COMPLETE','status',1),('COMPLETE','valid',1),('COMPLETE','hls',1),('COMPLETE','read',1),
        ('COMPLETE','residual',1),('COMPLETE','max_quiet',1<<40),('REVOKE','mode',1),('ABORT','mode',1),('PASS','pairs',1),('PASS','r',1)]
    def reject(name,bad):
        try:review_events(bad,values,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted control '+name)
    for tag,key,mask in changes:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r)
        reject(tag+'.'+key,bad)
    for tag in ('AW','W','B','WRITE','START','DONE','COMPLETE','ACK','REVOKE','RESET'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    assert len(rejected)==49
    return rejected
