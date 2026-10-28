"""Review actual post-O arithmetic outputs and AXI transport; compilation is attested separately."""
import hashlib
import json
from pathlib import Path
import re

PLAN=[0]*4+[10,0,13,0]
SCHEMA={
    'BEGIN':'cycle job mode sample sequence epoch',
    'AR':'cycle job address beats','R':'cycle job index response data',
    'B':'cycle job index address response data','START':'cycle job','DONE':'cycle job return',
    'COMPLETE':'cycle job status hls valid read residual abi ar r b max_quiet',
    'ABORT':'cycle job mode','RESET':'cycle job',
    'PASS':'seed jobs publics pairs resets starts dones ar r b cycles',
}
HEX={'sequence','epoch','address','data'}

def fixture(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['model_sha256']=='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
    assert manifest['parent_manifest_sha256']=='fd756faeb7350a54243eacd6a4693e97ac98b446d6ed968919b76f18ed4deefc'
    values={}
    for name,sha in manifest['files'].items():
        raw=(folder/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        values[name]=[int(s,16) for s in raw.split()]
    assert len(values['source32.hex'])==24576 and len(values['residual32.hex'])==8192 and len(values['abi64.hex'])==2048
    return values

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|Fatal:|ERROR:|POSTREAL_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('POSTREAL_'):continue
        parts=line.split();tag=parts[0][9:];assert tag in SCHEMA
        fields=[s.split('=') for s in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields]==SCHEMA[tag].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in fields};assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows

def review_events(rows,values,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1
    summary=rows[-1][1];assert (summary['seed'],summary['jobs'],summary['publics'],summary['pairs'],summary['resets'])==(seed,8,8,6,1)
    previous_cycle=-1;last_job=0;buckets={j:{k:[] for k in SCHEMA} for j in range(1,9)}
    for k,r in rows:
        if 'cycle' in r:assert r['cycle']>=previous_cycle;previous_cycle=r['cycle']
        if k=='BEGIN':assert r['job']==last_job+1;last_job=r['job']
        if r.get('job',0):assert r['job']==last_job;buckets[last_job][k].append(r)
    assert last_job==8 and summary['cycles']>previous_cycle
    totals={k:0 for k in ('ar','r','b','starts','dones')};published=0
    statuses={1:2,2:2,3:11,4:1,5:3,6:5,7:1,8:3,9:7,10:7,11:2,12:7,13:7,20:7,21:7,22:8}
    for j,mode in enumerate(PLAN,1):
        e=buckets[j];assert len(e['BEGIN'])==1
        begin=e['BEGIN'][0];sample=(j-1)%4
        assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'])==(mode,sample,0x1234567800000000+j,0xfedcba9800000000+j)
        source=[0x8000000fc0+p*0x100000000+j*16384 for p in range(3)]
        residual=0x2200000fc0+j*16384;abi=0x3300000fc0+j*16384
        cursor=0
        for ar in e['AR']:
            assert cursor<384 and ar['address']==source[cursor//128]+(cursor%128)*64
            beats=min(128-cursor%128,(4096-ar['address']%4096)//64)
            assert ar['beats']==beats
            actual=e['R'][cursor:cursor+beats]
            # The only reset with a read in flight deliberately truncates its burst.
            assert len(actual)==beats or mode==17
            for r in actual:
                assert r['index']==cursor and ar['cycle']<r['cycle']
                want=sum(values['source32.hex'][sample*6144+cursor*16+lane]<<(lane*32) for lane in range(16))
                assert r['data']==want and r['response']==(2 if mode==5 and cursor==3 else 0)
                cursor+=1
        assert cursor==len(e['R'])
        for i,r in enumerate(e['B']):
            assert i<192 and r['index']==i and r['address']==(residual+i*64 if i<128 else abi+(i-128)*64)
            if i<128:want=sum(values['residual32.hex'][sample*2048+i*16+lane]<<(lane*32) for lane in range(16))
            else:want=sum(values['abi64.hex'][sample*512+(i-128)*8+lane]<<(lane*64) for lane in range(8))
            assert r['data']==want and r['response']==(2 if mode==8 and i==128 else 0)
            assert len(e['DONE'])==1 and e['DONE'][0]['cycle']<r['cycle']
            if i:assert e['B'][i-1]['cycle']<r['cycle']
        if e['START']:
            assert len(e['START'])==1 and len(e['R'])==384 and e['R'][-1]['cycle']<e['START'][0]['cycle']
        if e['DONE']:
            assert len(e['DONE'])==1 and e['START'][0]['cycle']<e['DONE'][0]['cycle']
            assert 0<=e['DONE'][0]['return']<=7
            if mode==0:assert e['DONE'][0]['return']==0
            assert 0<e['DONE'][0]['cycle']-e['START'][0]['cycle']<262144
        if mode in (17,18,19):assert not e['COMPLETE'] and len(e['ABORT'])==len(e['RESET'])==1
        else:
            assert len(e['COMPLETE'])==1;c=e['COMPLETE'][0];status=statuses.get(mode,0)
            assert (c['status'],c['hls'],c['valid'])==(status,e['DONE'][0]['return'],int(status==0))
            assert 0<=c['max_quiet']<262144
            # The AXI reader drains the rest of an errored burst without
            # delivering those beats to loader RAM. Both counters are checked.
            delivered=next((r['index'] for r in e['R'] if r['response']!=0),len(e['R']))
            assert (c['read'],c['ar'],c['r'],c['b'])==(delivered,len(e['AR']),len(e['R']),len(e['B']))
            good=[r for r in e['B'] if r['response']==0]
            assert c['residual']==sum(r['index']<128 for r in good) and c['abi']==sum(r['index']>=128 for r in good)
            for tag in ('R','B','DONE'):
                if e[tag]:assert e[tag][-1]['cycle']<c['cycle']
            if status==0:
                assert len(e['R'])==384 and len(e['B'])==192 and len(e['DONE'])==1
                published+=12288
        if mode in (1,2,3,4,9,20):assert not e['START'] and not e['B']
        for key,tag in [('ar','AR'),('r','R'),('b','B'),('starts','START'),('dones','DONE')]:totals[key]+=len(e[tag])
    assert all(summary[k]==v for k,v in totals.items()) and published==73728
    return dict(seed=seed,jobs=8,published_pairs=6,published_bytes=published,cycles=summary['cycles'],totals=totals,
        hls_peer_replays_reference=False,compiled_source_attestation_required=True,
        jobs_metrics=[dict(job=j,hls_cycles=buckets[j]['DONE'][0]['cycle']-buckets[j]['START'][0]['cycle'],max_quiet_cycles=buckets[j]['COMPLETE'][0]['max_quiet']) for j in range(1,9)],
        scope='Independent R and B payload/order review. AW/W stability, RAM input forwarding and late revocation are frozen-bank assertions, not separate trace events.')

def review(text,values,seed):return review_events(parse(text),values,seed)

def controls(text,values,seed):
    rows=parse(text);rejected=[]
    for tag,key,mask in [('BEGIN','sequence',1<<40),('BEGIN','sample',1),('AR','address',64),('R','data',1),('R','response',2),('B','data',1),('B','response',2),('B','cycle',1<<40),('DONE','return',1),('COMPLETE','status',1),('COMPLETE','residual',1),('COMPLETE','max_quiet',1<<40),('PASS','pairs',1)]:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r)
        try:review_events(bad,values,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(tag+'.'+key)
        else:raise AssertionError('Accepted control '+tag+'.'+key)
    i=next(i for i,(k,r) in enumerate(rows) if k=='B' and r['index']==191)
    try:review_events(rows[:i]+rows[i+1:],values,seed)
    except (AssertionError,KeyError,IndexError):rejected.append('missing-final-B')
    else:raise AssertionError('Accepted missing B')
    assert len(rejected)==14
    return rejected
