"""Reconcile original RMSNorm/Q8 output words, AXI transfers and public leases."""
import re
import struct
from check_ntrpu_input_norm_path_r1 import fixture as transport_fixture
from check_ntrpu_norm_q8_r1 import reference

PLAN = [0]*4 + [m for mode in (6, 10, 13, 14, 15, 18, 19, 22) for m in (mode, 0)]
SCHEMA = {
    'BEGIN':'cycle job mode sample sequence epoch', 'AR':'cycle job address beats',
    'R':'cycle job index response data', 'AW':'cycle job index address', 'W':'cycle job index data',
    'B':'cycle job index address response data', 'START':'cycle job', 'DONE':'cycle job return',
    'WRITE':'cycle job index bits', 'COMPLETE':'cycle job status hls valid read abi ar r b max_quiet',
    'ABORT':'cycle job mode', 'RESET':'cycle job', 'REVOKE':'cycle job mode',
    'ACK':'cycle job valid revoked', 'PASS':'seed jobs publics pairs resets starts dones ar r b cycles',
}
HEX = {'sequence':16, 'epoch':16, 'address':16, 'data':128, 'bits':16}
STATUS = {6:5, 10:7, 13:7, 22:8}


def fixture(folder):
    values = transport_fixture(folder)
    for sample in range(4):
        x = values['source32.hex'][sample*4096:sample*4096+2048]
        w = values['source32.hex'][sample*4096+2048:(sample+1)*4096]
        status, raw = reference(x,w,2048,0x3727c5ac)
        assert status == 0 and list(struct.unpack('<512Q',raw)) == values['abi64.hex'][sample*512:(sample+1)*512]
        x[17] = w[17] = 0x7f7fffff
        assert reference(x,w,2048,0x3727c5ac) == (4,b'')
    return values


def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|INREAL_FAIL)',text)
    rows = []
    for line in text.splitlines():
        if not line.startswith('INREAL_'):continue
        parts = line.split(); tag = parts[0][7:]; assert tag in SCHEMA
        fields = [s.split('=') for s in parts[1:]]
        assert all(len(f)==2 for f in fields) and [f[0] for f in fields] == SCHEMA[tag].split()
        for k,v in fields:
            assert re.fullmatch('[0-9a-fA-F]{'+str(HEX[k])+'}' if k in HEX else r'\d+',v)
        rows.append((tag,{k:int(v,16 if k in HEX else 10) for k,v in fields}))
    return rows


def review_events(rows,values,seed):
    assert rows[0][0]=='RESET' and rows[0][1]['job']==0 and rows[-1][0]=='PASS'
    assert sum(t=='PASS' for t,_ in rows)==1 and sum(t=='RESET' and r['job']==0 for t,r in rows)==1
    summary=rows[-1][1]
    assert (summary['seed'],summary['jobs'],summary['publics'],summary['pairs'],summary['resets'])==(seed,20,18,14,3)
    buckets={j:{k:[] for k in SCHEMA} for j in range(1,21)};last_job=0;last_cycle=-1
    for tag,row in rows:
        if 'cycle' in row:
            assert row['cycle']>=last_cycle;last_cycle=row['cycle']
        if tag=='BEGIN':
            assert row['job']==last_job+1;last_job=row['job']
        if row.get('job',0):
            assert row['job']==last_job;buckets[last_job][tag].append(row)
    assert last_job==20 and summary['cycles']>last_cycle
    totals={k:0 for k in ('ar','r','b','starts','dones')};accepted=0;exact_words=0;metrics=[]
    for job,mode in enumerate(PLAN,1):
        e=buckets[job];assert len(e['BEGIN'])==1;b=e['BEGIN'][0];sample=(job-1)%4
        assert (b['mode'],b['sample'],b['sequence'],b['epoch'])==(mode,sample,0x1234567800000000+job,0xfedcba9800000000+job)
        source=[0x8000000fc0+p*0x100000000+job*16384 for p in range(2)]
        dest=0x3300000fc0+job*16384;cursor=0
        for ar in e['AR']:
            assert cursor<256 and ar['address']==source[cursor//128]+(cursor%128)*64
            assert ar['beats']==min(128-cursor%128,(4096-ar['address']%4096)//64)
            responses=e['R'][cursor:cursor+ar['beats']];assert len(responses)==ar['beats']
            if cursor:assert e['R'][cursor-1]['cycle']<ar['cycle']
            for r in responses:
                assert r['index']==cursor and ar['cycle']<r['cycle'] and r['response']==0
                words=values['source32.hex'][sample*4096+cursor*16:sample*4096+cursor*16+16].copy()
                if mode==6:
                    for lane in range(16):
                        if (cursor*16+lane)%2048==17:words[lane]=0x7f7fffff
                assert r['data']==sum(w<<(32*l) for l,w in enumerate(words))
                if cursor:assert e['R'][cursor-1]['cycle']<r['cycle']
                cursor+=1
        assert cursor==len(e['R'])==256 and len(e['AR'])==6
        assert len(e['START'])==1 and e['R'][-1]['cycle']<e['START'][0]['cycle']
        if mode==18:assert not e['DONE'] and not e['WRITE']
        else:
            assert len(e['DONE'])==1 and e['START'][0]['cycle']<e['DONE'][0]['cycle']
            assert e['DONE'][0]['return']==(4 if mode==6 else 0)
            assert len(e['WRITE'])==(0 if mode==6 else 512)
        for i,w in enumerate(e['WRITE']):
            assert i<512 and w['index']==i and e['START'][0]['cycle']<w['cycle']<=e['DONE'][0]['cycle']
            if i:assert e['WRITE'][i-1]['cycle']<w['cycle']
            if mode!=10:
                assert w['bits']==values['abi64.hex'][sample*512+i];exact_words+=1
            elif i%8<4:assert all(v!=128 for v in w['bits'].to_bytes(8,'little'))
            elif i%8==4:assert w['bits']<=0x4780204070e1c387
            else:assert w['bits']==0
        for tag in ('AW','W','B'):
            for i,r in enumerate(e[tag]):
                assert i<64 and r['index']==i and len(e['DONE'])==1 and e['DONE'][0]['return']==0
                assert e['DONE'][0]['cycle']<r['cycle']
                if i:assert e['B'][i-1]['cycle']<r['cycle']
                if tag in ('AW','B'):assert r['address']==dest+i*64
                if tag in ('W','B'):
                    want=sum(values['abi64.hex'][sample*512+i*8+l]<<(64*l) for l in range(8))
                    assert r['data']==want
                if tag=='B':
                    assert e['AW'][i]['cycle']<r['cycle'] and e['W'][i]['cycle']<r['cycle']
                    assert r['response']==0 and r['data']==e['W'][i]['data'] and r['address']==e['AW'][i]['address']
        if mode==19:assert len(e['AW'])==1 and not e['W'] and not e['B']
        else:assert len(e['AW'])==len(e['W'])==len(e['B'])
        if mode in (18,19):
            assert len(e['ABORT'])==len(e['RESET'])==1 and not e['ACK'] and not e['COMPLETE']
            assert e['ABORT'][0]['mode']==mode and e['START'][0]['cycle']<e['ABORT'][0]['cycle']<e['RESET'][0]['cycle']
        else:
            assert not e['ABORT'] and not e['RESET'] and len(e['ACK'])==len(e['COMPLETE'])==1
            c=e['COMPLETE'][0];a=e['ACK'][0];status=STATUS.get(mode,0)
            assert (c['status'],c['hls'],c['valid'])==(status,4 if mode==6 else 0,int(status==0))
            assert 0<=c['max_quiet']<262144
            assert (c['read'],c['ar'],c['r'],c['b'])==(256,len(e['AR']),len(e['R']),len(e['B']))
            # A late OKAY response still drains the bus after the writer times out.
            assert c['abi']==len(e['B'])-(1 if mode==22 else 0)
            for tag in ('R','B','DONE'):
                if e[tag]:assert e[tag][-1]['cycle']<c['cycle']
            valid=int(status==0 and mode not in (14,15))
            assert (a['valid'],a['revoked'])==(valid,1-valid) and c['cycle']+5<=a['cycle'];accepted+=valid
            if status==0:
                assert len(e['B'])==64 and len(e['WRITE'])==512
                metrics.append(dict(job=job,sample=sample,accepted=bool(valid),
                    hls_cycles=e['DONE'][0]['cycle']-e['START'][0]['cycle'],
                    command_to_completion_cycles=c['cycle']-b['cycle'],max_hls_quiet_cycles=c['max_quiet'],
                    first_ar_to_last_b_cycles=e['B'][-1]['cycle']-e['AR'][0]['cycle']+1))
        if mode in (14,15):
            assert len(e['REVOKE'])==1 and e['REVOKE'][0]['mode']==mode
            assert e['COMPLETE'][0]['cycle']<e['REVOKE'][0]['cycle']<e['ACK'][0]['cycle']
        else:assert not e['REVOKE']
        if mode in (6,10,18):assert not e['AW'] and not e['W'] and not e['B']
        if mode==13:assert len(e['B'])==8
        if mode==22:
            assert len(e['B'])==64
            assert e['B'][-1]['cycle']-max(e['AW'][-1]['cycle'],e['W'][-1]['cycle'])>=1024
        for key,tag in [('ar','AR'),('r','R'),('b','B'),('starts','START'),('dones','DONE')]:totals[key]+=len(e[tag])
    assert all(summary[k]==v for k,v in totals.items())
    assert totals==dict(ar=120,r=5120,b=968,starts=20,dones=19) and accepted==12 and exact_words==8704
    return dict(seed=seed,jobs=20,completions=18,successful_offers=14,accepted_results=12,accepted_bytes=49152,
                exact_hls_words=exact_words,cycles=summary['cycles'],totals=totals,jobs_metrics=metrics,
                compiled_source_attestation_required=True,
                scope='Original norm/Q8 arithmetic output words, independent numerical references and AXI R/AW/W/B causality. No live QKV consumer or physical DDR/timing qualification.')


def review(text,values,seed):return review_events(parse(text),values,seed)


def controls(text,values,seed):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<40),('BEGIN','sample',1),('AR','address',64),('AR','beats',1),
        ('R','data',1),('R','response',2),('R','index',1),('AW','address',64),('AW','index',1),('W','data',1),('W','index',1),
        ('B','data',1),('B','response',2),('B','cycle',1<<40),('WRITE','bits',1),('WRITE','index',1),('DONE','return',1),
        ('DONE','cycle',1<<40),('COMPLETE','status',1),('COMPLETE','valid',1),('COMPLETE','hls',1),('COMPLETE','read',1),
        ('COMPLETE','abi',1),('COMPLETE','max_quiet',1<<40),('REVOKE','mode',1),('ABORT','mode',1),
        ('ACK','valid',1),('ACK','revoked',1),('PASS','pairs',1),('PASS','r',1)]
    def reject(name,bad):
        try:review_events(bad,values,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted control '+name)
    for tag,key,mask in changes:
        i=next(i for i,(t,_) in enumerate(rows) if t==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r)
        reject(tag+'.'+key,bad)
    for tag in ('AW','W','B','WRITE','START','DONE','COMPLETE','ACK','REVOKE','RESET'):
        i=next(i for i,(t,_) in enumerate(rows) if t==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    assert len(rejected)==51
    return rejected
