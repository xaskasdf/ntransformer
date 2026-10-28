"""Review actual root -> O AXI B -> post-O arithmetic and paired publication."""
import copy
import re
import check_ntrpu_attention_projected_root_real_r2 as root
import check_ntrpu_post_attention_real_r1 as post

SCHEMA = {
    'HANDOFF': 'cycle job sequence epoch address bytes',
    'AR': 'cycle job address beats', 'R': 'cycle job index address data',
    'START': 'cycle job', 'DONE': 'cycle job return',
    'B': 'cycle job index address data',
    'COMPLETE': 'cycle job status valid bytes ar r b hls max_quiet',
    'PASS': 'seed jobs pairs bytes ar r b cycles',
}
HEX = {'sequence', 'epoch', 'address', 'data'}


def fixture(output_folder, root_folder, post_folder):
    values = root.load_fixture(output_folder)
    values = root.load_root_fixture(root_folder, values)
    paired = post.fixture(post_folder)
    # Validate independent fixture agreement; the DUT reads actual committed O.
    for sample in range(4):
        words = paired['source32.hex'][sample*6144+2048:sample*6144+4096]
        packed = [sum(words[i+k] << (32*k) for k in range(16)) for i in range(0,2048,16)]
        assert packed == values['samples'][sample]['fp32']
    return dict(root=values, post=paired)


def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|APREAL_FAIL|PRREAL_FAIL)', text)
    rows = []
    for line in text.splitlines():
        if not line.startswith('APREAL_'):
            continue
        parts = line.split(); tag = parts[0][7:]
        assert tag in SCHEMA
        fields = [p.split('=') for p in parts[1:]]
        assert all(len(p)==2 for p in fields) and [p[0] for p in fields]==SCHEMA[tag].split()
        row = {k:int(v,16 if k in HEX else 10) for k,v in fields}
        assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows


def review_post(rows, root_rows, values, seed):
    assert rows and rows[-1][0]=='PASS' and sum(k=='PASS' for k,r in rows)==1
    summary = rows[-1][1]
    assert tuple(summary[k] for k in ('seed','jobs','pairs','bytes','ar','r','b'))==(seed,3,2,24576,18,768,384)
    buckets = {j:{k:[] for k in SCHEMA} for j in range(1,4)}
    root_buckets = {j:{k:[] for k in root.SCHEMA} for j in range(1,4)}
    for k,r in root_rows:
        if 'job' in r: root_buckets[r['job']][k].append(r)
    cycle=-1;last_job=1
    for k,r in rows[:-1]:
        assert r['cycle']>=cycle and r['job']>=last_job
        cycle=r['cycle'];last_job=r['job'];buckets[r['job']][k].append(r)
    assert summary['cycles']==root_rows[-1][1]['cycles'] and summary['cycles']>cycle
    metrics=[]
    for j,(mode,sample) in enumerate(root.PLAN,1):
        e=buckets[j];r=root_buckets[j]
        assert len(e['COMPLETE'])==1
        c=e['COMPLETE'][0];rc=r['COMPLETE'][0]
        assert c['cycle']+7==rc['cycle'] and (c['status'],c['valid'])==(rc['status'],rc['valid'])
        assert (c['ar'],c['r'],c['b'])==tuple(len(e[k]) for k in ('AR','R','B'))
        assert c['hls']==0 and 0<=c['max_quiet']<262144
        if mode:
            assert c['status']==7 and c['bytes']==c['ar']==c['r']==c['b']==c['max_quiet']==0
            assert all(not e[k] for k in ('HANDOFF','AR','R','START','DONE','B'))
            continue
        assert c['status']==0 and c['bytes']==12288
        assert all(len(e[k])==1 for k in ('HANDOFF','START','DONE'))
        h=e['HANDOFF'][0];start=e['START'][0];done=e['DONE'][0]
        assert (h['sequence'],h['epoch'],h['address'],h['bytes'])==(rc['sequence'],rc['epoch'],root.OBASE+4032,8192)
        assert r['MATRIX_DONE'][0]['cycle']<h['cycle']<start['cycle']<done['cycle']<c['cycle']
        assert done['return']==0 and 0<done['cycle']-start['cycle']<262144
        # Reconstruct the only legal post-O branch from accepted root W and B.
        committed={};cursor=0
        for aw,b in zip(r['AW'],r['B']):
            assert b['response']==0 and b['cycle']<h['cycle']
            for word in r['W'][cursor:cursor+aw['beats']]:
                address=aw['address']+(word['word']-cursor)*64
                assert address not in committed and word['cycle']<b['cycle']
                committed[address]=(word['data'],b['cycle'])
            cursor+=aw['beats']
        assert len(committed)==128
        bases=[0xa100000fc0,0xa000000fc0,0xa200000fc0];cursor=0
        for ar in e['AR']:
            plane=cursor//128;offset=cursor%128
            assert plane<3 and ar['address']==bases[plane]+64*offset
            beats=min(128-offset,(4096-ar['address']%4096)//64)
            assert ar['beats']==beats and h['cycle']<ar['cycle']<start['cycle']
            if cursor: assert e['R'][cursor-1]['cycle']<ar['cycle']
            for beat in range(beats):
                reply=e['R'][cursor];address=ar['address']+beat*64
                assert (reply['index'],reply['address'])==(cursor,address)
                assert ar['cycle']<reply['cycle']<start['cycle']
                if cursor: assert e['R'][cursor-1]['cycle']<reply['cycle']
                if plane==1:
                    want,commit_cycle=committed[address]
                    assert commit_cycle<reply['cycle']
                else:
                    index=sample*6144+plane*2048+(offset+beat)*16
                    want=sum(values['source32.hex'][index+lane]<<(32*lane) for lane in range(16))
                assert reply['data']==want
                cursor+=1
        assert cursor==len(e['R'])==384 and len(e['AR'])==9 and len(e['B'])==192
        for i,b in enumerate(e['B']):
            address=0xa300000fc0+64*i if i<128 else 0xa400000fc0+64*(i-128)
            assert (b['index'],b['address'])==(i,address) and done['cycle']<b['cycle']<c['cycle']
            if i: assert e['B'][i-1]['cycle']<b['cycle']
            if i<128:
                want=sum(values['residual32.hex'][sample*2048+i*16+k]<<(32*k) for k in range(16))
            else:
                want=sum(values['abi64.hex'][sample*512+(i-128)*8+k]<<(64*k) for k in range(8))
            assert b['data']==want
        assert r['OUT_ACK'][0]['cycle']>c['cycle']
        metrics.append(dict(job=j,sample=sample,hls_cycles=done['cycle']-start['cycle'],
                            post_cycles=c['cycle']-h['cycle'],max_quiet=c['max_quiet']))
    return dict(jobs=3,published_pairs=2,published_bytes=24576,cycles=summary['cycles'],jobs_metrics=metrics,
                committed_O_bytes_consumed=16384,post_read_bytes=49152,post_written_bytes=24576,
                physical_timing_qualified=False,full_inference_qualified=False)


def review(text,values,seed=1):
    proof=root.review(text,values['root'],seed)
    result=review_post(parse(text),root.parse(text),values['post'],seed)
    result['projected_root']=proof
    return result


def controls(text,values,seed=1):
    rows=parse(text);rr=root.parse(text);rejected=[]
    changes=[('HANDOFF','sequence',1<<40),('HANDOFF','epoch',1<<40),('HANDOFF','address',64),
             ('HANDOFF','bytes',64),('HANDOFF','cycle',1<<40),('AR','address',64),('AR','beats',1),
             ('R','data',1),('R','index',1),('R','address',64),('START','cycle',1<<40),
             ('DONE','return',1),('DONE','cycle',1<<40),('B','data',1),('B','address',64),
             ('B','index',1),('B','cycle',1<<40),('COMPLETE','status',1),('COMPLETE','valid',1),
             ('COMPLETE','bytes',64),('COMPLETE','cycle',1),('COMPLETE','max_quiet',1<<40),('PASS','pairs',1),('PASS','cycles',1)]
    for tag,key,mask in changes:
        bad=copy.deepcopy(rows);i=next(i for i,(k,r) in enumerate(bad) if k==tag);bad[i][1][key]^=mask
        try:review_post(bad,rr,values['post'],seed)
        except (AssertionError,KeyError,IndexError):rejected.append(tag+'.'+key)
        else:raise AssertionError('Accepted '+tag+'.'+key)
    for tag,selector in [('R',lambda r:r['index']==128),('B',lambda r:r['index']==191),('HANDOFF',lambda r:True)]:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag and selector(r))
        try:review_post(rows[:i]+rows[i+1:],rr,values['post'],seed)
        except (AssertionError,KeyError,IndexError):rejected.append('missing-'+tag)
        else:raise AssertionError('Accepted missing '+tag)
    return dict(root=root.controls(text,values['root'],seed),post=rejected)
