"""Review the full real attention/O/post-O/FFN/residual trace and retained ownership."""
import hashlib,re,struct
from pathlib import Path
import check_ntrpu_attention_ffn_upstream_r3 as upstream
import check_ntrpu_attention_post_real_r2 as post
import check_ntrpu_attention_projected_root_real_r2 as projection
import check_ntrpu_ffn_residual_real_r2 as downstream
from check_ntrpu_residual_r1 import exact_add

MASK=(1<<64)-1
SEQ=0x1234567800000001
EPOCH=0xfedcba9800000001
ABI=0xa400000fc0
RESIDUAL=0xa300000fc0
FINAL=0x710000000fc0
AF_SCHEMA={
    'CONFIG':'dim hidden sample token attention_position full_weight_bytes input_abi_requantized',
    'PROGRESS':'cycle ffn phase bytes','SWITCH':'cycle ffn',
    'ATTENTION_COMPLETE':'cycle sequence epoch residual abi',
    'FFN_ADMIT':'cycle sequence epoch abi residual',
    'SOURCE_R':'cycle buffer offset data',
    'COMPLETE':'cycle sequence epoch status attention ffn_residual ffn child residual stages phase ffn_bytes stage_bytes attention_hls post_hls ffn_hls hls post_reads post_residual post_abi reads writes valid address bytes',
    'ACK':'cycle sequence epoch attention ffn private_ffn residual',
    'PASS':'jobs seed cycles ar r aw w b bytes attention_done ffn_done first_res_ar last_res_b abi_reads residual_reads',
    'REUSABLE':'jobs',
}
FF_SCHEMA={k:v for k,v in downstream.SCHEMA.items() if k in ('FFN_CHILD','SWITCH','FFN_COMPLETE','RES_START','HLS_START','HLS_WRITE','HLS_DONE','AR','RES_R','AW','W','B','READ','PHASE')}
HEX=downstream.HEX|{'abi','attention_hls','post_hls'}

def reference(output_fixture,root_fixture,post_fixture,ffn_fixture,extra):
    values=post.fixture(output_fixture,root_fixture,post_fixture)
    ref=downstream.reference(ffn_fixture,extra)
    abi=b''.join(x.to_bytes(8,'little') for x in values['post']['abi64.hex'][:512])
    residual=b''.join(x.to_bytes(4,'little') for x in values['post']['residual32.hex'][:2048])
    assert abi==ref['buffers'][0] and residual==ref['buffers'][8]
    return values,ref

def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|FAIL\b|AFREAL_FAIL|PRREAL_FAIL|APREAL_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if line.startswith('AFREAL_'):prefix='A';schema=AF_SCHEMA;parts=line[7:].split()
        elif line.startswith('FFRREAL_'):prefix='F';schema=FF_SCHEMA;parts=line[8:].split()
        else:continue
        tag=parts[0];assert tag in schema,(prefix,tag)
        fields=[p.split('=') for p in parts[1:]]
        assert all(len(p)==2 for p in fields) and [p[0] for p in fields]==schema[tag].split(),line
        row={k:int(v,16 if k in HEX or (prefix=='A' and tag in ('ATTENTION_COMPLETE','FFN_ADMIT') and k=='residual') else 10) for k,v in fields}
        assert all(v>=0 for v in row.values());rows.append((prefix+tag,row))
    return rows

def review_link(rows,post_rows,root_rows,ref,seed=1):
    by={p+k:[] for p,schema in [('A',AF_SCHEMA),('F',FF_SCHEMA)] for k in schema}
    last=-1
    for tag,r in rows:
        assert tag in by
        if 'cycle' in r:assert last<=r['cycle']<33000000;last=r['cycle']
        by[tag].append(r)
    for tag in ('ACONFIG','AATTENTION_COMPLETE','AFFN_ADMIT','ACOMPLETE','AACK','APASS','AREUSABLE',
                'FFFN_CHILD','FFFN_COMPLETE','FRES_START','FHLS_START','FHLS_DONE'):
        assert len(by[tag])==1,tag
    assert rows[0][0]=='ACONFIG' and rows[-1]==('AREUSABLE',dict(jobs=1))
    assert by['ACONFIG'][0]==dict(dim=2048,hidden=8192,sample=1,token=0,attention_position=0,full_weight_bytes=57933824,input_abi_requantized=0)
    posts=[r for tag,r in post_rows if tag=='B'];assert len(posts)==192
    source={};source_cycles={}
    for i,r in enumerate(posts):
        k=8 if i<128 else 0;offset=i*64 if i<128 else (i-128)*64
        assert (r['job'],r['index'],r['address'])==(1,i,(RESIDUAL if k==8 else ABI)+offset)
        raw=r['data'].to_bytes(64,'little');assert raw==ref['buffers'][k][offset:offset+64]
        source[k,offset]=raw;source_cycles[k,offset]=r['cycle']
    producer=by['AATTENTION_COMPLETE'][0];admit=by['AFFN_ADMIT'][0];end=by['ACOMPLETE'][0];ack=by['AACK'][0]
    assert producer==dict(cycle=producer['cycle'],sequence=SEQ,epoch=EPOCH,residual=RESIDUAL,abi=ABI)
    assert admit==dict(cycle=admit['cycle'],sequence=SEQ,epoch=EPOCH,abi=ABI,residual=RESIDUAL)
    assert posts[-1]['cycle']<producer['cycle']<admit['cycle']<end['cycle']<ack['cycle']
    switches=by['ASWITCH'];assert len(switches)==2 and [r['ffn'] for r in switches]==[1,0]
    assert producer['cycle']<switches[0]['cycle']<=admit['cycle'] and ack['cycle']<switches[1]['cycle']
    fswitches=by['FSWITCH'];assert len(fswitches)==2 and [r['residual'] for r in fswitches]==[1,0]
    assert ack==dict(cycle=end['cycle']+7,sequence=SEQ,epoch=EPOCH,attention=1,ffn=1,private_ffn=1,residual=1)
    assert fswitches[1]['cycle']>ack['cycle'] and fswitches[1]['cycle']<=switches[1]['cycle']
    for kind in ('QACK','PACK','OUT_ACK','RELEASE'):
        events=[r for tag,r in root_rows if tag==kind];assert len(events)==1 and events[0]['cycle']==ack['cycle'],kind
    actual_reads=[dict(id=1,**{k:r[k] for k in ('phase','buffer','offset','beats')}) for r in by['FAR']]
    assert actual_reads==ref['reads'],'Full five-phase AXI read plan'
    expected_source=[]
    for i,ar in enumerate(by['FAR']):
        if ar['buffer'] not in (0,8):continue
        until=by['FAR'][i+1]['cycle'] if i+1<len(by['FAR']) else end['cycle']
        for beat in range(ar['beats']):expected_source.append((ar['buffer'],ar['offset']+64*beat,ar['cycle'],until))
    actual_source=by['ASOURCE_R'];assert len(actual_source)==len(expected_source)==32896
    for r,(k,off,start,until) in zip(actual_source,expected_source):
        assert (r['buffer'],r['offset'])==(k,off)
        assert producer['cycle']<admit['cycle']<start<r['cycle']<until<ack['cycle']
        assert source_cycles[k,off]<r['cycle'] and r['data']==int.from_bytes(source[k,off],'little')
    outputs=(4,5,6,7,9)
    expected_writes=[dict(buffer=k,offset=off,strb=MASK,data=int.from_bytes(ref['buffers'][k][off:off+64],'little')) for k in outputs for off in range(0,len(ref['buffers'][k]),64)]
    assert [{k:r[k] for k in ('buffer','offset','strb','data')} for r in by['FW']]==expected_writes
    assert by['FREAD']==[dict(buffer=k,beats=ref['read_beats'][k],hash=ref['read_hashes'][k]) for k in range(10)]
    produced={k:bytearray(len(ref['buffers'][k])) for k in outputs}
    committed={k:bytearray(len(raw)) for k,raw in produced.items()}
    pending=None;child=ffn=res=hls_start=hls_done=completion=None;res_pending=None;res_reads=0;hls_words=0
    for tag,r in rows:
        if tag=='FFFN_CHILD':
            assert admit['cycle']<r['cycle'] and r['tag']==0xfffffffc;child=r
        elif tag=='FAR':
            assert child and child['cycle']<r['cycle']<ack['cycle']
            k,off,size=r['buffer'],r['offset'],r['beats']*64
            if k in produced:
                assert all(committed[k][off:off+size]),'Read before B commit'
                assert produced[k][off:off+size]==ref['buffers'][k][off:off+size]
            if r['phase']==4:
                assert res and res['cycle']<r['cycle'] and res_pending is None and hls_start is None
                res_pending=dict(buffer=k,offset=off,left=r['beats'],cycle=r['cycle'])
        elif tag=='FRES_R':
            assert res_pending and res_pending['cycle']<r['cycle'] and r['buffer']==res_pending['buffer'] and r['offset']==res_pending['offset']
            k,off=r['buffer'],r['offset'];raw=bytes(produced[7][off:off+64]) if k==7 else source[8,off]
            assert r['data']==int.from_bytes(raw,'little');res_reads+=1;res_pending['offset']+=64;res_pending['left']-=1
            if not res_pending['left']:res_pending=None
        elif tag=='FAW':
            assert child and pending is None and r['buffer']==(9 if r['phase']==4 else r['phase']+4)
            assert r['offset']%64==0 and 1<=r['beats']<=4 and (4032+r['offset'])%4096+r['beats']*64<=4096
            if r['phase']==4:assert hls_done and hls_done['cycle']<r['cycle'] and r['beats']==1
            pending=dict(r,words=[])
        elif tag=='FW':
            assert pending and len(pending['words'])<pending['beats'] and pending['cycle']<=r['cycle']
            assert r['buffer']==pending['buffer'] and r['offset']==pending['offset']+len(pending['words'])*64
            pending['words'].append(r['data'].to_bytes(64,'little'))
        elif tag=='FB':
            assert pending and len(pending['words'])==pending['beats'] and pending['cycle']<r['cycle']
            assert {k:r[k] for k in ('phase','buffer','offset','beats')}=={k:pending[k] for k in ('phase','buffer','offset','beats')} and r['resp']==0
            k=r['buffer'];off=r['offset'];raw=b''.join(pending['words']);stop=off+len(raw)
            assert stop<=len(produced[k]) and not any(committed[k][off:stop])
            produced[k][off:stop]=raw;committed[k][off:stop]=b'\1'*len(raw);pending=None
        elif tag=='FFFN_COMPLETE':
            assert pending is None and all(all(committed[k]) for k in (4,5,6,7))
            assert r==dict(cycle=r['cycle'],sequence=SEQ,epoch=EPOCH,status=0,child=0,stages=4,phase=3,bytes=8192,stage_bytes=155648,hls=0);ffn=r
        elif tag=='FSWITCH' and r['residual']==1:
            assert ffn and r['cycle']>ffn['cycle'] and pending is None
        elif tag=='FRES_START':
            assert ffn and ffn['cycle']<r['cycle'] and fswitches[0]['cycle']<=r['cycle']
            assert r==dict(cycle=r['cycle'],sequence=SEQ,epoch=EPOCH,input=RESIDUAL,branch=0x1e0000fc0,output=FINAL);res=r
        elif tag=='FHLS_START':
            assert res and res_reads==256 and res_pending is None and r['cycle']>by['FRES_R'][-1]['cycle'];hls_start=r
        elif tag=='FHLS_WRITE':
            assert hls_start and hls_done is None and r['cycle']>hls_start['cycle'] and r['index']==hls_words
            off=hls_words*4;beat=off//64*64;lane=off%64
            x=int.from_bytes(source[8,beat][lane:lane+4],'little');y=int.from_bytes(produced[7][off:off+4],'little')
            assert r['bits']==exact_add(x,y)==int.from_bytes(ref['buffers'][9][off:off+4],'little');hls_words+=1
        elif tag=='FHLS_DONE':
            assert hls_start and hls_words==2048 and r['return']==0 and r['cycle']>=by['FHLS_WRITE'][-1]['cycle'];hls_done=r
        elif tag=='ACOMPLETE':
            assert ffn and hls_done and pending is None and res_pending is None and all(all(v) for v in committed.values())
            assert all(bytes(v)==ref['buffers'][k] for k,v in produced.items())
            assert r==dict(cycle=r['cycle'],sequence=SEQ,epoch=EPOCH,status=0,attention=0,ffn_residual=0,ffn=0,child=0,residual=0,stages=4,phase=3,ffn_bytes=8192,stage_bytes=155648,attention_hls=0,post_hls=0,ffn_hls=0,hls=0,post_reads=384,post_residual=128,post_abi=64,reads=256,writes=128,valid=1,address=FINAL,bytes=8192)
            assert r['cycle']>by['FB'][-1]['cycle'];completion=r
        elif tag=='AACK':assert completion is not None
    assert completion and res_reads==256 and hls_words==2048 and pending is None and res_pending is None
    summary=by['APASS'][0];phases=by['FPHASE'];assert [p['phase'] for p in phases]==list(range(5))
    assert sum(p['cycles'] for p in phases)==switches[1]['cycle']-switches[0]['cycle'] and all(p['cycles']>0 for p in phases)
    assert [p['reads']*64 for p in phases]==[18874368,18874368,131072,18874368,16384]
    assert [p['writes']*64 for p in phases]==[65536,65536,16384,8192,8192]
    assert all(max(p['reads'],p['writes'])<=p['cycles'] for p in phases)
    assert {k:summary[k] for k in ('jobs','seed','ar','r','aw','w','b','bytes','abi_reads','residual_reads')}==dict(jobs=1,seed=seed,ar=16486,r=887040,aw=994,w=2560,b=994,bytes=163840,abi_reads=32768,residual_reads=128)
    assert len(by['FAW'])==len(by['FB'])==994 and len(by['FW'])==2560
    assert summary['attention_done']==producer['cycle'] and summary['ffn_done']==ffn['cycle']
    assert summary['first_res_ar']==next(r['cycle'] for r in by['FAR'] if r['phase']==4)
    assert summary['last_res_b']==[r['cycle'] for r in by['FB'] if r['phase']==4][-1]
    assert summary['cycles']>=switches[1]['cycle']>ack['cycle']
    progress=by['APROGRESS'];assert [r['cycle'] for r in progress]==list(range(100000,summary['cycles']+1,100000))
    for r in progress:
        assert r['ffn']==int(switches[0]['cycle']<=r['cycle']<switches[1]['cycle']) and 0<=r['phase']<=4
        assert r['bytes']%64==0 and 0<=r['bytes']<=163840
        written=sum(64 for w in by['FW'] if w['cycle']<r['cycle'])
        assert r['bytes'] in (written,written+64)  # W and progress monitors can run in either order at this edge.
    return dict(status='PASS_ATTENTION_FFN_REAL_R1',jobs=1,source_weight_bytes=57933824,downstream_read_bytes=56770560,
        downstream_written_bytes=163840,final_bytes=8192,final_sha256=hashlib.sha256(produced[9]).hexdigest(),
        live_abi_read_bytes=32768*64,live_residual_read_bytes=8192,summary=summary,phases=phases,
        attention_publication_cycle=producer['cycle'],ffn_admission_cycle=admit['cycle'],ffn_completion_cycle=ffn['cycle'],
        final_publication_cycle=end['cycle'],final_ack_cycle=ack['cycle'],residual_hls_cycles=hls_done['cycle']-hls_start['cycle'],
        physical_timing_qualified=False,full_token_inference_qualified=False)

def review(text,values,ref,seed=1):
    images=[{k:int(v,16 if k=='fnv' else 10) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)} for line in text.splitlines() if line.startswith('NTRPU_LLAMA32_WEIGHT_IMAGE ')]
    assert images==[dict(buffer=k,bytes=17829824,fnv=h) for k,h in enumerate((0xd22214a430288927,0x2f357a92b56df048,0x0ad507f50ff8e421),1)]
    rr=projection.parse(text);pr=post.parse(text);rows=parse(text)
    proof=upstream.review_events(rr,pr,values,seed)
    result=review_link(rows,pr,rr,ref,seed)
    result['upstream']={k:v for k,v in proof.items() if k!='post'}
    result['post']={k:v for k,v in proof['post'].items() if k!='source_beats'}
    return result

def controls(text,values,ref,seed=1):
    review(text,values,ref,seed)
    rows=parse(text);rr=projection.parse(text);pr=post.parse(text);rejected=[]
    def reject(label,changed,p=pr,r=rr):
        try:review_link(changed,p,r,ref,seed)
        except (AssertionError,KeyError,IndexError,StopIteration,OverflowError):rejected.append(label)
        else:raise AssertionError('Accepted '+label)
    changes=[('ACONFIG','sample',1),('ACONFIG','input_abi_requantized',1),('AATTENTION_COMPLETE','sequence',1<<40),('AATTENTION_COMPLETE','residual',64),
        ('AFFN_ADMIT','epoch',1<<40),('AFFN_ADMIT','abi',64),('ASOURCE_R','data',1),('ASOURCE_R','offset',64),('ASOURCE_R','buffer',8),
        ('FFFN_CHILD','tag',4),('FAR','offset',64),('FAR','beats',1),('FAR','buffer',1),('FAW','offset',64),('FW','data',1),('FW','strb',1),('FB','resp',1),
        ('FFFN_COMPLETE','stage_bytes',1),('FRES_START','input',64),('FRES_START','epoch',1<<40),('FRES_R','data',1),('FHLS_WRITE','bits',1),('FHLS_WRITE','index',1),('FHLS_DONE','return',1),
        ('ACOMPLETE','sequence',1<<40),('ACOMPLETE','status',1),('ACOMPLETE','attention',1),('ACOMPLETE','post_abi',1),('ACOMPLETE','reads',1),('ACOMPLETE','address',64),('ACOMPLETE','valid',1),
        ('AACK','cycle',1),('AACK','attention',1),('AACK','ffn',1),('AACK','private_ffn',1),('AACK','residual',1),('FREAD','hash',1),('FPHASE','cycles',1),('APASS','bytes',1)]
    for tag,key,mask in changes:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);changed=rows.copy();r=dict(rows[i][1]);r[key]^=mask;changed[i]=(tag,r);reject(tag+'.'+key,changed)
    for tag in ('AATTENTION_COMPLETE','AFFN_ADMIT','ASOURCE_R','FB','FRES_R','FHLS_WRITE','FFFN_COMPLETE','FRES_START','FHLS_START','FHLS_DONE','ACOMPLETE','AACK','ASWITCH','FSWITCH','AREUSABLE'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    for k,off in ((0,0),(0,4032),(8,0),(8,8128)):
        i=next(i for i,(tag,r) in enumerate(rows) if tag=='ASOURCE_R' and r['buffer']==k and r['offset']==off)
        changed=rows.copy();r=dict(rows[i][1]);r['data']^=1;changed[i]=(rows[i][0],r);reject(f'live-source-{k}-{off}',changed)
    for index in (0,127,128,191):
        i=next(i for i,(tag,r) in enumerate(pr) if tag=='B' and r['index']==index)
        changed=pr.copy();r=dict(pr[i][1]);r['data']^=1;changed[i]=(pr[i][0],r);reject('producer-B-'+str(index),rows,p=changed)
    i=next(i for i,(tag,r) in enumerate(rr) if tag=='PACK');changed=rr.copy();r=dict(rr[i][1]);r['cycle']+=1;changed[i]=(rr[i][0],r);reject('split-upstream-ACK',rows,r=changed)
    done=next(i for i,(k,r) in enumerate(rows) if k=='FHLS_DONE')
    last=max(i for i,(k,r) in enumerate(rows) if k=='FHLS_WRITE')
    assert last<done
    bad=rows.copy();kind,row=bad[done];row=dict(row);row['cycle']=rows[last][1]['cycle']-1;bad[done]=(kind,row)
    reject('done-before-final-write-cycle',bad)
    bad=rows.copy();event=bad.pop(done);bad.insert(last,event)
    reject('done-before-final-write-event',bad)
    reject('missing-final-hls-write',rows[:last]+rows[last+1:])
    bad=rows.copy();kind,row=bad[last];row=dict(row);row['bits']^=1;bad[last]=(kind,row)
    reject('corrupt-final-hls-write',bad)
    assert len(rejected)==82
    return rejected
