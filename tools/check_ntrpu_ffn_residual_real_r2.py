"""Reconcile real full FFN, B-committed residual inputs, exact FP32 sum and final lease."""
import hashlib,json,re,struct
from pathlib import Path
from check_ntrpu_llama32_owned_ffn_arithmetic_r1 import reference as ffn_reference,metadata as ffn_metadata
from check_ntrpu_residual_r1 import exact_add

MASK=(1<<64)-1
SCHEMA={
    'CONFIG':'dim hidden tile_k weight_bytes input_abi_requantized',
    'BEGIN':'cycle sequence epoch source_sample token residual output','ADMIT':'cycle',
    'FFN_CHILD':'cycle tag','SWITCH':'cycle residual',
    'FFN_COMPLETE':'cycle sequence epoch status child stages phase bytes stage_bytes hls',
    'RES_START':'cycle sequence epoch input branch output','RELEASE':'cycle private_ffn ffn residual',
    'HLS_START':'cycle','HLS_WRITE':'cycle index bits','HLS_DONE':'cycle return',
    'AR':'cycle phase buffer offset beats','RES_R':'cycle buffer offset data',
    'AW':'cycle phase buffer offset beats','W':'cycle buffer offset strb data','B':'cycle phase buffer offset beats resp',
    'PROGRESS':'cycle phase bytes',
    'COMPLETE':'cycle sequence epoch status ffn child residual stages phase ffn_bytes stage_bytes ffn_hls hls reads writes valid address bytes',
    'ACK':'cycle','READ':'buffer beats hash','PHASE':'phase cycles reads writes',
    'PASS':'jobs seed cycles ar r aw w b bytes ffn_done first_res_ar last_res_b ar_stalls aw_stalls w_stalls done_stalls',
    'REUSABLE':'jobs',
}
HEX={'sequence','epoch','tag','input','branch','output','address','data','strb','hash','bits','hls','ffn_hls'}

def reference(fixture,extra):
    ref=ffn_reference(Path(fixture));extra=Path(extra);m=json.loads((extra/'manifest.json').read_text())
    assert m['format']=='NTRFFNRESIDUALEXTRA01' and (m['dim'],m['hidden'],m['source_sample'],m['token'])==(2048,8192,1,0)
    assert m['model_sha256']=='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
    assert m['parent_manifest_sha256']=='f2e3edda6ce1e3b5a6fca72886501e118584cd6004412772c279b7c615df5d09'
    assert m['ffn_manifest_sha256']=='bf36c6afbe46df663aa9823769f6c6c74e1608826fce7182e21ffe42646e0194'
    raw={n:(extra/n).read_bytes() for n in m['files']}
    for n,h in m['files'].items():assert Path(n).name==n and hashlib.sha256(raw[n]).hexdigest()==h
    assert hashlib.sha256(raw['parent-manifest.json']).hexdigest()==m['parent_manifest_sha256']
    parent=json.loads(raw['parent-manifest.json'])
    for suffix in ('residual-fp32','final-fp32','layer-output-fp32'):
        n='job-001-'+suffix+'.bin';assert hashlib.sha256(raw[n]).hexdigest()==parent['files'][n] and len(raw[n])==8192
    assert raw['job-001-final-fp32.bin']==ref['buffers'][7]
    ref['buffers'][8]=raw['job-001-residual-fp32.bin'];ref['buffers'][9]=raw['job-001-layer-output-fp32.bin']
    x,y,z=[struct.unpack('<2048I',ref['buffers'][b]) for b in (8,7,9)]
    assert [exact_add(a,b) for a,b in zip(x,y)]==list(z)
    assert bytes.fromhex(raw['residual-input.hex'].decode())==b'\xa5'*4032+ref['buffers'][8]
    assert bytes.fromhex(raw['layer-output.hex'].decode())==ref['buffers'][9]
    for b in (8,9):ref['read_hashes'][b]=0xcbf29ce484222325;ref['read_beats'][b]=0
    for b in (8,7):
        offset=0
        while offset<8192:
            beats=min((8192-offset)//64,(4096-(4032+offset)%4096)//64)
            ref['reads'].append(dict(id=1,phase=4,buffer=b,offset=offset,beats=beats))
            for word, in struct.iter_unpack('<Q',ref['buffers'][b][offset:offset+64*beats]):
                ref['read_hashes'][b]=((ref['read_hashes'][b]^word)*0x100000001b3)&MASK
            ref['read_beats'][b]+=beats;offset+=64*beats
    ref['extra_manifest_sha256']=hashlib.sha256((extra/'manifest.json').read_bytes()).hexdigest()
    return ref

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|ERROR:|FAIL\b)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('FFRREAL_'):continue
        parts=line.split();tag=parts[0][8:];assert tag in SCHEMA,tag
        fields=[f.split('=') for f in parts[1:]]
        assert all(len(f)==2 for f in fields) and [k for k,v in fields]==SCHEMA[tag].split(),line
        # residual is an address only in BEGIN, a status/ownership bit elsewhere.
        row={k:int(v,16 if k in HEX or (tag=='BEGIN' and k=='residual') else 10) for k,v in fields}
        assert all(v>=0 for v in row.values());rows.append((tag,row))
    return rows

def review_events(rows,ref,seed=1):
    by={k:[r for tag,r in rows if tag==k] for k in SCHEMA}
    for tag in ('CONFIG','BEGIN','ADMIT','FFN_CHILD','FFN_COMPLETE','RES_START','SWITCH','HLS_START','HLS_DONE','COMPLETE','RELEASE','ACK','PASS','REUSABLE'):
        assert len(by[tag])==1,tag
    assert rows[0][0]=='CONFIG' and rows[-1]==('REUSABLE',dict(jobs=1))
    assert by['CONFIG'][0]==dict(dim=2048,hidden=8192,tile_k=32,weight_bytes=53477376,input_abi_requantized=0)
    seq=0x12345678fffffffe;epoch=0xfedcba9800000259
    assert by['BEGIN'][0]==dict(cycle=0,sequence=seq,epoch=epoch,source_sample=1,token=0,residual=0x700000000fc0,output=0x710000000fc0)
    actual_reads=[dict(id=1,**{k:r[k] for k in ('phase','buffer','offset','beats')}) for r in by['AR']]
    assert actual_reads==ref['reads'],'Complete five-phase AR plan'
    outputs=(4,5,6,7,9)
    expected_writes=[dict(buffer=b,offset=off,strb=MASK,data=int.from_bytes(ref['buffers'][b][off:off+64],'little')) for b in outputs for off in range(0,len(ref['buffers'][b]),64)]
    assert [{k:r[k] for k in ('buffer','offset','strb','data')} for r in by['W']]==expected_writes
    assert by['READ']==[dict(buffer=b,beats=ref['read_beats'][b],hash=ref['read_hashes'][b]) for b in range(10)]
    produced={b:bytearray(len(ref['buffers'][b])) for b in outputs}
    committed={b:bytearray(len(raw)) for b,raw in produced.items()}
    last=-1;pending=None;child=ffn=res=hls_start=hls_done=completion=ack=None
    residual_read_cursor=0;res_pending=None;hls_words=0
    for tag,r in rows:
        if 'cycle' in r:assert last<=r['cycle']<30000000;last=r['cycle']
        if tag=='FFN_CHILD':
            assert by['ADMIT'][0]['cycle']<r['cycle'] and r['tag']==0xfffffffc;child=r
        elif tag=='AR':
            assert child and child['cycle']<r['cycle'] and ack is None
            b,off,size=r['buffer'],r['offset'],r['beats']*64
            if b in produced:
                assert all(committed[b][off:off+size]),'Read before B commit'
                assert produced[b][off:off+size]==ref['buffers'][b][off:off+size]
            if r['phase']==4:
                assert res and res['cycle']<r['cycle'] and res_pending is None and hls_start is None
                res_pending=dict(buffer=b,offset=off,left=r['beats'],cycle=r['cycle'])
        elif tag=='RES_R':
            assert res_pending and res_pending['cycle']<r['cycle'] and r['buffer']==res_pending['buffer'] and r['offset']==res_pending['offset']
            b=r['buffer'];off=r['offset'];raw=bytes(produced[7][off:off+64]) if b==7 else ref['buffers'][8][off:off+64]
            assert r['data']==int.from_bytes(raw,'little')
            residual_read_cursor+=1;res_pending['offset']+=64;res_pending['left']-=1
            if not res_pending['left']:res_pending=None
        elif tag=='AW':
            assert child and pending is None and r['buffer']==(9 if r['phase']==4 else r['phase']+4)
            assert r['offset']%64==0 and 1<=r['beats']<=4 and (4032+r['offset'])%4096+r['beats']*64<=4096
            if r['phase']==4:assert hls_done and hls_done['cycle']<r['cycle'] and r['beats']==1
            pending=dict(r,words=[])
        elif tag=='W':
            assert pending and len(pending['words'])<pending['beats'] and pending['cycle']<=r['cycle']
            assert r['buffer']==pending['buffer'] and r['offset']==pending['offset']+len(pending['words'])*64
            pending['words'].append(r['data'].to_bytes(64,'little'))
        elif tag=='B':
            assert pending and len(pending['words'])==pending['beats'] and pending['cycle']<r['cycle']
            assert {k:r[k] for k in ('phase','buffer','offset','beats')}=={k:pending[k] for k in ('phase','buffer','offset','beats')} and r['resp']==0
            b=r['buffer'];off=r['offset'];raw=b''.join(pending['words']);end=off+len(raw)
            assert end<=len(produced[b]) and not any(committed[b][off:end])
            produced[b][off:end]=raw;committed[b][off:end]=b'\1'*len(raw);pending=None
        elif tag=='FFN_COMPLETE':
            assert pending is None and all(all(committed[b]) for b in (4,5,6,7))
            assert r==dict(cycle=r['cycle'],sequence=seq,epoch=epoch,status=0,child=0,stages=4,phase=3,bytes=8192,stage_bytes=155648,hls=0);ffn=r
        elif tag=='SWITCH':assert ffn and r['cycle']>ffn['cycle'] and r['residual']==1 and pending is None
        elif tag=='RES_START':
            assert ffn and ffn['cycle']<r['cycle'] and by['SWITCH'][0]['cycle']<=r['cycle']
            assert r==dict(cycle=r['cycle'],sequence=seq,epoch=epoch,input=0x700000000fc0,branch=0x1e0000fc0,output=0x710000000fc0);res=r
        elif tag=='HLS_START':
            assert res and residual_read_cursor==256 and res_pending is None and r['cycle']>by['RES_R'][-1]['cycle'];hls_start=r
        elif tag=='HLS_WRITE':
            assert hls_start and hls_done is None and r['cycle']>hls_start['cycle'] and r['index']==hls_words
            off=hls_words*4;x=int.from_bytes(ref['buffers'][8][off:off+4],'little');y=int.from_bytes(produced[7][off:off+4],'little')
            assert r['bits']==exact_add(x,y)==int.from_bytes(ref['buffers'][9][off:off+4],'little');hls_words+=1
        elif tag=='HLS_DONE':assert hls_start and hls_words==2048 and r['return']==0 and r['cycle']>=by['HLS_WRITE'][-1]['cycle'];hls_done=r
        elif tag=='COMPLETE':
            assert ffn and hls_done and pending is None and res_pending is None and all(all(v) for v in committed.values())
            assert all(bytes(v)==ref['buffers'][b] for b,v in produced.items())
            assert r==dict(cycle=r['cycle'],sequence=seq,epoch=epoch,status=0,ffn=0,child=0,residual=0,stages=4,phase=3,ffn_bytes=8192,stage_bytes=155648,ffn_hls=0,hls=0,reads=256,writes=128,valid=1,address=0x710000000fc0,bytes=8192)
            assert r['cycle']>by['B'][-1]['cycle'];completion=r
        elif tag=='RELEASE':
            assert completion and r==dict(cycle=completion['cycle']+6,private_ffn=1,ffn=1,residual=1)
        elif tag=='ACK':assert completion and r['cycle']==completion['cycle']+6==by['RELEASE'][0]['cycle'];ack=r
    assert ack and residual_read_cursor==256 and hls_words==2048 and pending is None
    summary=by['PASS'][0];phases=by['PHASE'];assert [p['phase'] for p in phases]==list(range(5))
    assert sum(p['cycles'] for p in phases)==summary['cycles']==ack['cycle'] and all(p['cycles']>0 for p in phases)
    assert [p['reads']*64 for p in phases]==[18874368,18874368,131072,18874368,16384]
    assert [p['writes']*64 for p in phases]==[65536,65536,16384,8192,8192]
    assert all(max(p['reads'],p['writes'])<=p['cycles'] for p in phases)
    assert {k:summary[k] for k in ('jobs','seed','ar','r','aw','w','b','bytes')}==dict(jobs=1,seed=seed,ar=16486,r=56770560//64,aw=994,w=2560,b=994,bytes=163840)
    assert len(by['AW'])==len(by['B'])==994 and summary['ffn_done']==ffn['cycle']
    res_ars=[r for r in by['AR'] if r['phase']==4];res_bs=[r for r in by['B'] if r['phase']==4]
    assert summary['first_res_ar']==res_ars[0]['cycle'] and summary['last_res_b']==res_bs[-1]['cycle']
    for k in ('ar_stalls','aw_stalls','w_stalls'):assert 0<summary[k]<=summary['cycles']
    assert summary['done_stalls']==6
    progress=by['PROGRESS'];assert [r['cycle'] for r in progress]==list(range(100000,summary['cycles']+1,100000))
    ends=[];acc=0
    for p in phases:acc+=p['cycles'];ends.append(acc)
    sizes=[65536,65536,16384,8192,8192]
    for r in progress:
        phase=next(i for i,end in enumerate(ends) if r['cycle']<=end)
        assert r['phase']==phase and sum(sizes[:phase])<=r['bytes']<=sum(sizes[:phase+1]) and r['bytes']%64==0
    return dict(status='PASS_FFN_RESIDUAL_REAL_R1',jobs=1,source_weight_bytes=53477376,verified_input_bytes=56770560,verified_output_bytes=163840,
        final_output_bytes=8192,final_output_sha256=hashlib.sha256(produced[9]).hexdigest(),summary=summary,phases=phases,
        ffn_completion_cycle=ffn['cycle'],residual_first_ar_to_last_b_cycles=summary['last_res_b']-summary['first_res_ar']+1,
        residual_hls_cycles=hls_done['cycle']-hls_start['cycle'],public_completion_cycle=completion['cycle'],ack_cycle=ack['cycle'],
        physical_timing_qualified=False,scope='One complete actual 2048/8192 FFN with original Q8 matrices, exact post-O reference ABI1, B-committed output feeding actual FP32 residual. Live post-O, full inference and physical timing remain pending.')

def review(text,ref,seed=1):
    images=[{k:int(v,16 if k=='fnv' else 10) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)} for line in text.splitlines() if line.startswith('NTRPU_LLAMA32_WEIGHT_IMAGE ')]
    assert images==[dict(buffer=b,bytes=17829824,fnv=h) for b,h in enumerate((0xd22214a430288927,0x2f357a92b56df048,0x0ad507f50ff8e421),1)]
    return review_events(parse(text),ref,seed)

def controls(text,ref,seed=1):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<40),('BEGIN','source_sample',1),('BEGIN','residual',64),
        ('FFN_CHILD','tag',4),('AR','buffer',1),('AR','offset',64),('AR','beats',1),('AW','offset',64),('W','data',1),('W','strb',1),('B','resp',1),
        ('FFN_COMPLETE','stage_bytes',1),('FFN_COMPLETE','status',1),('RES_START','branch',64),('RES_START','epoch',1<<40),
        ('RES_R','data',1),('RES_R','offset',64),('HLS_WRITE','bits',1),('HLS_WRITE','index',1),('HLS_DONE','return',1),
        ('COMPLETE','sequence',1<<40),('COMPLETE','status',1),('COMPLETE','residual',1),('COMPLETE','reads',1),('COMPLETE','writes',1),('COMPLETE','address',64),('COMPLETE','valid',1),
        ('RELEASE','ffn',1),('RELEASE','private_ffn',1),('RELEASE','residual',1),('ACK','cycle',1),('PHASE','cycles',1),('READ','hash',1),('PASS','bytes',1)]
    def reject(name,bad):
        try:review_events(bad,ref,seed)
        except (AssertionError,KeyError,IndexError,StopIteration):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    for tag,key,mask in changes:
        i=next(i for i,(k,r) in enumerate(rows) if k==tag);bad=rows.copy();r=dict(rows[i][1]);r[key]^=mask;bad[i]=(tag,r);reject(tag+'.'+key,bad)
    for tag in ('B','RES_R','HLS_WRITE','FFN_COMPLETE','RES_START','HLS_START','HLS_DONE','COMPLETE','RELEASE','ACK','SWITCH','REUSABLE'):
        i=next(i for i,(k,r) in enumerate(rows) if k==tag)
        reject('missing-'+tag,rows[:i]+rows[i+1:]);reject('duplicate-'+tag,rows[:i]+[rows[i]]+rows[i:])
    i=next(i for i,(k,r) in enumerate(rows) if k=='AR' and r['phase']==4 and r['buffer']==7)
    moved=rows[i];bad=rows[:i]+rows[i+1:];at=next(i for i,(k,r) in enumerate(bad) if k=='B' and r['buffer']==7);bad.insert(at,moved);reject('residual-read-before-ffn-commit',bad)
    # The final memory write and ap_done may be asserted on the same edge.
    # All 2048 sequential writes must still precede DONE in the event stream.
    done=next(i for i,(k,r) in enumerate(rows) if k=='HLS_DONE')
    last=max(i for i,(k,r) in enumerate(rows) if k=='HLS_WRITE')
    assert last<done and rows[last][1]['cycle']==rows[done][1]['cycle']
    bad=rows.copy();kind,row=bad[done];row=dict(row);row['cycle']-=1;bad[done]=(kind,row)
    reject('done-before-final-write-cycle',bad)
    bad=rows.copy();event=bad.pop(done);bad.insert(last,event)
    reject('done-before-final-write-event',bad)
    reject('missing-final-hls-write',rows[:last]+rows[last+1:])
    bad=rows.copy();kind,row=bad[last];row=dict(row);row['bits']^=1;bad[last]=(kind,row)
    reject('corrupt-final-hls-write',bad)
    assert len(rejected)==64
    return rejected
