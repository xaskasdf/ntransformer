"""Independent post-O arithmetic, ordered transcript and actual HLS frame review."""
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
from check_ntrpu_residual_r1 import exact_add
from check_ntrpu_norm_q8_r1 import reference as norm_reference
from read_ntrpu_hls_tv_r1 import transactions

TOP = 'ntrpu_post_attention_r1'
PREFIX = 'output/project/solution1/'
R_SENTINEL = 0xa55a3cc3
Q_SENTINEL = 0xa55a5aa53cc3c33c

def need(ok, why):
    if not ok: raise ValueError(why)


def fixture(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    raw=(folder/'llama32_post_attention_r1.bin').read_bytes()
    need(hashlib.sha256(raw).hexdigest()==manifest['files']['llama32_post_attention_r1.bin']=='a8f7797f71bc37bce6d5ff7a3107337172f63e87d8b5328c629b7fb9d21e1310','Fixture SHA')
    need(raw[:12]==b'NTRPOST1'+struct.pack('<I',26),'Fixture format/count')
    need(manifest['model_sha256']=='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1','Model SHA')
    need(manifest['parent_manifest_sha256']=='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e','Attention fixture')
    offset=12; jobs=[]
    def words(n,width=32):
        nonlocal offset
        data=raw[offset:offset+n*(width//8)]; offset+=len(data)
        return list(struct.unpack('<'+('I' if width==32 else 'Q')*n,data))
    for spec in manifest['jobs']:
        ident,n,eps,status=words(4)
        need(ident==len(jobs)+1==spec['id'] and (n,eps,status)==tuple(spec[k] for k in ('elements','epsilon','status')),'Case identity')
        x,branch,weight=words(2048),words(2048),words(2048)
        residual,abi=words(n if status==0 else 0),words(n//4 if status==0 else 0,64)
        data=dict(residual_input=x,branch=branch,norm_weight=weight,residual_output=residual,ffn_input=abi)
        for key,values in data.items():
            encoded=struct.pack('<'+('Q' if key=='ffn_input' else 'I')*len(values),*values)
            need(hashlib.sha256(encoded).hexdigest()==spec['hashes'][key],'Section SHA '+key)
        expected_r=[];expected_q=b''
        if not 0<n<=2048 or n%32: want=1
        elif any(v&0x7f800000==0x7f800000 for v in x[:n]+branch[:n]): want=2
        else:
            expected_r=[exact_add(a,b) for a,b in zip(x[:n],branch[:n])]
            if any(v&0x7f800000==0x7f800000 for v in expected_r): want=3
            else:
                norm_status,expected_q=norm_reference(expected_r,weight,n,eps)
                want=norm_status+2 if norm_status else 0
        if want: expected_r=[];expected_q=b''
        need(want==status and residual==expected_r,'Independent residual/status')
        need(struct.pack('<'+'Q'*len(abi),*abi)==expected_q,'Independent norm ABI1')
        need((len(residual),len(abi))==(spec['residual_words'],spec['abi_words']),'Output extents')
        jobs.append(dict(spec,**data))
    need(offset==len(raw) and len(jobs)==26,'Complete fixture')
    need({j['status'] for j in jobs}==set(range(7)),'Reachable status coverage')
    need([j.get('token') for j in jobs[:4]]==[0,42,128000,128255],'Real samples')
    for key in ('residual_words','abi_words'): need(sum(j[key] for j in jobs)==manifest[key],'Summary '+key)
    need(sum(j['status']==0 for j in jobs)==manifest['successful']==11,'Success count')
    return jobs


def review(text, jobs):
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    actual=[line for line in text.splitlines() if line.startswith('POST_')]; expected=[]
    for j in jobs:
        for label,key,width in [('R','residual_output',8),('Q','ffn_input',16)]:
            for i,v in enumerate(j[key]): expected.append(f"POST_{label} case={j['id']} index={i} bits={v:0{width}x}")
        expected.append(f"POST_CASE id={j['id']} elements={j['elements']} epsilon={j['epsilon']:08x} status={j['status']} residual={j['residual_words']} abi={j['abi_words']}")
    counts=dict(cases=len(jobs),successful=sum(j['status']==0 for j in jobs),
        residual=sum(j['residual_words'] for j in jobs),abi=sum(j['abi_words'] for j in jobs))
    expected.append('POST_PASS '+' '.join(f'{k}={v}' for k,v in counts.items()))
    need(actual==expected,'Exact transcript, identities and ordering')
    return dict(status='PASS_POST_ATTENTION_R1_NUMERICAL',**counts,max_fp32_ulp=0,
        independent_integer_and_ordered_decimal=True)


def controls(text,jobs):
    first=re.search(r'(?m)^POST_R .*$',text)[0]
    q=re.search(r'(?m)^POST_Q .*$',text)[0]
    corrupt=lambda line:re.sub(r'bits=([0-9a-f]+)',lambda m:'bits='+format(int(m[1],16)^1,'0'+str(len(m[1]))+'x'),line)
    changes=dict(residual=text.replace(first,corrupt(first),1),abi=text.replace(q,corrupt(q),1),
        missing=text.replace(first+'\n','',1),duplicate=text.replace(first,first+'\n'+first,1),
        status=text.replace(' status=6 ',' status=0 ',1),epsilon=text.replace('epsilon=3727c5ac','epsilon=3727c5ad',1),
        summary=text.replace('POST_PASS','REMOVED'),failure=text+'\nFAIL: injected\n',extra=text+'\nPOST_CASE unexpected\n')
    for key in ('4','5'):
        line=re.search(r'(?m)^POST_Q case=1 index='+key+r' .*$',text)[0]
        changes['scale' if key=='4' else 'reserved']=text.replace(line,corrupt(line),1)
    rejected=[]
    for name,bad in changes.items():
        need(bad!=text,'Control changed')
        try: review(bad,jobs)
        except (ValueError,AssertionError): rejected.append(name)
        else: raise ValueError('Accepted corrupt '+name)
    return rejected


def review_frames(encoded,jobs):
    raw={n:base64.b64decode(v,validate=True) for n,v in encoded.items()}
    count=len(jobs); frames={}; facts=[]
    outputs={'residual_output':(32,2048,R_SENTINEL),'ffn_input':(64,512,Q_SENTINEL),'ap_return':(32,1,None)}
    readonly=('residual_input','branch','norm_weight')
    for side,sub in [('c','cdatafile'),('rtl','rtldatafile')]:
        stem=PREFIX+'sim/tv/'+sub+'/'+side+'.'+TOP+'.autotvout_'
        selected={n[len(stem):-4]:n for n in raw if n.startswith(stem) and n.endswith('.dat')}
        need(set(selected)==set(outputs)|(set(readonly) if side=='rtl' else set()),'Exact output port set '+side)
        for port,(width,depth,sentinel) in outputs.items():
            name=selected[port]
            frames[side,port]=transactions(raw[name],count,width=width,depth=depth)
            for j,frame in zip(jobs,frames[side,port]):
                values=[j['status']] if port=='ap_return' else j[port]
                expected=values+[sentinel]*(depth-len(values))
                need(frame==[f'0x{v:0{width//4}x}' for v in expected],f'{side} {port} case {j["id"]}')
            facts.append(dict(path=name,sha256=hashlib.sha256(raw[name]).hexdigest(),width=width,depth=depth,frames=count))
        if side=='rtl':
            for port in readonly:
                c=PREFIX+'sim/tv/cdatafile/c.'+TOP+'.autotvin_'+port+'.dat'
                for name in (c,selected[port]):
                    decoded=transactions(raw[name],count,width=32,depth=2048)
                    need(decoded==[[f'0x{v:08x}' for v in j[port]] for j in jobs],'Read-only actual frame '+port)
                    facts.append(dict(path=name,sha256=hashlib.sha256(raw[name]).hexdigest(),width=32,depth=2048,frames=count))
    # Scalar inputs also pin the actual shape and epsilon on every transaction.
    for port,key in [('elements','elements'),('epsilon_bits','epsilon')]:
        name=PREFIX+'sim/tv/cdatafile/c.'+TOP+'.autotvin_'+port+'.dat'
        need(transactions(raw[name],count,width=32,depth=1)==[[f'0x{j[key]:08x}'] for j in jobs],'Actual scalar '+port)
        facts.append(dict(path=name,sha256=hashlib.sha256(raw[name]).hexdigest(),width=32,depth=1,frames=count))
    latency=raw[PREFIX+'sim/report/verilog/result.transaction.rpt'].decode('ascii')
    lines=[s.strip() for s in latency.splitlines() if s.strip()]
    need(lines[0].split()==['latency','interval'] and len(lines)==count+1,'Latency header/count')
    durations=[]; intervals=[]
    for i,line in enumerate(lines[1:]):
        match=re.fullmatch(r'transaction\s+(\d+):\s+(\d+)\s+(\d+|x)',line)
        need(match is not None,'Latency row format')
        ident,duration,interval=match.groups()
        need(int(ident)==i and int(duration)>0,'Latency identity/value')
        need((interval=='x')==(i==count-1),'Final interval unavailable')
        need(interval=='x' or int(interval)>0,'Positive interval')
        durations.append(int(duration)); intervals.append(None if interval=='x' else int(interval))
    return dict(status='PASS_POST_ATTENTION_R1_ACTUAL_C_RTL',transactions=count,
        successful=sum(j['status']==0 for j in jobs),residual_words=sum(j['residual_words'] for j in jobs),
        abi_words=sum(j['abi_words'] for j in jobs),all_failure_output_frames_unchanged=True,
        latency_cycles=durations,interval_cycles=intervals,min_cycles=min(durations),max_cycles=max(durations),source_files=facts,
        physical_timing_qualified=False)


def frame_controls(encoded,jobs):
    from read_ntrpu_hls_tv_r1 import text_transactions
    rejected=[]
    def mutate(name,side,port,index=0,lane=0,width=32,depth=2048,both=False):
        bad=dict(encoded)
        for s in (('c','rtl') if both else (side,)):
            sub='cdatafile' if s=='c' else 'rtldatafile'
            key=PREFIX+'sim/tv/'+sub+'/'+s+'.'+TOP+'.autotvout_'+port+'.dat'
            frames=transactions(base64.b64decode(bad[key]),len(jobs),width=width,depth=depth)
            frames[index][lane]=f'0x{int(frames[index][lane],16)^1:0{width//4}x}'
            bad[key]=base64.b64encode(text_transactions(frames).encode()).decode()
        try: review_frames(bad,jobs)
        except (ValueError,AssertionError,KeyError): rejected.append(name)
        else: raise ValueError('Accepted corrupt port '+name)
    mutate('residual','rtl','residual_output')
    mutate('matching_wrong_residual','rtl','residual_output',both=True)
    for lane,name in [(0,'abi'),(4,'scale'),(5,'padding')]: mutate(name,'rtl','ffn_input',lane=lane,width=64,depth=512)
    mutate('status','rtl','ap_return',width=32,depth=1)
    failure=next(i for i,j in enumerate(jobs) if j['status']==6)
    mutate('failed_residual_publication','rtl','residual_output',index=failure)
    mutate('failed_abi_publication','rtl','ffn_input',index=failure,width=64,depth=512)
    mutate('readonly_change','rtl','residual_input')
    key=PREFIX+'sim/report/verilog/result.transaction.rpt'
    bad=dict(encoded);bad[key]=base64.b64encode(b'transaction 0: 0 1 1\n').decode()
    try: review_frames(bad,jobs)
    except (ValueError,AssertionError): rejected.append('bad_latency')
    else: raise ValueError('Accepted bad latency')
    return rejected
