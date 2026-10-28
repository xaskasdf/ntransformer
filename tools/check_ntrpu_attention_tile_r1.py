"""Audit every tile against rational scores and offline Decimal90 softmax."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

def need(ok,why):
    if not ok:raise ValueError(why)
def f32(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def f64(x):return struct.unpack('<d',struct.pack('<Q',x))[0]
def bits32(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def bits64(x):return struct.unpack('<Q',struct.pack('<d',x))[0]
def order(x):return (~x)&0xffffffff if x&0x80000000 else x|0x80000000
def ulp(a,b):return 0 if f32(a)==f32(b)==0 else abs(order(a)-order(b))

def fixture(folder):
    raw=(folder/'llama32_attention_r1.bin').read_bytes();manifest=json.loads((folder/'manifest.json').read_text())
    need(hashlib.sha256(raw).hexdigest()==manifest['files']['llama32_attention_r1.bin']=='6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361','Pinned fixture')
    need(raw[:12]==b'NTRATTN1'+struct.pack('<I',143),'Format/count');pos=12;jobs=[]
    with localcontext() as ctx:
        ctx.prec=90
        for ident,spec in enumerate(manifest['jobs'],1):
            h=struct.unpack_from('<6I2Q',raw,pos);pos+=40
            need(h==tuple(spec[k] for k in ('id','tokens','layer','query_head','kv_head','tile_width','sequence','epoch')),'Case metadata')
            need(ident==h[0] and h[4]==h[3]//4 and h[2]==0,'Llama GQA identity');n=h[1]
            chunks={}
            for key,length in [('query',256),('keys',n*256),('values',n*256),('expected',256)]:
                chunk=raw[pos:pos+length];pos+=length;need(len(chunk)==length,'Truncated fixture')
                need(hashlib.sha256(chunk).hexdigest()==spec['hashes'][key],'Section digest')
                chunks[key]=list(struct.unpack('<'+'I'*(length//4),chunk))
            need(all(math.isfinite(f32(w)) for name,words in chunks.items() for w in words),'Nonfinite fixture')
            q=[Fraction(f32(w)) for w in chunks['query']];scores=[]
            for token in range(n):
                total=Fraction(0)
                for d in range(64):
                    product=Fraction(float(q[d]*Fraction(f32(chunks['keys'][token*64+d]))))
                    total=Fraction(float(total+product))
                scores.append(float(total/8))
            prefixes=[]
            for start in range(0,n,h[5]):
                end=min(start+h[5],n);maximum=max(scores[:end]);dm=Decimal.from_float(maximum)
                weights=[(Decimal.from_float(s)-dm).exp() for s in scores[:end]];den=sum(weights,Decimal(0))
                numerators=[];absolute=[]
                for d in range(64):
                    terms=[w*Decimal.from_float(f32(chunks['values'][t*64+d])) for t,w in enumerate(weights)]
                    numerators.append(sum(terms,Decimal(0)));absolute.append(sum(map(abs,terms),Decimal(0)))
                expected=[bits32(float(v/den)) for v in numerators]
                if end==n:need(all(ulp(a,b)<=1 for a,b in zip(expected,chunks['expected'])),'Offline fixture vs independent reference')
                prefixes.append(dict(start=start,count=end-start,end=end,maximum=bits64(maximum),normalizer=den,
                    accumulators=numerators,absolute=absolute,expected=expected))
            jobs.append(dict(spec,q=chunks['query'],prefixes=prefixes,fixture_expected=chunks['expected']))
    need(pos==len(raw) and len(jobs)==143,'Complete fixture');need(sum(len(j['prefixes']) for j in jobs)==291,'Tile coverage')
    real=[j for j in jobs if j['category']=='real_embedding_norm_q8_qkv_rope_layer0']
    need(len(real)==128 and {(j['tokens'],j['query_head']) for j in real}=={(n,h) for n in range(1,5) for h in range(32)},'All real prefixes and Q heads')
    return jobs

def review(text,jobs):
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    lines=[s for s in text.splitlines() if s.startswith('NTRPU_ATTENTION_')];pos=0;words=0;max_ulp=0;offline_ulp=0
    relative=Decimal(0);state_words=0
    def take(pattern):
        nonlocal pos
        need(pos<len(lines),'Truncated transcript');m=re.fullmatch(pattern,lines[pos]);pos+=1
        need(m is not None,'Transcript record/order');return m
    with localcontext() as ctx:
        ctx.prec=90
        for job in jobs:
            ident=job['id']
            for prefix in job['prefixes']:
                start=prefix['start'];count=prefix['count'];end=prefix['end']
                take(rf'NTRPU_ATTENTION_TILE case={ident} start={start} count={count} status=0')
                state=[]
                for i in range(104):
                    m=take(rf'NTRPU_ATTENTION_STATE case={ident} start={start} index={i} bits=([0-9a-f]{{16}})')
                    state.append(int(m[1],16));state_words+=1
                need(state[:6]==[0x4e54524154543031,job['sequence'],job['epoch'],job['tokens']-1,
                    (job['layer']<<32)|job['query_head'],end],'State ownership/count')
                need(state[6]==prefix['maximum'],'Ordered rational maximum')
                need(state[8:40]==[job['q'][i*2]|(job['q'][i*2+1]<<32) for i in range(32)],'Exact stored query')
                for actual,want,magnitude in [(state[7],prefix['normalizer'],prefix['normalizer'])]+[
                    (state[40+d],prefix['accumulators'][d],prefix['absolute'][d]) for d in range(64)]:
                    number=f64(actual);need(math.isfinite(number),'Nonfinite state')
                    error=abs(Decimal.from_float(number)-want)/max(magnitude,Decimal('1e-300'))
                    relative=max(relative,error);need(error<=Decimal('2e-12'),'Independent softmax state tolerance')
                if end==job['tokens']:
                    for i,want in enumerate(prefix['expected']):
                        got=int(take(rf'NTRPU_ATTENTION_WORD case={ident} index={i} bits=([0-9a-f]{{8}})')[1],16)
                        need(math.isfinite(f32(got)),'Nonfinite output');delta=ulp(got,want)
                        need(delta<=1,'Independent FP32 output tolerance');max_ulp=max(max_ulp,delta)
                        offline_ulp=max(offline_ulp,ulp(got,job['fixture_expected'][i]));words+=1
        take(rf'NTRPU_ATTENTION_PASS cases=143 transactions=291 words=9152 offline_max_ulp={offline_ulp}')
    need(pos==len(lines) and words==9152 and state_words==30264,'Trailing/missing records')
    return dict(status='PASS_ATTENTION_TILE_R1_NUMERICAL_TRANSCRIPT',cases=143,real_cases=128,transactions=291,
        output_words=words,state_words=state_words,decimal90_max_output_ulp=max_ulp,
        decimal90_max_scaled_state_error=str(relative),state_acceptance_scaled_tolerance='2e-12',
        scope='Native attention unit transcript; independently checked rounded scores and all prefix states. No RTL, KV memory adapter, full inference or physical timing qualification.')

def controls(text,jobs):
    first=re.search(r'(?m)^NTRPU_ATTENTION_STATE case=1 start=0 index=0 .*$',text)[0]
    def alter(index,mask):
        pattern=rf'(?m)^(NTRPU_ATTENTION_STATE case=1 start=0 index={index} bits=)([0-9a-f]{{16}})$'
        return re.sub(pattern,lambda m:m[1]+f'{int(m[2],16)^mask:016x}',text,count=1)
    bad={
        'identity':alter(1,1),'query':alter(8,1),'maximum':alter(6,0x0010000000000000),
        'mass':alter(7,0x0010000000000000),'accumulator':alter(40,0x0100000000000000),
        'missing_state':text.replace(first+'\n','',1),'duplicate_state':text.replace(first,first+'\n'+first,1),
        'wrong_status':text.replace('count=1 status=0','count=1 status=1',1),
        'missing_summary':text.replace('NTRPU_ATTENTION_PASS','REMOVED'),
        'extra_record':text+'\nNTRPU_ATTENTION_WORD extra\n','diagnostic':text+'\nFAIL: injected\n'}
    bad['output']=re.sub(r'(?m)^(NTRPU_ATTENTION_WORD case=1 index=0 bits=)([0-9a-f]{8})$',
        lambda m:m[1]+f'{int(m[2],16)^0x01000000:08x}',text,count=1)
    rejected=[]
    for name,mutation in bad.items():
        need(mutation!=text,'Control unchanged')
        try:review(mutation,jobs)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',action='store_true');a=p.parse_args()
    jobs=fixture(a.fixture);text=a.log.read_text();result=review(text,jobs)
    if a.controls:result['rejected_controls']=controls(text,jobs)
    result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
