"""Audit RMSNorm-to-ABI1 against ordered rational/Decimal rounding references."""
import argparse
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

def need(ok,why):
    if not ok:raise ValueError(why)
def value(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def pack32(v):return struct.pack('<'+'I'*len(v),*v)
def rounded32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def reference(x,w,n,eps):
    if not 1<=n<=2048 or n%32:return 1,b''
    epsilon=value(eps)
    if not math.isfinite(epsilon) or epsilon<=0:return 2,b''
    if any(not math.isfinite(value(v)) for v in x[:n]+w[:n]):return 3,b''
    total=Fraction(0)
    for word in x[:n]:
        v=Fraction(value(word));square=Fraction(float(v*v));total=Fraction(float(total+square))
    variance=float(Fraction(float(total/n))+Fraction(epsilon))
    with localcontext() as ctx:
        ctx.prec=90;root=float(Decimal.from_float(variance).sqrt())
    inverse=Fraction(float(Fraction(1)/Fraction(root)));normalized=[]
    for a,b in zip(x[:n],w[:n]):
        scaled=Fraction(float(Fraction(value(a))*inverse))
        product=float(scaled*Fraction(value(b)))
        if abs(product)>value(0x7f7fffff):return 4,b''
        # Rational zero lacks a sign, irrelevant to Q8 integer zero and +0 scale.
        normalized.append(rounded32(product))
    result=bytearray()
    for base in range(0,n,32):
        block=normalized[base:base+32];maximum=max(abs(v) for v in block)
        scale=float(Fraction(maximum)/127)
        need(math.isfinite(scale) and 0<=scale<=value(0x7f7fffff)/127,'ABI1 scale range')
        quantized=[]
        for v in block:
            ratio=float(Fraction(v)/Fraction(scale)) if scale else 0.0
            magnitude=abs(ratio);integer=math.floor(magnitude)
            if magnitude-integer>=0.5:integer+=1
            quantized.append((-1 if ratio<0 else 1)*min(integer,127))
        result.extend(struct.pack('<32bd',*quantized,scale)+bytes(24))
    return 0,bytes(result)

def fixture(folder):
    raw=(folder/'llama32_norm_q8_r1.bin').read_bytes();manifest=json.loads((folder/'manifest.json').read_text())
    need(hashlib.sha256(raw).hexdigest()==manifest['files']['llama32_norm_q8_r1.bin']=='4aa438737577deefa119076c4ba02114ba2e7c35cc56af2395aee04acd4943fe','Pinned fixture')
    need(raw[:12]==b'NTRNQ8R1'+struct.pack('<I',61),'Format/count')
    need((manifest['cases'],manifest['successful_cases'],manifest['words'],manifest['records'])==(61,49,22880,2860),'Coverage')
    offset=12;jobs=[]
    def read32(n):
        nonlocal offset
        words=list(struct.unpack_from('<'+'I'*n,raw,offset));offset+=4*n;return words
    for ident,spec in enumerate(manifest['jobs'],1):
        case,n,eps,status,count=read32(5);x,w=read32(2048),read32(2048)
        out=raw[offset:offset+count*8];offset+=count*8
        need(case==ident==spec['id'] and (n,eps,status,count)==tuple(spec[k] for k in ('elements','epsilon','status','expected_words')),'Identity')
        for key,data in [('input',pack32(x)),('weight',pack32(w)),('expected',out)]:need(hashlib.sha256(data).hexdigest()==spec['hashes'][key],'Section hash')
        want,expected=reference(x,w,n,eps);need(want==status and expected==out,'Independent numerical reference')
        # Decode every physical record separately from the producer packing path.
        for pos in range(0,len(out),64):
            record=out[pos:pos+64];need(len(record)==64 and record[40:]==bytes(24),'Reserved bytes/extent')
            need(all(v!=-128 for v in struct.unpack('<32b',record[:32])),'INT8 domain')
            need(int.from_bytes(record[32:40],'little')<=0x4780204070e1c387,'Binary64 scale domain')
        jobs.append(dict(spec,expected=list(struct.unpack('<'+'Q'*count,out))))
    need(offset==len(raw) and len(jobs)==61,'Complete fixture')
    need({j['status'] for j in jobs}==set(range(5)),'Reachable status coverage')
    need({j.get('norm_case') for j in jobs if j['category']=='real_norm_synthetic_activation'}==set(range(1,34)),'All 33 norm vectors')
    need({j.get('token_id') for j in jobs if j['category']=='real_embedding_layer0_attn_norm'}=={0,42,128000,128255},'Embedding paths')
    return jobs

def review(text,folder,jobs=None):
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    jobs=fixture(folder) if jobs is None else jobs
    lines=[l for l in text.splitlines() if l.startswith('NTRPU_NORM_Q8_')];pos=0;cases=[];words=0
    for job in jobs:
        h=0xcbf29ce484222325
        for i,word in enumerate(job['expected']):
            need(pos<len(lines) and lines[pos]==f"NTRPU_NORM_Q8_WORD case={job['id']} index={i} bits={word:016x}",'Word/order');pos+=1
            for byte in struct.pack('<Q',word):h=((h^byte)*0x100000001b3)&0xffffffffffffffff
        expected=f"NTRPU_NORM_Q8_CASE id={job['id']} elements={job['elements']} epsilon={job['epsilon']:08x} status={job['status']} words={job['expected_words']} fnv={h:016x}"
        need(pos<len(lines) and lines[pos]==expected,'Case/hash');pos+=1;words+=len(job['expected'])
        cases.append({k:v for k,v in job.items() if k not in ('expected','hashes')})
    need(lines[pos:]==['NTRPU_NORM_Q8_PASS cases=61 successful=49 words=22880 records=2860'],'Summary/trailing records')
    return dict(status='PASS_NORM_Q8_R1_NUMERICAL_TRANSCRIPT',cases=cases,words=words,records=2860,successful_cases=49,
        exact_output_bytes=183040,ordered_rational_decimal_reference=True,
        scope='Numerical transcript and ABI1 bytes; separate actual C/RTL proof required. No GEMV/AXI or full inference.')

def controls(text,folder):
    jobs=fixture(folder);first=re.search(r'(?m)^NTRPU_NORM_Q8_WORD.*$',text);need(first is not None,'Positive word')
    def corrupt(line):return re.sub(r'bits=([0-9a-f]{16})',lambda m:f'bits={int(m[1],16)^1:016x}',line)
    scale=re.search(r'(?m)^NTRPU_NORM_Q8_WORD case=1 index=4 .*$',text)
    pad=re.search(r'(?m)^NTRPU_NORM_Q8_WORD case=1 index=5 .*$',text)
    mutations={'wrong_int8':text.replace(first[0],corrupt(first[0]),1),'wrong_scale':text.replace(scale[0],corrupt(scale[0]),1),
        'nonzero_padding':text.replace(pad[0],corrupt(pad[0]),1),'missing_word':text.replace(first[0]+'\n','',1),
        'duplicate_word':text.replace(first[0],first[0]+'\n'+first[0],1),'wrong_status':text.replace(' status=1 ',' status=0 ',1),
        'wrong_epsilon':text.replace(' epsilon=3727c5ac ',' epsilon=3727c5ad ',1),
        'missing_summary':text.replace('NTRPU_NORM_Q8_PASS','REMOVED'),
        'failure_diagnostic':text+'\nFAIL: injected\n','extra_case':text+'\nNTRPU_NORM_Q8_CASE unexpected\n'}
    rejected=[]
    for name,bad in mutations.items():
        need(bad!=text,'Unchanged control')
        try:review(bad,folder,jobs)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted corrupt '+name)
    return rejected
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',action='store_true');a=p.parse_args()
    text=a.log.read_text(errors='replace');result=review(text,a.fixture)
    result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.controls:result['rejected_controls']=controls(text,a.fixture)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
