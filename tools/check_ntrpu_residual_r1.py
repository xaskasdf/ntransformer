"""Exact integer reference in units of 2**-149; independent of HLS GRS logic."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

def need(ok,why):
    if not ok:raise ValueError(why)
def units(word):
    exponent=(word>>23)&255;sign=-1 if word>>31 else 1
    return sign*((word&0x7fffff)|(0x800000 if exponent else 0))*(1<<max(exponent-1,0))
def exact_add(a,b):
    value=units(a)+units(b)
    if value==0:return 0x80000000 if a==b==0x80000000 else 0
    sign=0x80000000 if value<0 else 0;value=abs(value)
    if value<0x800000:return sign|value
    shift=max(value.bit_length()-24,0);q,r=divmod(value,1<<shift)
    if shift and (r>(1<<(shift-1)) or (r==(1<<(shift-1)) and q&1)):q+=1
    if q==0x1000000:q>>=1;shift+=1
    exponent=shift+1
    if exponent>=255:return sign|0x7f800000
    return sign|(exponent<<23)|(q&0x7fffff)
def fixture(folder):
    m=json.loads((folder/'manifest.json').read_text());raw=(folder/'llama32_residual_r1.bin').read_bytes()
    need(hashlib.sha256(raw).hexdigest()==m['files']['llama32_residual_r1.bin'],'Fixture identity')
    need(m['files']['llama32_residual_r1.bin']=='809c8187806843e27d6d3c5776ea7c9dab69b2718b7efd9460dfe4e22519adec','Pinned fixture')
    need(raw[:12]==b'NTRRADD1'+struct.pack('<I',43),'Fixture format/count')
    need(m['cases']==43 and m['successful_cases']==33 and m['values']==51556,'Fixture coverage')
    offset=12;jobs=[]
    def words(n):
        nonlocal offset
        result=list(struct.unpack_from('<'+'I'*n,raw,offset));offset+=n*4;return result
    for i,spec in enumerate(m['jobs'],1):
        ident,n,status,count=words(4);x,y=words(2048),words(2048);out=words(count)
        need(ident==i==spec['id'] and (n,status,count)==tuple(spec[k] for k in ('elements','status','expected_elements')),'Case identity')
        for key,data in [('residual',x),('branch',y),('expected',out)]:
            need(hashlib.sha256(struct.pack('<'+'I'*len(data),*data)).hexdigest()==spec['hashes'][key],'Section hash')
        if not 1<=n<=2048:want,expected=1,[]
        elif any((v&0x7f800000)==0x7f800000 for v in x[:n]+y[:n]):want,expected=2,[]
        else:
            expected=[exact_add(a,b) for a,b in zip(x[:n],y[:n])]
            want=3 if any((v&0x7f800000)==0x7f800000 for v in expected) else 0
            if want:expected=[]
        need(want==status and expected==out,'Exact addition reference')
        jobs.append(dict(spec,expected=expected))
    need(len(jobs)==43 and offset==len(raw),'Fixture extent')
    need({j['status'] for j in jobs}=={0,1,2,3},'Status coverage')
    need({j.get('token_id') for j in jobs if j['category']=='real_embedding_synthetic_branch'}=={0,42,128000,128255},'Embedding coverage')
    return jobs
def review(text,folder):
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    lines=[l for l in text.splitlines() if l.startswith('NTRPU_RESIDUAL_') and not l.startswith('NTRPU_RESIDUAL_FUZZ_')]
    pos=0;cases=[];values=0
    for job in fixture(folder):
        h=0xcbf29ce484222325
        for i,word in enumerate(job['expected']):
            need(pos<len(lines) and lines[pos]==f"NTRPU_RESIDUAL_VALUE case={job['id']} index={i} bits={word:08x}",'Value/order');pos+=1
            for byte in struct.pack('<I',word):h=((h^byte)*0x100000001b3)&0xffffffffffffffff
        expected=f"NTRPU_RESIDUAL_CASE id={job['id']} elements={job['elements']} status={job['status']} values={job['expected_elements']} fnv={h:016x}"
        need(pos<len(lines) and lines[pos]==expected,'Case/hash');pos+=1;values+=len(job['expected'])
        cases.append({k:v for k,v in job.items() if k not in ('expected','hashes')})
    need(lines[pos:]==['NTRPU_RESIDUAL_PASS cases=43 successful=33 values=51556'],'Summary/extra records')
    return dict(status='PASS_RESIDUAL_R1_NUMERICAL_TRANSCRIPT',cases=cases,values=values,successful_cases=33,
        max_fp32_ulp=0,exact_integer_reference=True,scope='Native transcript only; actual C/RTL trace review required for hardware qualification.')
def controls(text,folder):
    first=re.search(r'(?m)^NTRPU_RESIDUAL_VALUE.*$',text);need(first is not None,'Positive value')
    altered=re.sub(r'bits=([0-9a-f]{8})',lambda m:f'bits={int(m[1],16)^1:08x}',first[0])
    mutations={'wrong_word':text.replace(first[0],altered,1),'missing_word':text.replace(first[0]+'\n','',1),
        'duplicate_word':text.replace(first[0],first[0]+'\n'+first[0],1),'wrong_status':text.replace(' status=1 ',' status=0 ',1),
        'wrong_extent':text.replace(' elements=2048 ',' elements=2047 ',1),'missing_summary':text.replace('NTRPU_RESIDUAL_PASS','REMOVED'),
        'failure_diagnostic':text+'\nFAIL: injected\n','extra_case':text+'\nNTRPU_RESIDUAL_CASE unexpected\n'}
    rejected=[]
    for name,bad in mutations.items():
        need(bad!=text,'Control did not mutate')
        try:review(bad,folder)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',action='store_true');a=p.parse_args()
    text=a.log.read_text(errors='replace');r=review(text,a.fixture)
    r.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.controls:r['rejected_controls']=controls(text,a.fixture)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k!='cases'}))
if __name__=='__main__':main()
