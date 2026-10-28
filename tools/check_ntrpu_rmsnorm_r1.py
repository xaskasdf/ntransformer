"""Independently reconstruct all inputs and audit the complete RMSNorm transcript."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def value(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def order(v):return (~v)&0xffffffff if v&0x80000000 else v|0x80000000
def distance(a,b):return abs(order(a)-order(b))
def fixture_jobs(folder):
    m=json.loads((folder/'manifest.json').read_text());raw=(folder/'llama32_norm_r1.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==m['files']['llama32_norm_r1.bin']=='66573f15c4070efaf91d5baf1c3c64fb58441a9d84ad4790e115c8d8a9ee3e60'
    assert raw[:12]==b'NTRNORM1'+struct.pack('<I',33) and len(raw)==811416
    jobs=[];pos=12
    for i,spec in enumerate(m['jobs'],1):
        ident,n,eps=struct.unpack_from('<III',raw,pos);pos+=12;assert (ident,n,eps)==(i,2048,0x3727c5ac)
        blocks=[]
        for key in ('input_sha256','weight_sha256','output_sha256'):
            block=raw[pos:pos+8192];pos+=8192;assert hashlib.sha256(block).hexdigest()==spec[key]
            blocks.append(list(struct.unpack('<2048I',block)))
        x,w,y=blocks;jobs.append((f'llama_norm_{i}',n,eps,0,x,w,y))
    assert pos==len(raw)
    return jobs
def synthetic_jobs():
    jobs=[];eps=0x3727c5ac
    def fill():return ([bits(((i%47)-23)/16) for i in range(2048)],[bits(((i%19)-9)/8) for i in range(2048)])
    def put(label,n,status=0,e=eps):jobs.append((label,n,e,status,x[:],w[:],None))
    x,w=fill()
    for n in (1,31,32,33,256,2048):put('shape',n)
    x=[0x80000000 if i&1 else 0 for i in range(2048)];put('signed_zero',2048)
    x=[(0x80000000 if i&1 else 0)|(i+1) for i in range(2048)];w=[bits(1)]*2048;put('subnormal_input_epsilon',2048,e=1)
    x=[0xff7fffff if i&1 else 0x7f7fffff for i in range(2048)];w=[bits(.5)]*2048;put('finite_extremes',2048)
    x,w=fill();x[2047]=0x7fc00001;w[2047]=0xff800000;put('outside_extent',31)
    x,w=fill();put('empty',0,1);put('too_large',2049,1)
    for e in (0,0x80000000,0xbf800000,0x7f800000,0xff800000,0x7fc00001):put('bad_epsilon',2048,2,e)
    for bad in (0x7fc00001,0x7f800000,0xff800000):
        x,w=fill();x[2047]=bad;put('late_input',2048,3)
        x,w=fill();w[2047]=bad;put('late_weight',2048,3)
    x=[0]*2048;w=[bits(1)]*2048;x[2047]=bits(1);w[2047]=0x7f7fffff;put('late_overflow',2048,4)
    x[0]=0x7fc00001;put('validation_precedence',2048,3)
    x,w=fill();put('recovery',2048)
    assert len(jobs)==27
    return jobs
def expected_values(x,w,n,eps):
    total=0.0
    for v in x[:n]:total=total+value(v)*value(v)
    inverse=1/math.sqrt(total/n+value(eps))
    return [bits((value(a)*inverse)*value(b)) for a,b in zip(x[:n],w[:n])]
def review(text,folder):
    need=lambda ok,why: None if ok else (_ for _ in ()).throw(ValueError(why))
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    lines=[l for l in text.splitlines() if l.startswith('NTRPU_RMSNORM_')]
    jobs=synthetic_jobs()+fixture_jobs(folder);pos=0;count=0;max_error=0;cases=[]
    for ident,(label,n,eps,status,x,w,golden) in enumerate(jobs,1):
        expected=expected_values(x,w,n,eps) if status==0 else []
        hash_value=0xcbf29ce484222325;max_ulp=0
        for index,want in enumerate(expected):
            need(pos<len(lines),'Missing value');line=lines[pos];pos+=1
            m=re.fullmatch(r'NTRPU_RMSNORM_VALUE case=(\d+) index=(\d+) bits=([0-9a-f]{8}) ref=([0-9a-f]{8}) ulp=(\d+)',line)
            need(m is not None,'Value grammar');case,row,got,ref,ulp=m.groups()
            got=int(got,16);ref=int(ref,16);ulp=int(ulp)
            need((int(case),int(row))==(ident,index),'Value order/count')
            need(math.isfinite(value(got)) and distance(got,want)<=1 and distance(ref,want)<=1,'Independent numerical value')
            need(ulp==distance(got,ref) and ulp<=1,'ULP evidence')
            if want&0x7fffffff==0:need(got==want and ref==want,'Signed zero')
            if golden is not None:need(distance(got,golden[index])<=1,'Original serialized reference')
            for byte in struct.pack('<I',got):hash_value=((hash_value^byte)*0x100000001b3)&0xffffffffffffffff
            count+=1;max_ulp=max(max_ulp,ulp);max_error=max(max_error,distance(got,want))
        need(pos<len(lines),'Missing case');line=lines[pos];pos+=1
        want=f'NTRPU_RMSNORM_CASE id={ident} label={label} n={n} status={status} max_ulp={max_ulp} fnv={hash_value:016x}'
        need(line==want,'Case identity/status/count/hash')
        cases.append(dict(id=ident,label=label,elements=n,status=status,max_ulp=max_ulp,output_fnv=f'{hash_value:016x}'))
    need(lines[pos:]==['NTRPU_RMSNORM_PASS cases=60 successful=44 values=78208 real_norms=33'],'Suite summary/trailing records')
    need(count==78208,'Complete numerical coverage')
    return dict(status='PASS_RMSNORM_R1_NUMERICAL_TRANSCRIPT',cases=cases,values=count,real_norm_vectors=33,
        max_fp32_ulp_to_python=max_error,scope='Native/C-simulation transcript audit only; no RTL or physical qualification without separate co-simulation reports.')
def controls(text,folder):
    lines=text.splitlines();first=next(i for i,l in enumerate(lines) if l.startswith('NTRPU_RMSNORM_VALUE'))
    bad=lines[:];m=re.search(r'bits=([0-9a-f]{8})',bad[first]);pattern=int(m[1],16)
    bad[first]=bad[first].replace('bits='+m[1],f'bits={(pattern+8)&0xffffffff:08x}')
    mutations={'wrong_output':'\n'.join(bad),'missing_value':'\n'.join(lines[:first]+lines[first+1:]),
        'duplicate_value':'\n'.join(lines[:first]+[lines[first]]+lines[first:]),
        'wrong_status':text.replace('label=late_input n=2048 status=3','label=late_input n=2048 status=0',1),
        'missing_summary':text.replace('NTRPU_RMSNORM_PASS','REMOVED_RMSNORM_PASS'),
        'false_failure':text+'\nFAIL: injected\n',
        'wrong_ulp':text.replace(' ulp=0',' ulp=2',1)}
    rejected=[]
    for name,value_ in mutations.items():
        try:review(value_,folder)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted corrupted evidence: '+name)
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
