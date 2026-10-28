"""Independent rational reference, exact ABI1 bytes and transcript coverage."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

def need(ok,why):
    if not ok:raise ValueError(why)
def value(word):return struct.unpack('<f',struct.pack('<I',word))[0]
def pack32(words):return struct.pack('<'+'I'*len(words),*words)

def reference(x,n):
    if not 0<n<=2048 or n%32:return 1,b''
    if not all(math.isfinite(value(word)) for word in x[:n]):return 2,b''
    out=bytearray()
    for start in range(0,n,32):
        block=[Fraction(value(w)) for w in x[start:start+32]]
        scale=float(max(abs(v) for v in block)/127)
        integers=[]
        for v in block:
            ratio=float(v/Fraction(scale)) if scale else 0.0
            magnitude=abs(ratio);integer=math.floor(magnitude)
            if magnitude-integer>=0.5:integer+=1
            integers.append((-1 if ratio<0 else 1)*min(integer,127))
        out.extend(struct.pack('<32bd',*integers,scale)+bytes(24))
    return 0,bytes(out)

def fixture(folder):
    raw=(folder/'attention_quantize_r1.bin').read_bytes();m=json.loads((folder/'manifest.json').read_text())
    need(hashlib.sha256(raw).hexdigest()==m['files']['attention_quantize_r1.bin']=='121d09f22261000eeacb9359c82cdf09f98e29e3106e2338f5bbd51f588c1767','Pinned fixture')
    need(raw[:12]==b'NTRAQ8R1'+struct.pack('<I',27),'Magic/count')
    need((m['cases'],m['successful_cases'],m['words'],m['records'])==(27,16,5984,748),'Coverage')
    offset=12;jobs=[]
    for ident,spec in enumerate(m['jobs'],1):
        case,n,status,count=struct.unpack_from('<4I',raw,offset);offset+=16
        x=list(struct.unpack_from('<2048I',raw,offset));offset+=8192
        out=raw[offset:offset+count*8];offset+=count*8
        need(case==ident==spec['id'] and (n,status,count)==tuple(spec[k] for k in ('elements','status','expected_words')),'Identity')
        for name,data in [('input',pack32(x)),('expected',out)]:need(hashlib.sha256(data).hexdigest()==spec['hashes'][name],'Section hash')
        need(reference(x,n)==(status,out),'Independent numerical reference')
        for base in range(0,len(out),64):
            rec=out[base:base+64]
            need(len(rec)==64 and rec[40:]==bytes(24),'ABI1 extent/padding')
            need(all(v!=-128 for v in struct.unpack('<32b',rec[:32])),'INT8 domain')
            need(int.from_bytes(rec[32:40],'little')<=0x4780204070e1c387,'Scale domain')
        jobs.append(dict(spec,expected=list(struct.unpack('<'+'Q'*count,out)),input=x))
    need(offset==len(raw) and len(jobs)==27,'Complete fixture')
    need({j.get('sample') for j in jobs if j['category']=='real_layer0_attention'}=={1,2,3,4},'Real attention coverage')
    return jobs

def review(text,folder,jobs=None):
    need(not re.search(r'(?mi)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    jobs=fixture(folder) if jobs is None else jobs
    lines=[s for s in text.splitlines() if s.startswith('NTRPU_ATTENTION_QUANTIZE_')];pos=0;cases=[];words=0
    for job in jobs:
        h=0xcbf29ce484222325
        for i,word in enumerate(job['expected']):
            need(pos<len(lines) and lines[pos]==f"NTRPU_ATTENTION_QUANTIZE_WORD case={job['id']} index={i} bits={word:016x}",'Word/order');pos+=1
            for byte in struct.pack('<Q',word):h=((h^byte)*0x100000001b3)&0xffffffffffffffff
        expected=f"NTRPU_ATTENTION_QUANTIZE_CASE id={job['id']} elements={job['elements']} status={job['status']} words={job['expected_words']} fnv={h:016x}"
        need(pos<len(lines) and lines[pos]==expected,'Case/hash');pos+=1;words+=len(job['expected'])
        cases.append({k:v for k,v in job.items() if k not in ('expected','input','hashes')})
    need(lines[pos:]==['NTRPU_ATTENTION_QUANTIZE_PASS cases=27 successful=16 words=5984 records=748'],'Summary/trailing records')
    return dict(status='PASS_ATTENTION_QUANTIZE_R1_NUMERICAL_TRANSCRIPT',cases=cases,words=words,records=748,
        successful_cases=16,exact_output_bytes=47872,ordered_rational_reference=True,
        scope='Numerical transcript and ABI1 bytes; actual C/RTL/AXI and root/GEMV composition need separate qualification.')

def controls(text,folder):
    jobs=fixture(folder);first=re.search(r'(?m)^NTRPU_ATTENTION_QUANTIZE_WORD.*$',text);need(first is not None,'Positive word')
    def corrupt(line):return re.sub(r'bits=([0-9a-f]{16})',lambda m:f'bits={int(m[1],16)^1:016x}',line)
    scale=re.search(r'(?m)^NTRPU_ATTENTION_QUANTIZE_WORD case=1 index=4 .*$',text)
    pad=re.search(r'(?m)^NTRPU_ATTENTION_QUANTIZE_WORD case=1 index=5 .*$',text)
    mutations={'wrong_int8':text.replace(first[0],corrupt(first[0]),1),'wrong_scale':text.replace(scale[0],corrupt(scale[0]),1),
        'nonzero_padding':text.replace(pad[0],corrupt(pad[0]),1),'missing_word':text.replace(first[0]+'\n','',1),
        'duplicate_word':text.replace(first[0],first[0]+'\n'+first[0],1),'wrong_status':text.replace(' status=1 ',' status=0 ',1),
        'wrong_extent':text.replace(' elements=32 ',' elements=64 ',1),
        'missing_summary':text.replace('NTRPU_ATTENTION_QUANTIZE_PASS','REMOVED'),
        'failure_diagnostic':text+'\nFAIL: injected\n','extra_case':text+'\nNTRPU_ATTENTION_QUANTIZE_CASE unexpected\n'}
    rejected=[]
    for name,bad in mutations.items():
        need(bad!=text,'Unchanged control')
        try:review(bad,folder,jobs)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
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
