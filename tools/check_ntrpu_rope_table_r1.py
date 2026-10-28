"""Rebuild RoPE statuses and words independently from the sealed input records."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

def value(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def packed(values):return struct.pack('<'+'I'*len(values),*values)
def need(ok,why):
    if not ok:raise ValueError(why)

def oracle(heads,position,phase_position,phase,x):
    if not 1<=heads<=40:return 1,[]
    if not 0<=position<131072 or phase_position!=position:return 2,[]
    if position==0 and phase!=[v for _ in range(32) for v in (0x3f800000,0)]:return 3,[]
    factors=[value(v) for v in phase]
    if not all(math.isfinite(v) and abs(v)<=1 for v in factors):return 3,[]
    if not all(abs(factors[i]**2+factors[i+1]**2-1)<=2**-22 for i in range(0,64,2)):return 3,[]
    vals=[value(v) for v in x[:heads*64]]
    if not all(math.isfinite(v) for v in vals):return 4,[]
    result=[]
    for i in range(0,len(vals),2):
        a,b=vals[i:i+2];c,s=factors[i%64:i%64+2]
        # Four independent products and two sums, with no fused operation.
        out=(a*c-b*s,a*s+b*c)
        if any(abs(v)>value(0x7f7fffff) for v in out):return 5,[]
        exact=(Fraction(a)*Fraction(c)-Fraction(b)*Fraction(s),
               Fraction(a)*Fraction(s)+Fraction(b)*Fraction(c))
        words=[bits(v) for v in out]
        for word,reference in zip(words,exact):
            # Rational zero has no sign; preserve the ordered IEEE zero above.
            if reference!=0:need(word==bits(float(reference)),'Rational reference discrepancy')
        result.extend(words)
    return 0,result

def fixture(folder):
    manifest=json.loads((folder/'manifest.json').read_text());raw=(folder/'llama32_rope_r1.bin').read_bytes()
    for name,sha in manifest['files'].items():need(hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,'Fixture file identity')
    need(manifest['files']['llama32_rope_r1.bin']=='35e80881c4b71531c47c21379f98017e2fae2419c9b29a78576f538495081253','Pinned fixture')
    need(raw[:12]==b'NTRROPE1'+struct.pack('<I',48),'Fixture format/count')
    need(manifest['cases']==48 and manifest['successful_cases']==28 and manifest['values']==61760,'Coverage summary')
    jobs=[];offset=12
    def words(count):
        nonlocal offset
        result=list(struct.unpack_from('<'+'I'*count,raw,offset));offset+=count*4
        return result
    for ident,spec in enumerate(manifest['jobs'],1):
        case,heads,p,pp,status,n=words(6);need(case==ident==spec['id'],'Fixture identity')
        phase,x,y=words(64),words(2560),words(n)
        need((heads,p,pp,status,n)==tuple(spec[k] for k in ('heads','position','coefficient_position','status','expected_elements')),'Fixture metadata')
        need(all(hashlib.sha256(packed(data)).hexdigest()==spec['hashes'][name] for name,data in [('coefficients',phase),('input',x),('expected',y)]),'Fixture sections')
        expected_status,expected=oracle(heads,p,pp,phase,x)
        need(expected_status==status and expected==y,'Independent status/numerical reference')
        jobs.append(dict(spec,expected=y))
    need(len(jobs)==48 and offset==len(raw),'Fixture extent')
    need({j['position'] for j in jobs if j['category']=='gguf_phase_synthetic_input'}=={0,1,2,31,127,1023,2047,8191,8192,32767,65535,131071},'Position coverage')
    need(sum(j['category']=='native_full_real_qk_projection' for j in jobs)==6,'Real projection coverage')
    need({j['status'] for j in jobs}==set(range(6)),'Status coverage')
    return jobs

def review(text,folder):
    need(not re.search(r'(?m)^(?:FAIL:|Fatal:|ERROR:)',text),'Failure diagnostic')
    lines=[line for line in text.splitlines() if line.startswith('NTRPU_ROPE_')]
    jobs=fixture(folder);pos=0;cases=[];values=0
    for job in jobs:
        hash_value=0xcbf29ce484222325
        for index,want in enumerate(job['expected']):
            need(pos<len(lines),'Missing word');line=lines[pos];pos+=1
            need(line==f"NTRPU_ROPE_VALUE case={job['id']} index={index} bits={want:08x}",'Actual value/order')
            for byte in struct.pack('<I',want):hash_value=((hash_value^byte)*0x100000001b3)&0xffffffffffffffff
        need(pos<len(lines),'Missing transaction');line=lines[pos];pos+=1
        expected=f"NTRPU_ROPE_CASE id={job['id']} heads={job['heads']} position={job['position']} coefficient_position={job['coefficient_position']} status={job['status']} values={job['expected_elements']} fnv={hash_value:016x}"
        need(line==expected,'Transaction fields/hash');values+=len(job['expected'])
        cases.append({k:v for k,v in job.items() if k not in ('expected','hashes')}|dict(output_fnv=f'{hash_value:016x}'))
    need(lines[pos:]==['NTRPU_ROPE_PASS cases=48 successful=28 values=61760'],'Missing summary/trailing records')
    need(values==61760,'Complete output coverage')
    return dict(status='PASS_ROPE_TABLE_R1_NUMERICAL_TRANSCRIPT',cases=cases,values=values,
        successful_cases=28,max_fp32_ulp=0,rational_reference_checked=True,
        scope='Numerical transcript only; a separate co-simulation and actual trace review are required for RTL qualification.')

def controls(text,folder):
    # Each mutation traverses the full production parser and numerical oracle.
    lines=text.splitlines();first=next(i for i,line in enumerate(lines) if line.startswith('NTRPU_ROPE_VALUE'))
    word=re.search(r'bits=([0-9a-f]{8})',lines[first])[1];bad=lines[:]
    bad[first]=bad[first].replace('bits='+word,f'bits={int(word,16)^1:08x}')
    mutations={'wrong_word':'\n'.join(bad),'missing_word':'\n'.join(lines[:first]+lines[first+1:]),
        'duplicate_word':'\n'.join(lines[:first]+[lines[first]]+lines[first:]),
        'wrong_status':text.replace(' status=1 ',' status=0 ',1),
        'stale_position':text.replace('coefficient_position=0','coefficient_position=1',1),
        'missing_summary':text.replace('NTRPU_ROPE_PASS','REMOVED_ROPE_PASS'),
        'false_failure':text+'\nFAIL: injected\n','trailing_case':text+'\nNTRPU_ROPE_CASE injected\n'}
    rejected=[]
    for name,bad in mutations.items():
        try:review(bad,folder)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise AssertionError('Accepted corrupt transcript: '+name)
    return rejected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--controls',action='store_true');a=p.parse_args()
    text=a.log.read_text(errors='replace');result=review(text,a.fixture)
    result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.controls:result['rejected_controls']=controls(text,a.fixture)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':main()
