"""Prepare full attention vectors and quantizer boundary cases, outside Git."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

PARENT_SHA='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'
MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack32(v):return struct.pack('<'+'I'*len(v),*v)
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def value(v):return struct.unpack('<f',struct.pack('<I',v))[0]

def reference(words,n):
    if not 0<n<=2048 or n%32:return 1,b''
    if any(w&0x7f800000==0x7f800000 for w in words[:n]):return 2,b''
    result=bytearray()
    for base in range(0,n,32):
        values=[value(w) for w in words[base:base+32]]
        scale=max(abs(v) for v in values)/127.0;quantized=[]
        for v in values:
            ratio=v/scale if scale else 0.0;f=Fraction(abs(ratio))
            rounded=(2*f.numerator+f.denominator)//(2*f.denominator)
            quantized.append((-1 if ratio<0 else 1)*min(rounded,127))
        result.extend(struct.pack('<32bd',*quantized,scale)+bytes(24))
    return 0,bytes(result)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attention-output',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and digest(a.attention_output/'manifest.json')==PARENT_SHA
    parent=json.loads((a.attention_output/'manifest.json').read_text());assert parent['model_sha256']==MODEL_SHA
    for name,sha in parent['files'].items():assert digest(a.attention_output/name)==sha
    jobs=[];data=bytearray()
    def add(label,n,x,category='synthetic',**extra):
        assert len(x)==2048;status,out=reference(x,n);ident=len(jobs)+1
        sections=dict(input=pack32(x),expected=out)
        jobs.append(dict(id=ident,label=label,elements=n,status=status,expected_words=len(out)//8,category=category,
            hashes={k:hashlib.sha256(v).hexdigest() for k,v in sections.items()},**extra))
        data.extend(pack32([ident,n,status,len(out)//8])+sections['input']+out)
    x=[bits(((i%47)-23)/16) for i in range(2048)]
    for n in (32,64,256,1024,2048):add('length_'+str(n),n,x)
    add('signed_zero',2048,[0x80000000 if i%2 else 0 for i in range(2048)])
    add('subnormal',2048,[(0x80000000 if i%2 else 0)|(i+1) for i in range(2048)])
    add('maximum',2048,[0xff7fffff if i%2 else 0x7f7fffff for i in range(2048)])
    y=x.copy();y[-1]=0x7fc12345;add('outside_extent',32,y)
    for delta in (-1,0,1):
        y=[]
        for block in range(64):
            for lane in range(32):
                if lane==31:y.append(bits(254.0));continue
                k=(block*31+lane)%127;sign=-1 if (block*31+lane)//127%2 else 1
                y.append(bits(sign*(2*k+1))+delta)
        add('halfway_neighbor_'+str(delta),2048,y)
    for sample in range(1,5):
        raw=(a.attention_output/f'sample-{sample:03d}-attention-fp32.bin').read_bytes()
        want=(a.attention_output/f'sample-{sample:03d}-attention-abi1.bin').read_bytes()
        y=list(struct.unpack('<2048I',raw));assert reference(y,2048)==(0,want)
        add('attention_'+str(sample),2048,y,'real_layer0_attention',sample=sample,
            parent_input_sha256=hashlib.sha256(raw).hexdigest(),parent_abi1_sha256=hashlib.sha256(want).hexdigest())
    for n in (0,1,31,33,2049):add('bad_shape_'+str(n),n,x)
    for label,index,word in [('first_nan',0,0x7fc12345),('last_nan',2047,0x7fc12345),
        ('last_pos_inf',2047,0x7f800000),('last_neg_inf',2047,0xff800000),('last_snan',2047,0x7f800001)]:
        y=x.copy();y[index]=word;add(label,2048,y)
    y=x.copy();y[0]=0x7fc12345;add('shape_precedence',33,y)
    assert len(jobs)==27 and sum(j['status']==0 for j in jobs)==16 and sum(j['expected_words'] for j in jobs)==5984
    a.output.mkdir(parents=True)
    raw=b'NTRAQ8R1'+struct.pack('<I',len(jobs))+data
    with (a.output/'attention_quantize_r1.bin').open('xb') as f:f.write(raw)
    manifest=dict(format='NTRAQ8R1',cases=27,successful_cases=16,words=5984,records=748,jobs=jobs,
        model_sha256=MODEL_SHA,parent_manifest_sha256=PARENT_SHA,generator_sha256=digest(Path(__file__)),
        files={'attention_quantize_r1.bin':hashlib.sha256(raw).hexdigest()},
        scope='Full 2048-value layer0 attention vectors and Q8 boundary cases. Reference only, no RTL/AXI or inference qualification.')
    with (a.output/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='jobs'}))
if __name__=='__main__':main()
