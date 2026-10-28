"""Freeze real embedding rows plus directed/random residual addition inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct

MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def packed(words):return struct.pack('<'+'I'*len(words),*words)
def value(word):return struct.unpack('<f',packed([word]))[0]
def bits(v):
    try:return struct.unpack('<I',struct.pack('<f',v))[0]
    except OverflowError:return 0xff800000 if v<0 else 0x7f800000
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and sha(a.model)==MODEL_SHA
    audit=json.loads(a.audit.read_text());assert audit['sha256']==MODEL_SHA
    tensor=next(t for t in audit['tensors'] if t['name']=='token_embd.weight')
    assert tensor['shape_gguf']==[2048,128256] and tensor['type_name']=='Q8_0'
    with a.model.open('rb') as f:f.seek(tensor['offset']);payload=f.read(tensor['bytes'])
    assert hashlib.sha256(payload).hexdigest()==tensor['sha256']
    jobs=[];binary=bytearray();rng=random.Random(0x51d32)
    def add(label,n,x,y,category='synthetic',**meta):
        x=(x*((2048+len(x)-1)//len(x)))[:2048];y=(y*((2048+len(y)-1)//len(y)))[:2048]
        if not 1<=n<=2048:status,out=1,[]
        elif any(not math.isfinite(value(v)) for v in x[:n]+y[:n]):status,out=2,[]
        else:
            out=[bits(value(u)+value(v)) for u,v in zip(x[:n],y[:n])]
            status=3 if any((v&0x7f800000)==0x7f800000 for v in out) else 0
            if status:out=[]
        ident=len(jobs)+1
        jobs.append(dict(id=ident,label=label,elements=n,status=status,expected_elements=len(out),category=category,
            hashes={k:hashlib.sha256(packed(v)).hexdigest() for k,v in [('residual',x),('branch',y),('expected',out)]},**meta))
        binary.extend(packed([ident,n,status,len(out)]+x+y+out))
    for n in [1,2,3,31,32,33,127,128,2047,2048]:
        x=[bits(rng.uniform(-4,4)) for _ in range(2048)];y=[bits(rng.uniform(-2,2)) for _ in range(2048)]
        if n<2048:x[n]=0x7fc12345;y[n]=0xff800000 # Inactive tail must not be consumed.
        add('length_'+str(n),n,x,y)
    patterns=[
        ('signed_zero',[0,0,0x80000000,0x80000000],[0,0x80000000,0,0x80000000]),
        ('subnormal',[1,0x80000001,0x7fffff,0x807fffff],[1,0x80000001,1,0x80000001]),
        ('cancellation',[0x3f800000,0xbf800000,0x00800000,0x7f7fffff],[0xbf800000,0x3f800000,0x807fffff,0xff7fffff]),
        ('even_tie',[0x3f800000,0xbf800000],[0x33800000,0xb3800000]),
        ('odd_tie',[0x3f800001,0xbf800001],[0x33800000,0xb3800000]),
        ('carry',[0x3fffffff,0xbfffffff],[0x33800000,0xb3800000]),
        ('below_overflow',[0x7f7fffff,0xff7fffff],[0x72800000,0xf2800000]),
        ('normal_subnormal_boundary',[0x00800000,0x80800000,0x007fffff],[0x80000001,1,1]),
        ('large_gap',[0x7f7fffff,0xff7fffff,0x3f800000],[1,0x80000001,0x00800000])]
    for label,x,y in patterns:add(label,2048,x,y)
    for sample in range(10):
        def word():return (rng.getrandbits(1)<<31)|(rng.randrange(201)<<23)|rng.getrandbits(23)
        add('exponent_spread_'+str(sample),2048,[word() for _ in range(2048)],[word() for _ in range(2048)])
    rows={}
    for token in [0,42,128000,128255]:
        raw=payload[token*2176:(token+1)*2176];rows[str(token)]=hashlib.sha256(raw).hexdigest();x=[]
        for i in range(0,len(raw),34):
            scale=struct.unpack_from('<e',raw,i)[0]
            x.extend(bits(scale*q) for q in struct.unpack_from('<32b',raw,i+2))
        y=[bits(rng.uniform(-0.125,0.125)) for _ in range(2048)]
        add('embedding_'+str(token),2048,x,y,'real_embedding_synthetic_branch',token_id=token,row_sha256=rows[str(token)])
    for n in [0,2049,0xffffffff]:add('invalid_shape_'+str(n),n,[0x7fc12345],[0x7f800000])
    for label,where,word,side in [('nan_first',0,0x7fc12345,0),('snan_last',2047,0x7f800001,1),
                                  ('positive_inf',1023,0x7f800000,0),('negative_inf',2047,0xff800000,1)]:
        pair=[[0x3f800000]*2048,[0xbf000000]*2048];pair[side][where]=word;add(label,2048,*pair)
    add('positive_overflow',2048,[0x7f7fffff],[0x73000000])
    add('negative_overflow',2048,[0xff7fffff],[0xf3000000])
    x=[0x7f7fffff]*2048;y=[0x7f7fffff]*2048;y[-1]=0x7fc12345
    add('nonfinite_precedes_overflow',2048,x,y)
    assert len(jobs)==43 and sum(j['status']==0 for j in jobs)==33
    a.output.mkdir()
    raw=b'NTRRADD1'+struct.pack('<I',len(jobs))+binary
    (a.output/'llama32_residual_r1.bin').write_bytes(raw)
    manifest=dict(format='NTRRADD1',cases=len(jobs),successful_cases=33,values=sum(j['expected_elements'] for j in jobs),jobs=jobs,
        model_sha256=MODEL_SHA,audit_sha256=sha(a.audit),embedding_tensor_sha256=tensor['sha256'],embedding_rows=rows,
        generator_sha256=sha(Path(__file__)),files={'llama32_residual_r1.bin':hashlib.sha256(raw).hexdigest()},
        scope='Residual primitive; four dequantized real embedding rows with synthetic branches, not full layer activations or inference.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='jobs'}))
if __name__=='__main__':main()
