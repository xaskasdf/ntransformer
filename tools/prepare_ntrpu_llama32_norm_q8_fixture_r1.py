"""Compose full Llama RMSNorm inputs and activation ABI1 numerical references."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct

MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pack32(v):return struct.pack('<'+'I'*len(v),*v)
def val(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def reference(x,w,n,eps):
    if not 1<=n<=2048 or n%32:return 1,b''
    epsilon=val(eps)
    if not math.isfinite(epsilon) or epsilon<=0:return 2,b''
    if not all(math.isfinite(val(v)) for v in x[:n]+w[:n]):return 3,b''
    total=0.0
    for word in x[:n]:total=total+val(word)*val(word)
    inverse=1.0/math.sqrt(total/n+epsilon);out=[]
    for a,b in zip(x[:n],w[:n]):
        v=(val(a)*inverse)*val(b)
        if abs(v)>val(0x7f7fffff):return 4,b''
        out.append(val(bits(v)))
    records=bytearray()
    for base in range(0,n,32):
        block=out[base:base+32];maximum=max(abs(v) for v in block);scale=maximum/127.0
        q=[]
        for v in block:
            ratio=v/scale if scale else 0.0;f=Fraction(abs(ratio))
            rounded=(2*f.numerator+f.denominator)//(2*f.denominator)
            q.append((-1 if ratio<0 else 1)*min(rounded,127))
        records.extend(struct.pack('<32bd',*q,scale)+bytes(24))
    return 0,bytes(records)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--norm-fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and sha(a.model)==MODEL_SHA
    audit=json.loads(a.audit.read_text());assert audit['sha256']==MODEL_SHA
    norm_data=(a.norm_fixture/'llama32_norm_r1.bin').read_bytes()
    assert hashlib.sha256(norm_data).hexdigest()=='66573f15c4070efaf91d5baf1c3c64fb58441a9d84ad4790e115c8d8a9ee3e60'
    norm_manifest=json.loads((a.norm_fixture/'manifest.json').read_text())
    jobs=[];data=bytearray();eps=0x3727c5ac
    def add(label,n,x,w,e=eps,category='synthetic',**extra):
        status,output=reference(x,w,n,e);ident=len(jobs)+1
        assert len(x)==len(w)==2048
        sections=dict(input=pack32(x),weight=pack32(w),expected=output)
        jobs.append(dict(id=ident,label=label,elements=n,epsilon=e,status=status,expected_words=len(output)//8,
            category=category,hashes={k:hashlib.sha256(v).hexdigest() for k,v in sections.items()},**extra))
        data.extend(pack32([ident,n,e,status,len(output)//8])+sections['input']+sections['weight']+output)
    def fill():return ([bits(((i%47)-23)/16) for i in range(2048)],[bits(((i%19)-9)/8) for i in range(2048)])
    x,w=fill()
    for n in [32,64,256,1024,2048]:add('length_'+str(n),n,x,w)
    add('signed_zero',2048,[0x80000000 if i%2 else 0 for i in range(2048)],w)
    add('subnormal',2048,[(0x80000000 if i%2 else 0)|(i+1) for i in range(2048)],[bits(1)]*2048,e=1)
    add('maximum',2048,[0xff7fffff if i%2 else 0x7f7fffff for i in range(2048)],[bits(.5)]*2048)
    x,w=fill();x[-1]=0x7fc12345;w[-1]=0xff800000;add('outside_extent',32,x,w)
    for delta in [-1,0,1]:
        w=[]
        for block in range(64):
            for lane in range(32):
                if lane==31:w.append(bits(254.0));continue
                k=(block*31+lane)%127;sign=-1 if (block*31+lane)//127%2 else 1
                word=bits(sign*(2*k+1));w.append(word+delta)
        add('quantization_halfway_neighbor_'+str(delta),2048,[bits(1)]*2048,w,e=bits(3))
    offset=12
    for index,spec in enumerate(norm_manifest['jobs'],1):
        ident,n,e=struct.unpack_from('<III',norm_data,offset);offset+=12
        assert (ident,n,e)==(index,2048,eps)
        x=list(struct.unpack_from('<2048I',norm_data,offset));offset+=8192
        w=list(struct.unpack_from('<2048I',norm_data,offset));offset+=8192
        offset+=8192
        assert hashlib.sha256(pack32(x)).hexdigest()==spec['input_sha256']
        assert hashlib.sha256(pack32(w)).hexdigest()==spec['weight_sha256']
        add('original_norm_'+str(index),n,x,w,category='real_norm_synthetic_activation',norm_case=index)
    assert offset==len(norm_data)
    embedding=next(t for t in audit['tensors'] if t['name']=='token_embd.weight')
    norm=next(t for t in audit['tensors'] if t['name']=='blk.0.attn_norm.weight')
    with a.model.open('rb') as f:
        f.seek(embedding['offset']);payload=f.read(embedding['bytes']);assert hashlib.sha256(payload).hexdigest()==embedding['sha256']
        f.seek(norm['offset']);weight=f.read(norm['bytes']);assert hashlib.sha256(weight).hexdigest()==norm['sha256']
    w=list(struct.unpack('<2048I',weight))
    for token in [0,42,128000,128255]:
        row=payload[token*2176:(token+1)*2176];x=[]
        for i in range(0,len(row),34):
            scale=struct.unpack_from('<e',row,i)[0]
            x.extend(bits(scale*q) for q in struct.unpack_from('<32b',row,i+2))
        add('token_embedding_attn_norm_'+str(token),2048,x,w,category='real_embedding_layer0_attn_norm',token_id=token,
            embedding_row_sha256=hashlib.sha256(row).hexdigest(),norm_tensor_sha256=norm['sha256'])
    x,w=fill()
    for n in [0,1,31,33,2049]:add('bad_shape_'+str(n),n,x,w)
    for e in [0,0xbf800000,0x7fc00001]:add('bad_epsilon_'+str(e),2048,x,w,e=e)
    x,w=fill();x[-1]=0x7fc12345;add('late_input_nan',2048,x,w)
    x,w=fill();w[-1]=0x7f800000;add('late_weight_inf',2048,x,w)
    x=[0]*2048;w=[bits(1)]*2048;x[-1]=bits(1);w[-1]=0x7f7fffff
    add('late_overflow',2048,x,w);x[0]=0x7fc12345;add('nonfinite_precedence',2048,x,w)
    assert len(jobs)==61 and sum(j['status']==0 for j in jobs)==49
    a.output.mkdir()
    raw=b'NTRNQ8R1'+struct.pack('<I',len(jobs))+data
    (a.output/'llama32_norm_q8_r1.bin').write_bytes(raw)
    manifest=dict(format='NTRNQ8R1',cases=61,successful_cases=49,words=sum(j['expected_words'] for j in jobs),
        records=sum(j['expected_words']//8 for j in jobs),jobs=jobs,model_sha256=MODEL_SHA,audit_sha256=sha(a.audit),
        original_norm_fixture_sha256=hashlib.sha256(norm_data).hexdigest(),generator_sha256=sha(Path(__file__)),
        files={'llama32_norm_q8_r1.bin':hashlib.sha256(raw).hexdigest()},
        scope='Local normalization and ABI1 quantization; four actual embedding/layer0 norm paths, 33 real norm vectors with synthetic inputs. No GEMV or full inference.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='jobs'}))
if __name__=='__main__':main()
