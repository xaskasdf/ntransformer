"""Extract all 33 real FP32 norm vectors and independent ordered references."""
import argparse
from decimal import Decimal,localcontext
import hashlib
import json
import math
from pathlib import Path
import struct

MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    audit=json.loads(a.audit.read_text());model=Path(audit['local_path'])
    assert audit['status']=='PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1' and digest(model)==audit['sha256']==MODEL_SHA
    epsilon=audit['metadata']['llama.attention.layer_norm_rms_epsilon'];assert struct.pack('<f',epsilon)==bytes.fromhex('acc52737')
    tensors={v['name']:v for v in audit['tensors']}
    names=[f'blk.{layer}.{kind}_norm.weight' for layer in range(16) for kind in ('attn','ffn')]+['output_norm.weight']
    payload=bytearray(b'NTRNORM1'+struct.pack('<I',33));jobs=[]
    with model.open('rb') as source,localcontext() as ctx:
        ctx.prec=80
        for ident,name in enumerate(names,1):
            t=tensors[name];assert t['type_name']=='F32' and t['shape_gguf']==[2048] and t['bytes']==8192
            source.seek(t['offset']);raw=source.read(t['bytes']);assert hashlib.sha256(raw).hexdigest()==t['sha256']
            weights=struct.unpack('<2048f',raw)
            inputs=[f32(((i*29+ident*17)%509-254)/64) for i in range(2048)]
            inputs[0]=-0.0;inputs[1]=struct.unpack('<f',struct.pack('<I',ident))[0]
            total=0.0
            for v in inputs:total=total+v*v
            inverse=1.0/math.sqrt(total/2048+epsilon)
            outputs=[f32((v*inverse)*w) for v,w in zip(inputs,weights)]
            exact_sum=sum(Decimal(v)*Decimal(v) for v in inputs)
            exact_inverse=1/(exact_sum/Decimal(2048)+Decimal(epsilon)).sqrt()
            exact=[f32(float(Decimal(v)*exact_inverse*Decimal(w))) for v,w in zip(inputs,weights)]
            # Decimal conversion does not retain the sign of zero through all
            # operations; compare magnitudes for zero and exact bits otherwise.
            for lhs,rhs in zip(outputs,exact):
                assert (lhs==rhs==0) or struct.pack('<f',lhs)==struct.pack('<f',rhs)
            xbytes=struct.pack('<2048f',*inputs);ybytes=struct.pack('<2048f',*outputs)
            payload+=struct.pack('<II',ident,2048)+struct.pack('<f',epsilon)+xbytes+raw+ybytes
            jobs.append(dict(id=ident,tensor=name,elements=2048,epsilon_bits=struct.pack('>f',epsilon).hex(),
                source_offset=t['offset'],weight_sha256=t['sha256'],input_sha256=hashlib.sha256(xbytes).hexdigest(),
                output_sha256=hashlib.sha256(ybytes).hexdigest(),decimal80_reference_agrees_fp32=True))
    a.output.mkdir(parents=True)
    fixture=a.output/'llama32_norm_r1.bin';fixture.write_bytes(payload)
    record=dict(format='NTRNORM1',model_sha256=MODEL_SHA,revision=audit['revision'],audit_sha256=digest(a.audit),
        generator_sha256=digest(Path(__file__)),files={fixture.name:digest(fixture)},bytes=len(payload),jobs=jobs,
        numerical_contract='Ordered FP64 sum of squared FP32 values; 1/sqrt(mean+FP32 epsilon); (x*inverse)*weight; round to FP32. Positive finite epsilon. No FMA/reassociation.',
        tolerance='At most 1 FP32 ULP to ordered reference, exact sign of zero. C/RTL comparison remains exact against AMD HLS C.',
        scope='All 33 original norm vectors, synthetic FP32 activations. Not hidden states, complete transformer execution, or CUDA/BF16 bit equivalence.')
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(fixture=str(fixture),sha256=digest(fixture),jobs=len(jobs),values=len(jobs)*2048,bytes=len(payload))))
if __name__=='__main__':main()
