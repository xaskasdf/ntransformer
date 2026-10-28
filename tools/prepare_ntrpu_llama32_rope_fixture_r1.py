"""Pin GGUF factors, construct phase tables, and export guarded RoPE cases."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct

# Full artifact identity, including the original GGUF metadata/tokenizer.
MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
PI=Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
POSITIONS=(0,1,2,31,127,1023,2047,8191,8192,32767,65535,131071)
MAX=2560
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def value(v):return struct.unpack('<f',struct.pack('<I',v))[0]
def packed(words):return struct.pack('<'+'I'*len(words),*words)
def order(b):return (~b)&0xffffffff if b&0x80000000 else b|0x80000000
def ulp(a,b):return abs(order(a)-order(b))

def decimal_trig(x):
    tau=PI*2;x=x%tau
    if x>PI:x-=tau
    square=x*x;c=Decimal(1);s=x;ct=c;st=s
    for n in range(1,130):
        ct=-ct*square/Decimal((2*n-1)*(2*n));st=-st*square/Decimal((2*n)*(2*n+1))
        c+=ct;s+=st
        if abs(ct)<Decimal('1e-85') and abs(st)<Decimal('1e-85'):break
    else:raise AssertionError('Decimal series did not converge')
    return c,s

def reference(x,phase,heads):
    out=[]
    for head in range(heads):
        for pair in range(32):
            a,b=(value(v) for v in x[head*64+pair*2:head*64+pair*2+2])
            c,s=(value(v) for v in phase[pair*2:pair*2+2])
            products=(a*c,b*s,a*s,b*c)
            y=(products[0]-products[1],products[2]+products[3])
            if any(abs(v)>value(0x7f7fffff) for v in y):return None
            out.extend(bits(v) for v in y)
    return out

def project(raw,rows):
    # Complete native Q/K projections on a shared deterministic INT8 vector.
    # This supplies representative inputs; it does not qualify the GEMV RTL.
    out=[]
    for row in range(rows):
        total=0.0
        for block in range(64):
            offset=(row*64+block)*34;scale=struct.unpack_from('<e',raw,offset)[0]
            weights=struct.unpack_from('<32b',raw,offset+2)
            dot=sum(w*((block*13+i*7)%255-127) for i,w in enumerate(weights))
            scaled=float(scale)*(1.0/(block%5+1))
            total=total+scaled*float(dot)
        out.append(bits(total))
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists()
    audit=json.loads(a.audit.read_text());model=Path(audit['local_path'])
    assert digest(model)==audit['sha256']==MODEL_SHA
    spec=next(t for t in audit['tensors'] if t['name']=='rope_freqs.weight')
    assert spec['shape_gguf']==[32] and spec['type_name']=='F32'
    with model.open('rb') as f:f.seek(spec['offset']);factor_bytes=f.read(128)
    assert hashlib.sha256(factor_bytes).hexdigest()==spec['sha256']
    factors=struct.unpack('<32f',factor_bytes);assert all(v>=1 and math.isfinite(v) for v in factors)
    base=audit['metadata']['llama.rope.freq_base'];assert base==500000.0
    tables={};max_coefficient_ulp=0
    with localcontext() as ctx:
        ctx.prec=90
        for position in sorted(set(POSITIONS+(7,))):
            table=[]
            for i,factor in enumerate(factors):
                angle=(position*math.pow(base,-i/32))/factor
                pair=(bits(math.cos(angle)),bits(math.sin(angle)))
                high=Decimal(position)*(Decimal(500000)**(-Decimal(i)/32))/Decimal(factor)
                dc,ds=decimal_trig(high);exact=(bits(float(dc)),bits(float(ds)))
                errors=[ulp(x,y) for x,y in zip(pair,exact)]
                assert max(errors)==0,(position,i,errors)
                max_coefficient_ulp=max(max_coefficient_ulp,*errors);table.extend(pair)
            tables[position]=table
    jobs=[]
    def default():return [bits(((i*29+17)%509-254)/64) for i in range(MAX)]
    def add(label,heads=40,position=7,phase_position=None,x=None,phase=None,status=0,category='synthetic'):
        x=default() if x is None else x[:];phase=tables[position][:] if phase is None else phase[:]
        pp=position if phase_position is None else phase_position
        y=reference(x,phase,heads) if status==0 else []
        assert y is not None
        jobs.append(dict(id=len(jobs)+1,label=label,heads=heads,position=position,coefficient_position=pp,
            status=status,input=x,coefficients=phase,expected=y,category=category))
    for p in POSITIONS:add('position_'+str(p),position=p,category='gguf_phase_synthetic_input')
    for heads in (1,3,8,32):add('heads_'+str(heads),heads=heads,position=31)
    x=[0x80000000 if i&1 else 0 for i in range(MAX)];add('signed_zero',position=0,x=x)
    x=[(0x80000000 if i&1 else 0)|(i+1) for i in range(MAX)];add('subnormal',position=8192,x=x)
    x=[0xff7fffff if i&1 else 0x7f7fffff for i in range(MAX)];add('finite_extremes',position=0,x=x)
    x=default();x[-1]=0x7fc00001;add('outside_extent',heads=1,x=x)
    add('quarter_turn',position=1,phase=[v for _ in range(32) for v in (0,0x3f800000)])
    for heads in (0,41,0xffffffff):add('bad_heads_'+str(heads),heads=heads,status=1)
    add('stale_phase_position',phase_position=6,status=2)
    for p in (131072,0xffffffff):add('bad_position_'+str(p),position=p,phase=tables[7],status=2)
    for bad in (0x7fc00001,0x7f800000,0xff800000,0x3f800001):
        ph=tables[7][:];ph[-1]=bad;add('bad_coefficient_'+hex(bad),phase=ph,status=3)
    ph=tables[7][:];ph[-2:]=[0,0];add('nonunit_coefficient',phase=ph,status=3)
    ph=tables[0][:];ph[-1]=0x80000000;add('noncanonical_position_zero',position=0,phase=ph,status=3)
    for bad in (0x7fc00001,0x7f800000,0xff800000):
        x=default();x[-1]=bad;add('late_input_'+hex(bad),x=x,status=4)
    ph=[bits(math.sqrt(.5))]*64;x=default();x[-2:]=[0x7f7fffff,0xff7fffff]
    assert reference(x,ph,40) is None;add('late_overflow',x=x,phase=ph,status=5)
    x[0]=0x7fc00001;add('input_precedes_overflow',x=x,phase=ph,status=4)
    ph[-1]=0x7fc00001;add('coefficient_precedes_input',x=x,phase=ph,status=3)
    add('shape_precedes_position',heads=0,position=0xffffffff,phase=ph,x=x,status=1)
    add('position_precedes_coefficient',phase_position=6,phase=ph,x=x,status=2)
    add('recovery')
    provenance=[];tensors={t['name']:t for t in audit['tensors']}
    with model.open('rb') as source:
        for layer in (0,15):
            vector=[]
            for kind,rows in [('attn_q',2048),('attn_k',512)]:
                t=tensors[f'blk.{layer}.{kind}.weight'];assert t['shape_gguf']==[2048,rows]
                source.seek(t['offset']);raw=source.read(t['bytes']);assert hashlib.sha256(raw).hexdigest()==t['sha256']
                vector+=project(raw,rows);provenance.append(dict(tensor=t['name'],sha256=t['sha256'],offset=t['offset'],bytes=t['bytes']))
            for p in (1,8192,131071):
                add(f'layer_{layer}_position_{p}',position=p,x=vector,category='native_full_real_qk_projection')
    payload=bytearray(b'NTRROPE1'+struct.pack('<I',len(jobs)));records=[];max_decimal_ulp=0
    with localcontext() as ctx:
        ctx.prec=90
        for job in jobs:
            if job['status']==0:
                for h in range(job['heads']):
                    for i in range(0,64,2):
                        idx=h*64+i;x,y=[Decimal(value(v)) for v in job['input'][idx:idx+2]]
                        c,s=[Decimal(value(v)) for v in job['coefficients'][i:i+2]]
                        exact=[x*c-y*s,x*s+y*c]
                        for j,v in enumerate(exact):
                            got=job['expected'][idx+j];want=bits(float(v))
                            error=0 if value(got)==value(want)==0 else ulp(got,want)
                            assert error<=1;max_decimal_ulp=max(max_decimal_ulp,error)
            chunks=[packed(job[k]) for k in ('coefficients','input','expected')]
            payload+=struct.pack('<6I',job['id'],job['heads'],job['position'],job['coefficient_position'],job['status'],len(job['expected']))
            for chunk in chunks:payload+=chunk
            row={k:v for k,v in job.items() if k not in ('coefficients','input','expected')}
            row.update(expected_elements=len(job['expected']),hashes={k:hashlib.sha256(chunk).hexdigest() for k,chunk in zip(('coefficients','input','expected'),chunks)})
            records.append(row)
    a.output.mkdir(parents=True);fixture=a.output/'llama32_rope_r1.bin';fixture.write_bytes(payload)
    (a.output/'rope_freqs.f32.bin').write_bytes(factor_bytes)
    manifest=dict(format='NTRROPE1',model_sha256=MODEL_SHA,audit_sha256=digest(a.audit),revision=audit['revision'],
        generator_sha256=digest(Path(__file__)),factor_sha256=spec['sha256'],source_tensors=provenance,
        files={p.name:digest(p) for p in (fixture,a.output/'rope_freqs.f32.bin')},jobs=records,
        cases=len(jobs),successful_cases=sum(j['status']==0 for j in jobs),values=sum(len(j['expected']) for j in jobs),
        phase_positions=list(POSITIONS),coefficient_decimal90_max_ulp=max_coefficient_ulp,rotation_decimal90_max_ulp=max_decimal_ulp,
        numerical_contract='Adjacent GGUF pairs; FP32 table cos/sin(position*500000**(-pair/32)/original_f32_factor). Widen FP32 exactly, four ordered FP64 products, subtract/add separately, final finite FP32 rounding. No FMA/reassociation.',
        scope='Real GGUF phase factors; six cases use complete native Q/K projections with synthetic INT8 activations. No full attention, original BF16 equivalence or RTL projection qualification.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('cases','successful_cases','values','coefficient_decimal90_max_ulp','rotation_decimal90_max_ulp')}|dict(fixture_sha256=digest(fixture))))
if __name__=='__main__':main()
