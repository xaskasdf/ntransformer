"""Independently rebuild full Q/K/V and bind them to existing attention inputs."""
import argparse,copy,hashlib,json,struct
from decimal import Decimal,localcontext
from fractions import Fraction
from pathlib import Path
from check_ntrpu_norm_q8_r1 import reference as normalize
from check_ntrpu_rope_table_r1 import oracle as rotate
from check_ntrpu_ffn_memory_fixture import decode
from prepare_ntrpu_llama32_rope_fixture_r1 import decimal_trig

MODEL='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
AUDIT='5f41baa4d9d09a243b38a544a648fce8dc02e14035b8f357958a4ba6b35bea3a'
ATTENTION='6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361'
TOKENS=(0,42,128000,128255)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def words(raw):return list(struct.unpack('<'+'I'*(len(raw)//4),raw))
def pack(v):return struct.pack('<'+'I'*len(v),*v)

def reference(base):
    audit_path=base/'audit-r1/audit.json';assert file_sha(audit_path)==AUDIT
    audit=json.loads(audit_path.read_text());model=Path(audit['local_path']);assert file_sha(model)==MODEL==audit['sha256']
    tensors={t['name']:t for t in audit['tensors']};sources={};raws={};embeddings={}
    with model.open('rb') as f:
        for name in ['blk.0.attn_q.weight','blk.0.attn_k.weight','blk.0.attn_v.weight','blk.0.attn_norm.weight','rope_freqs.weight']:
            t=tensors[name];f.seek(t['offset']);data=f.read(t['bytes']);assert sha(data)==t['sha256'];raws[name]=data
            sources[name.split('.')[2] if name.startswith('blk.0.attn_') and name!='blk.0.attn_norm.weight' else name]=t
        for token in TOKENS:
            f.seek(tensors['token_embd.weight']['offset']+2176*token);embeddings[token]=f.read(2176)
    expected={};samples=[]
    def image(stem,data):
        expected[stem+'.bin']=data
        expected[stem+'.hex']=(''.join(f'{int.from_bytes(data[i:i+64],"little"):0128x}\n' for i in range(0,len(data),64))).encode()
    expected['attn-norm-fp32.bin']=raws['blk.0.attn_norm.weight'];expected['rope-factors-fp32.bin']=raws['rope_freqs.weight']
    factors=struct.unpack('<32f',raws['rope_freqs.weight']);projected=[]
    for position,token in enumerate(TOKENS):
        stem=f'sample-{position+1:03d}';x=[]
        for off in range(0,2176,34):
            factor=struct.unpack_from('<e',embeddings[token],off)[0]
            x.extend(bits(factor*q) for q in struct.unpack_from('<32b',embeddings[token],off+2))
        expected[stem+'-embedding-fp32.bin']=pack(x)
        status,abi=normalize(x,words(raws['blk.0.attn_norm.weight']),2048,0x3727c5ac);assert status==0 and len(abi)==4096
        image(stem+'-input-abi1',abi)
        qs=[struct.unpack_from('<32b',abi,b*64) for b in range(64)]
        scales=[Fraction(struct.unpack_from('<d',abi,b*64+32)[0]) for b in range(64)]
        raw_outputs={}
        for name,rows in [('attn_q',2048),('attn_k',512),('attn_v',512)]:
            weight=raws['blk.0.'+name+'.weight'];out=[];rounded=[]
            for row in range(rows):
                total=Fraction(0)
                for k in range(64):
                    off=(row*64+k)*34;factor=Fraction(struct.unpack_from('<e',weight,off)[0])
                    coeff=struct.unpack_from('<32b',weight,off+2);dot=sum(a*w for a,w in zip(qs[k],coeff))
                    scale_product=Fraction(float(factor*scales[k]));partial=Fraction(float(scale_product*dot));total=Fraction(float(total+partial))
                out.append(float(total));rounded.append(bits(float(total)))
            expected[stem+'-'+name+'-fp64.bin']=struct.pack('<'+'d'*rows,*out);image(stem+'-'+name+'-fp32',pack(rounded));raw_outputs[name]=rounded
        phase=[]
        with localcontext() as ctx:
            ctx.prec=90
            for pair,factor in enumerate(factors):
                angle=Decimal(position)*Decimal(500000)**(-Decimal(pair)/32)/Decimal.from_float(factor)
                phase.extend(bits(float(v)) for v in decimal_trig(angle))
        expected[stem+'-rope-phase-fp32.bin']=pack(phase)
        st,q=rotate(32,position,position,phase,raw_outputs['attn_q']);assert st==0
        st,k=rotate(8,position,position,phase,raw_outputs['attn_k']);assert st==0
        image(stem+'-q-rope-fp32',pack(q));image(stem+'-k-rope-fp32',pack(k))
        projected.append(dict(q=q,k=k,v=raw_outputs['attn_v']))
        samples.append(dict(id=position+1,token_id=token,position=position,layer=0,input_bytes=4096,q_bytes=8192,k_bytes=2048,v_bytes=2048))
    # All 128 real cases must consume these exact vectors and prefix ordering.
    attention=(base/'attention-r1/llama32_attention_r1.bin').read_bytes();assert sha(attention)==ATTENTION
    assert attention[:12]==b'NTRATTN1'+struct.pack('<I',143);offset=12;matched=0
    for case in range(1,144):
        ident,n,layer,head,kv,width,seq,epoch=struct.unpack_from('<6I2Q',attention,offset);offset+=40
        assert ident==case
        q=attention[offset:offset+256];offset+=256
        k=attention[offset:offset+256*n];offset+=256*n
        v=attention[offset:offset+256*n];offset+=256*n
        offset+=256
        if case<=128:
            position=(case-1)//32;h=(case-1)%32;assert (n,layer,head,kv)==(position+1,0,h,h//4)
            assert q==pack(projected[position]['q'][h*64:(h+1)*64])
            assert k==b''.join(pack(projected[p]['k'][kv*64:(kv+1)*64]) for p in range(n))
            assert v==b''.join(pack(projected[p]['v'][kv*64:(kv+1)*64]) for p in range(n));matched+=1
    assert matched==128 and offset==len(attention)
    return dict(expected=expected,raws=raws,sources=sources,samples=samples,attention_cases=matched)

def review(manifest,files,ref):
    assert manifest['format']=='NTRLLAMA32QKV01' and manifest['model_sha256']==MODEL and manifest['audit_sha256']==AUDIT
    assert manifest['norm_fixture_sha256']=='4aa438737577deefa119076c4ba02114ba2e7c35cc56af2395aee04acd4943fe'
    assert manifest['source_tensors']==ref['sources'] and manifest['samples']==ref['samples']
    assert manifest['original_weight_bytes']==6684672 and manifest['fp32_projection_values']==12288
    assert set(files)==set(manifest['files']) and all(sha(files[n])==h for n,h in manifest['files'].items())
    expected_names=set(ref['expected']);assert len(manifest['matrices'])==3
    for index,(name,rows) in enumerate([('attn_q',2048),('attn_k',512),('attn_v',512)]):
        dense=ref['raws']['blk.0.'+name+'.weight'];assert files[name+'-dense-q8.bin']==dense
        image=files[name+'-weights.bin'];assert decode(image,rows,64,32)==dense
        heximage=files[name+'-weights.hex'].decode().splitlines();assert len(heximage)*64==len(image)
        assert b''.join(int(line,16).to_bytes(64,'little') for line in heximage)==image
        base=0x100000000+index*0x20000000;offset=4032;tiles=[]
        for r in range(0,rows,32):
            for k in (0,32):
                tiles.append(dict(first_row=r,rows=32,k_base=k,k_count=32,address=base+offset,offset=offset,byte_count=34816));offset+=34816
        assert manifest['matrices'][index]==dict(name=name,rows=rows,k_blocks=64,tile_k=32,weight_addr=base+4032,region_base=base,region_bytes=offset,
            dense_sha256=sha(dense),weight_bytes=len(dense),tiles=tiles,output_bytes=rows*4)
        expected_names.update((name+'-dense-q8.bin',name+'-weights.bin',name+'-weights.hex'))
    assert set(files)==expected_names and len(files)==79
    for n,raw in ref['expected'].items():assert files[n]==raw,n
    return dict(status='PASS_LLAMA32_QKV_REFERENCE_R1',samples=4,full_projections=12,fp64_values=12288,fp32_values=12288,
        original_weight_bytes=6684672,rotated_qk_values=10240,attention_cases=ref['attention_cases'],rtl_executed=False,
        scope='Independent ordered rational-to-binary64 projection, normalization and RoPE reference; complete packed GGUF byte identity and exact existing attention-input binding.')

def controls(manifest,files,ref):
    variants=[]
    for name,offset in [('attn_q-dense-q8.bin',34),('attn_k-weights.bin',4032),('attn_v-weights.bin',0),
        ('sample-001-input-abi1.bin',32),('sample-002-attn_q-fp64.bin',64),('sample-003-attn_k-fp32.bin',0),
        ('sample-004-attn_v-fp32.bin',2044),('sample-004-rope-phase-fp32.bin',0),('sample-003-q-rope-fp32.bin',128),
        ('sample-001-k-rope-fp32.bin',256),('attn-norm-fp32.bin',4),('rope-factors-fp32.bin',0)]:
        m=copy.deepcopy(manifest);f=dict(files);raw=bytearray(f[name]);raw[offset]^=1;f[name]=bytes(raw);m['files'][name]=sha(f[name]);variants.append((name,m,f))
    for name in ('position','token','rows','address','extent','source','coverage','unknown-file'):
        m=copy.deepcopy(manifest);f=dict(files)
        if name=='position':m['samples'][2]['position']=0
        elif name=='token':m['samples'][1]['token_id']=0
        elif name=='rows':m['matrices'][1]['rows']=2048
        elif name=='address':m['matrices'][2]['weight_addr']^=1<<40
        elif name=='extent':m['matrices'][0]['region_bytes']-=64
        elif name=='source':m['source_tensors']['attn_q']['sha256']='0'*64
        elif name=='coverage':m['fp32_projection_values']-=1
        else:f['foreign.bin']=b'x';m['files']['foreign.bin']=sha(b'x')
        variants.append((name,m,f))
    rejected=[]
    for name,m,f in variants:
        try:review(m,f,ref)
        except (AssertionError,ValueError,KeyError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    assert len(rejected)==20
    return rejected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=reference(a.model_dir);manifest=json.loads((a.fixture/'manifest.json').read_text())
    assert all(Path(n).name==n for n in manifest['files'])
    files={n:(a.fixture/n).read_bytes() for n in manifest['files']};report=review(manifest,files,ref);report['rejected_controls']=controls(manifest,files,ref)
    report.update(manifest_sha256=file_sha(a.fixture/'manifest.json'),checker_sha256=file_sha(Path(__file__)),
        helper_sha256={n:file_sha(Path(__file__).parent/n) for n in ['check_ntrpu_norm_q8_r1.py','check_ntrpu_rope_table_r1.py','check_ntrpu_ffn_memory_fixture.py','prepare_ntrpu_llama32_rope_fixture_r1.py']})
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
