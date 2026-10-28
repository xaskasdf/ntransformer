"""Prepare full layer-0 Q/K/V matrices and four real normalized embedding paths.

This is reference material for a future FPGA producer, not RTL execution.
Original Q8 blocks are reordered only; numerical accumulation remains ordered.
"""
import argparse,hashlib,json,math,struct
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ntrpu_matrix_layout import pack_matrix
from prepare_ntrpu_ffn_memory_fixture import project
from prepare_ntrpu_llama32_rope_fixture_r1 import reference as rotate

MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
NORM_SHA='4aa438737577deefa119076c4ba02114ba2e7c35cc56af2395aee04acd4943fe'
TOKENS=(0,42,128000,128255)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bits(v):return struct.unpack('<I',struct.pack('<f',v))[0]
def pack(words):return struct.pack('<'+'I'*len(words),*words)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    audit_path=a.model_dir/'audit-r1/audit.json';audit=json.loads(audit_path.read_text());model=Path(audit['local_path'])
    assert audit['status']=='PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1' and digest(model)==audit['sha256']==MODEL_SHA
    tensors={t['name']:t for t in audit['tensors']};payload={};specs={}
    with model.open('rb') as f:
        for name,rows in [('attn_q',2048),('attn_k',512),('attn_v',512)]:
            t=tensors['blk.0.'+name+'.weight'];assert t['type_name']=='Q8_0' and t['shape_gguf']==[2048,rows]
            f.seek(t['offset']);raw=f.read(t['bytes']);assert hashlib.sha256(raw).hexdigest()==t['sha256']
            payload[name]=raw;specs[name]=t
        for name in ('rope_freqs.weight','blk.0.attn_norm.weight'):
            t=tensors[name];f.seek(t['offset']);raw=f.read(t['bytes']);assert hashlib.sha256(raw).hexdigest()==t['sha256']
            payload[name]=raw;specs[name]=t
    nr=a.model_dir/'norm-q8-r1/llama32_norm_q8_r1.bin';raw=nr.read_bytes();assert digest(nr)==NORM_SHA
    nm_path=nr.parent/'manifest.json';nm=json.loads(nm_path.read_text());inputs={};offset=12
    assert raw[:8]==b'NTRNQ8R1' and struct.unpack_from('<I',raw,8)[0]==len(nm['jobs'])==61
    for j in nm['jobs']:
        ident,n,eps,status,words=struct.unpack_from('<5I',raw,offset);offset+=20
        assert (ident,n,eps,status,words)==(j['id'],j['elements'],j['epsilon'],j['status'],j['expected_words'])
        x=raw[offset:offset+8192];w=raw[offset+8192:offset+16384];offset+=16384
        abi=raw[offset:offset+8*words];offset+=8*words
        if j['category']=='real_embedding_layer0_attn_norm':
            assert status==0 and n==2048 and len(abi)==4096 and w==payload['blk.0.attn_norm.weight']
            for key,data in [('input',x),('weight',w),('expected',abi)]:assert hashlib.sha256(data).hexdigest()==j['hashes'][key]
            inputs[j['token_id']]=(x,abi)
    assert offset==len(raw) and set(inputs)==set(TOKENS)
    a.output.mkdir();files={}
    def write(name,data):
        with (a.output/name).open('xb') as f:f.write(data)
        files[name]=hashlib.sha256(data).hexdigest()
    def image(name,data):
        assert len(data)%64==0
        write(name+'.bin',data)
        write(name+'.hex',(''.join(data[i:i+64][::-1].hex()+'\n' for i in range(0,len(data),64))).encode())
    matrices=[]
    for index,(name,rows) in enumerate([('attn_q',2048),('attn_k',512),('attn_v',512)]):
        base=0x100000000+index*0x20000000;packed,tiles=pack_matrix(payload[name],rows,64,32,region_base=base)
        write(name+'-dense-q8.bin',payload[name]);image(name+'-weights',packed)
        matrices.append(dict(name=name,rows=rows,k_blocks=64,tile_k=32,weight_addr=base+4032,region_base=base,region_bytes=len(packed),
            dense_sha256=specs[name]['sha256'],weight_bytes=len(payload[name]),tiles=[asdict(t) for t in tiles],output_bytes=rows*4))
    write('attn-norm-fp32.bin',payload['blk.0.attn_norm.weight']);write('rope-factors-fp32.bin',payload['rope_freqs.weight'])
    factors=struct.unpack('<32f',payload['rope_freqs.weight']);assert audit['metadata']['llama.rope.freq_base']==500000.0
    samples=[]
    for position,token in enumerate(TOKENS):
        stem=f'sample-{position+1:03d}';x,abi=inputs[token]
        image(stem+'-input-abi1',abi);write(stem+'-embedding-fp32.bin',x)
        q=np.array([struct.unpack_from('<32b',abi,k*64) for k in range(64)],dtype=np.int8)
        scale=np.array([struct.unpack_from('<d',abi,k*64+32)[0] for k in range(64)],dtype=np.float64)
        projections={}
        for spec in matrices:
            name=spec['name'];y=project(payload[name],spec['rows'],q,scale)
            write(stem+'-'+name+'-fp64.bin',y.astype('<f8').tobytes())
            out=y.astype('<f4').tobytes();image(stem+'-'+name+'-fp32',out);projections[name]=list(struct.unpack('<'+'I'*spec['rows'],out))
        phase=[]
        for pair,factor in enumerate(factors):
            angle=position*math.pow(500000.,-pair/32)/factor
            phase.extend([bits(math.cos(angle)),bits(math.sin(angle))])
        write(stem+'-rope-phase-fp32.bin',pack(phase))
        image(stem+'-q-rope-fp32',pack(rotate(projections['attn_q'],phase,32)))
        image(stem+'-k-rope-fp32',pack(rotate(projections['attn_k'],phase,8)))
        samples.append(dict(id=position+1,token_id=token,position=position,layer=0,input_bytes=4096,q_bytes=8192,k_bytes=2048,v_bytes=2048))
    record=dict(format='NTRLLAMA32QKV01',model_sha256=MODEL_SHA,audit_sha256=digest(audit_path),norm_fixture_sha256=NORM_SHA,
        norm_manifest_sha256=digest(nm_path),source_tensors=specs,matrices=matrices,samples=samples,
        original_weight_bytes=sum(len(payload[n]) for n in ('attn_q','attn_k','attn_v')),fp32_projection_values=12288,
        generator_sha256=digest(Path(__file__)),helper_sha256={n:digest(Path(__file__).parent/n) for n in ['ntrpu_matrix_layout.py','prepare_ntrpu_ffn_memory_fixture.py','prepare_ntrpu_llama32_rope_fixture_r1.py']},files=files,
        scope='Reference only. Full Q/K/V for layer0, four actual embedding/norm ABI1 inputs, and RoPE at positions0..3. No producer RTL, AXI transfer, integrated token inference or physical throughput qualification.')
    with (a.output/'manifest.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='PREPARED_LLAMA32_QKV_REFERENCE_R1',files=len(files),weight_bytes=record['original_weight_bytes'],projection_values=12288,manifest_sha256=digest(a.output/'manifest.json'))),flush=True)
if __name__=='__main__':main()
