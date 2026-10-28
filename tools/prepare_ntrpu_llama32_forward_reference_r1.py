"""Complete one-token, position-zero NTRPU arithmetic reference, all 16 layers.

This exports a CPU oracle, not an RTL or board result. It preserves GGUF Q8_0
weights and NTRPU ABI1 activation quantization; it does not emulate BF16/CUDA.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import numpy as np
from prepare_ntrpu_ffn_memory_fixture import project, quantize
from prepare_ntrpu_llama32_ffn_reference_r1 import activate
from check_ntrpu_residual_r1 import exact_add
from prepare_ntrpu_llama32_rope_fixture_r1 import reference as rope_reference

MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
LAYER0_SHA='e431a82afd68b25ebc2c6883f0efbbba697d04770a8b030ff49a48c2b9b04123'
D,H,VOCAB=2048,8192,128256


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def normalized(values,weight,epsilon):
    total=0.0
    for value in values:
        value=float(value);total=total+value*value
    inverse=1.0/math.sqrt(total/D+epsilon)
    return np.array([(float(x)*inverse)*float(w) for x,w in zip(values,weight)],dtype='<f4')


def residual(left,right):
    words=[exact_add(int(a),int(b)) for a,b in zip(left.view('<u4'),right.view('<u4'))]
    result=np.array(words,dtype='<u4').view('<f4')
    if not np.isfinite(result).all():raise ValueError('Nonfinite residual')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    audit=json.loads(a.audit.read_text());assert audit['status']=='PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1'
    assert digest(a.model)==audit['sha256']==MODEL_SHA and a.model.stat().st_size==1321082528
    tensors={t['name']:t for t in audit['tensors']}
    assert len(tensors)==147 and 'output.weight' not in tensors and audit['output_weight_present'] is False
    epsilon=audit['metadata']['llama.attention.layer_norm_rms_epsilon']
    assert struct.pack('<f',epsilon)==bytes.fromhex('acc52737')
    a.output.mkdir(parents=True);source=a.model.open('rb');used=set();files={}
    def read(name):
        t=tensors[name];source.seek(t['offset']);raw=source.read(t['bytes'])
        assert hashlib.sha256(raw).hexdigest()==t['sha256'];used.add(name);return raw
    def weight(name):return np.frombuffer(read(name),dtype='<f4').copy()
    def write(name,value):
        raw=value if isinstance(value,bytes) else value.tobytes()
        with (a.output/name).open('xb') as f:f.write(raw)
        files[name]=hashlib.sha256(raw).hexdigest();return files[name]
    raw=read('token_embd.weight')
    assert len(raw)==VOCAB*64*34 and tensors['token_embd.weight']['shape_gguf']==[D,VOCAB]
    embedding=np.frombuffer(raw[:64*34],dtype=np.uint8).reshape(64,34)
    scales=embedding[:,:2].copy().reshape(-1).view('<f2').astype(np.float64)
    current=(embedding[:,2:].copy().view(np.int8).astype(np.float64)*scales[:,None]).astype('<f4').reshape(-1)
    del raw
    write('embedding-fp32.bin',current)
    factors=weight('rope_freqs.weight');assert len(factors)==32 and (factors>=1).all()
    phase=[0x3f800000,0]*32;write('rope-phase-fp32.bin',struct.pack('<64I',*phase))
    layers=[];started=time.monotonic()
    for layer in range(16):
        before=time.monotonic();prefix='layer-%02d-'%layer;model_prefix='blk.%d.'%layer
        def save(name,value):return write(prefix+name+'.bin',value)
        input_hash=save('input-fp32',current)
        norm=normalized(current,weight(model_prefix+'attn_norm.weight'),epsilon)
        save('input-norm-fp32',norm);q,scale,abi=quantize(norm);save('input-abi',abi)
        projections=[]
        for name,rows in [('q',D),('k',512),('v',512)]:
            value=project(read(model_prefix+'attn_'+name+'.weight'),rows,q,scale)
            save(name+'-fp64',value.astype('<f8'));value=value.astype('<f4')
            save(name+'-fp32',value);projections.append(value)
        qv,kv,vv=projections
        qk=np.concatenate([qv,kv]).view('<u4').tolist()
        rotated=rope_reference(qk,phase,40);assert rotated is not None
        rotated=np.array(rotated,dtype='<u4').view('<f4');save('qk-rope-fp32',rotated)
        attention=np.zeros(D,dtype='<f4');scores=[]
        for head in range(32):
            kh=head//4;dot=0.0
            for x,y in zip(rotated[head*64:(head+1)*64],rotated[D+kh*64:D+(kh+1)*64]):
                dot=dot+float(x)*float(y)
            scores.append(dot*0.125)
            # Position zero has one key; normalized softmax weight is exactly 1.
            attention[head*64:(head+1)*64]=vv[kh*64:(kh+1)*64]
        save('attention-score-fp64',np.array(scores,dtype='<f8'));save('attention-fp32',attention)
        oq,os,oabi=quantize(attention);save('o-abi',oabi)
        o=project(read(model_prefix+'attn_output.weight'),D,oq,os)
        save('o-fp64',o.astype('<f8'));o=o.astype('<f4');save('o-fp32',o)
        post=residual(current,o);save('post-residual-fp32',post)
        ffn_norm=normalized(post,weight(model_prefix+'ffn_norm.weight'),epsilon)
        save('ffn-norm-fp32',ffn_norm);fq,fs,fabi=quantize(ffn_norm);save('ffn-abi',fabi)
        gate=project(read(model_prefix+'ffn_gate.weight'),H,fq,fs)
        up=project(read(model_prefix+'ffn_up.weight'),H,fq,fs)
        save('gate-fp64',gate.astype('<f8'));save('up-fp64',up.astype('<f8'))
        act=activate(gate,up);save('swiglu-fp32',act);aq,asc,actabi=quantize(act);save('swiglu-abi',actabi)
        down=project(read(model_prefix+'ffn_down.weight'),D,aq,asc)
        save('down-fp64',down.astype('<f8'));down=down.astype('<f4');save('down-fp32',down)
        current=residual(post,down);output_hash=save('output-fp32',current)
        assert np.isfinite(current).all()
        if layer==0:assert output_hash==LAYER0_SHA,'Layer 0 differs from approved RTL vector'
        row=dict(layer=layer,input_sha256=input_hash,output_sha256=output_hash,
                 max_abs=float(np.max(np.abs(current))),reference_wall_seconds=time.monotonic()-before)
        layers.append(row);print(json.dumps(row),flush=True)
    final_norm=normalized(current,weight('output_norm.weight'),epsilon)
    write('output-norm-fp32.bin',final_norm);q,s,abi=quantize(final_norm);write('output-abi.bin',abi)
    t=tensors['token_embd.weight'];logits=np.empty(VOCAB,dtype='<f8');chunks=[]
    for start in range(0,VOCAB,8192):
        rows=min(8192,VOCAB-start);offset=t['offset']+start*2176
        source.seek(offset);raw=source.read(rows*2176)
        logits[start:start+rows]=project(raw,rows,q,s)
        chunks.append(dict(index=len(chunks),first_row=start,rows=rows,gguf_offset=offset,
            weight_bytes=len(raw),weight_sha256=hashlib.sha256(raw).hexdigest(),
            output_offset_bytes=start*4,output_bytes=rows*4))
    source.close();assert len(used)==147 and len(chunks)==16 and chunks[-1]['rows']==5376
    write('logits-fp64.bin',logits);narrow=logits.astype('<f4');write('logits-fp32.bin',narrow)
    ranking=np.argsort(-narrow,kind='stable')[:10]
    top=[dict(token=int(index),logit=float(narrow[index])) for index in ranking]
    deps=['prepare_ntrpu_ffn_memory_fixture.py','prepare_ntrpu_llama32_ffn_reference_r1.py',
          'check_ntrpu_residual_r1.py','prepare_ntrpu_llama32_rope_fixture_r1.py','ntrpu_matrix_layout.py']
    manifest=dict(status='PREPARED_LLAMA32_FORWARD_REFERENCE_R1',model_sha256=MODEL_SHA,
        audit_sha256=digest(a.audit),model_revision=audit['revision'],token_id=0,position=0,layers=layers,
        dimensions=dict(dim=D,hidden=H,vocabulary=VOCAB,layers=16),tensor_sources=[tensors[n] for n in sorted(used)],
        epsilon_bits=struct.pack('>f',epsilon).hex(),output_chunks=chunks,top10=top,
        argmax_token=int(ranking[0]),files=files,generator_sha256=digest(Path(__file__)),
        helpers={n:digest(Path(__file__).with_name(n)) for n in deps},numpy_version=np.__version__,
        reference_wall_seconds=time.monotonic()-started,layer0_matches_approved_rtl=True,
        native_kernel_crosscheck_pending=True,whole_model_rtl_qualified=False,board_qualified=False,
        scope='CPU oracle for one input token 0 at position 0 through all 16 layers and tied output. '
              'Custom NTRPU activation Q8/ordered FP64 contract. No multi-token cache, tokenizer, '
              'sampling generation, BF16/CUDA equivalence, RTL latency or hardware throughput.')
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(status=manifest['status'],layers=16,logits=VOCAB,top10=top,
        manifest_sha256=digest(a.output/'manifest.json'))),flush=True)


if __name__=='__main__':main()
