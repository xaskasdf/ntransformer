"""First-layer Q/K/V from four real embeddings, plus bounded synthetic attention."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from decimal import Decimal, localcontext
import prepare_ntrpu_llama32_rope_fixture_r1 as rope

MODEL_SHA=rope.MODEL_SHA
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pack32(words):return struct.pack('<'+'I'*len(words),*words)
bits=rope.bits
value=rope.value

def projection(raw, rows, activation):
    result=[]
    for row in range(rows):
        total=0.0
        for block in range(64):
            w=(row*64+block)*34;a=block*64
            weight_scale=struct.unpack_from('<e',raw,w)[0]
            activation_scale=struct.unpack_from('<d',activation,a+32)[0]
            dot=sum(x*y for x,y in zip(struct.unpack_from('<32b',raw,w+2),struct.unpack_from('<32b',activation,a)))
            total=total+(weight_scale*activation_scale)*float(dot)
        result.append(bits(total))
    return result

def offline(q,k,v):
    scores=[]
    for row in k:
        total=0.0
        for a,b in zip(q,row):total=total+value(a)*value(b)
        scores.append(total*.125)
    maximum=max(scores);weights=[math.exp(x-maximum) for x in scores];den=math.fsum(weights)
    # Dividing weights first keeps the offline FP64 reference within range.
    return [bits(math.fsum(value(row[d])*(w/den) for row,w in zip(v,weights))) for d in range(64)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--norm-q8',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();audit=json.loads(a.audit.read_text());model=Path(audit['local_path'])
    assert sha(model)==audit['sha256']==MODEL_SHA
    nr=a.norm_q8/'llama32_norm_q8_r1.bin';raw=nr.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='4aa438737577deefa119076c4ba02114ba2e7c35cc56af2395aee04acd4943fe'
    manifest=json.loads((a.norm_q8/'manifest.json').read_text());activations={};offset=12
    for job in manifest['jobs']:
        ident,n,eps,status,words=struct.unpack_from('<5I',raw,offset);offset+=20
        assert ident==job['id'];offset+=16384;output=raw[offset:offset+words*8];offset+=words*8
        if job['category']=='real_embedding_layer0_attn_norm':
            assert status==0 and n==2048 and hashlib.sha256(output).hexdigest()==job['hashes']['expected']
            activations[job['token_id']]=output
    assert offset==len(raw) and set(activations)=={0,42,128000,128255}
    tensors={t['name']:t for t in audit['tensors']};sources={};matrices={}
    with model.open('rb') as f:
        for name,rows in [('attn_q',2048),('attn_k',512),('attn_v',512)]:
            t=tensors['blk.0.'+name+'.weight'];assert t['shape_gguf']==[2048,rows] and t['type_name']=='Q8_0'
            f.seek(t['offset']);data=f.read(t['bytes']);assert hashlib.sha256(data).hexdigest()==t['sha256']
            matrices[name]=data;sources[t['name']]=t['sha256']
        t=tensors['rope_freqs.weight'];f.seek(t['offset']);data=f.read(128)
        assert hashlib.sha256(data).hexdigest()==t['sha256'];sources[t['name']]=t['sha256'];factors=struct.unpack('<32f',data)
    assert audit['metadata']['llama.rope.freq_base']==500000.
    projected=[]
    with localcontext() as ctx:
        ctx.prec=90
        for position,token in enumerate((0,42,128000,128255)):
            phase=[]
            for pair,factor in enumerate(factors):
                angle=(position*math.pow(500000,-pair/32))/factor
                actual=[bits(math.cos(angle)),bits(math.sin(angle))]
                exact=rope.decimal_trig(Decimal(position)*Decimal(500000)**(-Decimal(pair)/32)/Decimal(factor))
                assert actual==[bits(float(x)) for x in exact];phase.extend(actual)
            q=projection(matrices['attn_q'],2048,activations[token]);k=projection(matrices['attn_k'],512,activations[token])
            v=projection(matrices['attn_v'],512,activations[token])
            projected.append(dict(q=rope.reference(q,phase,32),k=rope.reference(k,phase,8),v=v))
    jobs=[];payload=bytearray();provenance=[]
    def add(label,q,k,v,head=0,width=64,category='synthetic'):
        ident=len(jobs)+1;n=len(k);assert len(v)==n and len(q)==64 and all(len(row)==64 for row in k+v)
        expected=offline(q,k,v)
        chunks=[pack32(q),pack32(sum(k,[])),pack32(sum(v,[])),pack32(expected)]
        job=dict(id=ident,label=label,tokens=n,layer=0,query_head=head,kv_head=head//4,tile_width=width,category=category,
            sequence=0x1234000000000000+ident,epoch=0xfedc000000000001,
            hashes={name:hashlib.sha256(data).hexdigest() for name,data in zip(('query','keys','values','expected'),chunks)})
        payload.extend(struct.pack('<6I2Q',ident,n,0,head,head//4,width,job['sequence'],job['epoch']))
        for data in chunks:payload.extend(data)
        jobs.append(job)
    for position in range(4):
        for head in range(32):
            kv=head//4
            add(f'embedding_prefix_{position+1}_head_{head}',projected[position]['q'][head*64:(head+1)*64],
                [x['k'][kv*64:(kv+1)*64] for x in projected[:position+1]],
                [x['v'][kv*64:(kv+1)*64] for x in projected[:position+1]],head=head,width=(head%3)+1,
                category='real_embedding_norm_q8_qkv_rope_layer0')
    for n in (1,2,3,31,63,64,65,127,129):
        q=[bits((d%17-8)/16) for d in range(64)]
        k=[[bits(((t*73+d*19)%257-128)/32) for d in range(64)] for t in range(n)]
        v=[[bits(((t*41+d*13)%509-254)/16) for d in range(64)] for t in range(n)]
        add('synthetic_'+str(n),q,k,v,width=31)
    for word in (0x7f7fffff,0xff7fffff,1,0x80000001):
        n=63;q=[bits(8)]+[0]*63;k=[[bits((t*37%193-96)/16)]+[0]*63 for t in range(n)]
        add('constant_'+hex(word),q,k,[[word]*64 for _ in range(n)],width=7)
    for sign in (-1,1):
        q=[bits(8)]+[0]*63;k=[[bits(sign*x)]+[0]*63 for x in (-1000,0,1000)]
        v=[[bits((t+1)*(d+1)) for d in range(64)] for t in range(3)]
        add('softmax_underflow_'+str(sign),q,k,v,width=1)
    a.output.mkdir();fixture=a.output/'llama32_attention_r1.bin'
    fixture.write_bytes(b'NTRATTN1'+struct.pack('<I',len(jobs))+payload)
    record=dict(format='NTRATTN1',jobs=jobs,cases=len(jobs),real_cases=128,output_words=64*len(jobs),
        transactions=sum((j['tokens']+j['tile_width']-1)//j['tile_width'] for j in jobs),
        model_sha256=MODEL_SHA,audit_sha256=sha(a.audit),norm_q8_fixture_sha256=sha(nr),source_tensors=sources,
        generator_sha256=sha(Path(__file__)),rope_helper_sha256=sha(Path(rope.__file__)),
        files={fixture.name:sha(fixture)},scope='Native first-layer projections for token IDs [0,42,128000,128255]; custom ABI1 input quantization. Attention unit validation only; not full-model inference or RTL proof.')
    (a.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='jobs'}))
if __name__=='__main__':main()
