"""Pre-RoPE model projections tied to independently reviewed attention prefixes."""
import argparse,hashlib,json,math,struct
from decimal import Decimal,localcontext
from pathlib import Path
import prepare_ntrpu_llama32_attention_fixture_r1 as project
import check_ntrpu_rope_table_r1 as rotation
import check_ntrpu_attention_axi_reference_r1 as attention

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    rows=[]
    for sample,mode,bind,group,prefix in [(0,0,1,1,0),(1,0,0,1,1),(1,1,0,1,2),(3,1,0,1,2),
                                       (2,2,0,1,2),(2,0,0,1,2),(3,0,0,1,3),(0,0,1,2,0)]:
        rows.append(dict(sample=sample,mode=mode,rebind=bind,group=group,prefix=prefix,status=(0,1,3)[mode]))
    return rows
def build(base,output):
    assert not output.exists()
    audit_path=base/'audit-r1/audit.json';audit=json.loads(audit_path.read_text());model=Path(audit['local_path'])
    assert sha(model)==audit['sha256']==project.MODEL_SHA
    norm=base/'norm-q8-r1';raw=(norm/'llama32_norm_q8_r1.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='4aa438737577deefa119076c4ba02114ba2e7c35cc56af2395aee04acd4943fe'
    nm=json.loads((norm/'manifest.json').read_text());offset=12;activations={}
    for job in nm['jobs']:
        ident,n,eps,status,words=struct.unpack_from('<5I',raw,offset);offset+=20
        assert ident==job['id'];offset+=16384;data=raw[offset:offset+words*8];offset+=words*8
        if job['category']=='real_embedding_layer0_attn_norm':
            assert status==0 and n==2048 and hashlib.sha256(data).hexdigest()==job['hashes']['expected']
            activations[job['token_id']]=data
    assert offset==len(raw) and set(activations)=={0,42,128000,128255}
    tensors={t['name']:t for t in audit['tensors']};matrices={};sources={}
    with model.open('rb') as f:
        for name,rows in [('attn_q',2048),('attn_k',512),('attn_v',512)]:
            t=tensors['blk.0.'+name+'.weight'];assert t['shape_gguf']==[2048,rows] and t['type_name']=='Q8_0'
            f.seek(t['offset']);data=f.read(t['bytes']);assert hashlib.sha256(data).hexdigest()==t['sha256']
            matrices[name]=data;sources[t['name']]=t['sha256']
        t=tensors['rope_freqs.weight'];f.seek(t['offset']);data=f.read(128)
        assert hashlib.sha256(data).hexdigest()==t['sha256'];sources[t['name']]=t['sha256'];factors=struct.unpack('<32f',data)
    assert audit['metadata']['llama.rope.freq_base']==500000.
    inputs=[];rotated=[];values=[]
    with localcontext() as ctx:
        ctx.prec=90
        for position,token in enumerate((0,42,128000,128255)):
            phase=[]
            for pair,factor in enumerate(factors):
                exact=project.rope.decimal_trig(Decimal(position)*Decimal(500000)**(-Decimal(pair)/32)/Decimal(factor))
                phase.extend(project.bits(float(x)) for x in exact)
            q=project.projection(matrices['attn_q'],2048,activations[token]);k=project.projection(matrices['attn_k'],512,activations[token])
            v=project.projection(matrices['attn_v'],512,activations[token])
            status,out=rotation.oracle(40,position,position,phase,q+k);assert status==0 and len(out)==2560
            inputs.append(q+k+phase);rotated.append(out);values.append(v)
    parent=base/'attention-axi-r1/native-fixture';oracles=attention.fixture(parent)
    raw=(parent/'llama32_attention_r1.bin').read_bytes();offset=12
    for i in range(128):
        ident,n,layer,head,kv,width,seq,epoch=struct.unpack_from('<6I2Q',raw,offset);offset+=40
        assert (ident,n,layer,head,kv,width)==(i+1,i//32+1,0,i%32,(i%32)//4,64)
        q=list(struct.unpack_from('<64I',raw,offset));offset+=256
        k=list(struct.unpack_from('<'+'I'*(n*64),raw,offset));offset+=n*256
        v=list(struct.unpack_from('<'+'I'*(n*64),raw,offset));offset+=n*256
        y=list(struct.unpack_from('<64I',raw,offset));offset+=256
        assert q==rotated[n-1][head*64:(head+1)*64]==oracles[i]['q']
        assert k==sum([x[2048+kv*64:2048+(kv+1)*64] for x in rotated[:n]],[])
        assert v==sum([x[kv*64:(kv+1)*64] for x in values[:n]],[])
        assert all(attention.ulp(a,b)<=1 for a,b in zip(y,oracles[i]['prefixes'][-1]['expected']))
    files={}
    def hexfile(name,rows):files[name]=(''.join(f'{x:08x}\n' for row in rows for x in row)).encode()
    hexfile('input.hex',inputs);hexfile('rope-expected.hex',rotated);hexfile('v.hex',values)
    expected=[sum([oracles[p*32+h]['prefixes'][-1]['expected'] for h in range(32)],[]) for p in range(4)]
    hexfile('attention-expected.hex',expected)
    for name in ('manifest.json','llama32_attention_r1.bin'):files['attention-reference/'+name]=(parent/name).read_bytes()
    manifest=dict(format='NTRPU_ROPE_ATTENTION_FIXTURE_R1',model_sha256=audit['sha256'],audit_sha256=sha(audit_path),
        norm_fixture_sha256=sha(norm/'llama32_norm_q8_r1.bin'),source_tensors=sources,
        token_ids=[0,42,128000,128255],layer=0,positions=list(range(4)),plan=plan(),
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},generator_sha256=sha(Path(__file__)),
        helpers={Path(m.__file__).name:sha(Path(m.__file__)) for m in (project,project.rope,rotation,attention)},
        scope='First-layer Q/K/V native projections from four actual model embeddings using custom ABI1 quantization; pre-RoPE inputs and independent Decimal90 attention references. No RTL or full-model inference claim.')
    output.mkdir()
    for n,data in files.items():p=output/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(data)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return load(output)
def load(folder):
    m=json.loads((folder/'manifest.json').read_text());assert m['format']=='NTRPU_ROPE_ATTENTION_FIXTURE_R1' and m['plan']==plan()
    for n,h in m['files'].items():assert sha(folder/n)==h,n
    def words(n,count):
        vals=[int(x,16) for x in (folder/n).read_text().split()];assert len(vals)==count;return vals
    x=words('input.hex',4*2624);v=words('v.hex',4*512);y=words('rope-expected.hex',4*2560);o=words('attention-expected.hex',4*2048)
    refs=attention.fixture(folder/'attention-reference');rows=[]
    for p in range(4):
        source=x[p*2624:(p+1)*2624];rot=y[p*2560:(p+1)*2560]
        assert rotation.oracle(40,p,p,source[2560:],source[:2560])==(0,rot)
        assert all(refs[p*32+h]['q']==rot[h*64:(h+1)*64] for h in range(32))
        assert o[p*2048:(p+1)*2048]==sum([refs[p*32+h]['prefixes'][-1]['expected'] for h in range(32)],[])
        rows.append(dict(position=p,input=source,v=v[p*512:(p+1)*512],rope=rot,attention=o[p*2048:(p+1)*2048]))
    return m,rows,refs
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m,rows,refs=build(a.base,a.output)
    print(json.dumps(dict(manifest_sha256=sha(a.output/'manifest.json'),samples=len(rows),jobs=len(m['plan']),model_sha256=m['model_sha256'],parent_heads_verified=128)))
