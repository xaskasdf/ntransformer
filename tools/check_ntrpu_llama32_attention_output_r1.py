"""Independent full O/residual/norm reference and packed-weight byte audit."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import check_ntrpu_attention_axi_reference_r1 as attention
from check_ntrpu_norm_q8_r1 import reference as norm_reference
from check_ntrpu_residual_r1 import exact_add

MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
AUDIT_SHA = '5f41baa4d9d09a243b38a544a648fce8dc02e14035b8f357958a4ba6b35bea3a'
TOKENS = (0,42,128000,128255)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pack32(words):return struct.pack('<'+'I'*len(words),*words)
def bits32(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def val32(word):return struct.unpack('<f',struct.pack('<I',word))[0]


def quantize(words):
    raw = bytearray()
    for offset in range(0,len(words),32):
        block = [val32(word) for word in words[offset:offset+32]]
        scale = float(Fraction(max(map(abs,block)))/127)
        q = []
        for value in block:
            ratio = float(Fraction(value)/Fraction(scale)) if scale else 0.0
            ratio_exact = Fraction(abs(ratio))
            rounded = (2*ratio_exact.numerator+ratio_exact.denominator)//(2*ratio_exact.denominator)
            q.append(min(127,rounded)*(-1 if ratio < 0 else 1))
        raw.extend(struct.pack('<32bd',*q,scale)+bytes(24))
    return bytes(raw)


def oracle(audit_path,attention_path):
    assert sha(audit_path) == AUDIT_SHA
    audit = json.loads(audit_path.read_text()); model = Path(audit['local_path'])
    assert sha(model) == audit['sha256'] == MODEL_SHA
    tensors = {t['name']:t for t in audit['tensors']}
    dense = {}; embeddings = {}
    with model.open('rb') as f:
        for name in ('blk.0.attn_output.weight','blk.0.ffn_norm.weight'):
            t = tensors[name]; f.seek(t['offset']); raw = f.read(t['bytes'])
            assert hashlib.sha256(raw).hexdigest() == t['sha256']; dense[name] = raw
        embedding = tensors['token_embd.weight']
        for token in TOKENS:
            f.seek(embedding['offset']+token*2176); embeddings[token] = f.read(2176)
    jobs = attention.fixture(attention_path/'native-fixture')
    expect = {'output-weight-dense.bin':dense['blk.0.attn_output.weight'],
              'ffn-norm-weight.bin':dense['blk.0.ffn_norm.weight']}
    values_compared = 0
    for position,token in enumerate(TOKENS):
        words = []
        for head in range(32):
            job = jobs[position*32+head]
            assert job['tokens'] == position+1 and job['query_head'] == head and job['layer'] == 0
            words.extend(job['prefixes'][-1]['expected'])
        abi = quantize(words); output = []; rounded = []
        scales = [struct.unpack_from('<d',abi,k*64+32)[0] for k in range(64)]
        activation = [struct.unpack_from('<32b',abi,k*64) for k in range(64)]
        for row in range(2048):
            total = 0.0
            for k in range(64):
                offset = (row*64+k)*34
                factor = struct.unpack_from('<e',dense['blk.0.attn_output.weight'],offset)[0]
                weights = struct.unpack_from('<32b',dense['blk.0.attn_output.weight'],offset+2)
                dot = sum(a*w for a,w in zip(activation[k],weights))
                scale_product = float(Fraction(factor)*Fraction(scales[k]))
                partial = float(Fraction(scale_product)*dot)
                total = float(Fraction(total)+Fraction(partial))
            output.append(total); rounded.append(bits32(total)); values_compared += 1
        embed = []
        for offset in range(0,2176,34):
            factor = struct.unpack_from('<e',embeddings[token],offset)[0]
            embed.extend(bits32(float(Fraction(factor)*q)) for q in struct.unpack_from('<32b',embeddings[token],offset+2))
        residual = [exact_add(a,b) for a,b in zip(embed,rounded)]
        status,norm = norm_reference(residual,list(struct.unpack('<2048I',dense['blk.0.ffn_norm.weight'])),2048,0x3727c5ac)
        assert status == 0 and all(math.isfinite(val32(w)) for w in residual)
        stem = f'sample-{position+1:03d}'
        for key,raw in [('attention-fp32',pack32(words)),('attention-abi1',abi),
                        ('output-fp64',struct.pack('<2048d',*output)),('output-fp32',pack32(rounded)),
                        ('embedding-q8',embeddings[token]),('embedding-fp32',pack32(embed)),
                        ('residual-fp32',pack32(residual)),('ffn-input-abi1',norm)]:
            expect[stem+'-'+key+'.bin'] = raw
    assert values_compared == 8192
    return expect,tensors


def review_payload(manifest,files,expect,tensors):
    assert manifest['format'] == 'NTRLLAMA_ATTENTION_OUTPUT_R1'
    assert manifest['model_sha256'] == MODEL_SHA and manifest['audit_sha256'] == AUDIT_SHA
    assert manifest['native_archive_sha256'] == 'c0e02eae126c68c8649e2c4ebe49b0ddab3a703f0ad00732bd9ddc8b3966e089'
    assert manifest['native_review_sha256'] == '6cf4dbcfde78538d15f31e0af3aae3646c2beca3ae3b07442313d500477ed6e5'
    assert len(manifest['samples']) == 4 and manifest['epsilon'] == 0x3727c5ac
    assert manifest['embedding_tensor'] == tensors['token_embd.weight']
    for name in ('blk.0.attn_output.weight','blk.0.ffn_norm.weight'):
        assert manifest['sources'][name] == tensors[name]
    for position,(sample,token) in enumerate(zip(manifest['samples'],TOKENS)):
        assert tuple(sample[k] for k in ('id','token','position','layer','heads','dim','first_attention_case','last_attention_case')) == (
            position+1,token,position,0,32,2048,position*32+1,(position+1)*32)
    for name,raw in expect.items():
        assert files[name] == raw, ('Independent arithmetic/source',name)
    image = files['output-weight-image.bin']; dense = expect['output-weight-dense.bin']
    assert len(dense) == 4456448 and len(image) == 4456448+4032 and image[:4032] == b'\xa5'*4032
    for row in range(2048):
        for k in range(64):
            offset = 4032+((row//32)*2+k//32)*34816+((k%32)*32+row%32)*34
            assert image[offset:offset+34] == dense[(row*64+k)*34:(row*64+k+1)*34], 'Physical O block order'
    layout = manifest['output_matrix']
    assert tuple(layout[k] for k in ('rows','columns','k_blocks','tile_k','first_offset','image_bytes','weight_bytes')) == (
        2048,2048,64,32,4032,len(image),len(dense))
    assert len(layout['tiles']) == 128
    for n,tile in enumerate(layout['tiles']):
        assert tile == dict(first_row=(n//2)*32,rows=32,k_base=(n%2)*32,k_count=32,
                            address=0x100000000+4032+n*34816,offset=4032+n*34816,byte_count=34816)
    return dict(status='PASS_ATTENTION_OUTPUT_R1_INDEPENDENT_CPU_REFERENCE',samples=4,heads=128,
                ordered_fp64_projection_values=8192,residual_fp32_values=8192,
                attention_abi1_bytes=16384,ffn_input_abi1_bytes=16384,
                original_q8_weight_bytes=4456448,packed_weight_blocks=131072,
                max_projection_fp64_ulp=0,max_projection_fp32_ulp=0,
                rtl_qualified=False,full_inference_qualified=False)


def controls(manifest,files,expect,tensors):
    import copy
    rejected = []
    changes = ['head_order','attention_abi','output64','output32','residual','norm_abi','norm_weight',
               'packed_weight','packed_gap','embedding','token','tile','epsilon']
    mapping = {'attention_abi':'sample-001-attention-abi1.bin','output64':'sample-001-output-fp64.bin',
               'output32':'sample-001-output-fp32.bin','residual':'sample-001-residual-fp32.bin',
               'norm_abi':'sample-001-ffn-input-abi1.bin','norm_weight':'ffn-norm-weight.bin',
               'packed_weight':'output-weight-image.bin','packed_gap':'output-weight-image.bin',
               'embedding':'sample-001-embedding-q8.bin'}
    for name in changes:
        m = copy.deepcopy(manifest); f = files.copy()
        if name == 'head_order':
            key='sample-004-attention-fp32.bin';raw=f[key];f[key]=raw[256:512]+raw[:256]+raw[512:]
        elif name == 'token':m['samples'][1]['token'] += 1
        elif name == 'tile':m['output_matrix']['tiles'][1]['k_base'] = 0
        elif name == 'epsilon':m['epsilon'] ^= 1
        else:
            key=mapping[name];raw=bytearray(f[key]);index=4032 if name == 'packed_weight' else 0
            raw[index] ^= 1;f[key]=bytes(raw)
        assert m != manifest or f != files
        try: review_payload(m,f,expect,tensors)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--attention',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    manifest=json.loads((args.fixture/'manifest.json').read_text())
    files={name:(args.fixture/name).read_bytes() for name in manifest['files']}
    for name,want in manifest['files'].items():assert hashlib.sha256(files[name]).hexdigest() == want,name
    expect,tensors=oracle(args.audit,args.attention)
    result=review_payload(manifest,files,expect,tensors)
    if args.controls:result['rejected_controls']=controls(manifest,files,expect,tensors)
    result.update(manifest_sha256=sha(args.fixture/'manifest.json'),checker_sha256=sha(Path(__file__)))
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
