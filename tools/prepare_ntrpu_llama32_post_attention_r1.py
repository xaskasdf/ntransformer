"""Freeze post-O reference cases using independent integer/Decimal arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from check_ntrpu_residual_r1 import exact_add
from check_ntrpu_norm_q8_r1 import reference as norm_reference

ATTENTION_MANIFEST = '73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'
MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
EPS = 0x3727c5ac
def digest(data): return hashlib.sha256(data).hexdigest()
def pack32(words): return struct.pack('<' + 'I' * len(words), *words)


def reference(x, branch, weight, n, eps):
    if not 0 < n <= 2048 or n % 32: return 1, [], b''
    if any((v & 0x7f800000) == 0x7f800000 for v in x[:n] + branch[:n]): return 2, [], b''
    residual = [exact_add(a, b) for a, b in zip(x[:n], branch[:n])]
    if any((v & 0x7f800000) == 0x7f800000 for v in residual): return 3, [], b''
    status, abi = norm_reference(residual, weight, n, eps)
    if status: return (1 if status == 1 else status + 2), [], b''
    return 0, residual, abi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    raw_manifest = (args.attention/'manifest.json').read_bytes()
    assert digest(raw_manifest) == ATTENTION_MANIFEST
    parent = json.loads(raw_manifest)
    for name, sha in parent['files'].items(): assert digest((args.attention/name).read_bytes()) == sha
    jobs = []; payload = bytearray()
    def read32(name): return list(struct.unpack('<2048I', (args.attention/name).read_bytes()))
    def add(label, x, branch, weight, n=2048, eps=EPS, **extra):
        assert len(x) == len(branch) == len(weight) == 2048
        status, residual, abi = reference(x, branch, weight, n, eps)
        sections = dict(residual_input=pack32(x), branch=pack32(branch), norm_weight=pack32(weight),
                        residual_output=pack32(residual), ffn_input=abi)
        ident = len(jobs) + 1
        jobs.append(dict(id=ident, label=label, elements=n, epsilon=eps, status=status,
            residual_words=len(residual), abi_words=len(abi)//8,
            hashes={k:digest(v) for k,v in sections.items()}, **extra))
        payload.extend(pack32([ident,n,eps,status]))
        for data in sections.values(): payload.extend(data)
        return sections
    weight = read32('ffn-norm-weight.bin')
    for sample, token in enumerate([0,42,128000,128255], 1):
        prefix = f'sample-{sample:03d}-'
        sections = add('real_attention_prefix_'+str(sample), read32(prefix+'embedding-fp32.bin'),
                       read32(prefix+'output-fp32.bin'), weight, token=token, prefix=sample)
        assert sections['residual_output'] == (args.attention/(prefix+'residual-fp32.bin')).read_bytes()
        assert sections['ffn_input'] == (args.attention/(prefix+'ffn-input-abi1.bin')).read_bytes()
    def base(): return [0x3f800000]*2048, [0]*2048, [0x3f800000]*2048
    x,b,w = base(); add('short32',x,b,w,n=32)
    x,b,w = base(); x[-1]=b[-1]=w[-1]=0x7fc12345; add('inactive_tail_poison',x,b,w,n=64)
    x,b,w = base(); x[:32]=[0x80000000]*32; b[:32]=[0x80000000]*32; add('negative_zeros',x,b,w,n=32)
    x,b,w = base(); x[:32]=list(range(1,33)); b[:32]=[1]*32; add('subnormal',x,b,w,n=32,eps=1)
    x,b,w = base(); b[:32]=[0xbf800000]*32; add('exact_cancellation',x,b,w,n=32)
    x,b,w = base(); b[:32]=[0x33800000]*32; x[1]=0x3f800001; add('rne_ties',x,b,w,n=32)
    for n in [0,1,31,33,2049]:
        x,b,w=base(); x[0]=0x7fc00001; add('shape_precedence_'+str(n),x,b,w,n=n,eps=0)
    for port in ('x','branch'):
        x,b,w=base(); (x if port=='x' else b)[-1]=0x7fc00001; add('late_nonfinite_'+port,x,b,w)
    x,b,w=base(); x[-1]=b[-1]=0x7f7fffff; add('residual_overflow',x,b,w,eps=0)
    x[0]=0x7f800000; add('nonfinite_before_overflow',x,b,w)
    for eps in [0,0xbf800000,0x7fc00001]:
        x,b,w=base(); add('epsilon_'+str(eps),x,b,w,eps=eps)
    x,b,w=base(); w[-1]=0x7f800000; add('late_norm_weight_inf',x,b,w)
    x,b,w=base(); x=[0]*2048; x[-1]=0x3f800000; w[-1]=0x7f7fffff; add('late_norm_overflow',x,b,w)
    w[0]=0x7fc00001; add('norm_nonfinite_before_overflow',x,b,w)
    # Recover after all rejected transactions with a different valid extent.
    x,b,w=base(); add('recovery',x,b,w,n=64)
    assert len(jobs)==26 and {j['status'] for j in jobs}==set(range(7))
    raw=b'NTRPOST1'+pack32([len(jobs)])+payload
    args.output.mkdir(parents=True)
    (args.output/'llama32_post_attention_r1.bin').write_bytes(raw)
    manifest=dict(format='NTRPOST1',jobs=jobs,cases=len(jobs),successful=sum(j['status']==0 for j in jobs),
        residual_words=sum(j['residual_words'] for j in jobs),abi_words=sum(j['abi_words'] for j in jobs),
        parent_manifest_sha256=ATTENTION_MANIFEST,model_sha256=MODEL_SHA,
        files={'llama32_post_attention_r1.bin':digest(raw)},
        generator_sha256=digest(Path(__file__).read_bytes()),
        scope='Local post-O residual and FFN normalization, four real layer0 attention prefixes. No AXI or full inference.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='jobs'}))


if __name__=='__main__': main()
