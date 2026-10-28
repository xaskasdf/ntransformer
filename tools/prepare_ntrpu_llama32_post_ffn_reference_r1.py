"""Continue the four qualified post-O reference prefixes through full FFN and residual."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from check_ntrpu_post_attention_real_r1 import fixture as post_fixture
from check_ntrpu_residual_r1 import exact_add
from prepare_ntrpu_ffn_memory_fixture import project, quantize
from prepare_ntrpu_llama32_ffn_reference_r1 import activate, MODEL_SHA


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--post',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    post=post_fixture(a.post)
    parent=json.loads((a.weights/'manifest.json').read_text())
    assert parent['model_sha256']==MODEL_SHA and parent['layer']==0 and (parent['dim'],parent['hidden'])==(2048,8192)
    matrices=[]
    for i,tensor in enumerate(parent['weights'],1):
        name=f'weight-{i}-dense.bin';raw=(a.weights/name).read_bytes()
        assert len(raw)==17825792 and hashlib.sha256(raw).hexdigest()==parent['files'][name]==tensor['sha256']
        matrices.append(raw)
    assert len(matrices)==3
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,raw):
        with (a.output/name).open('xb') as f:f.write(raw)
    for i,raw in enumerate(matrices,1):write(f'weight-{i}-dense.bin',raw)
    samples=[]
    for sample,token in enumerate((0,42,128000,128255)):
        input_abi=struct.pack('<512Q',*post['abi64.hex'][sample*512:(sample+1)*512])
        records=np.frombuffer(input_abi,np.uint8).reshape(64,64)
        assert not records[:,40:].any()
        q=records[:,:32].copy().view(np.int8)
        scale=records[:,32:40].copy().reshape(-1).view('<f8')
        assert np.isfinite(scale).all() and (scale>=0).all() and (q>=-127).all()
        # Deliberately consume the exact preceding ABI1; no second quantization.
        gate=project(matrices[0],8192,q,scale);up=project(matrices[1],8192,q,scale)
        activated=activate(gate,up);aq,ascale,abi=quantize(activated)
        down=project(matrices[2],2048,aq,ascale);final=down.astype('<f4')
        residual=post['residual32.hex'][sample*2048:(sample+1)*2048]
        final_bits=final.view('<u4').tolist()
        output=[exact_add(x,y) for x,y in zip(residual,final_bits)]
        assert all((w&0x7f800000)!=0x7f800000 for w in output)
        assert all(np.isfinite(x).all() for x in (gate,up,activated,down,final))
        prefix=f'job-{sample+1:03d}'
        chunks={'input-abi':input_abi,'gate-fp64':gate.astype('<f8').tobytes(),
                'up-fp64':up.astype('<f8').tobytes(),'activation-fp32':activated.tobytes(),
                'activation-abi':abi,'down-fp64':down.astype('<f8').tobytes(),'final-fp32':final.tobytes(),
                'residual-fp32':struct.pack('<2048I',*residual),'layer-output-fp32':struct.pack('<2048I',*output)}
        for suffix,raw in chunks.items():write(prefix+'-'+suffix+'.bin',raw)
        samples.append(dict(id=sample+1,token_id=token,post_sample=sample,
            input_sha256=hashlib.sha256(input_abi).hexdigest(),input_abi_requantized=False,
            residual_sha256=hashlib.sha256(chunks['residual-fp32']).hexdigest(),
            layer_output_sha256=hashlib.sha256(chunks['layer-output-fp32']).hexdigest(),
            down_range=[float(final.min()),float(final.max())],nonzero_swiglu_blocks=int(np.count_nonzero(ascale))))
    manifest=dict(format='NTRPOSTFFN01',model_sha256=MODEL_SHA,revision=parent['revision'],layer=0,dim=2048,hidden=8192,
        post_manifest_sha256=digest(a.post/'manifest.json'),weight_parent_manifest_sha256=digest(a.weights/'manifest.json'),
        weights=parent['weights'],samples=samples,mac_blocks_per_sample=1572864,reference_values_per_sample=30720,
        input_abi_requantized=False,numpy_version=np.__version__,generator_sha256=digest(Path(__file__)),
        reference_contract='Original Q8_0 weights and exact post-O ABI1; ordered binary64 block sums, existing SwiGLU narrowing/quantization, FP32 down, exact integer RNE residual addition. NumPy exp requires native AMD comparison.',
        scope='Four layer-0 reference prefixes continued through FFN and second residual. Native/RTL execution and full-model inference not established by generation.',
        dependencies={n:digest(Path(__file__).with_name(n)) for n in ['check_ntrpu_post_attention_real_r1.py','check_ntrpu_residual_r1.py','prepare_ntrpu_ffn_memory_fixture.py','prepare_ntrpu_llama32_ffn_reference_r1.py']},
        files={p.name:digest(p) for p in sorted(a.output.iterdir())})
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(status='PREPARED_POST_FFN_REFERENCE_R1',manifest_sha256=digest(a.output/'manifest.json'),samples=samples)))


if __name__=='__main__':main()
