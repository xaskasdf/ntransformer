"""Full layer-0 FFN reference from pinned Q8 weights; synthetic FP32 inputs.

No attention, RMSNorm, residual addition or model inference is performed.
Preserves raw GGUF block bytes and ordered binary64 GEMV accumulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

from prepare_ntrpu_ffn_memory_fixture import project, quantize

MODEL_SHA='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def activate(gate,up):
    g=gate.astype('<f4').astype(np.float64);u=up.astype('<f4').astype(np.float64)
    exponent=np.exp(-np.abs(g));exponent[np.abs(g)>256]=0
    sigmoid=np.where(g>=0,1.0/(1.0+exponent),exponent/(1.0+exponent))
    return ((g*sigmoid)*u).astype('<f4')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    audit=json.loads(args.audit.read_text(encoding='utf-8'))
    if audit['status']!='PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1' or audit['sha256']!=MODEL_SHA or digest(args.model)!=MODEL_SHA:
        raise ValueError('Pinned model/audit identity')
    by_name={t['name']:t for t in audit['tensors']}
    matrices=[];provenance=[]
    with args.model.open('rb') as stream:
        for name,shape in [('ffn_gate',[2048,8192]),('ffn_up',[2048,8192]),('ffn_down',[8192,2048])]:
            tensor=by_name['blk.0.'+name+'.weight']
            if tensor['type']!=8 or tensor['shape_gguf']!=shape:raise ValueError('Tensor type/shape')
            stream.seek(tensor['offset']);raw=stream.read(tensor['bytes'])
            if hashlib.sha256(raw).hexdigest()!=tensor['sha256']:raise ValueError('Tensor bytes')
            matrices.append(raw);provenance.append(tensor)
    args.output.mkdir(parents=True,exist_ok=False)
    def write(name,raw):
        with (args.output/name).open('xb') as stream:stream.write(raw)
    for index,raw in enumerate(matrices,1):write(f'weight-{index}-dense.bin',raw)
    samples=[]
    for sample,amplitude in enumerate((0.25,1.0,4.0),1):
        x=((((np.arange(2048,dtype=np.int32)*73+sample*11)%257)-128)/64.0*amplitude).astype('<f4')
        q,s,input_abi=quantize(x)
        gate=project(matrices[0],8192,q,s);up=project(matrices[1],8192,q,s)
        activated=activate(gate,up)
        aq,ascale,abi=quantize(activated)
        down=project(matrices[2],2048,aq,ascale);final=down.astype('<f4')
        if not all(np.isfinite(v).all() for v in (x,gate,up,activated,ascale,down,final)):
            raise ValueError('Nonfinite FFN reference')
        prefix=f'job-{sample:03d}'
        for suffix,raw in [('input-fp32',x.tobytes()),('input-abi',input_abi),
                           ('gate-fp64',gate.astype('<f8').tobytes()),('up-fp64',up.astype('<f8').tobytes()),
                           ('activation-fp32',activated.tobytes()),('activation-abi',abi),
                           ('down-fp64',down.astype('<f8').tobytes()),('final-fp32',final.tobytes())]:
            write(prefix+'-'+suffix+'.bin',raw)
        samples.append(dict(id=sample,input_pattern='((i*73+sample*11)%257-128)/64*amplitude',amplitude=amplitude,
                            input_rms=float(np.sqrt(np.mean(x.astype(np.float64)**2))),
                            gate_range=[float(gate.min()),float(gate.max())],
                            up_range=[float(up.min()),float(up.max())],
                            activation_range=[float(activated.min()),float(activated.max())],
                            final_range=[float(final.min()),float(final.max())],
                            nonzero_activation_blocks=int(np.count_nonzero(ascale))))
    write('jobs.txt',b'NTRLLAMAFFN01 3 2048 8192\n')
    manifest=dict(format='NTRLLAMAFFN01',model=audit['repo'],revision=audit['revision'],model_sha256=MODEL_SHA,
                  audit_sha256=digest(args.audit),layer=0,dim=2048,hidden=8192,samples=samples,weights=provenance,
                  mac_blocks_per_sample=1572864,reference_values_per_sample=36864,
                  scope='Full layer0 FFN numerical reference using real Unsloth Q8_0 weights and three synthetic FP32 inputs; no model inference or RTL result.',
                  numerical_contract='Original Q8_0 weights; maxabs/127 activation quantization with FP64 scales; ordered FP64 block sums; FP32 gate/up and SwiGLU narrowing; FP32 final output. NumPy exp is not assumed identical to HLS exp.',
                  tools={name:digest(Path(__file__).with_name(name)) for name in
                         ['prepare_ntrpu_llama32_ffn_reference_r1.py','prepare_ntrpu_ffn_memory_fixture.py']},
                  numpy_version=np.__version__,files={p.name:digest(p) for p in sorted(args.output.iterdir())})
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(status='PREPARED_FULL_LLAMA32_FFN_NUMPY_REFERENCE_R1',
                         manifest_sha256=digest(args.output/'manifest.json'),weight_bytes=sum(map(len,matrices)),samples=samples)))


if __name__=='__main__':main()
