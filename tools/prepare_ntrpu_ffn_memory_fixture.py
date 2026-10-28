"""Pack a reviewed real-weight Brandon FFN fixture for the shared AXI RTL bank.

Requires NumPy. Inputs are synthetic unit-RMS vectors from the preserved native
FFN fixture; model weights are real. Does not claim full-model inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from ntrpu_matrix_layout import pack_matrix

MATRIX_SHA='82ed3af643c6d17ff0260b68882cfac3d4a5464582460247ae20ca52df7f4709'


def digest(data):return hashlib.sha256(data).hexdigest()


def quantize(values):
    values=np.pad(np.asarray(values,dtype=np.float32),(0,(-len(values))%32)).astype(np.float64).reshape(-1,32)
    scale=np.max(np.abs(values),axis=1)/127.0
    ratio=np.divide(values,scale[:,None],out=np.zeros_like(values),where=scale[:,None]!=0)
    integer=np.floor(np.abs(ratio));integer+=(np.abs(ratio)-integer>=0.5)
    q=np.clip(np.copysign(integer,ratio),-127,127).astype(np.int8)
    image=b''.join(q[b].tobytes()+struct.pack('<d',s)+bytes(24) for b,s in enumerate(scale))
    return q,scale,image


def project(dense,rows,q,scale):
    blocks=len(scale);raw=np.frombuffer(dense,np.uint8).reshape(rows,blocks,34)
    sw=raw[:,:,:2].copy().reshape(-1).view('<f2').reshape(rows,blocks).astype(np.float64)
    weights=raw[:,:,2:].copy().view(np.int8).astype(np.int32)
    result=np.zeros(rows,np.float64)
    for b in range(blocks):
        dot=weights[:,b,:] @ q[b].astype(np.int32)
        result+=(sw[:,b]*scale[b])*dot.astype(np.float64)
    if not np.isfinite(result).all() or np.any(np.abs(result)>np.finfo(np.float32).max):
        raise ValueError('Projection cannot narrow to FP32')
    return result


def prepare(reference,matrix,output):
    if output.exists():raise FileExistsError(output)
    matrix_bytes=(matrix/'manifest.json').read_bytes()
    if digest(matrix_bytes)!=MATRIX_SHA:raise ValueError('Expected reviewed Brandon matrix manifest')
    mm=json.loads(matrix_bytes);report_bytes=(reference/'report.json').read_bytes();report=json.loads(report_bytes)
    if report['status']!='passed' or len(report['layers'])!=1 or report['layers'][0]['unique_layer']!=0:
        raise ValueError('Expected successful native layer0 reference')
    for old,new in [('model_id','model'),('revision','revision'),('weight_sha256','weights_sha256'),('config_sha256','config_sha256')]:
        if report[old]!=mm[new]:raise ValueError('Model provenance '+old)
    source=(reference/'ffn_00.bin').read_bytes()
    if digest(source)!=report['layers'][0]['fixture_sha256'] or source[:8]!=b'NTRFFN01':raise ValueError('Native fixture identity')
    dim,hidden,padded,samples=struct.unpack_from('<4I',source,8)
    if (dim,hidden,padded)!=(256,720,736) or not 1<=samples<=16:raise ValueError('Unsupported fixture shape/count')
    offset=24;dense=[];images=[];chunks=[4,4,8]
    for ident,(rows,blocks) in enumerate([(hidden,dim//32),(hidden,dim//32),(dim,padded//32)],1):
        size=rows*blocks*34;weights=source[offset:offset+size];offset+=size
        old=bytes.fromhex((matrix/f'matrix-{ident:03d}-dense.hex').read_text())
        if weights!=old or digest(weights)!=mm['jobs'][ident-1]['dense_sha256']:
            raise ValueError('Native and reviewed matrix weights differ')
        dense.append(weights);images.append(pack_matrix(weights,rows,blocks,chunks[ident-1])[0])
    count=samples*dim
    if len(source)!=offset+count*12:raise ValueError('Native fixture length')
    arrays=np.frombuffer(source,dtype='<f4',offset=offset).reshape(3,samples,dim)
    inputs,fp32,native_q8=arrays
    jobs=[];files={}
    def hexbytes(name,value):files[name]=''.join(f'{b:02x}\n' for b in value).encode('ascii')
    for ident,(weight,image) in enumerate(zip(dense,images),1):
        hexbytes(f'weight-{ident}.hex',image);hexbytes(f'weight-{ident}-dense.hex',weight)
    for index,x in enumerate(inputs,1):
        q,s,activation=quantize(x)
        gate=project(dense[0],hidden,q,s);up=project(dense[1],hidden,q,s)
        gf=gate.astype(np.float32).astype(np.float64);uf=up.astype(np.float32).astype(np.float64)
        exponent=np.exp(-np.abs(gf))
        sigmoid=np.where(gf>=0,1.0/(1.0+exponent),exponent/(1.0+exponent))
        activated=((gf*sigmoid)*uf).astype(np.float32)
        aq,ascale,abi=quantize(activated)
        down=project(dense[2],dim,aq,ascale);final=down.astype(np.float32)
        if final.tobytes()!=native_q8[index-1].tobytes():raise ValueError('Native Q8 final reference mismatch')
        prefix=f'job-{index:03d}'
        hexbytes(prefix+'-input.hex',b'\xa5'*4032+activation)
        hexbytes(prefix+'-input-fp32.hex',x.tobytes())
        for b,value in [(4,gate.astype('<f8').tobytes()),(5,up.astype('<f8').tobytes()),(6,abi),(7,down.astype('<f8').tobytes())]:
            hexbytes(prefix+f'-buffer-{b}.hex',value)
        hexbytes(prefix+'-activation-fp32.hex',activated.astype('<f4').tobytes())
        hexbytes(prefix+'-final-fp32.hex',final.astype('<f4').tobytes())
        hexbytes(prefix+'-baseline-fp32.hex',fp32[index-1].tobytes())
        error=final.astype(np.float64)-fp32[index-1].astype(np.float64)
        relative=float(np.linalg.norm(error)/np.linalg.norm(fp32[index-1].astype(np.float64)))
        jobs.append(dict(id=index,relative_l2=relative,max_abs_error=float(np.max(np.abs(error))),
                         gate_min=float(gate.min()),gate_max=float(gate.max()),native_q8_fp32_exact=True))
    files['jobs.txt']=(f'NTRFFNMEM01 {samples} {dim} {hidden}\n').encode('ascii')
    files['native-report.json']=report_bytes
    files['native-fixture.bin']=source
    manifest=dict(format='NTRFFNMEM01',layout=2,dim=dim,hidden=hidden,samples=samples,tile_k=chunks,
        model=mm['model'],revision=mm['revision'],weights_sha256=mm['weights_sha256'],config_sha256=mm['config_sha256'],
        matrix_manifest_sha256=MATRIX_SHA,native_fixture_sha256=digest(source),
        weight_dense_sha256=[digest(w) for w in dense],jobs=jobs,
        scope='Real Brandon layer0 weights, shared synthetic unit-RMS x, full FFN chain; no attention/norm/residual/logits/inference.',
        numerical_contract='Ordered binary64 block accumulation; checkedFP32 gate/up and SwiGLU; ABI1 quantization; raw binary64 down, explicit FP32 reference.',
        files={name:digest(value) for name,value in files.items()})
    output.mkdir(parents=True,exist_ok=False)
    for name,value in files.items():(output/name).write_bytes(value)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(samples=samples,manifest_sha256=digest((output/'manifest.json').read_bytes()),jobs=jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--matrix-fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.reference,a.matrix_fixture,a.output)
