"""Pinned Brandon projection fixtures for the real AXI/Q8/HLS RTL path.

Synthetic unit-RMS inputs, real layer-0 weights; these are isolated projections,
not an FFN or model inference. Requires the reference NumPy environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from run_ntrpu_brandon import (BrandonWeights, CONFIG_SHA256, MODEL_ID, REVISION,
                              WEIGHT_SHA256, quantize_q8, round_away)


def bits(value):
    return struct.pack('>d', float(value)).hex()


def prepare(model_dir, output):
    model = BrandonWeights(model_dir)
    audit = model.audit()
    output.mkdir(parents=True, exist_ok=False)
    jobs, matrices = [], []
    for matrix_index, name in enumerate(('w1', 'w3', 'w2')):
        tensor = f'_unique_layers.0.feed_forward.{name}.weight'
        weights = model.tensor(tensor)
        packed, values, scales = quantize_q8(weights)
        n, columns = weights.shape
        k = values.shape[1]
        rng = np.random.default_rng(20261003 + matrix_index)
        x = rng.standard_normal(columns).astype(np.float32)
        x /= np.sqrt(np.mean(x*x))
        padded = np.pad(x.astype(np.float64), (0, k*32-columns)).reshape(k,32)
        sx = np.max(np.abs(padded), axis=1) / 127.0
        divided = np.divide(padded,sx[:,None],out=np.zeros_like(padded),where=sx[:,None]!=0)
        qx = np.clip(round_away(divided),-127,127).astype(np.int8)
        expected = []
        # Scalar integer dot and ordered binary64 operations, independent of HLS.
        for row in range(n):
            acc = 0.0
            for block in range(k):
                dot = sum(int(w)*int(a) for w,a in zip(values[row,block],qx[block]))
                acc += (float(scales[row,block])*float(sx[block])) * float(dot)
            expected.append(acc)
        reference = np.asarray(expected,dtype=np.float64)
        fp32 = weights @ x
        relative = float(np.linalg.norm(reference-fp32)/np.linalg.norm(fp32))
        if not np.isfinite(reference).all() or relative>0.05:
            raise ValueError(f'{name}: Q8 reference exceeds 5% relative L2: {relative}')
        chunk = 4 if k==8 else 8
        matrices.append(dict(tensor=tensor,rows=n,columns=columns,padded_columns=k*32,
                             relative_l2_q8_vs_fp32=relative,
                             tensor_sha256=hashlib.sha256(model.raw(tensor)).hexdigest(),
                             input_fp32_bits=[struct.pack('>f',float(v)).hex() for v in x],
                             fp32_reference_bits=[struct.pack('>f',float(v)).hex() for v in fp32]))
        for first in range(0,n,32):
            rows = min(32,n-first)
            if rows*chunk%32 or rows*(k%chunk)%32:
                raise ValueError('Tile cannot be transported without explicit row padding')
            ident=len(jobs)+1
            stem=f'job-{ident:03d}'
            payload=packed[first*k*34:(first+rows)*k*34]
            files={stem+'-weights.hex': ''.join(f'{byte:02x}\n' for byte in payload),
                   stem+'-activations.hex': ''.join(v.tobytes()[::-1].hex()+'\n' for v in qx),
                   stem+'-scales.hex': ''.join(bits(v)+'\n' for v in sx),
                   stem+'-reference.hex': ''.join(bits(v)+'\n' for v in reference[first:first+rows])}
            for filename,text in files.items():
                (output/filename).write_text(text,encoding='ascii')
            # Independently decode transport bytes and compare every signed lane.
            decoded=np.frombuffer(bytes.fromhex(files[stem+'-weights.hex']),dtype=np.uint8).reshape(rows,k,34)
            if not np.array_equal(decoded[:,:,2:].view(np.int8),values[first:first+rows]):
                raise ValueError('Signed lane serialization mismatch')
            expected_bits=[bits(v) for v in reference[first:first+rows]]
            jobs.append(dict(id=ident,tensor=tensor,first_row=first,rows=rows,k_blocks=k,
                             tile_k=chunk,reference_fp64_bits=expected_bits))
    (output/'jobs.txt').write_text('NTRGEMV01 '+str(len(jobs))+'\n'+''.join(
        f"{j['id']} {j['rows']} {j['k_blocks']} {j['tile_k']}\n" for j in jobs),encoding='ascii')
    manifest=dict(format='NTRGEMV01',model=MODEL_ID,revision=REVISION,
                  weights_sha256=WEIGHT_SHA256,config_sha256=CONFIG_SHA256,audit=audit,
                  seed=20261003,matrices=matrices,jobs=jobs,
                  scope='Isolated real-weight projections with independent synthetic unit-RMS inputs; no FFN or inference',
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(jobs=len(jobs),rows=sum(j['rows'] for j in jobs),
                         mac_ops=sum(j['rows']*j['k_blocks'] for j in jobs),
                         relative_l2=[m['relative_l2_q8_vs_fp32'] for m in matrices])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    prepare(a.model_dir,a.output_dir)


if __name__=='__main__':main()
