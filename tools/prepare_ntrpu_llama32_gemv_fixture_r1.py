"""Export 28 real Q8_0 weight slices, preserving all original block bytes.

Layers 0/15; seven projections; first/last 32 rows. Activations are synthetic,
not hidden states from Llama inference. Only NumPy and the audited GGUF index
are required; this tool does not import a model runtime or re-quantize weights.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from check_ntrpu_gemv_fixture import load_fixture
from ntrpu_gemv_layout import pack_weights
from prepare_ntrpu_ffn_memory_fixture import project

MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
PROJECTIONS = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scalar_reference(dense, qx, sx):
    """Separate scalar decoder and integer dot, accumulating in block order."""
    k_blocks = len(sx)
    output = []
    for row in range(32):
        total = 0.0
        for k in range(k_blocks):
            pos = (row*k_blocks+k)*34
            scale = struct.unpack_from('<e', dense, pos)[0]
            weights = struct.unpack_from('<32b', dense, pos+2)
            activation = struct.unpack('<32b', qx[k])
            dot = sum(a*w for a, w in zip(activation, weights))
            total += (float(scale)*sx[k])*float(dot)
        output.append(struct.pack('>d', total).hex())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    if (audit['status'] != 'PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1' or
            audit['sha256'] != MODEL_SHA or digest(args.model) != MODEL_SHA or
            audit['audit_script_sha256'] != digest(Path(__file__).with_name('audit_ntrpu_llama32_gguf_r1.py'))):
        raise ValueError('Pinned model/audit identity')
    tensors = {t['name']: t for t in audit['tensors']}
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    def write(name, text):
        with (args.output/name).open('x', encoding='ascii', newline='\n') as stream:
            stream.write(text)
    with args.model.open('rb') as source:
        for layer in (0, 15):
            for projection in PROJECTIONS:
                name = f'blk.{layer}.{projection}.weight'
                tensor = tensors[name]
                columns, rows = tensor['shape_gguf']
                blocks = columns//32
                source.seek(tensor['offset'])
                entire_tensor = source.read(tensor['bytes'])
                if hashlib.sha256(entire_tensor).hexdigest() != tensor['sha256']:
                    raise ValueError('Tensor payload identity')
                for first in (0, rows-32):
                    ident = len(jobs)+1
                    stem = f'job-{ident:03d}'
                    offset = first*blocks*34
                    dense = entire_tensor[offset:offset+32*blocks*34]
                    # Reproducible signed INT8, deliberately includes -128.
                    qx = [bytes((k*13+lane*7+layer*3+ident) & 255 for lane in range(32)) for k in range(blocks)]
                    sx = [1.0/(k % 5+1) for k in range(blocks)]
                    refs = scalar_reference(dense, qx, sx)
                    vector = project(dense, 32, np.frombuffer(b''.join(qx), np.int8).reshape(blocks, 32), np.asarray(sx))
                    if [struct.pack('>d', float(v)).hex() for v in vector] != refs:
                        raise ValueError('Scalar and NumPy binary64 references differ')
                    packed = pack_weights(dense, 32, blocks, 32, layout=2)
                    for suffix, data in [('weights', dense), ('image', packed.image)]:
                        write(stem+'-'+suffix+'.hex', ''.join(f'{value:02x}\n' for value in data))
                    write(stem+'-activations.hex', ''.join(v[::-1].hex()+'\n' for v in qx))
                    write(stem+'-scales.hex', ''.join(struct.pack('>d', v).hex()+'\n' for v in sx))
                    write(stem+'-reference.hex', '\n'.join(refs)+'\n')
                    write(stem+'-tiles.hex', ''.join(f'{t.address:016x}{t.k_base:08x}{t.k_count:08x}\n' for t in packed.tiles))
                    write(stem+'-image.txt', f'{len(packed.image)} {len(packed.tiles)}\n')
                    jobs.append(dict(id=ident, tensor=name, first_row=first, rows=32, k_blocks=blocks,
                        tile_k=32, layout=2, category='real_llama32_q8_projection_slice',
                        source_tensor_sha256=tensor['sha256'], source_file_offset=tensor['offset']+offset,
                        reference_fp64_bits=refs, scalar_numpy_bit_exact=True,
                        image_bytes=len(packed.image), image_sha256=hashlib.sha256(packed.image).hexdigest(),
                        dense_source_sha256=hashlib.sha256(dense).hexdigest(), tiles=[asdict(t) for t in packed.tiles]))
    write('jobs.txt', 'NTRGEMV02 28\n'+''.join(f"{j['id']} 32 {j['k_blocks']} 32\n" for j in jobs))
    tools = ['prepare_ntrpu_llama32_gemv_fixture_r1.py', 'audit_ntrpu_llama32_gguf_r1.py',
             'ntrpu_gemv_layout.py', 'prepare_ntrpu_ffn_memory_fixture.py', 'check_ntrpu_gemv_fixture.py']
    manifest = dict(format='NTRGEMV02', workload='NTRLLAMA32Q8SLICES01', layout=2,
                    model=audit['repo'], revision=audit['revision'], weights_sha256=MODEL_SHA,
                    config_sha256=audit['config_sha256'], audit_sha256=digest(args.audit),
                    jobs=jobs, matrices=[], region_base=0x100000000, region_bytes=0x200000,
                    scope='28 real Unsloth Q8_0 slices from layers0/15, synthetic INT8/FP64 activations; no full projections, FFN, attention, logits, inference or physical DDR.',
                    numerical_contract='Signed INT8 dot; original FP16 weight scale; synthetic FP64 activation scale; ordered FP64 block accumulation, no FMA.',
                    tools={name:digest(Path(__file__).with_name(name)) for name in tools},
                    files={p.name:digest(p) for p in sorted(args.output.iterdir())})
    write('manifest.json', json.dumps(manifest, indent=2)+'\n')
    # Existing independent byte-position checker decodes image and descriptors,
    # including every Q8 byte, K coverage and all sentinel gaps.
    loaded = load_fixture(args.output)
    if loaded != manifest:
        raise ValueError('Manifest round trip')
    print(json.dumps(dict(status='PASS_LLAMA32_Q8_FIXTURE_CPU_R1', jobs=len(jobs), rows=896,
                          blocks=sum(j['rows']*j['k_blocks'] for j in jobs),
                          manifest_sha256=digest(args.output/'manifest.json'),
                          scalar_numpy_bit_exact=True, independent_byte_layout_checked=True)))


if __name__ == '__main__':
    main()
