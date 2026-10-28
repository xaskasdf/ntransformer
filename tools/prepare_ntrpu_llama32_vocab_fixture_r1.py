"""Full tied vocabulary fixture from the pinned GGUF and cross-checked CPU forward.

All 128256 rows are retained. Per-segment hex images have a simulation-only
4032-byte prefix; DMA payloads concatenate without those prefixes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from ntrpu_matrix_layout import pack_matrix

MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
REFERENCE_SHA = 'b64a48681d404dfd9255bd527bbf3c0c6000f3e15c92f6220970d9a40c18be5a'
LOGITS_SHA = '6d64b6ac5ae3696db987551eb5a59206c87319ac60cade7bc4372e26f071939e'
WEIGHT_BASE, ACTIVATION_BASE, OUTPUT_BASE = 0x100000000, 0x8000000000, 0xa000000000
ROWS, BLOCKS, STRIDE, PREFIX = 128256, 64, 17825792, 4032


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--forward-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest_path = args.reference/'manifest.json'
    assert sha(manifest_path) == REFERENCE_SHA
    reference = json.loads(manifest_path.read_text())
    forward = json.loads(args.forward_review.read_text())
    assert forward['status'] == 'PASS_LLAMA32_FULL_FORWARD_NATIVE_R1'
    assert forward['reference_manifest_sha256'] == REFERENCE_SHA
    assert forward['outputs']['logits-fp32.bin'] == reference['files']['logits-fp32.bin'] == LOGITS_SHA
    assert forward['amd_continuation_from_native_outputs'] is True
    assert sha(args.model) == reference['model_sha256'] == forward['model_sha256'] == MODEL_SHA
    tensor = next(t for t in reference['tensor_sources'] if t['name'] == 'token_embd.weight')
    assert tensor['shape_gguf'] == [2048, ROWS] and tensor['type_name'] == 'Q8_0'
    assert tensor['bytes'] == ROWS*BLOCKS*34 == 279085056
    args.output.mkdir(parents=True)
    files, segments, tensor_digest = {}, [], hashlib.sha256()
    start_time = time.monotonic()

    def write(name, data):
        with (args.output/name).open('xb') as stream: stream.write(data)
        files[name] = hashlib.sha256(data).hexdigest()

    def hex_image(name, data, width):
        assert len(data) % width == 0
        with (args.output/name).open('x', encoding='ascii', newline='\n') as stream:
            for pos in range(0, len(data), width):
                stream.write(data[pos:pos+width][::-1].hex()+'\n')
        files[name] = sha(args.output/name)

    with args.model.open('rb') as stream:
        for index, row in enumerate(reference['output_chunks']):
            first = index*8192
            rows = min(8192, ROWS-first)
            assert row['index'] == index and row['first_row'] == first and row['rows'] == rows
            assert row['gguf_offset'] == tensor['offset']+first*BLOCKS*34
            stream.seek(row['gguf_offset'])
            raw = stream.read(rows*BLOCKS*34)
            assert hashlib.sha256(raw).hexdigest() == row['weight_sha256']
            tensor_digest.update(raw)
            image, tiles = pack_matrix(raw, rows, BLOCKS, 32,
                region_base=WEIGHT_BASE+index*STRIDE, region_bytes=PREFIX+rows*BLOCKS*34,
                first_offset=PREFIX)
            compact = image[PREFIX:]
            # Independent inverse expressed as tensor axes, not the packer's loop.
            unpacked = np.frombuffer(compact, np.uint8).reshape(rows//32, 2, 32, 32, 34)
            recovered = unpacked.transpose(0, 3, 1, 2, 4).reshape(rows, BLOCKS, 34).tobytes()
            assert recovered == raw, ('Compact layout changed original block bytes', index)
            stem = 'segment-%02d'%index
            write(stem+'-weights.bin', compact)
            hex_image(stem+'-weights.hex', image, 64)
            # Check the emitted transport image independently of its in-memory input.
            with (args.output/(stem+'-weights.hex')).open() as source:
                decoded = b''.join(bytes.fromhex(line.strip())[::-1] for line in source)
            assert decoded == image and decoded[:PREFIX] == bytes([0xa5])*PREFIX
            segment = dict(index=index, first_row=first, rows=rows, k_blocks=BLOCKS,
                weight_address=WEIGHT_BASE+PREFIX+index*STRIDE,
                output_address=OUTPUT_BASE+PREFIX+first*4,
                gguf_offset=row['gguf_offset'], dense_sha256=row['weight_sha256'],
                dma_payload_bytes=len(compact), simulation_prefix_bytes=PREFIX,
                simulation_image_words=len(image)//64, output_bytes=rows*4,
                group_count=rows//32, tiles=[asdict(t) for t in tiles],
                inverse_recovers_every_original_byte=True)
            segments.append(segment)
            print(json.dumps({k:v for k,v in segment.items() if k != 'tiles'}), flush=True)
    assert len(segments) == 16 and segments[-1]['rows'] == 5376
    assert tensor_digest.hexdigest() == tensor['sha256']
    for name in ('output-abi.bin', 'logits-fp64.bin', 'logits-fp32.bin'):
        raw = (args.reference/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == reference['files'][name]
        write(name, raw)
        hex_image(name.removesuffix('.bin')+'.hex', raw, 64 if name == 'output-abi.bin' else 8 if 'fp64' in name else 4)
    abi = (args.output/'output-abi.bin').read_bytes()
    assert len(abi) == 4096
    for k in range(BLOCKS):
        assert abi[k*64+40:k*64+64] == b'\0'*24
        assert np.isfinite(struct.unpack_from('<d', abi, k*64+32)[0])
    logits = np.fromfile(args.output/'logits-fp32.bin', dtype='<f4')
    assert len(logits) == ROWS and np.isfinite(logits).all() and int(np.argmax(logits)) == 475
    result = dict(format='NTRVOCAB01', status='FULL_VOCAB_FIXTURE_BYTES_VERIFIED',
        model_sha256=MODEL_SHA, reference_manifest_sha256=REFERENCE_SHA,
        forward_review_sha256=sha(args.forward_review), source_tensor=tensor,
        input_token=0, input_position=0, rows=ROWS, k_blocks=BLOCKS, tile_k=32,
        activation_address=ACTIVATION_BASE+PREFIX, activation_base=ACTIVATION_BASE,
        activation_region_bytes=PREFIX+4096, weight_base=WEIGHT_BASE,
        weight_address=WEIGHT_BASE+PREFIX, weight_region_bytes=PREFIX+279085056,
        output_base=OUTPUT_BASE, output_address=OUTPUT_BASE+PREFIX,
        output_region_bytes=PREFIX+513024, segments=segments, files=files,
        logits_sha256=LOGITS_SHA, argmax_token=475, raw_weight_bytes_verified=279085056,
        simulated_rtl=False, live_normalization=False, ready_for_inference=False,
        generator_sha256=sha(Path(__file__)), matrix_layout_sha256=sha(Path(__file__).with_name('ntrpu_matrix_layout.py')),
        wall_seconds=time.monotonic()-start_time,
        scope='Complete vocabulary for a cross-checked single-token CPU final-normalization ABI1 input. '
              'No RTL result, live normalization, multi-token context or physical memory throughput.')
    with (args.output/'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status=result['status'], rows=ROWS, segments=16,
        weights_bytes=result['raw_weight_bytes_verified'], files=len(files), manifest_sha256=sha(args.output/'manifest.json'))), flush=True)


if __name__ == '__main__':
    main()
