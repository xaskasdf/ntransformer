"""Audit the pinned Unsloth Q8_0 artifact, without converting its tensors.

Run with: uv run --no-project --with gguf==0.19.0 python tools/audit_ntrpu_llama32_gguf_r1.py ...
The optional comparison only identifies tensor payloads; it does not certify
another file's tokenizer, metadata, or provenance.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import urllib.request

import gguf
import numpy as np

REPO = 'unsloth/Llama-3.2-1B-Instruct-GGUF'
REVISION = 'b69aef112e9f895e6f98d7ae0949f72ff09aa401'
FILENAME = 'Llama-3.2-1B-Instruct-Q8_0.gguf'
SHA256 = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
FILE_BYTES = 1321082528
EXPECTED_META = {
    'general.architecture': 'llama', 'general.file_type': 7,
    'general.quantization_version': 2, 'llama.block_count': 16,
    'llama.context_length': 131072, 'llama.embedding_length': 2048,
    'llama.feed_forward_length': 8192, 'llama.attention.head_count': 32,
    'llama.attention.head_count_kv': 8, 'llama.attention.key_length': 64,
    'llama.attention.value_length': 64, 'llama.vocab_size': 128256,
    'llama.rope.dimension_count': 64, 'llama.rope.freq_base': 500000.0,
}
MATRICES = {
    'attn_q': (2048, 2048), 'attn_k': (2048, 512),
    'attn_v': (2048, 512), 'attn_output': (2048, 2048),
    'ffn_gate': (2048, 8192), 'ffn_up': (2048, 8192),
    'ffn_down': (8192, 2048),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_contract():
    result = {'token_embd.weight': ((2048, 128256), 8),
              'output_norm.weight': ((2048,), 0), 'rope_freqs.weight': ((32,), 0)}
    for layer in range(16):
        for name, shape in MATRICES.items():
            result[f'blk.{layer}.{name}.weight'] = (shape, 8)
        for name in ('attn_norm', 'ffn_norm'):
            result[f'blk.{layer}.{name}.weight'] = ((2048,), 0)
    return result


def inspect(path):
    with path.open('rb') as stream:
        magic, version, tensors, fields = struct.unpack('<4sIQQ', stream.read(24))
    need(magic == b'GGUF' and version == 3, 'Expected little-endian GGUF v3')
    reader = gguf.GGUFReader(path, mode='r')
    metadata = {key: field.contents() for key, field in reader.fields.items()
                if key.startswith(('general.', 'llama.'))}
    for key, expected in EXPECTED_META.items():
        need(metadata.get(key) == expected, 'Metadata mismatch: '+key)
    need(len(reader.fields['tokenizer.ggml.tokens'].data) == 128256, 'Tokenizer vocabulary')
    inventory = []
    expected = tensor_contract()
    names = [tensor.name for tensor in reader.tensors]
    need(len(names) == len(set(names)) == tensors == len(expected) and set(names) == set(expected),
         'Missing, duplicate or unexpected tensors')
    previous_end = int(reader.data_offset)
    q8_blocks = 0
    for tensor in sorted(reader.tensors, key=lambda t: t.data_offset):
        shape, kind = expected[tensor.name]
        need(tuple(tensor.shape.tolist()) == shape and int(tensor.tensor_type) == kind,
             'Tensor shape/type mismatch: '+tensor.name)
        count = int(np.prod(shape))
        size = count*4 if kind == 0 else count//32*34
        start = int(tensor.data_offset)
        need(size == tensor.n_bytes and start % 32 == 0 and start >= previous_end and
             start+size <= path.stat().st_size, 'Tensor range/size/alignment')
        previous_end = start+size
        data = tensor.data
        if kind == 8:
            raw = data.reshape(-1, 34)
            scale = raw[:, :2].copy().view('<f2').reshape(-1)
            need(np.isfinite(scale).all(), 'Nonfinite Q8_0 scale: '+tensor.name)
            q8_blocks += len(scale)
            zero_scales = int(np.count_nonzero(scale == 0))
        else:
            need(np.isfinite(data).all(), 'Nonfinite F32 tensor: '+tensor.name)
            zero_scales = None
        inventory.append(dict(name=tensor.name, shape_gguf=list(shape), type=kind,
                              type_name='Q8_0' if kind == 8 else 'F32',
                              offset=start, bytes=size, sha256=hashlib.sha256(data).hexdigest(),
                              zero_q8_scales=zero_scales))
    need(previous_end == path.stat().st_size, 'Unexpected trailing payload')
    return dict(version=version, metadata=metadata, metadata_fields=fields,
                tensor_count=tensors, data_offset=int(reader.data_offset), tensors=inventory,
                total_elements=sum(int(np.prod(t['shape_gguf'])) for t in inventory),
                tensor_payload_bytes=sum(t['bytes'] for t in inventory),
                tensor_types=dict(Counter(t['type_name'] for t in inventory)),
                finite_q8_blocks=q8_blocks, tokenizer_tokens=128256,
                output_weight_present=False,
                output_reuses_embedding='No output.weight stored; config.json declares tie_word_embeddings=true')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'Output directory already exists')
    need(args.model.stat().st_size == FILE_BYTES and digest(args.model) == SHA256, 'Pinned artifact mismatch')
    url = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        hub_bytes = response.read()
    hub = json.loads(hub_bytes)
    record = next(f for f in hub['siblings'] if f['rfilename'] == FILENAME)
    need(hub['sha'] == REVISION and record['lfs']['sha256'] == SHA256 and
         record['size'] == record['lfs']['size'] == FILE_BYTES, 'Hub provenance mismatch')
    config_path = args.model.parent/'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config_record = next(f for f in hub['siblings'] if f['rfilename'] == 'config.json')
    config_bytes = config_path.read_bytes()
    git_blob = hashlib.sha1(b'blob '+str(len(config_bytes)).encode()+b'\0'+config_bytes).hexdigest()
    need(git_blob == config_record['blobId'], 'Pinned config blob mismatch')
    need(all(config.get(k) == v for k, v in dict(hidden_size=2048, intermediate_size=8192,
         num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
         vocab_size=128256, head_dim=64, tie_word_embeddings=True).items()), 'Config dimensions')
    result = inspect(args.model)
    result.update(status='PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1', repo=REPO, revision=REVISION,
                  file=FILENAME, local_path=str(args.model.resolve()), sha256=SHA256, bytes=FILE_BYTES,
                  hub_api_url=url, hub_api_sha256=hashlib.sha256(hub_bytes).hexdigest(),
                  hub_file=record, config_sha256=digest(config_path),
                  reader_version=importlib.metadata.version('gguf'), numpy_version=np.__version__,
                  audit_script_sha256=digest(__file__),
                  scope='Pinned quantized Unsloth artifact and all tensor payloads. No Meta BF16 equivalence, RTL simulation, inference, or physical qualification.')
    if args.compare:
        other = inspect(args.compare)
        other_tensors = {t['name']: t for t in other['tensors']}
        differences = [t['name'] for t in result['tensors'] if
                       t['sha256'] != other_tensors[t['name']]['sha256']]
        result['comparison'] = dict(path=str(args.compare.resolve()), sha256=digest(args.compare),
                                    bytes=args.compare.stat().st_size, differing_tensor_names=differences,
                                    same_tensor_payloads=not differences,
                                    metadata_and_tokenizer_equivalence_claimed=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'hub-response.json').write_bytes(hub_bytes)
    (args.output/'config.json').write_bytes(config_bytes)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'sha256', 'tensor_count', 'tensor_types',
                     'total_elements', 'tensor_payload_bytes', 'finite_q8_blocks')}))
    if args.compare:
        print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
