"""Read GGUF headers only; never touch weight payloads or start CUDA."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, '/home/ubuntu/ntransformer-h100')
from gguf_scan import read_string, read_value, TYPE_BLOCKS, TYPE_NAMES

root = Path('/home/ubuntu/ntransformer-h100/ds41-audit-20261015')
model_dir = Path('/home/ubuntu/ntransformer-h100/models/deepseek-v41-flash-q8')
groups = collections.defaultdict(lambda: dict(count=0, bytes=0, types=set(), shapes=set()))
shards, tensors, metadata = [], [], {}
for path in sorted(model_dir.glob('*.gguf')):
    with path.open('rb') as f:
        magic, version, nt, nk = struct.unpack('<IIQQ', f.read(24))
        assert magic == 0x46554747 and version == 3
        for _ in range(nk):
            name = read_string(f)
            value = read_value(f, struct.unpack('<I', f.read(4))[0])
            if name.startswith('deepseek41.') and not name.startswith('deepseek41.engram.'):
                metadata[name] = value
        for _ in range(nt):
            name = read_string(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<'+'Q'*nd, f.read(8*nd))
            dtype, offset = struct.unpack('<IQ', f.read(12))
            be, bb = TYPE_BLOCKS[dtype]
            assert dims[0] % be == 0
            size = math.prod(dims) // be * bb
            group = re.sub(r'blk\.\d+\.', 'blk.N.', name)
            g = groups[group]
            g['count'] += 1
            g['bytes'] += size
            g['types'].add(TYPE_NAMES[dtype])
            g['shapes'].add(dims)
            tensors.append(dict(name=name, shape=list(dims), dtype=TYPE_NAMES[dtype], bytes=size,
                shard=path.name, offset=offset))
        header_bytes = f.tell()
        f.seek(0)
        digest = hashlib.sha256(f.read(header_bytes)).hexdigest()
    shards.append(dict(name=path.name, file_bytes=path.stat().st_size,
        header_bytes=header_bytes, header_sha256=digest, tensors=nt))

categories = collections.defaultdict(int)
for t in tensors:
    n = t['name']
    if re.search(r'ffn_(gate|up|down)_exps\.weight$', n): k='routed_experts'
    elif n.endswith('engram_embd.weight'): k='engram_tables'
    elif 'shexp.weight' in n: k='shared_experts'
    elif n == 'output.weight': k='lm_head'
    elif n == 'token_embd.weight': k='embedding'
    elif '.attn_' in n or '.indexer.' in n: k='attention_and_indexer'
    else: k='other'
    categories[k] += t['bytes']
for g in groups.values():
    g['types'] = sorted(g['types'])
    g['shapes'] = sorted(g['shapes'])

layers = metadata['deepseek41.block_count']
used = metadata['deepseek41.expert_used_count']
expert_bytes = sum(t['bytes']//t['shape'][2] for t in tensors
    if t['name'].startswith('blk.0.') and re.search(r'ffn_(gate|up|down)_exps\.weight$',t['name']))
result = dict(note='GGUF header hashes only, not weight hashes. No payload reads.',
    metadata=metadata, shards=shards, tensor_count=len(tensors),
    tensor_payload_bytes=sum(t['bytes'] for t in tensors), categories_bytes=dict(categories),
    groups=dict(sorted(groups.items())), tensors=tensors,
    derived=dict(expert_triplet_bytes=expert_bytes, all_miss_expert_bytes_per_token=layers*used*expert_bytes,
        attention_boundary_bytes_per_layer=4*(metadata['deepseek41.embedding_length']*2 +
          metadata['deepseek41.attention.q_lora_rank'] + metadata['deepseek41.attention.key_length'] +
          2*metadata['deepseek41.attention.head_count']*metadata['deepseek41.attention.key_length'])))
(root/'layout.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['tensor_count','tensor_payload_bytes','categories_bytes','derived']},indent=2))
print('GiB:', {k:round(v/2**30,4) for k,v in categories.items()})
