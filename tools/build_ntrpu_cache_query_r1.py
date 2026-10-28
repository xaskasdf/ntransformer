"""Add an independent, read-only per-layer query without altering frozen R1 RTL."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {
    'fpga/rtl/ntrpu_attention_root_guard_r1.sv': '84074fb411731420d25cf71468c668f82b2061d74304b52e280a2ee563c7f219',
    'fpga/rtl/ntrpu_attention_root_r1.sv': '1bf181ccc3dc7e8e707055ddbfa86c822a2564799cd99aa38c78621018abe065',
    'fpga/rtl/ntrpu_rope_attention_r1.sv': '70f84838dcc9778d877c9cedffbc8604128f640b353fd52365fdbdb3742537b1',
}


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def render():
    parents = {}
    for name, sha in PARENTS.items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha, name
        parents[name] = raw.decode()
    result = {}
    guard = parents['fpga/rtl/ntrpu_attention_root_guard_r1.sv']
    guard = replace_once(guard, 'module ntrpu_attention_root_guard_r1', 'module ntrpu_attention_root_guard_query_r1')
    guard = replace_once(guard, '    output logic maintenance_busy, metadata_fault',
        '    input wire logic [31:0] query_layer,\n'
        '    output wire logic query_valid,\n'
        '    output wire logic [31:0] query_committed_tokens,\n'
        '    output logic maintenance_busy, metadata_fault')
    guard = replace_once(guard, '    assign identity_ok=',
        '    // Observation only: this port grants no ownership or admission.\n'
        '    // Prefix publication still occurs solely on the owned engine ACK.\n'
        '    assign query_valid=trusted_cache_valid && query_layer<16;\n'
        "    assign query_committed_tokens=query_valid ? {14'd0,prefixes[query_layer[3:0]]}:32'd0;\n"
        '    assign identity_ok=')
    result['fpga/rtl/ntrpu_attention_root_guard_query_r1.sv'] = guard
    root = parents['fpga/rtl/ntrpu_attention_root_r1.sv']
    root = replace_once(root, 'module ntrpu_attention_root_r1', 'module ntrpu_attention_root_query_r1')
    root = replace_once(root, '    output logic [31:0] committed_tokens\n',
        '    output logic [31:0] committed_tokens,\n'
        '    input wire logic [31:0] query_layer,\n'
        '    output wire logic query_valid,\n'
        '    output wire logic [31:0] query_committed_tokens\n')
    root = replace_once(root, '    ntrpu_attention_root_guard_r1 guard (',
        '    ntrpu_attention_root_guard_query_r1 guard (\n'
        '        .query_layer(query_layer),\n'
        '        .query_valid(query_valid),\n'
        '        .query_committed_tokens(query_committed_tokens),')
    result['fpga/rtl/ntrpu_attention_root_query_r1.sv'] = root
    top = parents['fpga/rtl/ntrpu_rope_attention_r1.sv']
    top = replace_once(top, 'module ntrpu_rope_attention_r1', 'module ntrpu_rope_attention_query_r1')
    top = replace_once(top, '    output wire [31:0] cache_committed_tokens,',
        '    input wire [31:0] cache_query_layer,\n'
        '    output wire cache_query_valid,\n'
        '    output wire [31:0] cache_query_committed_tokens,\n'
        '    output wire [31:0] cache_committed_tokens,')
    top = replace_once(top, '    ntrpu_attention_root_r1 #', '    ntrpu_attention_root_query_r1 #')
    top = replace_once(top, '        .committed_tokens(cache_committed_tokens)',
        '        .committed_tokens(cache_committed_tokens),\n'
        '        .query_layer(cache_query_layer),\n'
        '        .query_valid(cache_query_valid),\n'
        '        .query_committed_tokens(cache_query_committed_tokens)')
    result['fpga/rtl/ntrpu_rope_attention_query_r1.sv'] = top
    return result


if __name__ == '__main__':
    hashes = {}
    for name, text in render().items():
        path = ROOT / name
        if path.exists():
            assert path.read_text() == text, 'Refuse to alter existing revision: ' + name
        else:
            with path.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(hashes))
