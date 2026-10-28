"""Four real layer-0 attention outputs through full O, residual and FFN norm.

Uses archived native AMD attention words. This prepares reference data, not
an integrated RTL result or full-model inference.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import struct
import tarfile
import numpy as np
from ntrpu_matrix_layout import pack_matrix
from prepare_ntrpu_ffn_memory_fixture import quantize, project
from prepare_ntrpu_llama32_norm_q8_fixture_r1 import reference as norm_q8

MODEL_SHA = '3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1'
NATIVE_SHA = 'c0e02eae126c68c8649e2c4ebe49b0ddab3a703f0ad00732bd9ddc8b3966e089'
NATIVE_REVIEW_SHA = '6cf4dbcfde78538d15f31e0af3aae3646c2beca3ae3b07442313d500477ed6e5'
TOKENS = (0, 42, 128000, 128255)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--attention',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--native-review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); assert not args.output.exists()
    audit = json.loads(args.audit.read_text()); model = Path(audit['local_path'])
    assert audit['status'] == 'PASS_PINNED_LLAMA32_Q8_ARTIFACT_AUDIT_R1'
    assert digest(model) == audit['sha256'] == MODEL_SHA
    assert digest(args.native_review) == NATIVE_REVIEW_SHA
    review = json.loads(args.native_review.read_text())
    assert review['status'] == 'REVIEWED_ATTENTION_AXI_R1_NATIVE_AMD_AND_LAYOUT'
    assert review['archive_sha256'] == NATIVE_SHA and review['numerical']['decimal90_max_output_ulp'] == 0
    cert = json.loads((args.native/'VERIFIED.json').read_text())
    archive = args.native/Path(cert['archive']).name
    assert digest(archive) == cert['sha256'] == NATIVE_SHA
    with tarfile.open(archive) as tar:
        names = [m.name for m in tar.getmembers() if m.name.endswith('/positive.log')]
        assert len(names) == 1
        transcript = tar.extractfile(names[0]).read()
    attention_manifest = (args.attention/'manifest.json').read_bytes()
    assert hashlib.sha256(attention_manifest).hexdigest() == '4e80daca26e268eb84f07b1a7f749ba509454ec1ab3457467540d96432193711'
    am = json.loads(attention_manifest)
    expected_raw = (args.attention/'expected.hex').read_bytes()
    assert hashlib.sha256(expected_raw).hexdigest() == am['files']['expected.hex']
    expected = [int(line,16) for line in expected_raw.decode().splitlines()]
    words = {}
    for line in transcript.decode().splitlines():
        match = re.fullmatch(r'NTRPU_ATTENTION_WORD case=(\d+) index=(\d+) bits=([0-9a-f]{8})',line)
        if match:
            case,lane = int(match[1]),int(match[2]); assert (case,lane) not in words
            words[case,lane] = int(match[3],16)
    assert len(words) == 9152
    assert all(words[case,lane] == expected[(case-1)*64+lane] for case in range(1,129) for lane in range(64))
    tensors = {row['name']:row for row in audit['tensors']}
    sources = {}; payload = {}
    with model.open('rb') as stream:
        for name in ('blk.0.attn_output.weight','blk.0.ffn_norm.weight'):
            tensor = tensors[name]; stream.seek(tensor['offset']); data = stream.read(tensor['bytes'])
            assert hashlib.sha256(data).hexdigest() == tensor['sha256']; sources[name] = tensor
            payload[name] = data
        embedding = tensors['token_embd.weight']; assert embedding['shape_gguf'] == [2048,128256]
        for token in TOKENS:
            stream.seek(embedding['offset']+token*2176); payload[token] = stream.read(2176)
    weight = payload['blk.0.attn_output.weight']; norm = payload['blk.0.ffn_norm.weight']
    assert sources['blk.0.attn_output.weight']['shape_gguf'] == [2048,2048]
    assert sources['blk.0.attn_output.weight']['type_name'] == 'Q8_0'
    assert sources['blk.0.ffn_norm.weight']['shape_gguf'] == [2048]
    assert sources['blk.0.ffn_norm.weight']['type_name'] == 'F32'
    packed,tiles = pack_matrix(weight,2048,64,32)
    assert len(tiles) == 128
    args.output.mkdir(parents=True)
    def write(name,raw):
        with (args.output/name).open('xb') as stream: stream.write(raw)
    write('output-weight-dense.bin',weight); write('output-weight-image.bin',packed)
    write('ffn-norm-weight.bin',norm)
    write('attention-native-positive.log',transcript)
    write('attention-native-review.json',args.native_review.read_bytes())
    samples = []
    for position,token in enumerate(TOKENS):
        ident = position+1; stem = f'sample-{ident:03d}'
        values = []
        for head in range(32):
            case = position*32+head+1; spec = am['jobs'][case-1]
            assert (spec['head'],spec['position'],spec['layer']) == (head,position,0)
            values.extend(words[case,lane] for lane in range(64))
        attention = np.asarray(values,dtype='<u4').view('<f4')
        q,scale,abi = quantize(attention)
        output64 = project(weight,2048,q,scale); output32 = output64.astype('<f4')
        embedding_values = []
        for offset in range(0,2176,34):
            factor = struct.unpack_from('<e',payload[token],offset)[0]
            embedding_values.extend(factor*x for x in struct.unpack_from('<32b',payload[token],offset+2))
        embedding32 = np.asarray(embedding_values,dtype='<f4')
        residual = (embedding32.astype(np.float64)+output32.astype(np.float64)).astype('<f4')
        status,ffn_abi = norm_q8(list(residual.view('<u4')),list(struct.unpack('<2048I',norm)),2048,0x3727c5ac)
        assert status == 0 and len(abi) == len(ffn_abi) == 4096
        for name,data in [('attention-fp32',attention.tobytes()),('attention-abi1',abi),
                          ('output-fp64',output64.astype('<f8').tobytes()),('output-fp32',output32.tobytes()),
                          ('embedding-q8',payload[token]),('embedding-fp32',embedding32.tobytes()),
                          ('residual-fp32',residual.tobytes()),('ffn-input-abi1',ffn_abi)]:
            write(stem+'-'+name+'.bin',data)
        assert all(np.isfinite(a).all() for a in (attention,output64,output32,embedding32,residual))
        samples.append(dict(id=ident,token=token,position=position,layer=0,heads=32,dim=2048,
                            first_attention_case=position*32+1,last_attention_case=(position+1)*32,
                            embedding_row_sha256=hashlib.sha256(payload[token]).hexdigest(),
                            output_range=[float(output32.min()),float(output32.max())]))
    manifest = dict(format='NTRLLAMA_ATTENTION_OUTPUT_R1',model_sha256=MODEL_SHA,audit_sha256=digest(args.audit),
                    attention_fixture_manifest_sha256=hashlib.sha256(attention_manifest).hexdigest(),
                    native_archive_sha256=NATIVE_SHA,native_review_sha256=NATIVE_REVIEW_SHA,
                    native_transcript_sha256=hashlib.sha256(transcript).hexdigest(),
                    sources=sources,embedding_tensor=embedding,samples=samples,epsilon=0x3727c5ac,
                    output_matrix=dict(rows=2048,columns=2048,k_blocks=64,tile_k=32,tiles=[asdict(t) for t in tiles],
                                       first_offset=4032,image_bytes=len(packed),weight_bytes=len(weight)),
                    quantization='32 FP32 values per ABI1 record; maxabs/127 FP64 scale, nearest integer ties away from zero; 24 zero reserved bytes.',
                    arithmetic='Ordered FP64 Q8 block sums without FMA; O narrowed to FP32 before FP32 residual; FFN RMSNorm/Q8 uses epsilon 0x3727c5ac.',
                    scope='CPU reference across all 32 native-attention heads, full layer0 O, residual and FFN normalization for four tokens. No integrated RTL or full inference.',
                    tools={name:digest(Path(__file__).with_name(name)) for name in
                           ('prepare_ntrpu_llama32_attention_output_r1.py','prepare_ntrpu_ffn_memory_fixture.py',
                            'prepare_ntrpu_llama32_norm_q8_fixture_r1.py','ntrpu_matrix_layout.py')},
                    files={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(status='PREPARED_ATTENTION_OUTPUT_REFERENCE_R1',samples=4,heads=128,
                         output_values=8192,tiles_per_projection=128,weight_bytes=len(weight),
                         manifest_sha256=digest(args.output/'manifest.json'))))


if __name__ == '__main__': main()
