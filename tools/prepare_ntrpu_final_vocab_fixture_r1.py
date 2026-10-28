"""Small final RMSNorm inputs, bound to the complete CPU reference and vocabulary."""
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np
from prepare_ntrpu_llama32_forward_reference_r1 import normalized
from prepare_ntrpu_ffn_memory_fixture import quantize
from prepare_ntrpu_llama32_vocab_fixture_r1 import MODEL_SHA, REFERENCE_SHA, sha
from check_ntrpu_vocab_memory_r2 import MANIFEST_SHA

FORWARD_REVIEW_SHA='efdf3b0716fe37808dbe12670ed6745051aef47ce4d52530a073825687e385af'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','reference','forward-review','vocab-fixture','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    assert sha(a.model)==MODEL_SHA
    assert sha(a.reference/'manifest.json')==REFERENCE_SHA
    assert sha(a.forward_review)==FORWARD_REVIEW_SHA
    assert json.loads(a.forward_review.read_text())['status']=='PASS_LLAMA32_FULL_FORWARD_NATIVE_R1'
    assert sha(a.vocab_fixture/'manifest.json')==MANIFEST_SHA
    reference=json.loads((a.reference/'manifest.json').read_text())
    vocab=json.loads((a.vocab_fixture/'manifest.json').read_text())
    tensor=next(t for t in reference['tensor_sources'] if t['name']=='output_norm.weight')
    assert tensor['bytes']==8192 and tensor['shape_gguf']==[2048]
    with a.model.open('rb') as f:f.seek(tensor['offset']);weight=f.read(tensor['bytes'])
    assert hashlib.sha256(weight).hexdigest()==tensor['sha256']
    residual=(a.reference/'layer-15-output-fp32.bin').read_bytes()
    assert len(residual)==8192 and hashlib.sha256(residual).hexdigest()==reference['files']['layer-15-output-fp32.bin']
    epsilon=struct.unpack('>f',bytes.fromhex(reference['epsilon_bits']))[0]
    assert reference['epsilon_bits']=='3727c5ac'
    norm=normalized(np.frombuffer(residual,dtype='<f4'),np.frombuffer(weight,dtype='<f4'),epsilon)
    assert norm.tobytes()==(a.reference/'output-norm-fp32.bin').read_bytes()
    abi=quantize(norm)[2]
    assert abi==(a.reference/'output-abi.bin').read_bytes()==(a.vocab_fixture/'output-abi.bin').read_bytes()
    assert hashlib.sha256(abi).hexdigest()==reference['files']['output-abi.bin']==vocab['files']['output-abi.bin']
    a.output.mkdir(parents=True);files={}
    def write(name,raw):
        with (a.output/name).open('xb') as f:f.write(raw)
        files[name]=hashlib.sha256(raw).hexdigest()
    def hexfile(name,raw,width):
        write(name,''.join(raw[i:i+width][::-1].hex()+'\n' for i in range(0,len(raw),width)).encode('ascii'))
        assert b''.join(bytes.fromhex(line)[::-1] for line in (a.output/name).read_text().splitlines())==raw
    write('residual-fp32.bin',residual);write('norm-weight-fp32.bin',weight)
    write('norm-fp32.bin',norm.tobytes());write('output-abi.bin',abi)
    hexfile('source32.hex',residual+weight,4);hexfile('abi64.hex',abi,8)
    manifest=dict(status='PREPARED_FINAL_VOCAB_NORM_FIXTURE_R1',model_sha256=MODEL_SHA,
        reference_manifest_sha256=REFERENCE_SHA,forward_review_sha256=FORWARD_REVIEW_SHA,
        vocab_manifest_sha256=MANIFEST_SHA,source_tensor=tensor,files=files,
        epsilon_bits=reference['epsilon_bits'],input_token=0,input_position=0,input_layer=15,
        source_is_cpu_reference=True,live_normalization_executed=False,whole_model_rtl_qualified=False,
        generator_sha256=sha(Path(__file__)),scope='Final residual comes from CPU cross-check; real RTL must compute normalization, commit ABI1, and read it for all 128256 logits.')
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps(dict(status=manifest['status'],files=len(files),manifest_sha256=sha(a.output/'manifest.json'))))

if __name__=='__main__':main()
