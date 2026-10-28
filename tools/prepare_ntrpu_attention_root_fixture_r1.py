"""Bind the 32-head root vectors to both independently reviewed fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

AXI_SHA='4e80daca26e268eb84f07b1a7f749ba509454ec1ab3457467540d96432193711'
OUTPUT_SHA='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'


def build(axi, attention_output):
    raw=(axi/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==AXI_SHA
    am=json.loads(raw)
    raw=(attention_output/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==OUTPUT_SHA
    om=json.loads(raw);assert am['model_sha256']==om['model_sha256']
    for folder,manifest in [(axi,am),(attention_output,om)]:
        for name,sha in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
    query=[int(n,16) for n in (axi/'query.hex').read_text().split()]
    expected=[int(n,16) for n in (axi/'expected.hex').read_text().split()]
    assert len(query)==len(expected)==9152
    concatenated=[]
    for sample in range(4):
        row=om['samples'][sample]
        assert (row['position'],row['heads'],row['dim'],row['first_attention_case'],row['last_attention_case'])==(sample,32,2048,sample*32+1,sample*32+32)
        for head in range(32):
            job=am['jobs'][sample*32+head]
            assert (job['image'],job['position'],job['layer'],job['head'])==(1,sample,0,head)
        data=(attention_output/f'sample-{sample+1:03d}-attention-fp32.bin').read_bytes()
        bits=list(struct.unpack('<2048I',data))
        assert bits==expected[sample*2048:(sample+1)*2048]
        concatenated += bits
    files={name:(axi/name).read_bytes() for name in ('query.hex','expected.hex','kv_beats.hex','commands.hex','images.hex')}
    files['root-expected.hex']=(''.join(f'{n:08x}\n' for n in concatenated)).encode()
    return dict(format='NTRPU_ATTENTION_ROOT_FIXTURE_R1',model_sha256=am['model_sha256'],
        attention_axi_manifest_sha256=AXI_SHA,attention_output_manifest_sha256=OUTPUT_SHA,
        samples=4,heads=32,head_dim=64,total_reference_words=8192,
        root_order='sample, head, lane; root index = head*64+lane',
        files={n:hashlib.sha256(data).hexdigest() for n,data in files.items()}),files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--axi',type=Path,required=True)
    p.add_argument('--attention-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest,files=build(a.axi,a.attention_output);assert not a.output.exists()
    a.output.mkdir()
    for n,data in files.items():(a.output/n).write_bytes(data)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
