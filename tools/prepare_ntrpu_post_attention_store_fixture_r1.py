"""Freeze the real post-O residual/ABI1 references for paired AXI transport."""
import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=(args.parent/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    parent=json.loads(raw)
    content={'parent-manifest.json':raw}
    residual=[];abi=[]
    for sample in range(1,5):
        for kind,width,size,lines in [('residual-fp32',4,8192,residual),('ffn-input-abi1',8,4096,abi)]:
            name=f'sample-{sample:03d}-{kind}.bin';data=(args.parent/name).read_bytes()
            assert len(data)==size and hashlib.sha256(data).hexdigest()==parent['files'][name]
            content[name]=data
            lines.extend(f'{int.from_bytes(data[i:i+width],"little"):0{width*2}x}' for i in range(0,len(data),width))
    content['residual32.hex']=('\n'.join(residual)+'\n').encode()
    content['abi64.hex']=('\n'.join(abi)+'\n').encode()
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in content.items():(args.output/name).write_bytes(data)
    manifest=dict(status='FROZEN_POST_ATTENTION_STORE_R1_FIXTURE',parent_manifest_sha256=PARENT_SHA,
        samples=4,residual_words_per_sample=2048,abi_words_per_sample=512,
        files={name:hashlib.sha256(data).hexdigest() for name,data in content.items()},
        scope='Real layer-0 post-O reference outputs; using these as source RAM tests AXI transport, not arithmetic integration.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),manifest_sha256=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest(),files=len(content))))


if __name__=='__main__':main()
