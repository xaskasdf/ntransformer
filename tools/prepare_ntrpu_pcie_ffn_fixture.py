"""Package a qualified Brandon FFN sample for PCIe DMA, without running RTL.

Retains the real-weight source identity and numerical references. Host input
buffers use separate 1-MiB windows: the small synthetic bank's 64-KiB spacing
would overlap with these matrices. This is a simulation fixture, not a driver.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, output, sample=1):
    if output.exists():
        raise FileExistsError(output)
    raw=(source/'manifest.json').read_bytes()
    manifest=json.loads(raw)
    if (manifest['format'],manifest['layout'],manifest['dim'],manifest['hidden'],manifest['tile_k']) != \
            ('NTRFFNMEM01',2,256,720,[4,4,8]):
        raise ValueError('Expected qualified Brandon D256/H720 layout-2 fixture')
    if not 1 <= sample <= manifest['samples']:
        raise ValueError('Sample outside reference')
    for name,expected in manifest['files'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path.read_bytes())!=expected:
            raise ValueError('Source file identity: '+name)
    prefix=f'job-{sample:03d}'
    names=[prefix+'-input.hex','weight-1.hex','weight-2.hex','weight-3.hex']
    names += [prefix+f'-buffer-{b}.hex' for b in [4,5,6]]+[prefix+'-final-fp32.hex']
    images=[]
    for b,name in enumerate(names):
        data=bytes.fromhex((source/name).read_text())
        if b<4:
            if data[:4032]!=b'\xa5'*4032:
                raise ValueError('Expected protected prefix '+name)
            data=data[4032:]
        images.append(data)
    sizes=[512,195840,195840,200192,5760,5760,1472,1024]
    if list(map(len,images))!=sizes:
        raise ValueError('Unexpected packed buffer sizes')
    if not all(len(data)%64==0 for data in images):
        raise ValueError('AXI beat alignment')
    buffers=[];files={}
    for b,data in enumerate(images):
        base=0x100000000+b*0x20000000
        host=0x200000fc0+b*0x100000 if b<4 else 0x400000fc0 if b==7 else None
        buffers.append(dict(id=b,region_base=base,device_address=base+4032,
            registered_extent=4032+len(data),bytes=len(data),host_address=host,
            contents='DMA input' if b<4 else 'expected intermediate' if b<7 else 'expected FP32 output',
            source_file=names[b],data_sha256=sha(data)))
        files[f'buffer-{b}.hex']=''.join(f'{value:02x}\n' for value in data).encode()
    ranges=[(b['host_address'],b['host_address']+b['bytes']) for b in buffers if b['host_address'] is not None]
    if any(a<d and c<b for i,(a,b) in enumerate(ranges) for c,d in ranges[i+1:]):
        raise ValueError('Overlapping host buffers')
    descriptors=[]
    for b in range(4):
        descriptors.append(struct.pack('<IIQQQ',0xad4b0010 if b<3 else 0xad4b0013,
            sizes[b],buffers[b]['host_address'],buffers[b]['device_address'],
            0x100000100+(b+1)*32 if b<3 else 0))
    c2h=struct.pack('<IIQQQ',0xad4b0013,sizes[7],buffers[7]['device_address'],buffers[7]['host_address'],0)
    for name,data in [('h2c-descriptors.hex',b''.join(descriptors)),('c2h-descriptor.hex',c2h)]:
        files[name]=''.join(f'{value:02x}\n' for value in data).encode()
    files['reference-manifest.json']=raw
    result=dict(format='NTRPCIEFFN01',dim=256,hidden=720,layout=2,tile_k=[4,4,8],sample=sample,
        model=manifest['model'],revision=manifest['revision'],weights_sha256=manifest['weights_sha256'],
        config_sha256=manifest['config_sha256'],reference_manifest_sha256=sha(raw),
        buffers=buffers,h2c_descriptor_address=0x100000100,c2h_descriptor_address=0x300000300,
        h2c_bytes=sum(sizes[:4]),stage_bytes=sum(sizes[4:]),c2h_bytes=sizes[7],output_fp32=256,
        dimension_csr=0x02d00100,tile_csr=0x080404,
        files={name:sha(data) for name,data in files.items()},
        scope='Prepared PCIe data/reference only; real Brandon layer0 weights, synthetic unit-RMS input; no new simulation or full inference claim')
    output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():(output/name).write_bytes(data)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--sample',type=int,default=1)
    args=parser.parse_args()
    result=prepare(args.source,args.output,args.sample)
    print(json.dumps({key:result[key] for key in ['format','sample','h2c_bytes','stage_bytes','c2h_bytes','output_fp32','scope']}))
