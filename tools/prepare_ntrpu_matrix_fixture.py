"""Assemble complete immutable matrices for the AXI matrix sequencer bank."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import struct

from check_ntrpu_gemv_memory import load_memory_fixture
from ntrpu_matrix_layout import pack_matrix
from prepare_ntrpu_gemv_packed_fixture import synthetic

BASE_MANIFEST='ae519d84dd530d6e457bc44ae25f956e63cd2e62671c4fec808bc008ee2e4f81'


def prepare(base,output,profile):
    if output.exists():raise FileExistsError(output)
    jobs=[]
    if profile=='brandon':
        if hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest()!=BASE_MANIFEST:
            raise ValueError('Expected reviewed activation/weight fixture')
        m=load_memory_fixture(base)
        for matrix in m['matrices']:
            parts=[j for j in m['jobs'] if j['tensor']==matrix['tensor']]
            if [j['first_row'] for j in parts]!=list(range(0,matrix['rows'],32)):
                raise ValueError('Incomplete original matrix')
            stems=[base/f"job-{j['id']:03d}" for j in parts]
            dense=b''.join(bytes.fromhex(Path(str(p)+'-weights.hex').read_text()) for p in stems)
            activations=[bytes.fromhex(Path(str(p)+'-activation-image.hex').read_text()) for p in stems]
            if any(v!=activations[0] for v in activations):raise ValueError('Inputs differ between row groups')
            refs=[v for j in parts for v in j['reference_fp64_bits']]
            jobs.append((matrix['tensor'],matrix['rows'],parts[0]['k_blocks'],parts[0]['tile_k'],dense,activations[0],refs))
        provenance={k:m[k] for k in ['model','revision','weights_sha256','config_sha256','matrices']}
        provenance['base_fixture_manifest_sha256']=BASE_MANIFEST
        scope='Three complete Brandon projections with independent synthetic inputs; no chained FFN or inference'
    else:
        for pattern,(rows,blocks) in enumerate([(8192,64),(2048,256)],7):
            dense,qx,sx,refs=synthetic(rows,blocks,pattern)
            act=b'\xa5'*4032+b''.join(a+struct.pack('<d',s)+bytes(24) for a,s in zip(qx,sx))
            jobs.append((f'synthetic/{rows}x{blocks*32}',rows,blocks,32,dense,act,refs))
        provenance=dict(model=None,revision=None,matrices=[])
        scope='Complete synthetic Llama-shaped up/gate8192x2048 and down2048x8192 projections; no Meta weights or inference'
    output.mkdir(parents=True,exist_ok=False)
    descriptions=[]
    for ident,(tensor,rows,blocks,tile_k,dense,act,refs) in enumerate(jobs,1):
        image,tiles=pack_matrix(dense,rows,blocks,tile_k)
        stem=f'matrix-{ident:03d}'
        for name,content in [('dense',dense),('image',image),('activation',act)]:
            (output/(stem+'-'+name+'.hex')).write_text(''.join(f'{v:02x}\n' for v in content),encoding='ascii')
        (output/(stem+'-reference.hex')).write_text('\n'.join(refs)+'\n',encoding='ascii')
        descriptions.append(dict(id=ident,tensor=tensor,rows=rows,k_blocks=blocks,tile_k=tile_k,
                                 image_bytes=len(image),reference_fp64_bits=refs,
                                 dense_sha256=hashlib.sha256(dense).hexdigest(),image_sha256=hashlib.sha256(image).hexdigest(),
                                 tiles=[asdict(t) for t in tiles]))
    (output/'jobs.txt').write_text('NTRMATRIX01 '+str(len(jobs))+'\n'+''.join(
        f"{j['id']} {j['rows']} {j['k_blocks']} {j['tile_k']} {j['image_bytes']}\n" for j in descriptions),encoding='ascii')
    manifest=dict(provenance,format='NTRMATRIX01',layout=2,profile=profile,jobs=descriptions,scope=scope,
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(jobs=len(jobs),rows=sum(j[1] for j in jobs),manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-fixture',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=['brandon','llama-shapes'],default='brandon')
    a=p.parse_args()
    if a.profile=='brandon' and a.memory_fixture is None:p.error('--memory-fixture is required for Brandon')
    prepare(a.memory_fixture,a.output,a.profile)
