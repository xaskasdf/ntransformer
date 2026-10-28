"""Decode matrix byte images and recompute every ordered Q8 result independently."""
import hashlib
import json
import math
from pathlib import Path
import struct


def load_matrix_fixture(path):
    m=json.loads((path/'manifest.json').read_text())
    if m['format']!='NTRMATRIX01' or m['layout']!=2:raise ValueError('Matrix format')
    for name,digest in m['files'].items():
        if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('File identity '+name)
    if [j['id'] for j in m['jobs']]!=list(range(1,len(m['jobs'])+1)):raise ValueError('Job identities')
    descriptor_lines=['NTRMATRIX01 '+str(len(m['jobs']))]
    for j in m['jobs']:
        n,k,t=j['rows'],j['k_blocks'],j['tile_k'];stem=f"matrix-{j['id']:03d}"
        if not(1<=n<=8192 and 1<=k<=256 and 1<=t<=32):raise ValueError('Shape')
        dense=bytes.fromhex((path/(stem+'-dense.hex')).read_text())
        image=bytes.fromhex((path/(stem+'-image.hex')).read_text())
        act=bytes.fromhex((path/(stem+'-activation.hex')).read_text())
        refs=(path/(stem+'-reference.hex')).read_text().splitlines()
        if len(dense)!=n*k*34 or len(image)!=4032+len(dense) or len(act)!=4032+k*64:raise ValueError('Image size')
        if hashlib.sha256(dense).hexdigest()!=j['dense_sha256'] or hashlib.sha256(image).hexdigest()!=j['image_sha256']:
            raise ValueError('Byte image identity')
        if image[:4032]!=b'\xa5'*4032 or act[:4032]!=b'\xa5'*4032:raise ValueError('Prefix guards')
        cursor=4032;descriptors=[]
        for first_row in range(0,n,32):
            rows=min(32,n-first_row)
            for kb in range(0,k,t):
                count=min(t,k-kb);size=rows*count*34
                if size%64:raise ValueError('Partial AXI beat tile')
                descriptors.append(dict(first_row=first_row,rows=rows,k_base=kb,k_count=count,
                                        address=0x100000000+cursor,offset=cursor,byte_count=size))
                for block in range(count):
                    for row in range(rows):
                        a=cursor+(block*rows+row)*34;b=((first_row+row)*k+kb+block)*34
                        if image[a:a+34]!=dense[b:b+34]:raise ValueError('Physical full-matrix layout mismatch')
                cursor+=size
        if descriptors!=j['tiles'] or cursor!=len(image) or len(image)!=j['image_bytes']:raise ValueError('Tile coverage')
        if refs!=j['reference_fp64_bits'] or len(refs)!=n:raise ValueError('Reference identity')
        activations=[]
        for b in range(k):
            record=act[4032+b*64:4032+(b+1)*64];scale=struct.unpack('<d',record[32:40])[0]
            bits=int.from_bytes(record[32:40],'little');magnitude=bits&0x7fffffffffffffff
            if record[40:]!=bytes(24) or (bits>>63 and magnitude) or magnitude>0x4780204070e1c387:
                raise ValueError('Activation ABI')
            activations.append((struct.unpack('32b',record[:32]),scale))
        for r in range(n):
            acc=0.0
            for b,(a,sa) in enumerate(activations):
                off=(r*k+b)*34;sw=struct.unpack('<e',dense[off:off+2])[0]
                if not math.isfinite(sw):raise ValueError('Weight scale')
                w=struct.unpack('32b',dense[off+2:off+34]);dot=sum(x*y for x,y in zip(w,a))
                acc+=(sw*sa)*float(dot)
            if struct.pack('>d',acc).hex()!=refs[r]:raise ValueError('Independent ordered Q8 reference')
        descriptor_lines.append(f"{j['id']} {n} {k} {t} {len(image)}")
    if (path/'jobs.txt').read_text().splitlines()!=descriptor_lines:raise ValueError('Job descriptors')
    return m
