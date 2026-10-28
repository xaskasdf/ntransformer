"""Derive a fixed-40-head AXI fixture from the independently checked RoPE parent."""
import hashlib
import json
from pathlib import Path
import struct
from check_ntrpu_rope_table_r1 import fixture as parent_fixture,oracle,packed

PARENT_SHA='35e80881c4b71531c47c21379f98017e2fae2419c9b29a78576f538495081253'
FAULT_PLAN=[(41,7),(43,7),(48,0),(49,0),(50,None),(51,None),(52,0),
            (60,0),(62,7),(63,2),(64,7),(65,None),(66,None),(67,7),(68,7)]

def load_parent(folder):
    approved=parent_fixture(folder)
    raw=(folder/'llama32_rope_r1.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    offset=12;rows=[]
    def words(n):
        nonlocal offset
        v=list(struct.unpack_from('<'+'I'*n,raw,offset));offset+=n*4;return v
    for expected in approved:
        ident,heads,p,pp,status,n=words(6)
        phase,x,y=words(64),words(2560),words(n)
        assert ident==expected['id'] and y==expected['expected']
        finite=all((v&0x7f800000)!=0x7f800000 for v in x+phase)
        if p>=131072:public,hls,start=1,0,False
        elif pp!=p:public,hls,start=2,0,False
        elif not finite:public,hls,start=4,0,False
        else:public,hls,start=(0 if status==0 else 5),status,True
        rows.append(dict(expected,input=x,phase=phase,public_status=public,hls_status=hls,hls_started=start))
    assert offset==len(raw) and sum(r['heads']==40 for r in rows)==39
    return rows

def plan(rows):
    return [(100+r['id']-1,r['public_status']) for r in rows if r['heads']==40]+[
        item for pair in FAULT_PLAN for item in (pair,(0,0))]

def load(folder):
    m=json.loads((folder/'manifest.json').read_text())
    for n,h in m['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h,n
    rows=load_parent(folder/'parent')
    source=[v for r in rows for v in r['input']+r['phase']]
    expected=[v for r in rows for v in r['expected']+[0]*(2560-len(r['expected']))]
    metadata=[v for r in rows for v in (r['heads'],r['position'],r['coefficient_position'],r['public_status'],r['hls_status'],int(r['hls_started']))]
    for name,values in [('source.hex',source),('expected.hex',expected),('metadata.hex',metadata)]:
        assert (folder/name).read_text()==''.join(f'{v:08x}\n' for v in values),name
    assert m['plan']==[list(p) for p in plan(rows)] and len(m['plan'])==69
    assert m['selected_cases']==39 and m['parent_sha256']==PARENT_SHA
    return rows

def prepare(parent,dest):
    rows=load_parent(parent);dest.mkdir(exist_ok=False);(dest/'parent').mkdir()
    pm=json.loads((parent/'manifest.json').read_text())
    for n in ['manifest.json',*pm['files']]:
        p=dest/'parent'/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes((parent/n).read_bytes())
    values={'source.hex':[v for r in rows for v in r['input']+r['phase']],
        'expected.hex':[v for r in rows for v in r['expected']+[0]*(2560-len(r['expected']))],
        'metadata.hex':[v for r in rows for v in (r['heads'],r['position'],r['coefficient_position'],r['public_status'],r['hls_status'],int(r['hls_started']))]}
    for name,words in values.items():(dest/name).write_text(''.join(f'{v:08x}\n' for v in words))
    files={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
    manifest=dict(parent_sha256=PARENT_SHA,files=files,selected_cases=39,excluded_parent_cases=[r['id'] for r in rows if r['heads']!=40],
        exclusion_reason='Fixed 40-head integration; variable/invalid head-count cases remain covered only by the original standalone core bank.',
        plan=plan(rows),arithmetic_executed=False,source='Original RoPE fixture; memory and output references kept separate.')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert len(load(dest))==48
    return manifest
