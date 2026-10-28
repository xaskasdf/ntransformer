"""Decode packed memory and recompute the real-weight FFN with scalar Python.

No NumPy, HLS helpers or fixture-producer imports. The byte images and ordered
binary64 references are checked independently, including intermediate ABI1.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

DENSE_SHA=[
    '6aff2c7817fd708caf26be8dc39b99e3d573b2d289a95c11a73f2f8373e08e62',
    '0bdfa6e62351f57ca52595858507a9fb2c436ee2753282092e87ddb546e300a3',
    '3f1f1ab3f06bab287fe01f9f4bc5e34c90b045aba3e8abb24e2a003bcdadb69c']


def f32(value):
    if not math.isfinite(value) or abs(value)>float.fromhex('0x1.fffffep+127'):
        raise ValueError('FP32 narrowing overflow/nonfinite')
    return struct.unpack('<f',struct.pack('<f',value))[0]


def activation(values):
    values=list(values)+[0.0]*((-len(values))%32);records=[]
    for start in range(0,len(values),32):
        block=values[start:start+32];scale=max(map(abs,block))/127.0;quantized=[]
        for x in block:
            ratio=x/scale if scale else 0.0
            frac,integer=math.modf(abs(ratio));q=min(127,int(integer)+(frac>=0.5))
            quantized.append((-q if ratio<0 else q)&255)
        records.append(bytes(quantized)+struct.pack('<d',scale)+bytes(24))
    return b''.join(records)


def decode(image,rows,blocks,chunk):
    if image[:4032]!=b'\xa5'*4032 or len(image)!=4032+rows*blocks*34:raise ValueError('Weight image bounds/guard')
    dense=bytearray(rows*blocks*34);cursor=4032
    for r0 in range(0,rows,32):
        for b0 in range(0,blocks,chunk):
            for b in range(b0,min(b0+chunk,blocks)):
                for r in range(r0,min(r0+32,rows)):
                    off=(r*blocks+b)*34;dense[off:off+34]=image[cursor:cursor+34];cursor+=34
    if cursor!=len(image):raise ValueError('Weight image coverage')
    return bytes(dense)


def gemv(weight,rows,records):
    if len(records)%64:raise ValueError('Activation size')
    blocks=len(records)//64;out=[]
    for row in range(rows):
        value=0.0
        for b in range(blocks):
            x=records[b*64:(b+1)*64];q=struct.unpack('32b',x[:32]);sx=struct.unpack('<d',x[32:40])[0]
            if x[40:]!=bytes(24) or not math.isfinite(sx) or sx<0:raise ValueError('Activation ABI')
            off=(row*blocks+b)*34;sw=struct.unpack_from('<e',weight,off)[0]
            w=struct.unpack('32b',weight[off+2:off+34]);dot=0
            for a,coeff in zip(q,w):dot+=a*coeff
            value=value+(sw*sx)*float(dot)
        f32(value);out.append(value)
    return out


def load_fixture(path):
    m=json.loads((path/'manifest.json').read_text())
    if (m['format'],m['layout'],m['dim'],m['hidden'],m['samples'],m['tile_k'])!=('NTRFFNMEM01',2,256,720,3,[4,4,8]):
        raise ValueError('Fixture shape/layout')
    if m['model']!='xaskasdf/brandon-tiny-10m-instruct' or m['revision']!='ca6329de991a1cfc17b2a7a1aaae2cc38c08577a':
        raise ValueError('Model identity')
    if m['weights_sha256']!='2ff727cc8921d21e5e7b153babd288efc5524b166e9417e1b439ac8c11fe44cf' or \
       m['config_sha256']!='602cab332932b65003fd386c67816406d14d9dd548740c6dc0395734f45b39ae' or \
       m['weight_dense_sha256']!=DENSE_SHA or [j['id'] for j in m['jobs']]!=[1,2,3]:
        raise ValueError('Pinned provenance and sample identities')
    for name,sha in m['files'].items():
        if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=sha:raise ValueError('File identity '+name)
    def data(name):return bytes.fromhex((path/name).read_text())
    dense=[]
    for ident,(r,k,t) in enumerate([(720,8,4),(720,8,4),(256,23,8)],1):
        decoded=decode(data(f'weight-{ident}.hex'),r,k,t)
        if decoded!=data(f'weight-{ident}-dense.hex') or hashlib.sha256(decoded).hexdigest()!=DENSE_SHA[ident-1]:
            raise ValueError('Pinned real-weight layout/content')
        dense.append(decoded)
    native=(path/'native-fixture.bin').read_bytes();offset=24
    if native[:24]!=b'NTRFFN01'+struct.pack('<4I',256,720,736,3):raise ValueError('Native header')
    for weight in dense:
        if native[offset:offset+len(weight)]!=weight:raise ValueError('Native weight identity')
        offset+=len(weight)
    if len(native)!=offset+3*3*256*4:raise ValueError('Native length')
    if hashlib.sha256(native).hexdigest()!=m['native_fixture_sha256']:raise ValueError('Native hash')
    report=json.loads((path/'native-report.json').read_text())
    if report['status']!='passed' or report['layers'][0]['fixture_sha256']!=m['native_fixture_sha256']:
        raise ValueError('Native CPU validation identity')
    if (path/'jobs.txt').read_text()!='NTRFFNMEM01 3 256 720\n':raise ValueError('RTL descriptors')
    results=[]
    for ident in range(1,4):
        prefix=f'job-{ident:03d}';x=data(prefix+'-input-fp32.hex')
        if x!=native[offset+(ident-1)*1024:offset+ident*1024]:raise ValueError('Shared native x identity')
        records=activation(struct.unpack('<256f',x));image=data(prefix+'-input.hex')
        if image!=b'\xa5'*4032+records:raise ValueError('Input ABI reference')
        gate=gemv(dense[0],720,records);up=gemv(dense[1],720,records);activated=[]
        for gd,ud in zip(gate,up):
            g,u=f32(gd),f32(ud);e=math.exp(-abs(g)) if abs(g)<=256 else 0.0
            sig=1.0/(1.0+e) if g>=0 else e/(1.0+e)
            activated.append(f32((g*sig)*u))
        abi=activation(activated);down=gemv(dense[2],256,abi)
        buffers={4:struct.pack('<720d',*gate),5:struct.pack('<720d',*up),6:abi,7:struct.pack('<256d',*down)}
        for b,value in buffers.items():
            if value!=data(prefix+f'-buffer-{b}.hex'):raise ValueError('Independent FFN buffer '+str((ident,b)))
        if struct.pack('<720f',*activated)!=data(prefix+'-activation-fp32.hex'):raise ValueError('Activation FP32')
        final=struct.pack('<256f',*map(f32,down));fp32=data(prefix+'-baseline-fp32.hex')
        if final!=data(prefix+'-final-fp32.hex') or final!=native[offset+6144+(ident-1)*1024:offset+6144+ident*1024]:
            raise ValueError('Final/native Q8 FP32')
        if fp32!=native[offset+3072+(ident-1)*1024:offset+3072+ident*1024]:raise ValueError('FP32 baseline')
        v=struct.unpack('<256f',final);ref=struct.unpack('<256f',fp32)
        relative=math.sqrt(sum((a-b)**2 for a,b in zip(v,ref))/sum(b*b for b in ref))
        if not math.isclose(relative,m['jobs'][ident-1]['relative_l2'],rel_tol=1e-13):raise ValueError('Relative error metadata')
        results.append(dict(id=ident,reference_buffers={b:value.hex() for b,value in buffers.items()},relative_l2=relative))
    return dict(manifest=m,jobs=results)


def review_log(text,fixture,seed=1,fp32=False):
    from check_ntrpu_gemv_memory import bursts
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    def records(prefix):
        return [{k:int(v,16) if k in {'data','strb'} else int(v) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_FFN_MEMORY_SEED')==[dict(seed=seed)],'Seed')
    if fp32:need(records('NTRPU_FFN_MEMORY_FORMAT')==[dict(fp32=1)],'FP32 output format')
    cases=records('NTRPU_FFN_MEMORY_CASE');writes=records('NTRPU_FFN_MEMORY_W')
    need([c.get('id') for c in cases]==[1,2,3],'Case identity/order')
    ar=aw=0
    for rows,k,chunk in [(720,8,4),(720,8,4),(256,23,8)]:
        address=4032
        for first in range(0,rows,32):
            count=min(32,rows-first);ar+=bursts(4032,k*64)
            element_bytes=4 if fp32 and rows==256 else 8
            aw+=bursts(4032+first*element_bytes,((count*element_bytes+63)//64)*64)
            for kb in range(0,k,chunk):
                size=count*min(chunk,k-kb)*34;ar+=bursts(address,size);address+=size
    for first in range(0,720,32):ar+=2*bursts(4032+first*8,((min(32,720-first)+7)//8)*64)
    aw+=(720+31)//32
    expected_writes=[]
    for job,c in zip(fixture['jobs'],cases):
        ident=job['id'];need(tuple(c.get(k) for k in ['dim','hidden','mode','target','reset'])==(256,720,0,0,0),'Case contract')
        need(0<c.get('cycles',0)<1000000,'Cycle bound')
        need(sum(c.get('p'+str(p),0) for p in range(4))==c.get('cycles'),'Phase cycle accounting')
        need(tuple(c.get(k) for k in ['ar','r','aw','w','b','bytes','down_bytes'])==
             (ar,9980,aw,219 if fp32 else 235,aw,14016 if fp32 else 15040,1024 if fp32 else 2048),'Full AXI traffic')
        for b,hexbytes in job['reference_buffers'].items():
            raw=bytes.fromhex(hexbytes)
            if fp32 and b==7:raw=struct.pack('<256f',*struct.unpack('<256d',raw))
            for offset in range(0,len(raw),64):
                expected_writes.append(dict(id=ident,buffer=int(b),offset=offset,strb=2**64-1,data=int.from_bytes(raw[offset:offset+64],'little')))
    need(writes==expected_writes,'Complete independently computed physical byte trace')
    summaries=records('NTRPU_FFN_MEMORY_PASS');need(len(summaries)==1,'Suite completion')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==3 and summary.get('resets')==summary.get('cancels')==0,'Suite scope')
    need(summary.get('bytes')==(42048 if fp32 else 45120),'Total W bytes')
    need(summary.get('cycles')==sum(c.get('cycles',0) for c in cases),'Total cycles')
    for key in ['ar_stalls','aw_stalls','w_stalls','done_stalls']:need(summary.get(key,0)>0,'Stall coverage '+key)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,cases=cases,summary=summary,
                physical_bytes=len(expected_writes)*64,fp64_values=3*(720+720+(0 if fp32 else 256)),published_down_values=768,
                fp32_output=fp32,
                activation_records=69,relative_l2=[j['relative_l2'] for j in fixture['jobs']],
                scope='Three real-weight Brandon layer0 FFN chains with shared synthetic x; '+
                      ('physically stored FP32 down matches native Q8 reference' if fp32 else 'raw FP64 down and independently narrowed FP32 reference')+
                      ', no full-model inference.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture',type=Path)
    p.add_argument('--log',type=Path)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--fp32',action='store_true')
    a=p.parse_args();fixture=load_fixture(a.fixture)
    report=review_log(a.log.read_text(),fixture,a.seed,a.fp32) if a.log else dict(status='PASS',samples=3,
        relative_l2=[j['relative_l2'] for j in fixture['jobs']],scope=fixture['manifest']['scope'])
    print(json.dumps(report,indent=2));raise SystemExit(report['status']!='PASS')
