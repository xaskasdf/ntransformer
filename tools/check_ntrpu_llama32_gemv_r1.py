"""Review 28 pinned Unsloth Q8 slices, 896 outputs per seed, bit exactly.
Derived from the existing GEMV reviewer; no full-model inference claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def fp64(bits):
    return struct.unpack('>d',bytes.fromhex(bits))[0]


def review(text,manifest,seed):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    executed='\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    diagnostics=[l for l in executed.splitlines() if re.match(r'^\s*(Fatal:|ERROR:|FAIL\b|WARNING:)',l,re.I)]
    profile=manifest.get('format')
    packed=profile=='NTRGEMV02';maximum=profile=='NTRGEMV_MAX01'
    need(profile in {'NTRGEMV01','NTRGEMV02','NTRGEMV_MAX01'},'Fixture format')
    def known_wave(line):
        if not line.startswith('WARNING: [Wavedata 42-489]'):return False
        return ('"/tb_ntrpu_q8_gemv_stream/fixture_weights"' in line and
                '2228224 bits, which exceeds the display limit of 65536 bits' in line) or (
                packed and '"/tb_ntrpu_q8_gemv_stream/fixture_image"' in line and
                '4194304 bits, which exceeds the display limit of 65536 bits' in line)
    need(all(known_wave(l) for l in diagnostics),'Simulation diagnostic')
    need(re.findall(r'^NTRPU_GEMV_SEED seed=(\d+)\s*$',executed,re.M)==[str(seed)],'Actual seed')
    cases=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in executed.splitlines()
           if line.startswith('NTRPU_GEMV_CASE ')]
    jobs=manifest['jobs']
    count=28 if packed else 4 if maximum else 54
    need(len(jobs)==count and [j['id'] for j in jobs]==list(range(1,count+1)),'Fixture jobs')
    need([c.get('id') for c in cases]==[j['id'] for j in jobs],'Missing/duplicated/unordered jobs')
    total_bursts=0
    for c,j in zip(cases,jobs):
        need(all(c.get(k)==v for k,v in dict(id=j['id'],rows=j['rows'],columns=j['k_blocks']*32,
                 scenario=0,status=0,outputs=j['rows']).items()),'Case contract '+str(j['id']))
        bursts=sum((4032+j['rows']*min(j['tile_k'],j['k_blocks']-base)*34+4095)//4096
                   for base in range(0,j['k_blocks'],j['tile_k']))
        total_bursts+=bursts
        need(c.get('bursts')==bursts,'Page split count '+str(j['id']))
        limit=250000 if packed or maximum else 60000
        need(0<c.get('first_result',0)<c.get('cycles',0)<limit,'Latency '+str(j['id']))
    values=re.findall(r'^NTRPU_GEMV_VALUE id=(\d+) row=(\d+) bits=([0-9a-fA-F]{16})\s*$',executed,re.M)
    expected=[(j['id'],r,b) for j in jobs for r,b in enumerate(j['reference_fp64_bits'])]
    need([(int(i),int(r)) for i,r,b in values]==[(i,r) for i,r,b in expected],'Published value identity/count')
    max_error=0.0
    for (i,r,bits),(_,_,ref) in zip(values,expected):
        actual,want=fp64(bits),fp64(ref)
        delta=abs(actual-want)
        need(math.isfinite(actual) and math.isfinite(want) and bits.lower()==ref.lower(),f'Numerical value {i}/{r}')
        if math.isfinite(delta):max_error=max(max_error,delta)
    matrix_errors=[]
    by_id={j['id']:j for j in jobs}
    for matrix in manifest['matrices']:
        observed={by_id[int(i)]['first_row']+int(r):fp64(b) for i,r,b in values
                  if int(i) in by_id and by_id[int(i)]['tensor']==matrix['tensor']}
        need(sorted(observed)==list(range(matrix['rows'])),'Matrix rows '+matrix['tensor'])
        reference=[struct.unpack('>f',bytes.fromhex(b))[0] for b in matrix['fp32_reference_bits']]
        if len(observed)==len(reference):
            l2=math.sqrt(sum((observed[r]-ref)**2 for r,ref in enumerate(reference))/sum(ref*ref for ref in reference))
            need(math.isfinite(l2) and l2<=0.05,'Quantization L2 '+matrix['tensor'])
            matrix_errors.append(dict(tensor=matrix['tensor'],relative_l2_rtl_vs_fp32=l2))
    marker='NTRPU_GEMV_MAX_PASS' if maximum else 'NTRPU_GEMV_FIXTURE_PASS'
    footers=re.findall(r'^'+marker+r' (.+)$',executed,re.M)
    need(len(footers)==1,'Missing/duplicate completion marker')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',footers[0])} if footers else {}
    need(summary.get('cases')==len(jobs),'Completed job count')
    need(summary.get('bursts')==total_bursts,'Total bursts')
    need(summary.get('mac_ops')==sum(j['rows']*j['k_blocks'] for j in jobs),'Total MAC count')
    need(summary.get('result_stalls',0)>0 and summary.get('completion_stalls',0)>0,'Missing backpressure')
    need(sum(c.get('cycles',0) for c in cases)<=summary.get('cycles',0)<2000000,'Suite cycles')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=cases,
                summary=summary,rows_compared=len(values),max_abs_error=max_error,matrices=matrix_errors,
                scope=manifest['scope'])


def load_fixture(path):
    if hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()!='258ffcdad9e2cb1890b46526f67b0f0b92491c0db3dffcffa254c76ab0e038a5':
        raise ValueError('Pinned Llama fixture manifest mismatch')
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('workload')!='NTRLLAMA32Q8SLICES01' or manifest.get('weights_sha256')!='3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1':
        raise ValueError('Llama artifact/workload mismatch')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Fixture hash/path mismatch: '+name)
    if manifest.get('format')=='NTRGEMV02':
        # Decode every byte from the physical image independently of the packer.
        # A self-consistent manifest alone cannot prove its layout is correct.
        if manifest.get('layout')!=2:raise ValueError('Packed layout must be2')
        for j in manifest['jobs']:
            stem=f"job-{j['id']:03d}"
            dense=bytes.fromhex((path/(stem+'-weights.hex')).read_text())
            image=bytes.fromhex((path/(stem+'-image.hex')).read_text())
            descriptors=(path/(stem+'-tiles.hex')).read_text().splitlines()
            if hashlib.sha256(dense).hexdigest()!=j['dense_source_sha256'] or hashlib.sha256(image).hexdigest()!=j['image_sha256']:
                raise ValueError('Dense/image hash mismatch')
            if len(image)!=j['image_bytes'] or len(descriptors)!=len(j['tiles']):raise ValueError('Image/descriptor size')
            occupied=set();next_k=0
            for line,t in zip(descriptors,j['tiles']):
                encoded=int(line,16)
                if (encoded>>64,(encoded>>32)&0xffffffff,encoded&0xffffffff)!=(t['address'],t['k_base'],t['k_count']):raise ValueError('Descriptor encoding')
                if t['k_base']!=next_k or t['address']!=manifest['region_base']+t['offset']:raise ValueError('Descriptor order/address')
                next_k+=t['k_count']
                if t['byte_count']!=j['rows']*t['k_count']*34 or t['byte_count']%64 or t['address']%64:raise ValueError('Descriptor byte size/alignment')
                start=t['offset'];end=start+t['byte_count']
                if start<0 or end>len(image) or end>manifest['region_bytes']:raise ValueError('Image range')
                addresses=set(range(start,end))
                if occupied & addresses:raise ValueError('Overlapping tiles')
                occupied.update(addresses)
                if hashlib.sha256(image[start:end]).hexdigest()!=t['sha256']:raise ValueError('Tile hash')
                for r in range(j['rows']):
                    for k in range(t['k_count']):
                        a=start+(k*j['rows']+r)*34;b=(r*j['k_blocks']+t['k_base']+k)*34
                        if image[a:a+34]!=dense[b:b+34]:raise ValueError('Physical layout byte mismatch')
            if next_k!=j['k_blocks'] or len(dense)!=j['rows']*j['k_blocks']*34:raise ValueError('K/source coverage')
            if any(v!=0xa5 for i,v in enumerate(image) if i not in occupied):raise ValueError('Gap guard pattern')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--json',type=Path)
    a=p.parse_args();r=review(a.log.read_text(errors='replace'),load_fixture(a.fixture),a.seed)
    if a.json:
        with a.json.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r));return 0 if r['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
