"""Review memory-fed GEMV identities, AXI counts and independent dot results."""
import json
import hashlib
import math
import re
import struct

CASES=[(2,32,0,0),(2,128,0,0),(0,32,1,1),(2,32,2,1),(2,32,3,2),(2,32,4,2),
       (2,32,5,2),(2,32,6,2),(2,32,7,3),(2,32,8,3),(2,32,9,7),(2,32,10,7),
       (2,128,11,7),(2,32,12,8),(2,32,13,3),(2,32,14,7),(1,64,0,0),
       (2,128,15,0),(2,32,16,0),(2,32,0,0)]
EXTENDED_CASES=CASES+[(n,k,0,0) for n,k in [(1,32),(3,32),(4,32),(5,32),(31,32),(32,32),
                                         (3,64),(4,64),(5,64),(31,64),(32,64)]]+\
    [(n,32,13,3) for n in (4,5,31,32)]+[(n,32,14,7) for n in (4,5,31,32)]+[(n,32,16,0) for n in (4,5)]
MAC_FAULT_CASES=[(4,32,0,0)]
for fault in ([(n,32,17+lane,6) for n in (4,5) for lane in range(4)]+
              [(31,32,17,6),(32,32,20,6)]+[(n,32,21,8) for n in (4,5,32)]+
              [(n,32,22,7) for n in (4,5)]+[(n,32,23,0) for n in (4,5)]):
    MAC_FAULT_CASES += [fault,(4,32,0,0)]


def bursts(address,byte_count):
    count=0
    while byte_count:
        size=min(byte_count,4096-address%4096);address+=size;byte_count-=size;count+=1
    return count


def load_memory_fixture(path):
    from check_ntrpu_gemv_fixture import load_fixture
    m=load_fixture(path)  # Independently decode weight images and descriptors.
    if (m.get('activation_abi'),m.get('activation_base'),m.get('activation_offset'),
        m.get('activation_region_bytes'))!=(1,0x8000000000,4032,0x100000):
        raise ValueError('Activation fixture ABI/range')
    for j in m['jobs']:
        stem=f"job-{j['id']:03d}"
        image=bytes.fromhex((path/(stem+'-activation-image.hex')).read_text())
        source=[int(l,16).to_bytes(32,'little') for l in (path/(stem+'-activations.hex')).read_text().split()]
        scales=[int(l,16) for l in (path/(stem+'-scales.hex')).read_text().split()]
        if len(source)!=j['k_blocks'] or len(scales)!=len(source):raise ValueError('Activation source count')
        if len(image)!=4032+64*len(source) or len(image)!=j['activation_image_bytes']:raise ValueError('Activation image size')
        if hashlib.sha256(image).hexdigest()!=j['activation_image_sha256']:raise ValueError('Activation image hash')
        if image[:4032]!=bytes([0xa5])*4032:raise ValueError('Activation image guard')
        decoded=[]
        for k,values in enumerate(source):
            record=image[4032+64*k:4032+64*(k+1)]
            scale=int.from_bytes(record[32:40],'little');magnitude=scale&0x7fffffffffffffff
            if record[:32]!=values or scale!=scales[k] or record[40:]!=bytes(24):raise ValueError('Activation physical bytes')
            if (scale>>63 and magnitude) or magnitude>0x4780204070e1c387:raise ValueError('Activation numeric scale')
            decoded.append(([v if v<128 else v-256 for v in values],struct.unpack('<d',record[32:40])[0]))
        dense=bytes.fromhex((path/(stem+'-weights.hex')).read_text())
        references=(path/(stem+'-reference.hex')).read_text().split()
        if references!=j['reference_fp64_bits']:raise ValueError('Reference file/manifest mismatch')
        for row in range(j['rows']):
            accumulator=0.0
            for k,(a,sx) in enumerate(decoded):
                offset=(row*j['k_blocks']+k)*34
                sw=struct.unpack('<e',dense[offset:offset+2])[0]
                w=[v if v<128 else v-256 for v in dense[offset+2:offset+34]]
                accumulator+=(sw*sx)*float(sum(x*y for x,y in zip(w,a)))
            if struct.pack('>d',accumulator).hex()!=references[row]:raise ValueError('Independent ordered Q8 oracle')
    return m


def review(text,seed,fixture=None,extended=False,mac_faults=False):
    if sum((fixture is not None,bool(extended),bool(mac_faults)))>1:raise ValueError('Memory banks are mutually exclusive')
    errors=[]
    def need(ok,message):
        if not ok: errors.append(message)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(re.findall(r'^NTRPU_MEMORY_SEED seed=(\d+)\s*$',text,re.M)==[str(seed)],'Seed')
    records=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',l)} for l in text.splitlines() if l.startswith('NTRPU_MEMORY_CASE ')]
    contracts=[(j['rows'],j['k_blocks'],0,0) for j in fixture['jobs']] if fixture else MAC_FAULT_CASES if mac_faults else EXTENDED_CASES if extended else CASES
    need([r.get('id') for r in records]==list(range(1,len(contracts)+1)),'Case identities')
    expected_values={}
    for r,w in zip(records,contracts):
        ident=r.get('id');n,k,mode,status=w;ok=status==0 and mode not in {15,16,23}
        need(tuple(r.get(f) for f in ['rows','blocks','scenario','status'])==w,'Contract '+str(ident))
        need(set(r)=={'id','rows','blocks','scenario','status','cycles','first_result','activation_bursts','weight_bursts','activation_beats','weight_beats','outputs','tiles'},'Fields')
        need(0<r.get('cycles',0)<(250000 if fixture else 100000 if extended else 20000),'Latency bound')
        need(r.get('outputs')==(n if ok else 0),'Output count')
        first=r.get('first_result',0);need(0<first<r.get('cycles',0) if ok else first==0,'Publication timing')
        if ok:
            if fixture:
                job=fixture['jobs'][ident-1]
                wanted_bursts=sum(bursts(t['address'],t['byte_count']) for t in job['tiles'])
                wanted_tiles=len(job['tiles'])
            else: wanted_bursts=(k//32)*bursts(4032,n*32*34);wanted_tiles=k//32
            need((r.get('activation_beats'),r.get('activation_bursts'),r.get('weight_beats'),r.get('weight_bursts'),r.get('tiles'))==
                 (k,bursts(4032,k*64),n*k*34//64,wanted_bursts,wanted_tiles),'Successful memory counts')
            for row in range(n):
                value=struct.unpack('>d',bytes.fromhex(job['reference_fp64_bits'][row]))[0] if fixture else float(sum(
                    ((b+lane)%5-2)*((b+row*3+lane)%7-3) for b in range(k) for lane in range(32)))
                expected_values[(ident,row)]=value
        elif mode in {1,2,3,4,9}:
            need(all(r.get(f)==0 for f in ['activation_beats','activation_bursts','weight_beats','weight_bursts','tiles']),'No AXI allowed')
        elif 5<=mode<=12:
            ab,beats={5:(2,32),6:(2,32),7:(2,32),8:(1,1),10:(1,1),11:(2,65),12:(1,1)}[mode]
            need((r.get('activation_bursts'),r.get('activation_beats'))==(ab,beats),'Failed activation drain')
            need(r.get('weight_beats')==r.get('weight_bursts')==r.get('tiles')==0,'No weights after failed activation')
        else:
            need(r.get('activation_beats')==(4 if mode==15 else k),'Reset/compute activation count')
            need(r.get('weight_beats')==0 if mode==15 else 0<r.get('weight_beats',0)<=n*k*34//64,'Compute drain/reset')
    values=[]
    for l in text.splitlines():
        if not l.startswith('NTRPU_MEMORY_VALUE '):continue
        fields=dict(re.findall(r'(\w+)=([^ ]+)',l));key=(int(fields['case']),int(fields['row']))
        actual=float(fields['value']);ref=float(fields['reference']);wanted=expected_values.get(key)
        need(wanted is not None and math.isfinite(actual) and actual==wanted and ref==wanted,'Independent result '+str(key))
        values.append(key)
    need(values==list(expected_values),'Missing/duplicate/unordered results')
    footers=re.findall(r'^NTRPU_MEMORY_PASS (.+)$',text,re.M);need(len(footers)==1,'Suite marker')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',footers[0])} if footers else {}
    need(summary.get('cases')==len(contracts) and summary.get('resets')==(0 if fixture else 4 if extended else 2),'Cases/resets')
    for key in ['cycles','outputs','activation_bursts','weight_bursts']:
        need(summary.get(key)==sum(r.get(key,0) for r in records),'Total '+key)
    for key in ['ar_stalls','result_stalls','done_stalls']:need(summary.get(key,0)>0,'Missing '+key)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=records,summary=summary,
                values=len(values),max_absolute_error=0 if not errors else None)
