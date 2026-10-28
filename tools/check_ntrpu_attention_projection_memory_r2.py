"""Independently bind full O projection AXI traces to frozen real model bytes."""
import copy
import hashlib
import json
from pathlib import Path
import re

WBASE=0x100000000
ABASE=0x8000000000
OBASE=0xa000000000
FIRST_TAG=0xffffff80
PLAN=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,1),(0,2)]
SCHEMA={
 'CONFIG':'seed rows k_blocks tile_k watchdog',
 'BEGIN':'cycle job mode sample sequence epoch tag',
 'MATRIX':'cycle job tag','GROUP':'cycle job group tag','TILE':'cycle job tile address',
 'AR':'cycle job ar address beats group','R':'cycle job ar beat address data last',
 'MAC':'cycle job row data','AW':'cycle job aw address beats','W':'cycle job word data last',
 'B':'cycle job aw response','GROUP_DONE':'cycle job group status bytes',
 'MATRIX_DONE':'cycle job status bytes rows tag','CANCEL':'cycle job reads',
 'COMPLETE':'cycle job mode status sequence epoch valid bytes groups tiles results ar r aw w b max_quiet',
 'PASS':'seed jobs publics results ar r aw w b ar_stalls r_stalls aw_stalls w_stalls cycles'}
HEX={'sequence','epoch','tag','address','data'}
MANIFEST_SHA='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'


def load_fixture(folder):
    folder=Path(folder)
    raw=(folder/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA
    manifest=json.loads(raw)
    base=folder/'raw' if (folder/'raw').is_dir() else folder
    def words(name,size,width):
        raw=(base/name).read_bytes()
        assert len(raw)==size and hashlib.sha256(raw).hexdigest()==manifest['files'][name],name
        return [int.from_bytes(raw[n:n+width],'little') for n in range(0,len(raw),width)]
    return dict(weight=words('output-weight-image.bin',4460480,64),samples=[dict(
        activation=words(f'sample-{i:03d}-attention-abi1.bin',4096,64),
        fp64=words(f'sample-{i:03d}-output-fp64.bin',16384,8),
        fp32=words(f'sample-{i:03d}-output-fp32.bin',8192,64)) for i in range(1,5)])


def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|OPROJ_FAIL)',text),'Simulator diagnostic'
    events=[]
    for line in text.splitlines():
        if not line.startswith('OPROJ_'):continue
        parts=line.split();kind=parts[0][6:]
        assert kind in SCHEMA,kind
        fields=[part.split('=') for part in parts[1:]]
        assert all(len(pair)==2 for pair in fields)
        assert [pair[0] for pair in fields]==SCHEMA[kind].split(),kind
        row={key:int(value,16 if key in HEX else 10) for key,value in fields}
        assert all(value>=0 for value in row.values())
        events.append((kind,row))
    return events


def bursts(address,count):
    result=[]
    while count:
        beats=min(count//64,(4096-address%4096)//64)
        assert 0<beats<=64
        result.append((address,beats));address+=beats*64;count-=beats*64
    return result


def review_events(events,fixture,seed=1):
    assert events and events[0]==('CONFIG',dict(seed=seed,rows=2048,k_blocks=64,tile_k=32,watchdog=65536))
    assert events[-1][0]=='PASS'
    buckets={j:{k:[] for k in SCHEMA} for j in range(1,8)}
    global_rows={k:[] for k in SCHEMA};cycle=-1;active=None;last_job=0
    for kind,row in events:
        assert kind in SCHEMA and set(row)==set(SCHEMA[kind].split())
        if 'cycle' in row:
            assert row['cycle']>=cycle,'Cycle regression'
            cycle=row['cycle']
        if 'job' in row:
            j=row['job'];assert j in buckets
            if kind=='BEGIN':
                assert active is None and j==last_job+1
                active=j;last_job=j
            assert j==active,'Events outside admitted owner'
            buckets[j][kind].append(row)
            if kind=='COMPLETE':active=None
        else:global_rows[kind].append(row)
    assert active is None and last_job==7
    assert len(global_rows['CONFIG'])==len(global_rows['PASS'])==1
    summary=global_rows['PASS'][0]
    assert (summary['seed'],summary['jobs'],summary['publics'],summary['results'])==(seed,7,7,10240)
    totals={k:0 for k in ('ar','r','aw','w','b','results')};metrics=[]
    for j,(mode,sample_id) in enumerate(PLAN,1):
        e=buckets[j]
        for name in ('BEGIN','MATRIX','MATRIX_DONE','COMPLETE'):assert len(e[name])==1,(j,name)
        begin=e['BEGIN'][0];matrix=e['MATRIX'][0];done=e['MATRIX_DONE'][0];end=e['COMPLETE'][0]
        seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j;tag=FIRST_TAG+(j-1)*64
        assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'],begin['tag'])==(mode,sample_id,seq,epoch,tag)
        assert matrix['tag']==done['tag']==tag and matrix['cycle']==begin['cycle']+2
        status={0:0,1:1,2:7}[mode]
        assert (end['mode'],end['status'],end['sequence'],end['epoch'],end['valid'],end['bytes'])==(mode,status,seq,epoch,int(mode==0),8192 if mode==0 else 0)
        assert (done['status'],done['bytes'],done['rows'])==(status,8192 if mode==0 else 0,2048 if mode==0 else 0)
        assert done['cycle']>matrix['cycle'] and end['cycle']>=done['cycle']+7
        sample=fixture['samples'][sample_id]
        ng=64 if mode==0 else 0 if mode==1 else 1
        assert len(e['GROUP'])==len(e['GROUP_DONE'])==ng
        assert len(e['TILE'])==(128 if mode==0 else 0 if mode==1 else 1)
        for index,row in enumerate(e['GROUP']):
            assert (row['group'],row['tag'])==(index,tag+index)
            assert matrix['cycle']<row['cycle']<done['cycle']
            if index:assert row['cycle']>e['GROUP_DONE'][index-1]['cycle']
        for index,row in enumerate(e['TILE']):
            assert (row['tile'],row['address'])==(index,WBASE+4032+index*34816)
            assert e['GROUP'][index//2]['cycle']<row['cycle']<e['GROUP_DONE'][index//2]['cycle']
        for index,row in enumerate(e['GROUP_DONE']):
            assert (row['group'],row['status'],row['bytes'])==(index,7 if mode==2 else 0,0 if mode==2 else 128)
            assert e['GROUP'][index]['cycle']<row['cycle']<done['cycle']
        expected_ar=[]
        for group in range(ng):
            for address,count in [(ABASE+4032,4096),(WBASE+4032+group*69632,34816),(WBASE+4032+group*69632+34816,34816)]:
                expected_ar += [(address,beats,group) for address,beats in bursts(address,count)]
        ars=e['AR'];reads=e['R'];cursor=0
        if mode==0:assert len(ars)==1344 and len(reads)==73728
        if mode==1:assert not ars and not reads
        if mode==2:assert 3<=len(ars)<=21 and 70<=len(reads)<=1152
        for index,row in enumerate(ars):
            assert index<len(expected_ar)
            address,beats,group=expected_ar[index]
            assert (row['ar'],row['address'],row['beats'],row['group'])==(index,address,beats,group)
            assert e['GROUP'][group]['cycle']<row['cycle']<e['GROUP_DONE'][group]['cycle']
            if index:assert row['cycle']>reads[cursor-1]['cycle'],'More than one read outstanding'
            if address>=ABASE:
                values=sample['activation'];word0=(address-ABASE-4032)//64
            else:
                values=fixture['weight'];word0=(address-WBASE)//64
                tile=(address-WBASE-4032)//34816
                assert e['TILE'][tile]['cycle']<row['cycle']
            for beat in range(beats):
                assert cursor<len(reads),'Undrained read burst'
                r=reads[cursor];cursor+=1
                assert (r['ar'],r['beat'],r['address'],r['data'],r['last'])==(index,beat,address+beat*64,values[word0+beat],int(beat==beats-1))
                assert row['cycle']<r['cycle']<e['GROUP_DONE'][group]['cycle']
                if beat:assert reads[cursor-2]['cycle']<r['cycle']
        assert cursor==len(reads),'Unrequested read response'
        mac=e['MAC'];assert len(mac)==(2048 if mode==0 else 0)
        for index,row in enumerate(mac):
            assert (row['row'],row['data'])==(index,sample['fp64'][index])
            group=index//32
            assert reads[(group+1)*1152-1]['cycle']<row['cycle']<e['GROUP_DONE'][group]['cycle']
            if index:assert row['cycle']>mac[index-1]['cycle']
        expected_aw=[]
        if mode==0:
            for group in range(64):expected_aw += [(address,beats,group) for address,beats in bursts(OBASE+4032+128*group,128)]
        assert len(e['AW'])==len(e['B'])==len(expected_aw)
        assert len(e['W'])==(128 if mode==0 else 0)
        cursor=0
        for index,(address,beats,group) in enumerate(expected_aw):
            aw=e['AW'][index];b=e['B'][index];words=e['W'][cursor:cursor+beats]
            assert (aw['aw'],aw['address'],aw['beats'])==(index,address,beats)
            assert (b['aw'],b['response'])==(index,0)
            assert mac[(group+1)*32-1]['cycle']<aw['cycle']<b['cycle']<e['GROUP_DONE'][group]['cycle']
            if index:assert e['B'][index-1]['cycle']<aw['cycle']
            assert len(words)==beats
            for beat,row in enumerate(words):
                assert (row['word'],row['data'],row['last'])==(cursor+beat,sample['fp32'][cursor+beat],int(beat==beats-1))
                assert mac[(group+1)*32-1]['cycle']<row['cycle']<b['cycle']
                if index:assert row['cycle']>e['B'][index-1]['cycle']
                if beat:assert row['cycle']>words[beat-1]['cycle']
            cursor+=beats
        if mode==2:
            assert len(e['CANCEL'])==1
            cancel=e['CANCEL'][0]
            assert cancel['reads']==sum(r['cycle']<=cancel['cycle'] for r in reads)>=70
            assert cancel['cycle']<done['cycle'] and reads[-1]['cycle']>cancel['cycle'],'Cancel must drain an active read'
        else:assert not e['CANCEL']
        bus_cycles=sorted({r['cycle'] for k in ('AR','R','AW','W','B') for r in e[k]})
        start=begin['cycle']+2;stop=done['cycle']
        assert all(start<=c<=stop for c in bus_cycles)
        gaps=([bus_cycles[0]-start]+[b-a-1 for a,b in zip(bus_cycles,bus_cycles[1:])]+[stop-bus_cycles[-1]]) if bus_cycles else [stop-start+1]
        assert end['max_quiet']==max(gaps)<65536,'Watchdog gap differs from trace'
        for key,kind in [('ar','AR'),('r','R'),('aw','AW'),('w','W'),('b','B'),('results','MAC'),('groups','GROUP'),('tiles','TILE')]:
            assert end[key]==len(e[kind]),(j,key)
            if key in totals:totals[key]+=len(e[kind])
        metrics.append(dict(job=j,mode=mode,sample=sample_id,completion_cycles_including_hold=end['cycle']-begin['cycle'],
                            matrix_cycles=stop-matrix['cycle']+1,max_quiet_cycles=max(gaps),read_bytes=len(reads)*64,
                            written_bytes=len(e['W'])*64,published_bytes=end['bytes'],fp64_results=len(mac)))
    for key,total in totals.items():assert summary[key]==total,(key,summary[key],total)
    assert (totals['aw'],totals['w'],totals['b'])==(330,640,330)
    assert all(summary[name]>0 for name in ('ar_stalls','aw_stalls','w_stalls'))
    # The slave randomizes RVALID; the always-ready read path observed no held R.
    assert summary['r_stalls']==0
    assert summary['cycles']>buckets[7]['COMPLETE'][0]['cycle']
    return dict(seed=seed,jobs=7,successful_projections=5,fp64_results=totals['results'],fp32_output_bytes=40960,
                totals=totals,jobs_metrics=metrics,cycles=summary['cycles'],clock_ns=5,
                scope='Real layer-0 O weights and four frozen attention activations, AXI BFM; no DDR or end-to-end inference qualification.')


def review(text,fixture,seed=1):return review_events(parse(text),fixture,seed)


def controls(text,fixture,seed=1):
    events=parse(text);review_events(events,fixture,seed);rejected=[]
    # Change parsed semantic fields independently; parser diagnostics are checked separately.
    changes=[('BEGIN','sequence',1<<40,1),('BEGIN','epoch',1<<40,1),('MATRIX','tag',1<<32,3),
             ('GROUP','tag',1<<32,1),('TILE','address',64,1),('AR','address',64,1),('AR','beats',1,1),
             ('R','data',1,1),('R','last',1,1),('R','beat',1,1),('MAC','data',1,1),
             ('AW','address',64,1),('W','data',1,1),('W','last',1,1),('B','response',2,1),
             ('GROUP_DONE','bytes',64,1),('MATRIX_DONE','rows',1,1),('MATRIX_DONE','tag',1<<32,1),
             ('COMPLETE','status',1,5),('COMPLETE','valid',1,5),('COMPLETE','bytes',64,6),
             ('COMPLETE','max_quiet',1,1),('CANCEL','reads',1,6),('PASS','results',1,None),('PASS','r_stalls',1,None)]
    for kind,key,mask,job in changes:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and (job is None or r.get('job')==job))
        altered=events.copy();row=dict(events[index][1]);row[key]^=mask;altered[index]=(kind,row)
        try:review_events(altered,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'{kind}.{key}.job{job}')
        else:raise AssertionError('Mutation accepted: '+rejected[-1] if rejected else kind)
    for kind,job in [('R',6),('MAC',1),('B',1),('MATRIX_DONE',7)]:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and r.get('job')==job)
        altered=events[:index]+events[index+1:]
        try:review_events(altered,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'missing-{kind}.job{job}')
        else:raise AssertionError('Missing event accepted: '+kind)
    for label,bad in [('fatal','Fatal: injected diagnostic\n'),('unknown','OPROJ_UNKNOWN field=1\n'),('xz','OPROJ_MATRIX cycle=1 job=1 tag=xx\n')]:
        try:parse(bad)
        except (AssertionError,ValueError):rejected.append('parser-'+label)
        else:raise AssertionError('Bad parser input accepted')
    assert len(rejected)==32
    return rejected
