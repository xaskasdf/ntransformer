"""Independently bind full O projection AXI traces to frozen real model bytes."""
from check_ntrpu_attention_projected_root_materialization_r2 import SCHEMA as ROOT_SCHEMA, load_root_fixture, review_root
import copy
import hashlib
import json
from pathlib import Path
import re

WBASE=0x100000000
ABASE=0x8000000000
OBASE=0xa000000000
FIRST_TAG=0xffffffc0
PLAN=[(0,0),(1,1),(0,3)]
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
SCHEMA.update({
 'READ':'cycle job index data','INPUT':'cycle job index data','START':'cycle job',
 'WORD':'cycle job index data','DONE':'cycle job status words','ROOT_ACK':'cycle job sequence epoch',
 'REUSE':'cycle job sequence epoch','QAW':'cycle job record address','QW':'cycle job record data',
 'QB':'cycle job record response',
 'QUANT_PUBLIC':'cycle job status hls sequence epoch records valid address bytes',
 'PROJECTION_COMMAND':'cycle job sequence epoch address',
 'QACK':'cycle job sequence epoch','PACK':'cycle job sequence epoch','RELEASE':'cycle job'})
SCHEMA['PASS'] += ' qreads qwords root qacks packs qaw qw qb'
SCHEMA.update(ROOT_SCHEMA)
del SCHEMA['REUSE']
SCHEMA['READ']='cycle job index'
SCHEMA['CANCEL']='cycle job qbeats'
SCHEMA['PASS'] += ' heads copies qbeats binds appends kar kr kaw kw kb outacks'
HEX={'sequence','epoch','tag','address','data','base','lease'}
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
        input_fp32=words(f'sample-{i:03d}-attention-fp32.bin',8192,4),
        abi_words=words(f'sample-{i:03d}-attention-abi1.bin',4096,8),
        fp64=words(f'sample-{i:03d}-output-fp64.bin',16384,8),
        fp32=words(f'sample-{i:03d}-output-fp32.bin',8192,64)) for i in range(1,5)])


def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|PRREAL_FAIL)',text),'Simulator diagnostic'
    events=[]
    for line in text.splitlines():
        if not line.startswith('PRREAL_'):continue
        parts=line.split();kind=parts[0][7:]
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
    buckets={j:{k:[] for k in SCHEMA} for j in range(1,len(PLAN)+1)}
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
            if kind=='APPEND_DONE':
                # B is logged at posedge; registered done is observed at the
                # following negedge with the same cycle number. Require that
                # all accepted KAWs already have B events before publication.
                assert buckets[j]['KAW'] and len(buckets[j]['KB'])==len(buckets[j]['KAW'])
                assert buckets[j]['KAW'][-1]['position']==row['position']
                assert buckets[j]['KB'][-1]['cycle']<=row['cycle']
            buckets[j][kind].append(row)
            if kind=='RELEASE':active=None
        else:global_rows[kind].append(row)
    assert active is None and last_job==len(PLAN)
    assert len(global_rows['CONFIG'])==len(global_rows['PASS'])==1
    summary=global_rows['PASS'][0]
    assert (summary['seed'],summary['jobs'],summary['publics'],summary['results'])==(seed,3,3,4096)
    totals={k:0 for k in ('ar','r','aw','w','b','results')};metrics=[]
    for j,(mode,sample_id) in enumerate(PLAN,1):
        e=buckets[j]
        for name in ('BEGIN','COMPLETE','RELEASE'):assert len(e[name])==1,(j,name)
        begin=e['BEGIN'][0];end=e['COMPLETE'][0]
        seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j;tag=FIRST_TAG+(0 if j==1 else 64)
        assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'],begin['tag'])==(mode,sample_id,seq,epoch,tag)
        status=7 if mode else 0
        assert (end['mode'],end['status'],end['sequence'],end['epoch'],end['valid'],end['bytes'])==(mode,status,seq,epoch,int(mode==0),8192 if mode==0 else 0)
        root_metrics=review_root(e,fixture,j,mode,sample_id)
        if mode:
            absent=('MATRIX','MATRIX_DONE','PROJECTION_COMMAND','QUANT_PUBLIC','QACK','PACK','START','DONE',
                    'READ','INPUT','WORD','QAW','QW','QB','GROUP','GROUP_DONE','TILE','AR','R','MAC','AW','W','B')
            for name in absent:assert not e[name],(j,name)
            for key in ('groups','tiles','results','ar','r','aw','w','b','max_quiet'):assert end[key]==0,(j,key)
            metrics.append(dict(job=j,mode=mode,sample=sample_id,root=root_metrics,
                                completion_cycles_including_cache_setup_and_hold=end['cycle']-begin['cycle']))
            continue
        for name in ('MATRIX','MATRIX_DONE','PROJECTION_COMMAND','QUANT_PUBLIC','QACK','PACK','START','DONE'):assert len(e[name])==1,(j,name)
        matrix=e['MATRIX'][0];done=e['MATRIX_DONE'][0]
        assert matrix['tag']==done['tag']==tag and matrix['cycle']==e['PROJECTION_COMMAND'][0]['cycle']+1
        assert (done['status'],done['bytes'],done['rows'])==(0,8192,2048)
        assert done['cycle']>matrix['cycle'] and end['cycle']>=done['cycle']+7
        sample=fixture['samples'][sample_id]
        reads_q=e['READ'];inputs_q=e['INPUT'];words_q=e['WORD']
        assert len(reads_q)>=2048 and len(inputs_q)==len(reads_q)
        assert set(r['index'] for r in reads_q)==set(range(2048))
        requested={(r['cycle']+1,r['index']):sample['input_fp32'][r['index']] for r in reads_q}
        assert len(requested)==len(reads_q)
        for row in inputs_q:assert row['data']==requested[(row['cycle'],row['index'])]
        assert len({(r['cycle'],r['index']) for r in inputs_q})==len(inputs_q)
        assert len(words_q)==512 and [r['index'] for r in words_q]==list(range(512))
        assert [r['data'] for r in words_q]==sample['abi_words']
        qdone=e['DONE'][0];root=e['ROOT_ACK'][0];qp=e['QUANT_PUBLIC'][0];pc=e['PROJECTION_COMMAND'][0]
        assert e['START'][0]['cycle']>e['ROOT_PUBLIC'][0]['cycle'] and qdone['status']==0 and qdone['words']==512
        assert all(e['START'][0]['cycle']<=r['cycle']<=qdone['cycle'] for r in reads_q+inputs_q+words_q)
        assert (root['sequence'],root['epoch'])==(seq,epoch) and root['cycle']>qdone['cycle']
        assert root['cycle']<pc['cycle']
        assert (qp['status'],qp['hls'],qp['sequence'],qp['epoch'],qp['records'],qp['valid'],qp['address'],qp['bytes'])==(0,0,seq,epoch,64,1,ABASE+4032,4096)
        assert (pc['sequence'],pc['epoch'],pc['address'])==(seq,epoch,ABASE+4032) and pc['cycle']>qp['cycle']
        for kind in ('QAW','QW','QB'):
            assert len(e[kind])==64 and [r['record'] for r in e[kind]]==list(range(64))
        for index in range(64):
            aw=e['QAW'][index];w=e['QW'][index];reply=e['QB'][index]
            assert aw['address']==ABASE+4032+index*64 and w['data']==sample['activation'][index] and reply['response']==0
            assert root['cycle']<min(aw['cycle'],w['cycle']) and reply['cycle']>max(aw['cycle'],w['cycle'])
            if index:assert e['QB'][index-1]['cycle']<min(aw['cycle'],w['cycle'])
        assert qp['cycle']>e['QB'][-1]['cycle']
        assert e['QACK'][0]['cycle']==e['PACK'][0]['cycle']==e['RELEASE'][0]['cycle']>end['cycle']
        for kind in ('QACK','PACK'):assert (e[kind][0]['sequence'],e[kind][0]['epoch'])==(seq,epoch)
        ng=64 if mode==0 else 0
        assert len(e['GROUP'])==len(e['GROUP_DONE'])==ng
        assert len(e['TILE'])==(128 if mode==0 else 0)
        for index,row in enumerate(e['GROUP']):
            assert (row['group'],row['tag'])==(index,tag+index)
            assert matrix['cycle']<row['cycle']<done['cycle']
            if index:assert row['cycle']>e['GROUP_DONE'][index-1]['cycle']
        for index,row in enumerate(e['TILE']):
            assert (row['tile'],row['address'])==(index,WBASE+4032+index*34816)
            assert e['GROUP'][index//2]['cycle']<row['cycle']<e['GROUP_DONE'][index//2]['cycle']
        for index,row in enumerate(e['GROUP_DONE']):
            assert (row['group'],row['status'],row['bytes'])==(index,0,128)
            assert e['GROUP'][index]['cycle']<row['cycle']<done['cycle']
        expected_ar=[]
        for group in range(ng):
            for address,count in [(ABASE+4032,4096),(WBASE+4032+group*69632,34816),(WBASE+4032+group*69632+34816,34816)]:
                expected_ar += [(address,beats,group) for address,beats in bursts(address,count)]
        ars=e['AR'];reads=e['R'];cursor=0
        if mode==0:assert len(ars)==1344 and len(reads)==73728
        if mode==1:assert not ars and not reads
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
                if address>=ABASE:
                    qi=(r['address']-ABASE-4032)//64
                    assert r['data']==e['QW'][qi]['data'] and e['QB'][qi]['cycle']<r['cycle']<e['QACK'][0]['cycle']

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
        assert not e['CANCEL']
        bus_cycles=sorted({r['cycle'] for k in ('AR','R','AW','W','B') for r in e[k]})
        start=pc['cycle']+1;stop=done['cycle']
        assert all(start<=c<=stop for c in bus_cycles)
        gaps=([bus_cycles[0]-start]+[b-a-1 for a,b in zip(bus_cycles,bus_cycles[1:])]+[stop-bus_cycles[-1]]) if bus_cycles else [stop-start+1]
        assert end['max_quiet']==max(gaps)<65536,'Watchdog gap differs from trace'
        for key,kind in [('ar','AR'),('r','R'),('aw','AW'),('w','W'),('b','B'),('results','MAC'),('groups','GROUP'),('tiles','TILE')]:
            assert end[key]==len(e[kind]),(j,key)
            if key in totals:totals[key]+=len(e[kind])
        metrics.append(dict(job=j,mode=mode,sample=sample_id,completion_cycles_including_hold=end['cycle']-begin['cycle'],
                            root=root_metrics,matrix_cycles=stop-matrix['cycle']+1,max_quiet_cycles=max(gaps),read_bytes=len(reads)*64,
                            written_bytes=len(e['W'])*64,published_bytes=end['bytes'],fp64_results=len(mac)))
    for key,total in totals.items():assert summary[key]==total,(key,summary[key],total)
    assert (totals['aw'],totals['w'],totals['b'])==(132,256,132)
    assert all(summary[name]>0 for name in ('ar_stalls','aw_stalls','w_stalls'))
    assert summary['cycles']==buckets[3]['RELEASE'][0]['cycle']
    for key,kind in [('qreads','READ'),('qwords','WORD'),('root','ROOT_ACK'),('qacks','QACK'),('packs','PACK'),('qaw','QAW'),('qw','QW'),('qb','QB')]:
        assert summary[key]==sum(len(e[kind]) for e in buckets.values()),key
    assert (summary['qwords'],summary['root'],summary['qacks'],summary['packs'],summary['qaw'],summary['qw'],summary['qb'])==(1024,3,2,2,128,128,128)
    for key,kind in [('heads','HEAD'),('copies','COPY'),('qbeats','Q'),('binds','BIND_DONE'),('appends','APPEND_DONE'),
                     ('kar','KAR'),('kr','KR'),('kaw','KAW'),('kw','KW'),('kb','KB'),('outacks','OUT_ACK')]:
        assert summary[key]==sum(len(e[kind]) for e in buckets.values()),key
    assert tuple(summary[k] for k in ('heads','copies','qbeats','binds','appends','kar','kr','kaw','kw','kb','outacks'))==(64,4096,264,3,12,160,1280,204,768,204,2)
    return dict(seed=seed,jobs=3,successful_projections=2,fp64_results=totals['results'],fp32_output_bytes=16384,
                totals=totals,root_summary={k:summary[k] for k in ('heads','copies','qbeats','binds','appends','kar','kr','kaw','kw','kb','outacks')},quantization_summary={k:summary[k] for k in ('qreads','qwords','root','qacks','packs','qaw','qw','qb')},jobs_metrics=metrics,cycles=summary['cycles'],clock_ns=5,
                scope='Actual root reads committed KV, emits 4096 exact FP32 values; quantizer commits ABI1 consumed by full O GEMV. Two complete outputs, Q-input cancellation and recovery. No physical DDR or complete inference qualification.')


def review(text,fixture,seed=1):return review_events(parse(text),fixture,seed)


CHANGES=[('BEGIN','sequence',1<<40,1),('BEGIN','epoch',1<<40,1),('MATRIX','tag',1<<32,3),
 ('GROUP','tag',1<<32,1),('TILE','address',64,1),('AR','address',64,1),('R','data',1,1),('R','last',1,1),
 ('MAC','data',1,1),('AW','address',64,1),('W','data',1,1),('B','response',2,1),
 ('COMPLETE','max_quiet',1,1),('COMPLETE','valid',1,3),('PROJECTION_COMMAND','address',64,1),
 ('READ','index',1,1),('INPUT','data',1,1),('WORD','data',1,1),('QAW','address',64,1),('QW','data',1,1),
 ('QB','response',2,1),('ROOT_ACK','sequence',1<<40,1),('QUANT_PUBLIC','bytes',1,1),('QUANT_PUBLIC','records',1,1),
 ('QACK','sequence',1<<40,1),('PACK','epoch',1<<44,1),('RELEASE','cycle',1,1),('PASS','results',1,None)]
CHANGES += [('COMMAND','sequence',1<<40,1),('COMMAND','lease',1<<32,3),('Q','data',1,1),
 ('BIND','capacity',1,1),('BIND','base',64,1),('APPEND','position',1,1),('DATA','data',1,1),
 ('APPEND_DONE','committed',1,1),('KAW','address',64,1),('KAW','beats',1,1),('KW','data',1,1),('KW','last',1,1),
 ('KB','response',2,1),('KAR','address',64,1),('KAR','head',1,1),('KR','data',1,1),('KR','last',1,1),
 ('HEAD','lease',1<<32,3),('HEAD','epoch',1<<40,1),('COPY','data',1,1),('COPY','head',1,1),
 ('HEAD_DONE','reads',1,1),('ROOT_PUBLIC','valid',1,2),('CANCEL','qbeats',1,2),('OUT_ACK','cycle',1,1),('PASS','kr',1,None)]
MISSING=[('ROOT_ACK',1),('QUANT_PUBLIC',1),('PROJECTION_COMMAND',1),('PACK',1),('R',1),('DATA',1),('KB',1),('KR',1),('HEAD_DONE',1),('ROOT_PUBLIC',2)]
CONTROL_COUNT=len(CHANGES)+len(MISSING)+5

def controls(text,fixture,seed=1):
    events=parse(text);review_events(events,fixture,seed);rejected=[]
    for kind,key,mask,job in CHANGES:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and (job is None or r.get('job')==job))
        altered=events.copy();row=dict(events[index][1]);row[key]^=mask;altered[index]=(kind,row)
        try:review_events(altered,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'{kind}.{key}.job{job}')
        else:raise AssertionError('Mutation accepted: '+kind+'.'+key)
    for kind,job in MISSING:
        index=next(i for i,(k,r) in enumerate(events) if k==kind and r.get('job')==job)
        altered=events[:index]+events[index+1:]
        try:review_events(altered,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(f'missing-{kind}.job{job}')
        else:raise AssertionError('Missing event accepted: '+kind)
    for label,bad in [('fatal','Fatal: injected diagnostic\n'),('unknown','PRREAL_UNKNOWN field=1\n'),('xz','PRREAL_MATRIX cycle=1 job=1 tag=xx\n')]:
        try:parse(bad)
        except (AssertionError,ValueError):rejected.append('parser-'+label)
        else:raise AssertionError('Bad parser input accepted')
    index=next(i for i,(k,r) in enumerate(events) if k=='APPEND_DONE' and r['job']==1)
    prior=max(i for i,(k,r) in enumerate(events[:index]) if k=='KB' and r['job']==1)
    assert events[index][1]['cycle']==events[prior][1]['cycle']
    premature=events.copy();kind,row=events[index];row=dict(row);row['cycle']-=1;premature[index]=(kind,row)
    reordered=events.copy();row=reordered.pop(index);reordered.insert(prior,row)
    for label,altered in [('append-before-final-B-cycle',premature),('append-before-final-B-event',reordered)]:
        try:review_events(altered,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Premature append accepted: '+label)
    assert len(rejected)==CONTROL_COUNT==len(CHANGES)+len(MISSING)+5
    return rejected
