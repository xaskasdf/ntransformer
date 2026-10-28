"""Review actual QKV AXI bytes, arithmetic, phase ownership and retirement."""
import hashlib
import json
from pathlib import Path
import re

MANIFEST_SHA='f5b4413ece6c6bd9da2fa772fab69502eef3ce1c39d802afe4d1c44087af9626'
ABASE=0x8000000000
FIRST_TAG=0xffffffc0
PLAN=[(0,0,0),(0,1,0),(0,2,0),(3,3,0),(1,0,0),(2,0,0),(2,1,1),(2,2,2),(0,3,0)]
SCHEMA={
 'CONFIG':'seed q_rows kv_rows k_blocks tile_k watchdog phase_limit',
 'BEGIN':'cycle job mode sample target sequence epoch tag',
 'MATRIX':'cycle job phase tag','GROUP':'cycle job phase group tag','TILE':'cycle job phase tile address',
 'AR':'cycle job phase ar address beats group','R':'cycle job phase ar beat address data last',
 'MAC':'cycle job phase row data','AW':'cycle job phase aw address beats','W':'cycle job phase word data last',
 'B':'cycle job phase aw response','GROUP_DONE':'cycle job phase group status bytes',
 'MATRIX_DONE':'cycle job phase status bytes rows tag groups tiles results ar r aw w b elapsed',
 'MATRIX_ACK':'cycle job phase public','CHILD_CANCEL':'cycle job phase','CANCEL':'cycle job phase reads',
 'REVOKE':'cycle job','COMPLETE':'cycle job mode matrices good results ar r aw w b elapsed',
 'PUBLIC':'cycle job status sequence epoch valid revoked matrices bytes','SOURCE_ACK':'cycle job','END':'cycle job',
 'PASS':'seed jobs publics source_acks cancels matrices groups results ar r aw w b ar_stalls r_stalls aw_stalls w_stalls cycles'}
HEX={'sequence','epoch','tag','address','data'}


def weight_base(phase):return 0x100000000+phase*0x20000000
def output_base(phase):return 0xa000000000+phase*0x10000000
def rows_for(phase):return 2048 if phase==0 else 512


def load_fixture(folder):
    folder=Path(folder)
    if (folder/'raw').is_dir():folder=folder/'raw'
    raw=(folder/'manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==MANIFEST_SHA
    manifest=json.loads(raw)
    for name,sha in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
    def words(name,width):
        raw=(folder/name).read_bytes();assert len(raw)%width==0
        return [int.from_bytes(raw[i:i+width],'little') for i in range(0,len(raw),width)]
    names=('attn_q','attn_k','attn_v')
    return dict(weights=[words(n+'-weights.bin',64) for n in names],samples=[dict(
        activation=words(f'sample-{i:03d}-input-abi1.bin',64),
        fp64=[words(f'sample-{i:03d}-{n}-fp64.bin',8) for n in names],
        fp32=[words(f'sample-{i:03d}-{n}-fp32.bin',64) for n in names]) for i in range(1,5)])


def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|ERROR:|QKVM_FAIL)',text),'Simulator failure'
    events=[]
    for line in text.splitlines():
        if not line.startswith('QKVM_'):continue
        fields=line.split();kind=fields[0][5:];assert kind in SCHEMA,kind
        pairs=[x.split('=') for x in fields[1:]]
        assert all(len(p)==2 for p in pairs) and [p[0] for p in pairs]==SCHEMA[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in pairs};assert all(v>=0 for v in row.values())
        events.append((kind,row))
    return events


def bursts(address,count):
    result=[]
    while count:
        beats=min(count//64,(4096-address%4096)//64);assert 0<beats<=64
        result.append((address,beats));address+=beats*64;count-=beats*64
    return result


def phase_review(e,fixture,sample,phase,tag,cancelled):
    for kind in ('MATRIX','MATRIX_DONE','MATRIX_ACK'):assert len(e[kind])==1,(phase,kind)
    start=e['MATRIX'][0];done=e['MATRIX_DONE'][0];ack=e['MATRIX_ACK'][0]
    rows=rows_for(phase);ng=1 if cancelled else rows//32;wb=weight_base(phase);ob=output_base(phase)
    assert start['tag']==done['tag']==tag and start['cycle']<done['cycle']<=ack['cycle']
    assert done['elapsed']==done['cycle']-start['cycle']<16777216
    assert (done['status'],done['bytes'],done['rows'])==((7,0,0) if cancelled else (0,rows*4,rows))
    assert len(e['GROUP'])==len(e['GROUP_DONE'])==ng
    assert len(e['TILE'])==(1 if cancelled else ng*2)
    for i,g in enumerate(e['GROUP']):
        gd=e['GROUP_DONE'][i]
        assert (g['group'],g['tag'])==(i,tag+i)
        assert (gd['group'],gd['status'],gd['bytes'])==(i,7 if cancelled else 0,0 if cancelled else 128)
        assert start['cycle']<g['cycle']<gd['cycle']<done['cycle']
        if i:assert e['GROUP_DONE'][i-1]['cycle']<g['cycle']
    for i,t in enumerate(e['TILE']):
        assert (t['tile'],t['address'])==(i,wb+4032+i*34816)
        assert e['GROUP'][i//2]['cycle']<t['cycle']<e['GROUP_DONE'][i//2]['cycle']
    expected=[]
    for g in range(ng):
        for addr,count in [(ABASE+4032,4096),(wb+4032+g*69632,34816),(wb+4032+g*69632+34816,34816)]:
            expected += [(a,n,g) for a,n in bursts(addr,count)]
    if cancelled:
        assert 3<=len(e['AR'])<=21 and 70<=len(e['R'])<=1152
    else:
        assert len(e['AR'])==ng*21 and len(e['R'])==ng*1152
    cursor=0;reads=e['R']
    for i,ar in enumerate(e['AR']):
        address,count,g=expected[i]
        assert (ar['ar'],ar['address'],ar['beats'],ar['group'])==(i,address,count,g)
        assert e['GROUP'][g]['cycle']<ar['cycle']<e['GROUP_DONE'][g]['cycle']
        if i:assert reads[cursor-1]['cycle']<ar['cycle']
        if address>=ABASE:
            values=sample['activation'];offset=(address-ABASE-4032)//64
        else:
            values=fixture['weights'][phase];offset=(address-wb)//64;tile=(address-wb-4032)//34816
            assert e['TILE'][tile]['cycle']<ar['cycle']
        for beat in range(count):
            assert cursor<len(reads)
            row=reads[cursor];cursor+=1
            assert (row['ar'],row['beat'],row['address'],row['data'],row['last'])==(i,beat,address+64*beat,values[offset+beat],int(beat==count-1))
            assert ar['cycle']<row['cycle']<e['GROUP_DONE'][g]['cycle']
            if beat:assert reads[cursor-2]['cycle']<row['cycle']
    assert cursor==len(reads)
    mac=e['MAC'];assert len(mac)==(0 if cancelled else rows)
    for i,row in enumerate(mac):
        assert (row['row'],row['data'])==(i,sample['fp64'][phase][i])
        assert reads[(i//32+1)*1152-1]['cycle']<row['cycle']<e['GROUP_DONE'][i//32]['cycle']
        if i:assert mac[i-1]['cycle']<row['cycle']
    expected=[]
    if not cancelled:
        for g in range(ng):expected += [(a,n,g) for a,n in bursts(ob+4032+g*128,128)]
    assert len(e['AW'])==len(e['B'])==len(expected)
    assert len(e['W'])==(0 if cancelled else rows//16)
    cursor=0
    for i,(address,count,g) in enumerate(expected):
        aw=e['AW'][i];b=e['B'][i]
        assert (aw['aw'],aw['address'],aw['beats'])==(i,address,count)
        assert (b['aw'],b['response'])==(i,0)
        assert mac[(g+1)*32-1]['cycle']<aw['cycle']<b['cycle']<e['GROUP_DONE'][g]['cycle']
        if i:assert e['B'][i-1]['cycle']<aw['cycle']
        for beat in range(count):
            row=e['W'][cursor]
            assert (row['word'],row['data'],row['last'])==(cursor,sample['fp32'][phase][cursor],int(beat==count-1))
            assert mac[(g+1)*32-1]['cycle']<row['cycle']<b['cycle']
            if cursor:assert e['W'][cursor-1]['cycle']<row['cycle']
            if i:assert e['B'][i-1]['cycle']<row['cycle']
            cursor+=1
    assert len(e['CANCEL'])==len(e['CHILD_CANCEL'])==int(cancelled)
    if cancelled:
        cancel=e['CANCEL'][0];child=e['CHILD_CANCEL'][0]
        assert cancel['reads']==sum(r['cycle']<=cancel['cycle'] for r in reads)>=70
        assert cancel['cycle']<child['cycle']<=done['cycle'] and reads[-1]['cycle']>cancel['cycle']
    counters={k:len(e[v]) for k,v in [('groups','GROUP'),('tiles','TILE'),('results','MAC'),('ar','AR'),('r','R'),('aw','AW'),('w','W'),('b','B')]}
    assert all(done[k]==v for k,v in counters.items())
    return dict(phase=phase,status=done['status'],cycles=done['elapsed'],**counters,
                read_bytes=len(reads)*64,written_bytes=len(e['W'])*64)


def review_events(events,fixture,seed=1):
    assert events[0]==('CONFIG',dict(seed=seed,q_rows=2048,kv_rows=512,k_blocks=64,tile_k=32,watchdog=65536,phase_limit=16777216))
    assert events[-1][0]=='PASS' and sum(k=='CONFIG' for k,_ in events)==sum(k=='PASS' for k,_ in events)==1
    buckets={i:[] for i in range(1,10)};active=0;last=0;last_cycle=0
    for kind,row in events:
        assert kind in SCHEMA and set(row)==set(SCHEMA[kind].split())
        if 'cycle' in row:assert row['cycle']>=last_cycle;last_cycle=row['cycle']
        if kind=='BEGIN':assert not active and row['job']==last+1;active=row['job'];last=active
        if 'job' in row:
            assert row['job']==active and active in buckets;buckets[active].append((kind,row))
        if kind=='END':active=0
    assert not active and last==9
    metrics=[];next_tag=FIRST_TAG
    totals={k:0 for k in ('ar','r','aw','w','b','results','groups','matrices')}
    phase_kinds=[k for k,s in SCHEMA.items() if 'phase' in s.split()]
    for job,(mode,sample_id,target) in enumerate(PLAN,1):
        pairs=buckets[job];e={k:[r for t,r in pairs if t==k] for k in SCHEMA}
        for k in ('BEGIN','COMPLETE','PUBLIC','SOURCE_ACK','END'):assert len(e[k])==1,(job,k)
        begin=e['BEGIN'][0];end=e['COMPLETE'][0];public=e['PUBLIC'][0];src=e['SOURCE_ACK'][0];retired=e['END'][0]
        seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        assert (begin['mode'],begin['sample'],begin['target'],begin['sequence'],begin['epoch'],begin['tag'])==(mode,sample_id,target,seq,epoch,next_tag)
        n=0 if mode==1 else target+1 if mode==2 else 3
        good=target if mode==2 else n
        assert len(e['MATRIX'])==len(e['MATRIX_DONE'])==len(e['MATRIX_ACK'])==n
        assert [r['phase'] for r in e['MATRIX']]==[r['phase'] for r in e['MATRIX_DONE']]==[r['phase'] for r in e['MATRIX_ACK']]==list(range(n))
        for kind in phase_kinds:assert all(0<=r['phase']<n for r in e[kind])
        phases=[]
        for phase in range(n):
            pe={k:[r for r in e[k] if r['phase']==phase] for k in phase_kinds}
            phases.append(phase_review(pe,fixture,fixture['samples'][sample_id],phase,next_tag+phase*64,mode==2 and phase==target))
            start=pe['MATRIX'][0];done=pe['MATRIX_DONE'][0];ack=pe['MATRIX_ACK'][0]
            assert begin['cycle']<start['cycle'] and done['cycle']<end['cycle']
            assert ack['public']==int(phase==n-1)
            if phase==n-1:assert ack['cycle']==public['cycle']
            else:assert ack['cycle']<e['MATRIX'][phase+1]['cycle']
        if n:next_tag+=192
        expected_status=1 if mode==1 else 7 if mode==2 else 0
        expected_revoked=int(mode in (2,3))
        assert (public['status'],public['sequence'],public['epoch'],public['valid'],public['revoked'],public['matrices'],public['bytes'])==(
            expected_status,seq,epoch,int(mode==0),expected_revoked,good,[0,8192,10240,12288][good])
        assert end['mode']==mode and end['matrices']==n and end['good']==good
        assert begin['cycle']<end['cycle']<public['cycle']==src['cycle']<retired['cycle']
        assert end['elapsed']==end['cycle']-begin['cycle']
        assert len(e['REVOKE'])==int(mode==3)
        if mode==3:assert e['MATRIX_DONE'][-1]['cycle']<e['REVOKE'][0]['cycle']<end['cycle']
        counts={k:len(e[v]) for k,v in [('ar','AR'),('r','R'),('aw','AW'),('w','W'),('b','B'),('results','MAC'),('groups','GROUP'),('matrices','MATRIX')]}
        assert all(end[k]==v for k,v in counts.items() if k!='groups')
        for k,v in counts.items():totals[k]+=v
        metrics.append(dict(job=job,mode=mode,sample=sample_id,target=target,cycles=end['elapsed'],phases=phases,
                            read_bytes=counts['r']*64,written_bytes=counts['w']*64))
    summary=events[-1][1]
    assert all(summary[k]==v for k,v in totals.items())
    assert (summary['seed'],summary['jobs'],summary['publics'],summary['source_acks'],summary['cancels'])==(seed,9,9,9,3)
    assert (totals['matrices'],totals['groups'],totals['results'],totals['w'],totals['aw'],totals['b'])==(21,627,19968,1248,649,649)
    assert summary['cycles']==last_cycle
    for key in ('ar_stalls','aw_stalls','w_stalls'):assert 0<summary[key]<summary['cycles']
    assert 0<=summary['r_stalls']<summary['cycles']
    # RVALID gaps are visible in accepted beats; they do not prove RREADY stalls.
    prior_reads={};gaps=[]
    for kind,row in events:
        if kind!='R':continue
        key=(row['job'],row['phase'],row['ar'])
        if key in prior_reads:gaps.append(row['cycle']-prior_reads[key]-1)
        prior_reads[key]=row['cycle']
    assert gaps and max(gaps)>0 and min(gaps)==0
    read_coverage=dict(intraburst_gap_cycles=sum(gaps),max_intraburst_gap_cycles=max(gaps),
        nonconsecutive_beat_pairs=sum(g>0 for g in gaps),consecutive_beat_pairs=sum(g==0 for g in gaps),
        rvalid_backpressure_cycles=summary['r_stalls'],rvalid_backpressure_exercised=summary['r_stalls']>0,
        rvalid_backpressure_qualification_pending=summary['r_stalls']==0)
    return dict(seed=seed,jobs=9,summary=summary,metrics=metrics,read_coverage=read_coverage,clock_ns=5,physical_throughput_qualified=False,
                scope='Actual full layer-0 QKV with retained prepared ABI1 source. Live normalization, RoPE and complete inference remain pending.')


def review(text,fixture,seed=1):return review_events(parse(text),fixture,seed)


def controls_events(events,fixture,seed=1):
    cases=[]
    def changed(label,kind,field,value,phase=None,mode_job=None):
        index=next(i for i,(k,r) in enumerate(events) if k==kind and (phase is None or r.get('phase')==phase) and (mode_job is None or r.get('job')==mode_job))
        pair=events[index];row=dict(pair[1]);row[field]=value
        assert row!=pair[1];cases.append((label,index,(pair[0],row)))
    for phase in range(3):
        for kind,field,value in [('MATRIX','tag',1),('GROUP','tag',1),('TILE','address',0),('AR','address',0),
                                  ('R','data',1),('MAC','data',1),('AW','address',0),('W','data',1),
                                  ('B','response',2),('GROUP_DONE','bytes',0),('MATRIX_DONE','rows',1),('MATRIX_ACK','public',2)]:
            changed(f'{kind}_{field}_phase_{phase}',kind,field,value,phase)
    for field,value in [('sequence',1),('epoch',1),('valid',0),('revoked',1),('matrices',2),('bytes',12224)]:
        changed('public_'+field,'PUBLIC',field,value)
    changed('early_source_ack','SOURCE_ACK','cycle',0)
    changed('wrong_cancel_phase','CANCEL','phase',2,mode_job=6)
    changed('revoked_republished','PUBLIC','valid',1,mode_job=4)
    changed('summary_bytes','PASS','w',1)
    rejected=[]
    for label,index,pair in cases:
        copy=events.copy();copy[index]=pair
        try:review_events(copy,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted corruption: '+label)
    for kind in ('R','B','MATRIX_ACK','SOURCE_ACK','REVOKE','END','PASS'):
        index=next(i for i,(k,_) in enumerate(events) if k==kind)
        copy=events[:index]+events[index+1:]
        try:review_events(copy,fixture,seed)
        except (AssertionError,KeyError,IndexError):rejected.append('missing_'+kind)
        else:raise AssertionError('Accepted missing '+kind)
    return rejected


def controls(text,fixture,seed=1):return controls_events(parse(text),fixture,seed)
