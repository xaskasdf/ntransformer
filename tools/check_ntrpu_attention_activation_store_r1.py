"""Review source bytes, actual AXI transactions and atomic activation publication."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SCHEMA={
    'CONFIG':'seed exhaustion watchdog','RESET':'cycle job count',
    'BEGIN':'cycle job mode sample sequence epoch address tag',
    'READ':'cycle job index data','OFFER':'cycle job record tag',
    'WRITER':'cycle job record status bytes tag','AW':'cycle job record address',
    'W':'cycle job record data','B':'cycle job record response id',
    'ACK':'cycle job sequence epoch','FOREIGN':'cycle job mode','HELD':'cycle job mode',
    'RESTORE':'cycle job mode','ABORT':'cycle job mode aw w b','REVOKED':'cycle job mode',
    'COMPLETE':'cycle job mode status sequence epoch records valid address bytes aw w b',
    'PASS':'seed exhaustion jobs publics acks resets reads offers replies aw w b aw_stalls w_stalls held aw_first w_first cycles',
}
STATUS={1:2,2:2,3:2,4:3,5:11,6:1,7:1,8:1,9:1,10:7,11:2,12:11,13:4,
        14:4,15:4,16:4,17:3,18:3,19:8,20:8,21:8,22:7,23:2,24:7,33:11,34:9}
READS={**dict.fromkeys(range(1,10),0),10:20,11:20,12:20,13:18,14:30,15:25,16:29,
       17:64,18:64,**dict.fromkeys(range(19,25),8),27:20,28:8,29:8,30:8,32:8,34:0}


def load_fixture(folder):
    folder=Path(folder)
    raw=(folder/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'
    manifest=json.loads(raw);samples=[]
    for n in range(1,5):
        name=f'sample-{n:03d}-attention-abi1.bin';data=(folder/name).read_bytes()
        assert len(data)==4096 and hashlib.sha256(data).hexdigest()==manifest['files'][name]
        samples.append(data)
    return samples


def parse(text):
    assert not re.search(r'(?mi)^\s*(FATAL|Fatal:|ERROR|ASTORE_FAIL)',text),'Executed failure'
    events=[]
    for line in text.splitlines():
        if not line.startswith('ASTORE_'):continue
        tag,*fields=line.removeprefix('ASTORE_').split()
        assert tag in SCHEMA,'Unknown event'
        pairs=[f.split('=') for f in fields]
        assert all(len(p)==2 for p in pairs) and [p[0] for p in pairs]==SCHEMA[tag].split(),'Event schema'
        row={}
        for name,value in pairs:
            hexadecimal=name in ('sequence','epoch','address','tag','data')
            width=128 if tag=='W' and name=='data' else 16
            assert re.fullmatch('[0-9a-f]{'+str(width)+'}' if hexadecimal else '[0-9]+',value),'Field width'
            row[name]=int(value,16 if hexadecimal else 10)
        events.append((tag,row))
    return events


def review(text,samples,seed=1,exhaustion=0):
    events=parse(text);counts=Counter(t for t,_ in events)
    assert events[0]==('CONFIG',dict(seed=seed,exhaustion=exhaustion,watchdog=64))
    assert counts['CONFIG']==counts['PASS']==1 and events[-1][0]=='PASS'
    summary=events[-1][1];assert (summary['seed'],summary['exhaustion'])==(seed,exhaustion)
    aliases=dict(jobs='BEGIN',publics='COMPLETE',acks='ACK',resets='RESET',reads='READ',offers='OFFER',replies='WRITER',aw='AW',w='W',b='B')
    for k,tag in aliases.items():assert summary[k]==counts[tag],('Counter',k)
    assert tuple(summary[k] for k in ('jobs','publics','acks','resets','reads','offers','replies','aw','w','b'))==(
        (4,4,4,2,1024,128,128,128,128,128) if exhaustion else (70,64,63,7,21382,2669,2666,2668,2668,2666))
    assert summary['held']==(28 if exhaustion else 452)
    assert all(summary[k]>0 for k in ('aw_stalls','w_stalls','aw_first','w_first'))
    rows=events[1:-1];assert rows[0]==('RESET',dict(cycle=1,job=0,count=1))
    times=[v['cycle'] for _,v in rows];assert times==sorted(times) and summary['cycles']>times[-1]
    plan=[0,34,34,0] if exhaustion else [0]*4+[m for fault in range(1,34) for m in (fault,0)]
    begins=[i for i,(t,_) in enumerate(rows) if t=='BEGIN'];assert len(begins)==len(plan) and begins[0]==1
    first_tag=0xffffffffffffffc0 if exhaustion else 0x00000000ffffff80
    next_tag=first_tag;reset_count=1;published=[];metrics=[]
    for job,(offset,mode) in enumerate(zip(begins,plan),1):
        batch=rows[offset:begins[job] if job<len(plan) else len(rows)]
        assert all(v['job']==job for _,v in batch),'Cross-owner event'
        sequence=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        address=0x2200000fc0+job*8192;sample=(job-1)%4;begin=batch[0][1]
        assert tuple(begin[k] for k in ('mode','sample','sequence','epoch','address','tag'))==(mode,sample,sequence,epoch,address,next_tag)
        tag_base=next_tag
        if (mode==0 or mode>=6) and mode!=34:next_tag=(next_tag+64)&((1<<64)-1)
        seen=Counter();last={};successful=0;writes=[];b_done={};aw_done={};w_done={}
        for tag,row in batch[1:]:
            if 'mode' in row:assert row['mode']==mode
            if tag=='READ':
                index=seen[tag];assert row['index']==index and index<READS.get(mode,512),'Source extent/order'
                want=int.from_bytes(samples[sample][index*8:index*8+8],'little')
                if mode==14 and index==29:want=1
                elif mode==15 and index==24:want=(want&~255)|128
                elif mode==16 and index==28:want=0x8000000000000000
                assert row['data']==want,'Source ABI bytes'
                assert not seen['ACK'] and not seen['COMPLETE']
            elif tag=='OFFER':
                index=seen[tag];assert row['record']==index and seen['READ']==(index+1)*8
                assert row['tag']==tag_base+index,'Reserved writer identity'
                assert seen['WRITER']==index and not seen['ACK']
            elif tag in ('AW','W'):
                index=seen[tag];assert row['record']==index and seen['OFFER']==index+1 and seen['B']==index
                assert not seen['ACK'],'Source released before AXI'
                if tag=='AW':
                    assert row['address']==address+index*64 and row['address']%64==0,'AXI address'
                    assert row['address']%4096+64<=4096;aw_done[index]=row['cycle']
                else:
                    value=row['data'].to_bytes(64,'little')
                    assert value==samples[sample][index*64:index*64+64],'Actual AXI byte order'
                    writes.append(value);w_done[index]=row['cycle']
            elif tag=='B':
                index=seen[tag];assert row['record']==index and seen['AW']==seen['W']==index+1
                assert row['response']==(2 if mode==17 and index==7 else 0)
                assert row['id']==(15 if mode==18 and index==7 else 0)
                assert row['cycle']>max(aw_done[index],w_done[index]);b_done[index]=row['cycle']
            elif tag=='WRITER':
                index=seen[tag];assert row['record']==index and seen['OFFER']==index+1 and seen['B']==index+1
                status=3 if mode in (17,18) and index==7 else 7 if mode in (*range(19,25),32) else 0
                assert (row['tag'],row['status'],row['bytes'])==(tag_base+index,status,64),'Writer completion'
                assert row['cycle']>b_done[index];successful+=int(status==0)
            elif tag=='ACK':
                assert not seen[tag] and mode not in (1,2,27,28,29,30,32)
                assert row['sequence']==sequence and row['epoch']==epoch,'Source release identity'
                assert seen['OFFER']==seen['WRITER'] and seen['AW']==seen['W']==seen['B'],'Source release before drain'
                if seen['WRITER']:assert row['cycle']>last['WRITER']['cycle']
                if mode==33:assert seen['RESTORE']==1 and row['cycle']>last['RESTORE']['cycle']
            elif tag=='COMPLETE':
                assert not seen[tag] and not 27<=mode<=32
                assert seen['ACK']==int(mode not in (1,2))
                assert (row['status'],row['sequence'],row['epoch'],row['records'])==(STATUS.get(mode,0),sequence,epoch,successful)
                valid=int(STATUS.get(mode,0)==0 and mode not in (25,26))
                assert (row['valid'],row['address'],row['bytes'])==(valid,address if valid else 0,4096 if valid else 0),'Region publication'
                assert (row['aw'],row['w'],row['b'])==(seen['AW'],seen['W'],seen['B'])
                if valid:
                    assert successful==64 and b''.join(writes)==samples[sample]
                    published.append(dict(job=job,sample=sample,bytes=4096,sha256=hashlib.sha256(b''.join(writes)).hexdigest()))
            elif tag=='ABORT':
                assert not seen[tag] and 27<=mode<=32
                assert (row['aw'],row['w'],row['b'])==(seen['AW'],seen['W'],seen['B'])
            elif tag=='RESET':
                assert (27<=mode<=32 and seen['ABORT']==1) or (exhaustion and job==3 and seen['COMPLETE']==1)
                reset_count+=1;assert row['count']==reset_count;next_tag=first_tag
            elif tag in ('FOREIGN','HELD','RESTORE'):
                assert mode in (32,33) and not seen[tag]
                if tag=='FOREIGN':assert not seen['ACK']
                elif tag=='HELD':assert seen['FOREIGN']==1 and row['cycle']>=last['FOREIGN']['cycle']+32 and not seen['ACK']
                else:assert mode==33 and seen['HELD']==1
            elif tag=='REVOKED':
                assert mode in (25,26) and not seen[tag] and seen['ACK']==1 and seen['WRITER']==64
            else:raise AssertionError('Unexpected per-job event '+tag)
            seen[tag]+=1;last[tag]=row
        assert seen['READ']==READS.get(mode,512),'Rejected request source activity/extent'
        offers=seen['READ']//8
        if mode==15:offers=3
        assert seen['OFFER']==offers,'Assembled record count'
        assert seen['WRITER']==offers-int(mode in (28,29,30))
        assert seen['AW']==offers-int(mode==29) and seen['W']==offers-int(mode==28)
        assert seen['B']==offers-int(mode in (28,29,30))
        assert seen['COMPLETE']==int(not 27<=mode<=32) and seen['ABORT']==int(27<=mode<=32)
        assert seen['ACK']==int(mode not in (1,2,27,28,29,30,32))
        assert seen['REVOKED']==int(mode in (25,26))
        assert seen['FOREIGN']==seen['HELD']==int(mode in (32,33)) and seen['RESTORE']==int(mode==33)
        assert seen['RESET']==int(27<=mode<=32 or (exhaustion and job==3))
        metrics.append(dict(job=job,mode=mode,source_words=seen['READ'],axi_responses=seen['B'],cycles=batch[-1][1]['cycle']-begin['cycle']))
    assert reset_count==(2 if exhaustion else 7) and len(published)==(2 if exhaustion else 37)
    return dict(seed=seed,exhaustion=exhaustion,summary=summary,jobs=len(plan),published_regions=published,
        published_bytes=len(published)*4096,job_metrics=metrics,actual_axi_writer=True,
        generated_quantizer_integrated=False,physical_memory_qualified=False)


def controls(text,samples,seed=1,exhaustion=0):
    variants={}
    for tag in ('READ','OFFER','WRITER','AW','W','B','ACK','COMPLETE','RESET','ABORT','REVOKED','FOREIGN','HELD','RESTORE'):
        changed,n=re.subn(r'(?m)^ASTORE_'+tag+r' [^\n]*\n','',text,count=1)
        if n:variants['missing_'+tag]=changed
    for tag,field in [('BEGIN','sequence'),('BEGIN','epoch'),('BEGIN','tag'),('READ','index'),('READ','data'),
                      ('AW','address'),('W','data'),('B','response'),('WRITER','tag'),('WRITER','bytes'),
                      ('ACK','epoch'),('ACK','cycle'),('COMPLETE','records'),('COMPLETE','valid'),('COMPLETE','address'),('PASS','jobs')]:
        pattern=rf'(?m)^(ASTORE_{tag} .*?\b{field}=)([0-9a-f]+)'
        def mutate(m):
            base=16 if field in ('sequence','epoch','tag','data','address') else 10
            value=int(m[2],base)^1
            return m[1]+(f'{value:0{len(m[2])}x}' if base==16 else str(value))
        changed,n=re.subn(pattern,mutate,text,count=1);assert n==1
        variants[tag+'_'+field]=changed
    variants['diagnostic']=text+'\nASTORE_FAIL injected\n'
    variants['duplicate_pass']=text+next(l for l in text.splitlines() if l.startswith('ASTORE_PASS'))+'\n'
    rejected=[]
    for name,changed in variants.items():
        assert changed!=text
        try:review(changed,samples,seed,exhaustion)
        except (AssertionError,IndexError,KeyError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted corrupt '+name)
    return rejected
