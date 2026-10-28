"""Independently reconcile both source RAMs, accepted AXI writes and pair publication."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SCHEMA={
    'CONFIG':'seed watchdog','RESET':'cycle job count',
    'BEGIN':'cycle job mode sample sequence epoch raddr rbase rbytes qaddr qbase qbytes',
    'RREAD':'cycle job index data','QREAD':'cycle job index data',
    'AW':'cycle job record address','W':'cycle job record data','B':'cycle job record response id',
    'ACK':'cycle job sequence epoch','FOREIGN':'cycle job mode','HELD':'cycle job mode',
    'RESTORE':'cycle job mode','ABORT':'cycle job mode aw w b','REVOKED':'cycle job mode',
    'COMPLETE':'cycle job mode status sequence epoch rbeats qbeats valid raddr rbytes qaddr qbytes aw w b',
    'PASS':'seed jobs publics acks resets rreads qreads aw w b aw_stalls w_stalls held aw_first w_first cycles',
}
STATUS={1:2,2:2,3:2,4:3,5:11,**dict.fromkeys(range(6,19),1),19:7,20:2,21:11,
        **dict.fromkeys(range(22,28),4),**dict.fromkeys(range(28,32),3),32:8,33:8,34:8,
        35:7,36:2,37:7,38:7,46:11,48:4,49:4,50:7}
EXTENT={**dict.fromkeys(range(1,20),(0,0,0)),20:(20,0,1),21:(20,0,1),22:(18,0,1),23:(25,0,1),
        24:(2048,18,130),25:(2048,30,131),26:(2048,25,131),27:(2048,29,131),
        28:(16,0,1),29:(2048,0,128),30:(2048,8,129),31:(2048,512,192),
        32:(16,0,1),33:(16,0,1),34:(2048,512,192),35:(16,0,1),36:(16,0,1),37:(16,0,1),38:(2048,0,128),
        41:(20,0,1),42:(16,0,0),43:(16,0,0),44:(16,0,0),46:(16,0,1),47:(16,0,1),
        48:(2048,0,127),49:(2048,509,191),50:(16,0,1)}
RESET_MODES={41,42,43,44,45,47}


def load_fixture(folder):
    folder=Path(folder);raw=(folder/'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='5cb75b3ee030bc7233f0b49cdacdc48e50791c0896945d214524a883158e81f6'
    manifest=json.loads(raw)
    for name,sha in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha,name
    assert manifest['parent_manifest_sha256']=='73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e'
    parent=json.loads((folder/'parent-manifest.json').read_bytes())
    samples=[]
    for sample in range(1,5):
        pair={}
        for key,kind,size in [('r','residual-fp32',8192),('q','ffn-input-abi1',4096)]:
            name=f'sample-{sample:03d}-{kind}.bin';data=(folder/name).read_bytes()
            assert len(data)==size and hashlib.sha256(data).hexdigest()==parent['files'][name]
            pair[key]=data
        samples.append(pair)
    for key,name,width in [('r','residual32.hex',4),('q','abi64.hex',8)]:
        decoded=b''.join(int(line,16).to_bytes(width,'little') for line in (folder/name).read_text().splitlines())
        assert decoded==b''.join(sample[key] for sample in samples)
    return samples


def parse(text):
    assert not re.search(r'(?mi)^\s*(FATAL|Fatal:|ERROR|PSTORE_FAIL)',text),'Executed failure'
    events=[]
    for line in text.splitlines():
        if not line.startswith('PSTORE_'):continue
        tag,*fields=line.removeprefix('PSTORE_').split();assert tag in SCHEMA,'Unknown event'
        pairs=[f.split('=') for f in fields]
        assert all(len(p)==2 for p in pairs) and [p[0] for p in pairs]==SCHEMA[tag].split(),'Event schema'
        row={}
        for name,value in pairs:
            hexadecimal=name in ('sequence','epoch','address','data','raddr','qaddr','rbase','qbase') or (tag=='BEGIN' and name in ('rbytes','qbytes'))
            width=128 if tag=='W' and name=='data' else 8 if tag=='RREAD' and name=='data' else 16
            assert re.fullmatch('[0-9a-f]{'+str(width)+'}' if hexadecimal else '[0-9]+',value),'Field width'
            row[name]=int(value,16 if hexadecimal else 10)
        events.append((tag,row))
    return events


def descriptor(job,mode):
    r=0x2200000fc0+job*16384;q=0x3300000fc0+job*16384;max64=(1<<64)-1
    if mode==51:q=r+8192
    if mode==52:r=q+4096
    if mode==53:r=0xffffffffffffdfc0
    result=dict(raddr=r,rbase=r,rbytes=8192,qaddr=q,qbase=q,qbytes=4096)
    changes={6:dict(raddr=r+1),7:dict(rbytes=8191),8:dict(rbase=0xfffffffffffff000,rbytes=8192),
        9:dict(raddr=0xfffffffffffff040,rbase=0,rbytes=max64),10:dict(rbase=r+64),
        11:dict(qaddr=q+1),12:dict(qbytes=4095),13:dict(qbase=0xfffffffffffff800,qbytes=4096),
        14:dict(qaddr=0xfffffffffffff040,qbase=0,qbytes=max64),15:dict(qbase=q+64),
        16:dict(qaddr=r+4096,qbase=r,qbytes=8192),17:dict(rbytes=0),18:dict(qbytes=0)}
    result.update(changes.get(mode,{}))
    return result,r,q


def review(text,samples,seed=1):
    events=parse(text);counts=Counter(tag for tag,_ in events)
    assert events[0]==('CONFIG',dict(seed=seed,watchdog=64)) and events[-1][0]=='PASS'
    assert counts['CONFIG']==counts['PASS']==1
    summary=events[-1][1];assert summary['seed']==seed
    aliases=dict(jobs='BEGIN',publics='COMPLETE',acks='ACK',resets='RESET',rreads='RREAD',qreads='QREAD',aw='AW',w='W',b='B')
    for key,tag in aliases.items():assert summary[key]==counts[tag],('Counter',key)
    assert tuple(summary[k] for k in aliases)==(110,104,103,7,151847,33899,13722,13722,13720)
    assert summary['held']==732 and all(summary[k]>0 for k in ('aw_stalls','w_stalls','aw_first','w_first'))
    rows=events[1:-1];assert rows[0]==('RESET',dict(cycle=1,job=0,count=1))
    cycles=[row['cycle'] for _,row in rows]
    assert cycles==sorted(cycles) and summary['cycles']>cycles[-1]
    plan=[0]*4+[m for fault in range(1,54) for m in (fault,0)]
    begins=[i for i,(tag,_) in enumerate(rows) if tag=='BEGIN'];assert len(begins)==110 and begins[0]==1
    reset_count=1;published=[];metrics=[]
    for job,(offset,mode) in enumerate(zip(begins,plan),1):
        batch=rows[offset:begins[job] if job<len(plan) else len(rows)]
        assert all(row['job']==job for _,row in batch),'Foreign owner'
        sequence=0x1234567800000000+job;epoch=0xfedcba9800000000+job;sample=(job-1)%4
        desc,raddr,qaddr=descriptor(job,mode);begin=batch[0][1]
        assert {k:begin[k] for k in desc}==desc,'Exact admitted regions'
        assert (begin['mode'],begin['sample'],begin['sequence'],begin['epoch'])==(mode,sample,sequence,epoch)
        seen=Counter();last={};writes={};aw_cycles={};w_cycles={};good_r=0;good_q=0;committed={}
        rr,qr,bcount=EXTENT.get(mode,(2048,512,192))
        for tag,row in batch[1:]:
            if 'mode' in row:assert row['mode']==mode
            if tag in ('RREAD','QREAD'):
                plane='r' if tag=='RREAD' else 'q';width=4 if plane=='r' else 8
                index=seen[tag];assert row['index']==index and index<(rr if plane=='r' else qr),'Source extent/order'
                want=int.from_bytes(samples[sample][plane][index*width:index*width+width],'little')
                if plane=='r' and ((mode==23 and index==24) or (mode==48 and index==2047)):want=0x7f800000
                if plane=='q':
                    if mode==25 and index==29:want=1
                    if mode==26 and index==24:want=(want&~255)|128
                    if (mode==27 and index==28) or (mode==49 and index==508):want=1<<63
                assert row['data']==want,'Real source bytes'
                assert seen['B']==(index//16 if plane=='r' else 128+index//8),'Read before previous commit'
                assert not seen['ACK'] and not seen['COMPLETE']
            elif tag in ('AW','W'):
                index=seen[tag];assert row['record']==index and seen['B']==index and not seen['ACK']
                if index<128:assert seen['RREAD']==(index+1)*16 and seen['QREAD']==0
                else:assert seen['RREAD']==2048 and seen['QREAD']==(index-127)*8
                if tag=='AW':
                    address=raddr+index*64 if index<128 else qaddr+(index-128)*64
                    assert row['address']==address and address%64==0 and address%4096+64<=4096,'AXI address'
                    aw_cycles[index]=row['cycle']
                else:
                    data=row['data'].to_bytes(64,'little');plane='r' if index<128 else 'q';block=index if index<128 else index-128
                    assert data==samples[sample][plane][block*64:block*64+64],'Actual AXI bytes'
                    writes[index]=data;w_cycles[index]=row['cycle']
            elif tag=='B':
                index=seen[tag];assert row['record']==index and seen['AW']==seen['W']==index+1
                response=2 if (mode==28 and index==0) or (mode==30 and index==128) else 0
                identity=15 if (mode==29 and index==127) or (mode==31 and index==191) else 0
                assert (row['response'],row['id'])==(response,identity)
                assert row['cycle']>max(aw_cycles[index],w_cycles[index]),'Premature B'
                if response==0 and identity==0:
                    committed[index]=writes[index]
                    if index<128:good_r+=1
                    else:good_q+=1
            elif tag=='ACK':
                assert not seen[tag] and mode not in (1,2,41,42,43,44,47)
                assert (row['sequence'],row['epoch'])==(sequence,epoch)
                assert seen['AW']==seen['W']==seen['B'],'Early source release'
                if seen['B']:assert row['cycle']>last['B']['cycle']
                if mode==46:assert seen['RESTORE']==1 and row['cycle']>last['RESTORE']['cycle']
            elif tag=='COMPLETE':
                assert not seen[tag] and mode not in RESET_MODES
                assert seen['ACK']==int(mode not in (1,2))
                assert (row['status'],row['sequence'],row['epoch'],row['rbeats'],row['qbeats'])==(STATUS.get(mode,0),sequence,epoch,good_r,good_q)
                valid=int(STATUS.get(mode,0)==0 and mode not in (39,40))
                assert (row['valid'],row['raddr'],row['rbytes'],row['qaddr'],row['qbytes'])==(valid,raddr if valid else 0,8192 if valid else 0,qaddr if valid else 0,4096 if valid else 0),'Pair publication'
                assert (row['aw'],row['w'],row['b'])==(seen['AW'],seen['W'],seen['B'])
                if valid:
                    assert good_r==128 and good_q==64
                    assert b''.join(committed[i] for i in range(128))==samples[sample]['r']
                    assert b''.join(committed[i] for i in range(128,192))==samples[sample]['q']
                    published.append(dict(job=job,sample=sample,bytes=12288,
                        sha256=hashlib.sha256(b''.join(committed[i] for i in range(192))).hexdigest()))
            elif tag=='ABORT':
                assert not seen[tag] and mode in RESET_MODES
                assert (row['aw'],row['w'],row['b'])==(seen['AW'],seen['W'],seen['B'])
            elif tag=='RESET':
                assert mode in RESET_MODES and seen['ABORT']==1
                reset_count+=1;assert row['count']==reset_count
            elif tag in ('FOREIGN','HELD','RESTORE'):
                assert mode in (46,47) and not seen[tag]
                if tag=='FOREIGN':assert not seen['ACK']
                elif tag=='HELD':assert seen['FOREIGN']==1 and row['cycle']>=last['FOREIGN']['cycle']+32 and not seen['ACK']
                else:assert mode==46 and seen['HELD']==1
            elif tag=='REVOKED':assert mode in (39,40) and not seen[tag] and seen['ACK']==1 and seen['B']==192
            else:raise AssertionError('Unexpected event '+tag)
            seen[tag]+=1;last[tag]=row
        assert (seen['RREAD'],seen['QREAD'],seen['B'])==(rr,qr,bcount),'Scenario extent'
        assert seen['AW']==bcount+int(mode in (42,44)) and seen['W']==bcount+int(mode in (43,44))
        assert seen['COMPLETE']==int(mode not in RESET_MODES) and seen['ABORT']==seen['RESET']==int(mode in RESET_MODES)
        assert seen['ACK']==int(mode not in (1,2,41,42,43,44,47))
        assert seen['REVOKED']==int(mode in (39,40))
        assert seen['FOREIGN']==seen['HELD']==int(mode in (46,47)) and seen['RESTORE']==int(mode==46)
        metrics.append(dict(job=job,mode=mode,residual_words=rr,abi_words=qr,axi_responses=bcount,cycles=batch[-1][1]['cycle']-begin['cycle']))
    assert reset_count==7 and len(published)==60
    return dict(seed=seed,jobs=110,summary=summary,published_pairs=published,published_bytes=len(published)*12288,
                job_metrics=metrics,actual_axi_writer=True,post_attention_hls_integrated=False,physical_memory_qualified=False)


def controls(text,samples,seed=1):
    variants={}
    for tag in ('RREAD','QREAD','AW','W','B','ACK','COMPLETE','RESET','ABORT','REVOKED','FOREIGN','HELD','RESTORE'):
        changed,n=re.subn(r'(?m)^PSTORE_'+tag+r' [^\n]*\n','',text,count=1);assert n
        variants['missing_'+tag]=changed
    for tag,field in [('BEGIN','sequence'),('BEGIN','epoch')]+[('BEGIN',f) for f in ('raddr','rbase','rbytes','qaddr','qbase','qbytes')]+[
        ('RREAD','index'),('RREAD','data'),('QREAD','index'),('QREAD','data'),('AW','address'),('W','data'),
        ('B','response'),('B','id'),('ACK','epoch'),('ACK','cycle'),('COMPLETE','rbeats'),('COMPLETE','qbeats'),
        ('COMPLETE','valid'),('COMPLETE','raddr'),('COMPLETE','qaddr'),('COMPLETE','rbytes'),('COMPLETE','qbytes'),('PASS','jobs')]:
        pattern=rf'(?m)^(PSTORE_{tag} .*?\b{field}=)([0-9a-f]+)'
        def mutate(match):
            hexadecimal=field in ('sequence','epoch','data','address','raddr','qaddr','rbase','qbase') or (tag=='BEGIN' and field in ('rbytes','qbytes'))
            value=int(match[2],16 if hexadecimal else 10)^1
            if tag=='ACK' and field=='cycle':value=0
            return match[1]+(f'{value:0{len(match[2])}x}' if hexadecimal else str(value))
        changed,n=re.subn(pattern,mutate,text,count=1);assert n==1
        variants[tag+'_'+field]=changed
    variants['diagnostic']=text+'\nPSTORE_FAIL injected\n'
    variants['unknown']=text+'\nPSTORE_UNKNOWN cycle=0 job=0\n'
    variants['duplicate_pass']=text+next(line for line in text.splitlines() if line.startswith('PSTORE_PASS'))+'\n'
    rejected=[]
    for name,changed in variants.items():
        assert changed!=text
        try:review(changed,samples,seed)
        except (AssertionError,IndexError,KeyError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
