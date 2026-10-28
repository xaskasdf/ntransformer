"""Audit actual root/child ownership, AXI bytes and all concatenated results."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import check_ntrpu_attention_engine_r1 as engine

PLAN=[(1,0,0),(2,0,0),(3,0,0),(4,0,0),(1,1,1),(1,2,7),(1,3,2),(1,4,4),
      (1,5,7),(1,0,0),(2,6,2),(2,0,0),(3,7,3),(3,0,0),(4,8,0),
      (1,9,None),(1,0,0),(2,10,None),(2,0,0),(3,11,None),(3,0,0),(4,12,None),(4,0,0)]
CHILD_COUNTS={i:(32 if status==0 or mode==12 else 2 if mode in (5,6,7,10,11) else 0)
              for i,(_,mode,status) in enumerate(PLAN,1)}
SCHEMA={
 'BIND':'cycle id generation capacity base sequence epoch','BIND_DONE':'cycle id status',
 'APPEND':'cycle id generation layer position mode prefix','APPEND_DONE':'cycle id status bytes committed',
 'AW':'cycle id address beats','W':'cycle id last data','B':'cycle id ordinal resp',
 'AR':'cycle id address beats','R':'cycle id address resp last data',
 'BEGIN':'cycle root sample mode generation sequence epoch lease',
 'CHILD':'cycle root child head fixture position sequence epoch lease',
 'CHILD_DONE':'cycle root child status reads',
 'HLS_START':'cycle root child fixture count head sequence epoch',
 'STATE':'cycle child index data','OUTPUT':'cycle child lane data',
 'HLS_DONE':'cycle child status aborting','Q':'cycle child lane data',
 'K':'cycle child index valid data','V':'cycle child index valid data',
 'COPY':'cycle root child index data','RESULT':'cycle root index data',
 'RESET':'cycle count root','RESET_ABORT':'cycle root mode children',
 'REVOKED':'cycle root status','CANCEL':'cycle root','EPOCH':'cycle root',
 'COMPLETE':'cycle root status children elapsed',
 'PASS':'seed watchdog roots completions results children releases resets binds appends starts dones copies q k v state_w output_w held aw w b ar r cycles',
}
HEX={'base','sequence','epoch','lease','address','data'}


def parse(text):
    assert not re.search(r'(?mi)^\s*(?:Fatal:|FATAL:|ERROR:|FAIL:|AROOT_FAIL)',text),'Failure diagnostic'
    events=[]
    for line in text.splitlines():
        if not line.startswith('AROOT_'):continue
        tag,*fields=line[6:].split();assert tag in SCHEMA,('Unknown trace',tag)
        assert [f.split('=')[0] for f in fields]==SCHEMA[tag].split(),('Schema',tag)
        row={}
        for field in fields:
            key,value=field.split('=');assert re.fullmatch('[0-9a-f]+' if key in HEX else '[0-9]+',value)
            row[key]=int(value,16 if key in HEX else 10)
        events.append((tag,row))
    assert events and events[-1][0]=='PASS'
    return events


def load_fixture(folder,axi,parent):
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['format']=='NTRPU_ATTENTION_ROOT_FIXTURE_R1'
    for n,sha in manifest['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==sha
    proof=engine.load_fixture(axi,parent)
    expected=[int(n,16) for n in (folder/'root-expected.hex').read_text().split()]
    assert len(expected)==8192
    return dict(manifest=manifest,attention=proof,expected=expected)


def review(text,fixture,seed=1):
    events=parse(text);counts=Counter(k for k,_ in events)
    fixed=dict(BIND=7,BIND_DONE=7,APPEND=28,APPEND_DONE=28,BEGIN=23,CHILD=426,CHILD_DONE=424,
        COMPLETE=19,RESULT=24576,RESET=5,RESET_ABORT=4,REVOKED=1,CANCEL=1,EPOCH=1,PASS=1,W=1792)
    for tag,n in fixed.items():assert counts[tag]==n,('Coverage',tag,counts[tag],n)
    summary=events[-1][1];assert summary['seed']==seed and summary['watchdog']==65536 and summary['held']>0
    aliases=dict(roots='BEGIN',completions='COMPLETE',results='RESULT',children='CHILD',releases='CHILD_DONE',
        resets='RESET',binds='BIND',appends='APPEND',starts='HLS_START',dones='HLS_DONE',copies='COPY',
        q='Q',k='K',v='V',state_w='STATE',output_w='OUTPUT',aw='AW',w='W',b='B',ar='AR',r='R')
    assert all(summary[k]==counts[v] for k,v in aliases.items()),'Summary differs from trace'
    cache=binding=append=root=child=tile=aw=ar=None
    memory={};writes=[];applied=False;prefix=0;generation=0;operation=0;root_count=0;child_count=0
    last_cycle=-1;numeric_tiles=0;max_error=Decimal(0);max_ulp=0;resets=0;metrics=[]
    jobs=fixture['attention']['jobs'];beats=fixture['attention']['beats']
    for tag,row in events:
        if tag=='PASS':continue
        assert row['cycle']>=last_cycle,('Trace time order',tag);last_cycle=row['cycle']
        if tag=='RESET':
            resets+=1;assert row['count']==resets
            if resets==1:assert row['root']==0 and root is None
            else:assert root and root['reset'] and row['root']==root['root']
            cache=binding=append=root=child=tile=aw=ar=None;memory={};writes=[];applied=False;prefix=0
        elif tag=='BIND':
            assert not any((root,child,append,aw,ar,binding))
            generation+=1;operation+=1
            assert tuple(row[k] for k in ('id','generation','capacity','base','sequence','epoch'))==(
                operation,generation,4,0xa000000000+4032,0x1234567800000000+generation,0xfedcba9800000000+generation)
            binding=row;prefix=0;memory={}
        elif tag=='BIND_DONE':
            assert binding and row['id']==binding['id'] and row['status']==0
            cache=binding;binding=None
        elif tag=='APPEND':
            assert cache and not any((root,child,append,aw,ar)) and not writes
            operation+=1
            assert tuple(row[k] for k in ('id','generation','layer','position','mode','prefix'))==(
                operation,generation,0,prefix,0,prefix)
            assert prefix<4;append=dict(row,words=0,responses=0)
        elif tag=='APPEND_DONE':
            assert append and row['id']==append['id'] and append['words']==64 and not aw and not writes
            prefix+=1;assert (row['status'],row['bytes'],row['committed'])==(0,4096,prefix)
            append=None
        elif tag=='AW':
            assert append and not root and aw is None and row['id']==append['id']
            assert row['address']%64==0 and 1<=row['beats']<=4 and row['address']%4096+row['beats']*64<=4096
            aw=row;applied=False
        elif tag=='W':
            assert append and not root and row['id']==append['id'] and len(writes)<4
            assert not writes or not writes[-1]['last'];writes.append(row)
        elif tag=='B':
            assert append and aw and applied and row['id']==aw['id']==append['id']
            assert row['ordinal']==append['responses'] and row['resp']==0
            append['responses']+=1;aw=None;writes=[];applied=False
        elif tag=='BEGIN':
            assert cache and prefix==4 and not any((root,child,tile,aw,ar,append,binding))
            root_count+=1;sample,mode,status=PLAN[root_count-1]
            assert (row['root'],row['sample'],row['mode'],row['generation'])==(root_count,sample,mode,generation)
            assert (row['sequence'],row['epoch'],row['lease'])==(cache['sequence'],cache['epoch'],0x1122334400000000+root_count*64)
            root=dict(row,children=0,releases=0,copied={},results=[],reset=False,cancel=False,epoch_changed=False,revoked=False)
        elif tag=='CHILD':
            assert root and not any((child,tile,ar)) and row['root']==root['root']
            child_count+=1;head=root['children'];ident=(root['sample']-1)*32+head+1
            assert (row['child'],row['head'],row['fixture'],row['position'])==(child_count,head,ident,root['sample']-1)
            assert head<CHILD_COUNTS[root_count] and root['releases']==head
            assert (row['sequence'],row['epoch'],row['lease'])==(root['sequence'],root['epoch'],root['lease']+head)
            child=dict(row,read_addresses=[],responses=0,requested=0,copy=[],outputs={},hls_done=False)
            root['children']+=1
        elif tag=='AR':
            assert root and child and ar is None and row['id']==child['child']
            assert row['address']%64==0 and 1<=row['beats']<=256 and row['address']%4096+row['beats']*64<=4096
            wanted=[engine.physical(cache,0,plane,child['head']//4,pos,word)
                    for plane in range(2) for pos in range(root['sample']) for word in range(4)]
            addresses=[row['address']+64*i for i in range(row['beats'])]
            assert addresses==wanted[child['requested']:child['requested']+row['beats']],'Physical KV address order'
            child['requested']+=row['beats'];ar=dict(row,index=0)
        elif tag=='R':
            assert root and child and ar and row['id']==ar['id']==child['child']
            address=ar['address']+64*ar['index']
            assert row['address']==address and row['last']==int(ar['index']+1==ar['beats'])
            assert address in memory and row['data']==memory[address],'Physical KV content'
            expected_resp=2 if root['mode']==7 and child['head']==1 and child['responses']==1 else 0
            assert row['resp']==expected_resp,'AXI error injection/delivery'
            child['read_addresses'].append(address);child['responses']+=1;ar['index']+=1
            if ar['index']==ar['beats']:ar=None
        elif tag=='HLS_START':
            assert root and child and tile is None and ar is None
            assert len(child['read_addresses'])==root['sample']*8
            assert tuple(row[k] for k in ('root','child','fixture','count','head','sequence','epoch'))==(
                root['root'],child['child'],child['fixture'],root['sample'],child['head'],root['sequence'],root['epoch'])
            tile=dict(start=0,state={},outputs={},q=set(),k=set(),v=set(),done=False)
        elif tag in ('Q','K','V','STATE','OUTPUT','HLS_DONE'):
            assert root and child and tile and row['child']==child['child'] and not tile['done']
            job=jobs[child['fixture']-1]
            if tag=='Q':
                assert 0<=row['lane']<64 and row['data']==job['query'][row['lane']];tile['q'].add(row['lane'])
            elif tag in ('K','V'):
                index=row['index'];assert 0<=index<root['sample']*64
                assert row['valid'] in (0,1)
                if not row['valid']:assert root['mode']==5 and root['cancel'] and row['data']==0
                else:assert row['data']==job['keys' if tag=='K' else 'values'][index]
                tile[tag.lower()].add(index)
            elif tag=='STATE':
                assert 0<=row['index']<104 and row['index'] not in tile['state'];tile['state'][row['index']]=row['data']
            elif tag=='OUTPUT':
                assert 0<=row['lane']<64 and row['lane'] not in tile['outputs'];tile['outputs'][row['lane']]=row['data']
            else:
                abort=int(root['mode']==5 and child['head']==1)
                assert row['aborting']==abort
                if not abort:
                    assert row['status']==0 and tile['q']==set(range(64)) and tile['k']==tile['v']==set(range(root['sample']*64))
                    assert set(tile['state'])==set(range(104)) and set(tile['outputs'])==set(range(64))
                    q=dict(sequence=root['sequence'],epoch=root['epoch'],position=root['sample']-1,layer=0,head=child['head'])
                    error,ulp=engine.state_numeric(q,tile,fixture['attention']['oracle'][child['fixture']-1])
                    max_error=max(max_error,error);max_ulp=max(max_ulp,ulp);numeric_tiles+=1
                tile['done']=True;child['hls_done']=True;child['outputs']=tile['outputs']
        elif tag=='COPY':
            assert root and child and tile and tile['done'] and row['root']==root['root'] and row['child']==child['child']
            index=len(child['copy']);assert row['index']==index and index<64
            assert row['data']==child['outputs'][index],'Child copy differs from HLS write'
            child['copy'].append(row['data']);root['copied'][child['head']*64+index]=row['data']
        elif tag=='CHILD_DONE':
            assert root and child and row['root']==root['root'] and row['child']==child['child'] and ar is None
            status=7 if root['mode']==5 and child['head']==1 else 3 if root['mode']==7 and child['head']==1 else 0
            copies=8 if root['mode']==6 and child['head']==1 else 0 if status else 64
            assert (row['status'],row['reads'])==(status,copies) and len(child['copy'])==copies
            if status==3:assert tile is None
            else:assert child['hls_done']
            root['releases']+=1;child=tile=None
        elif tag=='RESULT':
            assert root and not any((child,tile,ar)) and root['children']==root['releases']==32
            index=len(root['results']);assert row['root']==root['root'] and row['index']==index and index<2048
            assert row['data']==root['copied'][index],'Concatenation differs from copied child'
            want=fixture['expected'][(root['sample']-1)*2048+index]
            assert engine.numeric.ulp(row['data'],want)<=1,'Root output differs from independent reference'
            root['results'].append(row['data'])
        elif tag in ('CANCEL','EPOCH','REVOKED'):
            assert root and row['root']==root['root']
            if tag=='CANCEL':assert root['mode']==5 and child and child['head']==1;root['cancel']=True
            elif tag=='EPOCH':assert root['mode']==6 and child and child['head']==1 and len(child['copy'])==8;root['epoch_changed']=True
            else:assert root['mode']==8 and row['status']==0 and len(root['results'])==2048;root['revoked']=True
        elif tag=='COMPLETE':
            assert root and not any((child,tile,ar)) and row['root']==root['root']
            _,mode,status=PLAN[root['root']-1]
            assert status is not None and row['status']==status and row['children']==root['children']==root['releases']==CHILD_COUNTS[root_count]
            assert len(root['results'])==(2048 if status==0 else 0)
            assert root['cancel']==(mode==5) and root['epoch_changed']==(mode==6) and root['revoked']==(mode==8)
            # Frozen TB R2 captures root_start on the preceding falling edge,
            # before start_root waits for the falling edge that emits BEGIN.
            # Keep the exact +1 contract; measure published cycles from events.
            assert row['elapsed']==row['cycle']-root['cycle']+1 and row['elapsed']>1
            metrics.append(dict(root=root_count,sample=root['sample'],mode=mode,status=status,cycles=row['cycle']-root['cycle'],bench_elapsed=row['elapsed']))
            root=None
        elif tag=='RESET_ABORT':
            assert root and PLAN[root_count-1][2] is None and row['root']==root_count and row['mode']==root['mode']
            assert row['children']==root['children']==CHILD_COUNTS[root_count] and not root['results']
            if root['mode']==9:assert not child and root['children']==0
            elif root['mode']==10:assert child and child['head']==1 and ar
            elif root['mode']==11:assert child and child['head']==1 and len(child['copy'])==8
            elif root['mode']==12:assert not child and root['releases']==32
            root['reset']=True
        if aw and writes and writes[-1]['last'] and not applied:
            assert len(writes)==aw['beats'] and append
            for i,w in enumerate(writes):
                n=append['words'];plane,offset=divmod(n,32);head,word=divmod(offset,4)
                address=engine.physical(cache,0,plane,head,append['position'],word)
                assert aw['address']+64*i==address and w['data']==beats[append['position']*64+n]
                memory[address]=w['data'];append['words']+=1
            applied=True
    assert not any((root,child,tile,append,aw,ar,binding)) and root_count==23 and child_count==426 and generation==7
    assert numeric_tiles==423,('Actual arithmetic coverage',numeric_tiles)
    return dict(summary=summary,numeric_tiles=numeric_tiles,max_scaled_state_error=str(max_error),max_output_ulp=max_ulp,
        root_metrics=metrics,scope='Actual 32-head wrapper with physical AXI BFM and generated HLS. No physical DDR or routed timing.')


def controls(text,fixture,seed):
    rejected=[]
    changes=[('BIND','epoch'),('APPEND','position'),('AW','address'),('W','data'),('B','resp'),('AR','address'),
             ('R','data'),('BEGIN','sample'),('CHILD','lease'),('CHILD','head'),('HLS_START','count'),
             ('Q','data'),('K','data'),('V','valid'),('STATE','data'),('OUTPUT','data'),('COPY','data'),
             ('COPY','index'),('CHILD_DONE','reads'),('RESULT','index'),('RESULT','data'),('COMPLETE','status'),
             ('RESET_ABORT','mode'),('REVOKED','status'),('BEGIN','cycle'),('COMPLETE','elapsed')]
    for tag,key in changes:
        pattern=rf'(?m)^(AROOT_{tag} .*?\b{key}=)([0-9a-f]+)'
        def mutate(m):
            n=int(m[2],16 if key in HEX else 10)^1
            return m[1]+(format(n,'x') if key in HEX else str(n))
        bad,n=re.subn(pattern,mutate,text,count=1);assert n==1 and bad!=text
        try:review(bad,fixture,seed)
        except (AssertionError,ValueError,KeyError,IndexError,StopIteration):rejected.append(tag+'_'+key)
        else:raise AssertionError('Accepted corrupt '+tag+'_'+key)
    return rejected
