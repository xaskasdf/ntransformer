"""Review the whole front owner and actual lease controller against a fixed plan."""
from collections import Counter,defaultdict
import re
from ntrpu_input_attention_plan_r1 import plan,regions

SCHEMA={
    'INPUT':'cycle job index address base bytes','BOUNDARY':'cycle job index address base bytes',
    'ADMIT':'cycle job mode sequence epoch layer position','REGION':'cycle job peer index address base bytes',
    'SOURCE':'cycle job sequence epoch','CONSUMER':'cycle job sequence epoch layer position lease',
    'LEASE':'cycle job status','LEASE_ACK':'cycle job','CANCEL':'cycle job peer',
    'INJECT':'cycle job phase','RESTORE':'cycle job peer','IDENTITY':'cycle job sequence epoch layer position',
    'COMPLETE':'cycle job status source child committed bytes tokens','WORD':'cycle job index data',
    'REVOKE':'cycle job status revoked valid ce','ACK':'cycle job','SOURCE_ACK':'cycle job','CONSUMER_ACK':'cycle job',
    'RELEASE':'cycle job sources consumers','PASS':'seed jobs sources consumers source_acks consumer_acks public_acks words cycles'}


def need(good,why):
    if not good:raise ValueError(why)


def parse(text):
    need(not re.search(r'(?mi)^\s*(?:FATAL:|Fatal:|ERROR:|IA_FAIL)',text),'No simulator failure')
    result=[]
    for line in text.splitlines():
        if not line.startswith('IA_'):continue
        tag,*fields=line[3:].split();need(tag in SCHEMA,'Known event')
        need([p.split('=')[0] for p in fields]==SCHEMA[tag].split(),'Exact schema');row={}
        for item in fields:
            key,value=item.split('=');hexa=key in ('address','base','sequence','epoch','lease','data') or (key=='bytes' and tag in ('INPUT','BOUNDARY','REGION'))
            need(re.fullmatch('[0-9a-f]+' if hexa else '[0-9]+',value),'Finite field');row[key]=int(value,16 if hexa else 10)
        result.append((tag,row))
    return result


def review(text,seed):
    events=parse(text);specs=plan();need(events and events[-1][0]=='PASS','Whole suite completion')
    summary=events[-1][1];counts=Counter(t for t,r in events)
    need(counts['PASS']==1 and summary['seed']==seed and summary['jobs']==len(specs)==298,'Planned extent')
    for field,tag in dict(sources='SOURCE',consumers='CONSUMER',source_acks='SOURCE_ACK',consumer_acks='CONSUMER_ACK',public_acks='ACK',words='WORD').items():
        need(summary[field]==counts[tag],'Summary agrees with events')
    need((summary['sources'],summary['consumers'],summary['source_acks'],summary['consumer_acks'],summary['public_acks'],summary['words'])==(173,167,173,167,298,3328),'Fixed coverage')
    groups=defaultdict(list);previous=-1
    for tag,row in events[:-1]:
        need(previous<=row['cycle'] and 1<=row['job']<=298,'Monotonic owned events');previous=row['cycle'];groups[row['job']].append((tag,row))
    need(previous==summary['cycles']<200000,'Terminal cycle');last_release=-1
    for job,spec in enumerate(specs,1):
        ev=groups[job];by=defaultdict(list)
        for tag,row in ev:by[tag].append(row)
        def one(tag,count=1):
            need(len(by[tag])==count,'Extent '+tag);return by[tag][0] if count else None
        mode=spec['mode'];seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job;layer=job%16;pos=layer%3
        identity=dict(sequence=seq,epoch=epoch^(0x100000000 if mode==18 else 0),layer=0x10000000 if mode==20 else layer,position=pos+(mode==23))
        admit=one('ADMIT');public_id=one('IDENTITY');complete=one('COMPLETE');release=one('RELEASE');ack=one('ACK')
        need(ev[0][0]=='INPUT' and ev[-1][0]=='RELEASE' and ev[0][1]['cycle']>=last_release,'Serialized jobs');last_release=release['cycle']
        need(admit==dict(cycle=admit['cycle'],job=job,mode=mode,**identity),'Admitted identity')
        need(public_id==dict(cycle=public_id['cycle'],job=job,**identity),'Public retained identity')
        expected=regions(mode)
        for tag in ('INPUT','BOUNDARY'):
            need(len(by[tag])==10,'Ten allocations')
            for i,row in enumerate(by[tag]):
                need({k:row[k] for k in ('index','address','base','bytes')}==expected[i],'Intended allocation reaches actual owner')
                need(row['cycle']<admit['cycle'] if tag=='INPUT' else row['cycle']==admit['cycle'],'Allocation capture boundary')
        source=one('SOURCE',int(spec['producer']));consumer=one('CONSUMER',int(spec['consumer']))
        lease_expected=spec['producer'] and mode not in (6,8,13,26)
        lease=one('LEASE',int(lease_expected));lease_ack=one('LEASE_ACK',int(lease_expected))
        source_ack=one('SOURCE_ACK',int(spec['producer']));consumer_ack=one('CONSUMER_ACK',int(spec['consumer']))
        if source:
            need(source==dict(cycle=source['cycle'],job=job,sequence=seq,epoch=epoch) and source['cycle']>=admit['cycle']+51,'Preflight before producer')
            need(complete['cycle']<ack['cycle']<source_ack['cycle']<release['cycle'],'Producer retirement order')
        if lease:
            need(source['cycle']<lease['cycle']<complete['cycle']<ack['cycle']<lease_ack['cycle']<source_ack['cycle'],'Lease order')
            need(lease['status']==(3 if mode==1 else 0),'Retained source status')
        if consumer:
            need(consumer==dict(cycle=consumer['cycle'],job=job,sequence=seq,epoch=epoch,layer=layer,position=pos,lease=0x2000000010000000+job*64),'Consumer descriptor')
            need(lease['cycle']<consumer['cycle']<complete['cycle'] and lease_ack['cycle']<consumer_ack['cycle']<source_ack['cycle'],'Consumer ownership order')
        need(len(by['REGION'])==9*int(spec['producer'])+4*int(spec['consumer']),'Physical allocation mappings')
        for peer,count,start in [(0,9,source),(1,4,consumer)]:
            rows=[r for r in by['REGION'] if r['peer']==peer]
            need(len(rows)==(count if start else 0),'Peer extent')
            for i,row in enumerate(rows):
                region=expected[i if peer==0 else i+6]
                need(row==dict(cycle=start['cycle'],job=job,peer=peer,index=i,address=region['address'],base=region['base'],bytes=region['bytes']),'Captured peer region')
        committed=spec['consumer'] and mode not in (3,7,9,27)
        source_status=3 if mode==1 else 7 if mode in (6,8,26) else 0
        child_status=3 if mode==3 else 7 if mode in (7,9,27) else 0
        want_complete=dict(cycle=complete['cycle'],job=job,status=spec['status'],source=source_status,child=child_status,
            committed=int(committed),bytes=(4032 if mode==4 else 4096) if committed else 0,
            tokens=pos+int(committed) if spec['consumer'] else 0)
        need(complete==want_complete,'Whole completion diagnostic')
        earliest=admit['cycle']+(0 if mode in (18,19,24,28) else 1)
        need(public_id['cycle']==complete['cycle'] and earliest<=complete['cycle']<ack['cycle']<release['cycle'],'Public response order')
        expected_words=(2048 if job==1 else 8) if spec['status']==0 else 0
        need(len(by['WORD'])==expected_words,'Result extent')
        for i,row in enumerate(by['WORD']):
            index=(i*257+31)%2048
            need(row['index']==index and row['data']==(0xbadd0000^index^(layer<<16)^pos),'Public result mapping')
            need(complete['cycle']<row['cycle']<ack['cycle'],'Read only while owned response')
        phase=0 if mode==5 else 1 if mode in (6,26) else 2 if mode in (7,27) else 3 if spec['late'] else None
        injected=one('INJECT',int(phase is not None))
        if injected:
            need(injected['phase']==phase,'Fault location')
            lower=admit if phase==0 else source if phase==1 else consumer if phase==2 else complete
            upper=ack if phase==3 else complete
            need(lower['cycle']<=injected['cycle']<upper['cycle'],'Fault occurs in intended owner phase')
        revoke=one('REVOKE',int(spec['late']))
        if revoke:need(revoke==dict(cycle=revoke['cycle'],job=job,status=0,revoked=1,valid=0,ce=0) and injected['cycle']<revoke['cycle']<ack['cycle'],'Sticky late revocation')
        restore=one('RESTORE',int(mode in (13,14)))
        if restore:need(restore['peer']==int(mode==14) and (source if mode==13 else consumer)['cycle']<restore['cycle']<complete['cycle'],'Foreign identity restored before response')
        expected_cancel={5:[2],6:[0,2],7:[1,2],8:[0],9:[1],12:[2],26:[0],27:[1]}.get(mode,[])
        need(sorted(r['peer'] for r in by['CANCEL'])==expected_cancel,'Cancellation handshakes')
        need(release['sources']==int(spec['producer']) and release['consumers']==int(spec['consumer']),'Release coverage')
    return dict(summary=summary,arithmetic_qualified=False,actual_owner_and_lease=True,actual_hierarchy_execution=False)


def controls(text,seed):
    events=parse(text);variants={}
    for tag in SCHEMA:
        line=next(s for s in text.splitlines() if s.startswith('IA_'+tag+' '))
        variants['missing_'+tag]=text.replace(line+'\n','',1)
        variants['duplicate_'+tag]=text.replace(line,line+'\n'+line,1)
    for tag,key in [('ADMIT','epoch'),('IDENTITY','layer'),('CONSUMER','lease'),('REGION','address'),('BOUNDARY','base'),('INPUT','bytes'),('COMPLETE','status'),('WORD','data'),('REVOKE','valid'),('CANCEL','peer')]:
        line=next(s for s in text.splitlines() if s.startswith('IA_'+tag+' '))
        match=re.search(r'\b'+key+r'=([0-9a-f]+)',line);need(match is not None,'Control field')
        value=match[1];replacement='0' if int(value,16) else '1'
        variants['changed_'+tag+'_'+key]=text.replace(line,line[:match.start(1)]+replacement+line[match.end(1):],1)
    rejected=[]
    for name,candidate in variants.items():
        need(candidate!=text,'Nonempty corruption')
        try:review(candidate,seed)
        except (ValueError,AssertionError):rejected.append(name)
        else:raise ValueError('Accepted trace corruption '+name)
    return rejected
