"""Check ordered AXI reads and every public private-RAM word independently."""
import re

PLAN=[(n,0) for n in range(3)]+[(n,1) for n in range(10,22)]+[(n,2) for n in range(22,25)]+[(25,11)]
PLAN += [(n,3) for n in range(30,38)]+[(n,4) for n in range(38,41)]+[(n,7) for n in range(41,45)]
PLAN += [(45,2),(46,8),(47,8),(48,0),(49,0),(50,None),(51,None),(52,0),(53,1),(54,1),(55,0)]
PLAN += [(56,1),(57,2),(58,2),(59,0),(60,0),(61,1)]
LENGTHS=[8192,2048,256]
BEATS=[n//64 for n in LENGTHS]
SCHEMA={
 'BEGIN':'job scenario sequence epoch expected source_sequence source_epoch source_valid ra rb rz ba bb bz wa wb wz position source_position coefficient_position',
 'AR':'cycle job plane address beats',
 'R':'cycle job address response rid last data',
 'COMPLETE':'cycle job status sequence epoch valid beats ar r position coefficient_position',
 'RAM':'cycle job qa q ca c cv','INVALID_READ':'cycle job qa ca',
 'CANCEL':'cycle job','EPOCH':'cycle job expected','REVOKE':'cycle job mode',
 'RESET':'cycle job ar r','RELEASE':'cycle job',
 'PASS':'seed jobs completions resets revocations ar r ram ar_stalls r_stalls cycles'}
HEX={
 'BEGIN':set('sequence epoch expected source_sequence source_epoch ra rb rz ba bb bz wa wb wz'.split()),
 'AR':{'address'},'R':{'address','data'},'COMPLETE':{'sequence','epoch'},'RAM':{'q','c'},'EPOCH':{'expected'}}

def need(ok,why):
    if not ok:raise ValueError(why)

def word(address,job):return 0x3f000000|(((address>>2)^(address>>33)^(address>>40)^(job*97))&0x7fffff)

def review(text,seed):
    need(not re.search(r'(?m)^(?:FAIL|FATAL|ERROR|Fatal:)',text),'Failure diagnostic')
    jobs={};summary=None;last_cycle=0
    for line in text.splitlines():
        if not line.startswith('RPELOAD_'):continue
        kind,*tokens=line[len('RPELOAD_'):].split();need(kind in SCHEMA,'Unknown trace kind')
        fields=[token.split('=') for token in tokens];need(all(len(p)==2 for p in fields),'Field syntax')
        need([p[0] for p in fields]==SCHEMA[kind].split(),'Exact trace fields')
        for k,v in fields:
            width=128 if kind=='R' and k=='data' else 8 if kind=='RAM' and k in ('q','c') else 16
            need(re.fullmatch('[0-9a-f]{'+str(width)+'}' if k in HEX.get(kind,set()) else '[0-9]+',v) is not None,'Canonical numeric field')
        row={k:int(v,16 if k in HEX.get(kind,set()) else 10) for k,v in fields}
        if 'cycle' in row:
            need(row['cycle']>=last_cycle,'Monotonic cycle');last_cycle=row['cycle']
        if kind=='PASS':need(summary is None,'Unique summary');summary=row;continue
        need(summary is None,'No event after summary')
        if kind=='BEGIN':
            need(row['job']==len(jobs)+1,'Ordered jobs');jobs[row['job']]=[]
        need(row['job'] in jobs and row['job']==len(jobs),'Event identity/order')
        jobs[row['job']].append((kind,row))
    need(len(jobs)==len(PLAN)==51 and summary is not None,'Complete bank')
    stats=dict(ar=0,r=0,ram=0,completions=0,resets=0,revocations=0);metrics=[];previous_end=1
    for ident,(mode,want) in enumerate(PLAN,1):
        events=jobs[ident];desc=events[0][1];need(events[0][0]=='BEGIN' and desc['scenario']==mode,'Plan identity')
        seq=0x1234567800000000+ident;epoch=0xfedcba9800000000+ident
        need(desc['sequence']==seq and desc['epoch']==epoch,'Full public identities')
        need(desc['source_sequence']==(seq^(1<<32) if mode==22 else seq),'Full source sequence')
        need(desc['source_epoch']==(epoch^(1<<32) if mode==23 else epoch),'Full source epoch')
        need(desc['expected']==(epoch^(1<<32) if mode==24 else epoch) and desc['source_valid']==int(mode!=25),'Admission')
        position=131072 if mode==56 else 131071 if mode==59 else 0 if mode==0 else ident*2017
        need(desc['position']==position and desc['source_position']==position+(mode==57) and desc['coefficient_position']==position+(mode==58),'Position admission')
        addresses=[desc[k] for k in ('ra','ba','wa')];bases=[desc[k] for k in ('rb','bb','wb')];sizes=[desc[k] for k in ('rz','bz','wz')]
        bounds=all(a%64==0 and z>0 and a>=b and a+n<=b+z and a+n<1<<64 and b+z<1<<64 for a,b,z,n in zip(addresses,bases,sizes,LENGTHS))
        need(bounds==(mode not in list(range(10,22))+[53,54,61]),'All regions validated before AXI')
        offset=0 if mode in (1,36,37) else 128 if mode==2 else 4032
        expected_bases=[0x100000000,0x8000000000,0xc000000000]
        expected_addresses=[a+offset for a in expected_bases];expected_sizes=[16384]*3
        if 10<=mode<=12:expected_addresses[mode-10]+=1
        if 13<=mode<=15:expected_sizes[mode-13]=offset+LENGTHS[mode-13]-1
        if 16<=mode<=18:expected_bases[mode-16]=expected_addresses[mode-16]+64
        if 19<=mode<=21:expected_addresses[mode-19]=expected_bases[mode-19]=0xfffffffffffff000
        if mode==53:expected_sizes[2]=0
        if mode==54:expected_addresses[1]=0xffffffffffffffc0;expected_bases[1]=0;expected_sizes[1]=0xffffffffffffffff
        if mode==61:expected_sizes[0]=0
        need(addresses==expected_addresses and bases==expected_bases and sizes==expected_sizes,'Exact command descriptor')
        by={k:[] for k in SCHEMA};ledger={};pending=None;plane=0;index=0;bursts=0;returned=0
        completed=False;rows=0
        for kind,row in events[1:]:
            by[kind].append(row)
            if kind=='AR':
                need(not completed and pending is None and plane<3,'Single outstanding burst before completion')
                need(row['plane']==plane and row['address']==addresses[plane]+index*64,'Ordered plane/address')
                n=min(BEATS[plane]-index,(4096-row['address']%4096)//64)
                need(row['beats']==n and 0<n<=64,'Page bounded burst')
                pending=dict(address=row['address'],beats=n,index=0,plane=plane);bursts+=1;index+=n
                if index==BEATS[plane]:plane+=1;index=0
            elif kind=='R':
                need(not completed and pending is not None,'R must belong to issued burst')
                a=pending['address']+pending['index']*64;bp=pending['plane']
                need(row['address']==a,'R beat address')
                last=(pending['index']==3) if mode==36 and bursts==1 else (pending['index']==pending['beats']) if mode==37 and bursts==1 else pending['index']==pending['beats']-1
                need(row['last']==int(last),'RLAST contract')
                late=a==addresses[bp]+LENGTHS[bp]-64
                need(row['response']==(2 if mode in (30,31,32) and bp==mode-30 and late else 0),'RRESP fault scenario')
                need(row['rid']==(1 if mode in (33,34,35) and bp==mode-33 and late else 0),'RID fault scenario')
                expected=[word(a+lane*4,ident) for lane in range(16)]
                if mode in (38,39,40) and bp==mode-38 and late:expected[-1]=0x7fc12345
                actual=[(row['data']>>(lane*32))&0xffffffff for lane in range(16)]
                need(actual==expected,'Actual memory data/byte order')
                for lane,v in enumerate(actual):
                    need(a+lane*4 not in ledger,'No duplicated bus words');ledger[a+lane*4]=v
                pending['index']+=1;returned+=1
                if last:pending=None
            elif kind=='COMPLETE':
                need(not completed and pending is None and want is not None,'Completion after drain')
                need((row['status'],row['sequence'],row['epoch'],row['valid'])==(want,seq,epoch,int(want==0)),'Completion/publication')
                need(row['ar']==bursts and row['r']==returned and 0<=row['beats']<=returned,'Transport counters')
                need(row['position']==position and row['coefficient_position']==desc['coefficient_position'],'Captured positions')
                if want==0:need((row['beats'],returned,plane,index)==(164,164,3,0),'Full Q/K/phase vector')
                completed=True
            elif kind=='RAM':
                need(completed and want==0 and not by['REVOKE'] and not by['RESET'],'Read only valid held output')
                qa=(rows*17+31)%2560;ca=(rows*7+63)%64;cv=int(rows<64)
                need((row['qa'],row['ca'],row['cv'])==(qa,ca,cv),'All address permutations and port enables')
                address=addresses[0]+qa*4 if qa<2048 else addresses[1]+(qa-2048)*4
                need(row['q']==ledger[address],'Q/K RAM equals accepted AXI')
                need(row['c']==(ledger[addresses[2]+ca*4] if cv else 0),'Coefficient RAM equals accepted AXI')
                rows+=1
            elif kind=='INVALID_READ':
                need(mode==60 and completed and rows==2560,'Invalid address placement')
                n=len(by[kind])-1
                need(n<2 and (row['qa'],row['ca'])==((2560,64) if n==0 else (4095,4095)),'Consumer range checks')
            elif kind=='CANCEL':need(mode in (41,42,43,44) and not completed,'Cancel placement')
            elif kind=='EPOCH':need(mode==45 and row['expected']==epoch^(1<<32) and not completed,'Epoch revocation')
            elif kind=='REVOKE':need(completed and mode in (48,49) and row['mode']==mode and rows==2560,'Held output revoked')
            elif kind=='RESET':
                need(mode in (50,51,52) and row['ar']==bursts and row['r']==returned,'Coordinated reset')
                pending=None
            elif kind=='RELEASE':need(completed and pending is None,'Release after drain')
            else:raise ValueError('Unexpected job event '+kind)
        need(len(by['COMPLETE'])==int(want is not None),'Completion cardinality')
        need(rows==(2560 if want==0 else 0),'Exact complete RAM readout')
        need(len(by['RESET'])==int(mode in (50,51,52)),'Reset coverage')
        need(len(by['RELEASE'])==int(mode not in (50,51,52)),'Release coverage')
        need(len(by['REVOKE'])==int(mode in (48,49)),'Revocation coverage')
        need(len(by['INVALID_READ'])==(2 if mode==60 else 0),'Consumer range coverage')
        need(len(by['CANCEL'])==int(mode in (41,42,43,44)) and len(by['EPOCH'])==int(mode==45),'Fault trigger coverage')
        if mode in list(range(10,26))+[41,53,54,56,57,58,61]:need(bursts==returned==0,'Admission rejected without AXI')
        end=events[-1][1]['cycle'];need(end>previous_end,'Job cycle extent')
        metrics.append(dict(job=ident,scenario=mode,status=want,ar=bursts,r=returned,ram_words=(2624 if rows else 0),
            completion_cycles=None if want is None else by['COMPLETE'][0]['cycle']-previous_end))
        previous_end=end
        for key,v in dict(ar=bursts,r=returned,ram=(2624 if rows else 0),completions=len(by['COMPLETE']),resets=len(by['RESET']),revocations=len(by['REVOKE'])).items():stats[key]+=v
    need(summary['seed']==seed and summary['jobs']==51,'Summary identity')
    for k,v in stats.items():need(summary[k]==v,'Summary counter '+k)
    need((stats['completions'],stats['resets'],stats['revocations'],stats['ram'])==(49,3,2,23616),'Coverage totals')
    need(summary['ar_stalls']>0 and summary['r_stalls']>0 and summary['cycles']==previous_end,'Stalls/cycles')
    return dict(status='PASS_ROPE_LOADER_R1_TRACE',seed=seed,jobs=51,**stats,cycles=summary['cycles'],jobs_metrics=metrics,
        scope='Actual AXI reader and private RAM loader; deterministic synthetic FP32 payloads. No HLS arithmetic, physical DDR or inference qualification.')

def controls(text,seed):
    changes={};rejected=[]
    def row(kind,fragment=''):
        found=re.search(r'(?m)^RPELOAD_'+kind+r' .*'+fragment+r'.*$',text);need(found is not None,'Control target');return found[0]
    def field(name,line,key,mask=1,base=10):
        m=re.search(r'\b'+key+r'=([0-9a-f]+)',line);need(m is not None,'Control field')
        value=int(m[1],base)^mask
        v=format(value,'0'+str(len(m[1]))+'x') if base==16 else str(value)
        bad=line[:m.start(1)]+v+line[m.end(1):];changes[name]=text.replace(line,bad,1)
    field('ar_high_address',row('AR'),'address',1<<32,16)
    field('ar_length',row('AR'),'beats')
    field('ar_plane',row('AR'),'plane')
    field('r_word',row('R'),'data',1,16)
    field('r_last',row('R'),'last')
    field('r_response',row('R'),'response')
    field('r_id',row('R'),'rid')
    field('ram_qk',row('RAM'),'q',1,16)
    field('ram_coefficient',row('RAM'),'c',1,16)
    field('ram_enable',row('RAM'),'cv')
    field('ram_address',row('RAM'),'qa')
    field('position',row('COMPLETE'),'position')
    field('coefficient_position',row('BEGIN'),'coefficient_position')
    field('consumer_range',row('INVALID_READ'),'qa')
    field('full_sequence',row('COMPLETE'),'sequence',1<<32,16)
    field('full_epoch',row('COMPLETE'),'epoch',1<<32,16)
    field('status',row('COMPLETE','status=3 '),'status')
    field('invalid_publication',row('COMPLETE','status=3 '),'valid')
    field('completion_beats',row('COMPLETE'),'beats')
    field('source_identity',row('BEGIN'),'source_sequence',1<<32,16)
    field('bounds',row('BEGIN'),'rz',1<<63,16)
    field('counter',row('PASS'),'ram')
    first=row('RAM');changes['missing_ram']=text.replace(first+'\n','',1);changes['duplicate_ram']=text.replace(first,first+'\n'+first,1)
    first=row('AR');changes['missing_ar']=text.replace(first+'\n','',1)
    first=row('CANCEL');changes['missing_cancel']=text.replace(first+'\n','',1)
    first=row('REVOKE');changes['missing_revocation']=text.replace(first+'\n','',1)
    changes['missing_pass']=text.replace('RPELOAD_PASS','REMOVED');changes['failure']=text+'\nFAIL injected\n'
    changes['extra_event']=text+'\nRPELOAD_RELEASE cycle=999999 job=51\n'
    for name,bad in changes.items():
        need(bad!=text,'Changed control')
        try:review(bad,seed)
        except (ValueError,AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise ValueError('Accepted corrupt '+name)
    return rejected
