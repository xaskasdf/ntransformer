"""Check ordered AXI reads and every public private-RAM word independently."""
import re

from ntrpu_rope_input_fixture_r1 import load as fixture,plan

LENGTHS=[8192,2048,256]
BEATS=[n//64 for n in LENGTHS]
SCHEMA={
 'BEGIN':'job scenario sequence epoch expected source_sequence source_epoch source_valid ra rb rz ba bb bz wa wb wz position source_position coefficient_position',
 'AR':'cycle job plane address beats',
 'R':'cycle job address response rid last data',
 'COMPLETE':'cycle job status sequence epoch valid beats ar r position coefficient_position hls',
 'RAM':'cycle job address data','INVALID_READ':'cycle job address',
 'HSTART':'cycle job','HDONE':'cycle job status reads writes quiet',
 'WRITE':'cycle job index data checked',
 'CANCEL':'cycle job','EPOCH':'cycle job expected','REVOKE':'cycle job mode',
 'RESET':'cycle job ar r','RELEASE':'cycle job',
 'PASS':'seed jobs completions resets revocations ar r ram ar_stalls r_stalls cycles'}
HEX={
 'BEGIN':set('sequence epoch expected source_sequence source_epoch ra rb rz ba bb bz wa wb wz'.split()),
 'AR':{'address'},'R':{'address','data'},'COMPLETE':{'sequence','epoch'},'RAM':{'data'},'WRITE':{'data'},'EPOCH':{'expected'}}

def need(ok,why):
    if not ok:raise ValueError(why)

def review(text,values,seed):
    PLAN=plan(values)
    need(not re.search(r'(?m)^(?:FAIL|FATAL|ERROR|Fatal:)',text),'Failure diagnostic')
    jobs={};summary=None;last_cycle=0
    for line in text.splitlines():
        if not line.startswith('RPEREAL_'):continue
        kind,*tokens=line[len('RPEREAL_'):].split();need(kind in SCHEMA,'Unknown trace kind')
        fields=[token.split('=') for token in tokens];need(all(len(p)==2 for p in fields),'Field syntax')
        need([p[0] for p in fields]==SCHEMA[kind].split(),'Exact trace fields')
        for k,v in fields:
            width=128 if kind=='R' and k=='data' else 8 if kind in ('RAM','WRITE') and k=='data' else 16
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
    need(len(jobs)==len(PLAN)==69 and summary is not None,'Complete bank')
    stats=dict(ar=0,r=0,ram=0,completions=0,resets=0,revocations=0);metrics=[];previous_end=1
    for ident,(mode,want) in enumerate(PLAN,1):
        events=jobs[ident];desc=events[0][1];need(events[0][0]=='BEGIN' and desc['scenario']==mode,'Plan identity')
        seq=0x1234567800000000+ident;epoch=0xfedcba9800000000+ident
        need(desc['sequence']==seq and desc['epoch']==epoch,'Full public identities')
        need(desc['source_sequence']==(seq^(1<<32) if mode==22 else seq),'Full source sequence')
        need(desc['source_epoch']==(epoch^(1<<32) if mode==23 else epoch),'Full source epoch')
        need(desc['expected']==(epoch^(1<<32) if mode==24 else epoch) and desc['source_valid']==int(mode!=25),'Admission')
        sample=mode-100 if mode>=100 else 41;record=values[sample]
        position=record['position']
        need(desc['position']==position and desc['source_position']==position+(mode==57) and desc['coefficient_position']==record['coefficient_position'],'Position admission')
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
        completed=False;rows=0;writes=0;checked_words=0;computed=None
        expected_start=record['hls_started'] if mode>=100 else want==0 or mode in (62,63,64,65,66)
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
                data=record['input'][:2048] if bp==0 else record['input'][2048:] if bp==1 else record['phase']
                base_index=(a-addresses[bp])//4
                expected=data[base_index:base_index+16]
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
                need(row['hls']==(by['HDONE'][0]['status'] if by['HDONE'] else 0),'Core diagnostic retained')
                if mode>=100:need(row['hls']==record['hls_status'],'Fixture core diagnostic')
                completed=True
            elif kind=='RAM':
                need(completed and want==0 and not by['REVOKE'] and not by['RESET'],'Read only valid held output')
                index=(rows*17+31)%2560
                need(row['address']==index,'All output addresses')
                need(computed is not None and computed[0]==0,'Arithmetic oracle available')
                need(row['data']==computed[1][index],'Real RoPE from accepted AXI words')
                rows+=1
            elif kind=='INVALID_READ':
                need(mode==60 and completed and rows==2560,'Invalid address placement')
                n=len(by[kind])-1
                need(n<2 and row['address']==(2560 if n==0 else 4095),'Consumer range checks')
            elif kind=='HSTART':
                need(not completed and pending is None and returned==164 and plane==3 and index==0,'HLS starts after complete AXI load')
                need(len(by[kind])==1 and not by['HDONE'] and expected_start,'One HLS ownership interval')
                from check_ntrpu_rope_table_r1 import oracle
                x=[ledger[addresses[0]+i*4] for i in range(2048)]+[ledger[addresses[1]+i*4] for i in range(512)]
                phase=[ledger[addresses[2]+i*4] for i in range(64)]
                computed=oracle(40,position,desc['coefficient_position'],phase,x)
                need(computed==(record['status'],record['expected']),'Accepted bytes preserve fixture arithmetic')
            elif kind=='WRITE':
                need(not completed and len(by['HSTART'])==1 and not by['HDONE'],'Write inside active core interval')
                need(row['index']==writes and writes<2560,'All sequential core writes')
                trusted=not by['CANCEL'] and not by['EPOCH']
                need(row['checked']==int(trusted),'Trust only non-revoked arithmetic')
                if trusted:
                    need(computed is not None and computed[0]==0 and row['data']==computed[1][writes],'Exact core arithmetic')
                    checked_words+=1
                writes+=1
            elif kind=='HDONE':
                need(not completed and pending is None and len(by['HSTART'])==1 and len(by[kind])==1,'HLS completion before publication')
                need(row['cycle']>by['HSTART'][0]['cycle'],'HLS cycle order')
                need(0<=row['status']<=5 and row['writes']==writes and row['quiet']<262144 and row['reads']<=row['cycle']-by['HSTART'][0]['cycle']+1,'Core counts and watchdog')
                if not by['CANCEL'] and not by['EPOCH']:
                    need(row['status']==computed[0],'Independent core status')
                    need(writes==(2560 if computed[0]==0 else 0),'Atomic core output extent')
                    if computed[0]==0:need(row['reads']>=2560,'Full Q/K input consumed')
                if mode in (62,63):need(row['reads']>=64,'Read cancellation coverage')
                if mode==64:need(writes>=64,'Output cancellation coverage')
            elif kind=='CANCEL':need(mode in (41,42,43,44,62,64,67,68) and not completed,'Cancel placement')
            elif kind=='EPOCH':need(mode in (45,63) and row['expected']==epoch^(1<<32) and not completed,'Epoch revocation')
            elif kind=='REVOKE':need(completed and mode in (48,49) and row['mode']==mode and rows==2560,'Held output revoked')
            elif kind=='RESET':
                need(mode in (50,51,52,65,66) and row['ar']==bursts and row['r']==returned,'Coordinated reset')
                pending=None
            elif kind=='RELEASE':need(completed and pending is None,'Release after drain')
            else:raise ValueError('Unexpected job event '+kind)
        need(len(by['COMPLETE'])==int(want is not None),'Completion cardinality')
        need(len(by['HSTART'])==int(expected_start),'HLS admission coverage')
        need(len(by['HDONE'])==int(expected_start and mode not in (65,66)),'HLS drain coverage')
        if by['COMPLETE'] and by['HDONE']:need(by['COMPLETE'][0]['cycle']>by['HDONE'][0]['cycle'],'Public completion follows drain')
        need(rows==(2560 if want==0 else 0),'Exact complete RAM readout')
        need(len(by['RESET'])==int(mode in (50,51,52,65,66)),'Reset coverage')
        need(len(by['RELEASE'])==int(mode not in (50,51,52,65,66)),'Release coverage')
        need(len(by['REVOKE'])==int(mode in (48,49)),'Revocation coverage')
        need(len(by['INVALID_READ'])==(2 if mode==60 else 0),'Consumer range coverage')
        need(len(by['CANCEL'])==int(mode in (41,42,43,44,62,64,67,68)) and len(by['EPOCH'])==int(mode in (45,63)),'Fault trigger coverage')
        if mode in list(range(10,26))+[41,53,54,56,57,58,61]:need(bursts==returned==0,'Admission rejected without AXI')
        end=events[-1][1]['cycle'];need(end>previous_end,'Job cycle extent')
        metrics.append(dict(job=ident,scenario=mode,status=want,ar=bursts,r=returned,ram_words=(2560 if rows else 0),
            exact_hls_words=checked_words,hls_start=int(expected_start),
            hls_cycles=None if not by['HDONE'] else by['HDONE'][0]['cycle']-by['HSTART'][0]['cycle'],
            max_core_quiet=None if not by['HDONE'] else by['HDONE'][0]['quiet'],completion_cycles=None if want is None else by['COMPLETE'][0]['cycle']-previous_end))
        previous_end=end
        for key,v in dict(ar=bursts,r=returned,ram=(2560 if rows else 0),completions=len(by['COMPLETE']),resets=len(by['RESET']),revocations=len(by['REVOKE'])).items():stats[key]+=v
    need(summary['seed']==seed and summary['jobs']==69,'Summary identity')
    for k,v in stats.items():need(summary[k]==v,'Summary counter '+k)
    need((stats['completions'],stats['resets'],stats['revocations'],stats['ram'])==(65,5,2,107520),'Coverage totals')
    need(summary['ar_stalls']>0 and summary['r_stalls']>0 and summary['cycles']==previous_end,'Stalls/cycles')
    return dict(status='PASS_ROPE_INPUT_REAL_R1_TRACE',seed=seed,jobs=69,**stats,cycles=summary['cycles'],jobs_metrics=metrics,
        exact_hls_words=sum(m['exact_hls_words'] for m in metrics),
        scope='Original RoPE arithmetic over actual AXI loader and owner. Synthetic memory and fault peer; physical DDR and inference pending.')

def controls(text,values,seed):
    changes={};rejected=[]
    def row(kind,fragment=''):
        found=re.search(r'(?m)^RPEREAL_'+kind+r' .*'+fragment+r'.*$',text);need(found is not None,'Control target');return found[0]
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
    field('ram_result',row('RAM'),'data',1,16)
    field('ram_address',row('RAM'),'address')
    field('position',row('COMPLETE'),'position')
    field('coefficient_position',row('BEGIN'),'coefficient_position')
    field('consumer_range',row('INVALID_READ'),'address')
    field('early_hls',row('HSTART'),'cycle',1<<30)
    field('full_sequence',row('COMPLETE'),'sequence',1<<32,16)
    field('full_epoch',row('COMPLETE'),'epoch',1<<32,16)
    field('status',row('COMPLETE','status=5 '),'status')
    field('invalid_publication',row('COMPLETE','status=5 '),'valid')
    field('completion_beats',row('COMPLETE'),'beats')
    field('source_identity',row('BEGIN'),'source_sequence',1<<32,16)
    field('bounds',row('BEGIN'),'rz',1<<63,16)
    field('counter',row('PASS'),'ram')
    first=row('RAM');changes['missing_ram']=text.replace(first+'\n','',1);changes['duplicate_ram']=text.replace(first,first+'\n'+first,1)
    first=row('AR');changes['missing_ar']=text.replace(first+'\n','',1)
    first=row('CANCEL');changes['missing_cancel']=text.replace(first+'\n','',1)
    first=row('REVOKE');changes['missing_revocation']=text.replace(first+'\n','',1)
    changes['missing_pass']=text.replace('RPEREAL_PASS','REMOVED');changes['failure']=text+'\nFAIL injected\n'
    changes['extra_event']=text+'\nRPEREAL_RELEASE cycle=999999 job=65\n'
    for kind in ('HSTART','HDONE'):
        first=row(kind);changes['missing_'+kind]=text.replace(first+'\n','',1)
        changes['duplicate_'+kind]=text.replace(first,first+'\n'+first,1)
    field('computed_word',row('WRITE'),'data',1,16)
    field('computed_address',row('WRITE'),'index')
    field('unchecked_live_output',row('WRITE'),'checked')
    field('core_return',row('HDONE'),'status')
    field('core_write_count',row('HDONE'),'writes')
    field('core_read_count',row('HDONE'),'reads',1<<31)
    field('core_watchdog',row('HDONE'),'quiet',1<<31)
    field('public_core_status',row('COMPLETE'),'hls')
    first=row('WRITE');changes['missing_computed_word']=text.replace(first+'\n','',1)
    changes['duplicate_computed_word']=text.replace(first,first+'\n'+first,1)
    for name,bad in changes.items():
        need(bad!=text,'Changed control')
        try:review(bad,values,seed)
        except (ValueError,AssertionError,KeyError,IndexError):rejected.append(name)
        else:raise ValueError('Accepted corrupt '+name)
    return rejected
