"""Independent transaction/byte replay for the20-job fragment collector suite."""
import re

CASES=[
 ('complete',[0,0],0),('old_identity_and_identical_duplicate',[0,0,2,2,0],0),
 ('conflicting_duplicate',[0,2,2],4),('frame_local_invalids',[2,2,2,3,2,2,2,2,0,0],0),
 ('unsupported_size',[2,0,0],0),('unsupported_burst',[2,0,0],0),
 ('early_last',[2,0,0],0),('late_last',[2,0,0],0),
 ('stop_after_aw',[2],1),('stop_after_w',[2],1),('stop_exposed_line',[0,0],1),
 ('stop_waiting_destination',[0,0],1),('destination_error',[0,0],5),
 ('foreign_destination_response',[0,0],5),('write_after_publication',[0,0,2],6),
 ('missing_half_then_stop',[0],1),('queue_pressure',[0,0],0),
 ('missing_last_long_drain',[2,0,0],0),('late_after_retirement',[0,0,2],0),
 ('stale_after_retirement_rearm',[2,0,0],0),
]
LINE_JOBS=set(range(1,21))-{3,9,10,16}
WINDOW=0x1000fffc0
DESTINATION=0x1fffffffc0
MASK=(1<<64)-1


def need(ok,message):
    if not ok:raise ValueError(message)


def pattern(job,seed):return bytes((job*37+j*19+seed*7)&255 for j in range(64))


def fixture(job,index,seed,prior_b):
    epoch=0x100000000+job if job else 0;tag=0x8000000100001000+job if job else 0
    half=index if index<2 else 0;mask=MASK;address=WINDOW;length=0;size=6;payload_job=job
    magic=0x313046555052544e;bad_offset=False;conflict=False
    beat_halves=None;ident=(prior_b+3)%16
    if job==2:
        half=1 if index<2 else 0
        if index==2:epoch-=1;tag-=1;payload_job=1
        if index==3:tag^=1<<40
    if job==3:half=1 if index==2 else 0;conflict=index==1
    if job==4:
        half=1 if index==9 else 0
        if index==0:mask>>=1
        if index==1:magic^=1
        if index==2:bad_offset=True
        if index==3:address^=1<<32
        if index==4:epoch^=1<<40
        if index==5:tag^=1<<40
        if index==6:mask^=1
        if index==7:mask=0
    if job in {5,6,7,8,18,20}:
        half=1 if index==2 else 0
        if index==0:
            if job==5:size=2
            if job in {6,7}:length=1
            if job in {6,8}:beat_halves=[0,1];ident=9 if job==6 else 7
            if job==18:beat_halves=[i%2 for i in range(260)]+[1];ident=15
            if job==20:epoch-=1;tag-=1;payload_job=19
    if job==9:ident=5
    if job==10:ident=6
    if job==16:half=1
    if job==17:ident=index+1
    if job in {15,19} and index==2:half=0
    beats=[]
    for i,h in enumerate(beat_halves if beat_halves is not None else [half]):
        raw=b''.join(x.to_bytes(8,'little') for x in [magic,epoch,tag,64 if bad_offset else h*32])+pattern(payload_job,seed)[h*32:h*32+32]
        data=int.from_bytes(raw,'little')^((1<<256) if conflict else 0)
        beats.append(dict(data=data,strb=mask,last=int(i==(len(beat_halves)-1 if beat_halves is not None else 0))))
    return dict(addr=address,id=ident,len=length,size=size,burst=1,lock=0,beats=beats)


def review(text,seed):
    need(not re.search(r'^(Fatal:|ERROR:|WARNING:|CRITICAL WARNING:)',text,re.M),'positive diagnostic')
    events=[]
    for l in text.splitlines():
        if not l.startswith('NTRPU_FRAGMENT_'):continue
        name,*values=l.split();f=dict(v.split('=',1) for v in values)
        need(len(f)==len(values),'duplicate field');events.append((name.removeprefix('NTRPU_FRAGMENT_'),f))
    need(events and events[-1][0]=='PASS' and sum(n=='PASS' for n,_ in events)==1,'one terminal PASS')
    job=0;armed=False;address=None;beats=[];bank=[None]*64;responses=0;result=False;line_cycle=None;done_cycle=None
    counts=dict(aw=0,w=0,b=0,lines=0,results=0);foreign=0;last_cycle=0;last_b=0
    rejects=0;identity=0;last_code=0;last_faults=0;stats_pending=False;byte_writes=0;max_beats=0;measurements=[];passes=None
    for kind,f in events:
        if 'cycle' in f:
            cycle=int(f['cycle']);need(cycle>=last_cycle,'event chronology');last_cycle=cycle
        if kind=='CASE':
            need(not stats_pending and address is None and not beats,'pending transport at CASE')
            if job:need(result and responses==len(CASES[job-1][1]),'previous job incomplete')
            else:need(counts==dict(aw=1,w=1,b=1,lines=0,results=0),'unarmed negative')
            job+=1;need(job<=20 and int(f['job'])==job and f['name']==CASES[job-1][0],'case order')
            armed=False;bank=[None]*64;responses=0;result=False;line_cycle=done_cycle=None;foreign=0
        elif kind=='ARM':
            need(job>0 and not armed and not result,'ARM phase')
            need(int(f['epoch'],16)==0x100000000+job and int(f['tag'],16)==0x8000000100001000+job,'full64 ARM identity')
            need(int(f['window'],16)==WINDOW and int(f['dest'],16)==DESTINATION,'ARM address')
            armed=True
        elif kind=='AW':
            need(address is None and (armed or job==0),'AW capture ownership')
            address={k:int(v,16 if k in ['addr','id'] else 10) for k,v in f.items()}
            expected=fixture(job,responses,seed,counts['b'])
            need(all(address[k]==expected[k] for k in ['addr','id','len','size','burst','lock']),'AW fixture')
            counts['aw']+=1
        elif kind=='W':
            need(not beats or beats[-1]['last']==0,'W after frame end')
            beat={k:int(v,16 if k in ['data','strb'] else 10) for k,v in f.items()}
            expected=fixture(job,responses,seed,counts['b'])['beats']
            need(len(beats)<len(expected) and all(beat[k]==expected[len(beats)][k] for k in ['data','strb','last']),'W fixture')
            beats.append(beat);counts['w']+=1
        elif kind=='B':
            need(not stats_pending and address is not None and beats and beats[-1]['last']==1,'B paired frame')
            need(cycle>address['cycle'] and cycle>beats[-1]['cycle'] and int(f['id'],16)==address['id'],'B identity/order')
            intent=fixture(job,responses,seed,counts['b']);need(len(beats)==len(intent['beats']),'full frame drain')
            planned=CASES[job-1][1] if job else [3]
            need(responses<len(planned) and int(f['resp'])==planned[responses],'B response fixture')
            resp=int(f['resp']);first=beats[0];raw=first['data'].to_bytes(64,'little')
            words=[int.from_bytes(raw[i:i+8],'little') for i in range(0,32,8)]
            e=0x100000000+job if job else 0;t=0x8000000100001000+job if job else 0
            faults=int(not armed)
            faults|=int(address['len']!=0 or address['size']!=6 or address['burst']!=1 or address['lock']!=0 or address['addr']%64!=0 or not first['last'] or first['strb']!=MASK)<<1
            faults|=int(address['addr']!=(WINDOW if armed else 0))<<2
            faults|=int(words[0]!=0x313046555052544e)<<3
            faults|=int(words[1]!=e)<<4;faults|=int(words[2]!=t)<<5;faults|=int(words[3] not in [0,32])<<6
            if resp==0:
                need(faults==0 and len(beats)==1 and line_cycle is None,'accepted authorized frame')
                offset=words[3]
                for lane,value in enumerate(raw[32:],offset):
                    need(bank[lane] is None or bank[lane]==value,'conflicting bytes accepted')
                    need(value==pattern(job,seed)[lane],'accepted byte reference')
                    bank[lane]=value;byte_writes+=1
            else:
                rejects+=1;last_faults=faults
                if faults and not faults&~0x30:identity+=1
                if job in [9,10]:last_code=1
                elif job==3:last_code=4
                elif faults&5:last_code=3
                elif faults&(2|8|64):last_code=2
                elif faults&0x30:last_code=7
                elif job in [15,19] and responses==2:last_code=6
                else:raise ValueError('unclassified rejected fixture')
            max_beats=max(max_beats,len(beats));address=None;beats=[];responses+=1;counts['b']+=1;last_b=cycle;stats_pending=True
        elif kind=='STATS':
            need(stats_pending and cycle==last_b,'STATS tied to B')
            need(int(f['rejected'])==rejects and int(f['identity'])==identity and int(f['code'])==last_code and int(f['faults'],16)==last_faults,'rejection telemetry')
            bitmap=sum((v is not None)<<i for i,v in enumerate(bank))
            need(int(f['bitmap'],16)==bitmap,'reference bitmap after accepted/rejected frame')
            error=4 if job==3 and responses>=2 else 1 if job in [9,10] else 6 if job==15 and responses==3 else 0
            need(int(f['failed'])==bool(error) and int(f['error'])==error,'frame-local versus lease failure')
            stats_pending=False
        elif kind=='LINE':
            need(not stats_pending and job in LINE_JOBS and line_cycle is None and all(v is not None for v in bank),'complete unique line')
            need(address is None and not beats and cycle>last_b,'line after transport drain')
            need(int(f['epoch'],16)==0x100000000+job and int(f['tag'],16)==0x8000000100001000+job and int(f['addr'],16)==DESTINATION,'published line descriptor')
            need(int(f['data'],16).to_bytes(64,'little')==bytes(bank),'published line bytes')
            line_cycle=cycle;counts['lines']+=1
        elif kind=='DONE':
            need(line_cycle is not None and cycle>line_cycle and done_cycle is None,'DONE phase')
            need(int(f['epoch'],16)==0x100000000+job,'completion epoch')
            tag=0x8000000100001000+job
            if int(f['tag'],16)!=tag:
                need(job==14 and foreign==0 and int(f['tag'],16)==tag^0x10000 and f['error']=='0','foreign destination fixture');foreign+=1
            else:
                need(int(f['error'])==(job==13),'destination error');done_cycle=cycle
        elif kind=='RESULT':
            n=2 if job==19 else len(CASES[job-1][1])
            need(not result and not stats_pending and address is None and not beats and responses==n,'RESULT after transport')
            need(int(f['epoch'],16)==0x100000000+job and int(f['tag'],16)==0x8000000100001000+job,'RESULT identity')
            need(int(f['status'])==CASES[job-1][2],'lease result status')
            need((done_cycle is not None)==(job in LINE_JOBS) and foreign==(job==14),'all destination responses')
            if done_cycle is not None:need(cycle>done_cycle,'RESULT after memory completion')
            measurements.append(dict(job=job,line_cycle=line_cycle,destination_cycle=done_cycle,result_cycle=cycle,status=int(f['status'])))
            result=True;counts['results']+=1
        elif kind=='PASS':passes={k:int(v) for k,v in f.items()}
        else:raise ValueError('unknown trace event '+kind)
    expected_aw=1+sum(len(c[1]) for c in CASES)
    expected_w=1+sum(len(fixture(j,i,seed,0)['beats']) for j,c in enumerate(CASES,1) for i in range(len(c[1])))
    need(job==20 and result and address is None and not beats and not stats_pending,'terminal state')
    need(counts==dict(aw=expected_aw,w=expected_w,b=expected_aw,lines=16,results=20),'exact transaction counts')
    need(passes['jobs']==20 and passes['seed']==seed and all(passes[k]==v for k,v in counts.items()),'summary counts')
    need(passes['rejected']==rejects==1+sum(r!=0 for c in CASES for r in c[1]) and passes['identity']==identity==5,'summary rejection counts')
    need(max_beats==261 and all(passes[k]>0 for k in ['aw_stall','w_stall','b_stall','line_stall','result_stall']),'drain and backpressure coverage')
    finish=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns|us)\s*:',text,re.M)
    need(len(finish)==1,'one simulator finish')
    final_ps=int(finish[0][0])*{'ps':1,'ns':1000,'us':1000000}[finish[0][1]]
    need(final_ps==(200+passes['cycles']*5)*1000,'cycle/time accounting')
    return dict(status='PASS_FRAGMENT_INGRESS_TRACE',seed=seed,counts=counts,rejected_frames=rejects,rejected_identity=identity,
        accepted_byte_writes=byte_writes,published_bytes=16*64,max_frame_beats=max_beats,summary=passes,final_ps=final_ps,measurements=measurements)
