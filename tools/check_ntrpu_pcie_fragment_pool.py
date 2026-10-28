"""Independent port/transaction oracle for the exclusive F01/pool composition.

Reconstructs full frames, payloads, lease lifetime and transport cookies. Does
not use DUT phase encodings to decide ownership or successful completion.
"""
import re
from collections import Counter

FIELDS='cycle case rst stop stopped isolated qe admit restart pf drained av ar car bar ae at aw aa ac kv kr ke kt rr awv awr awa awi awl aws awb awk wv wr wd ws wl bv br bi bs lv lr le lt la ld cv cr cc ca ct cl dv dr data pv pr pt ps ov ore oe ot os ops err rv rc re rt rs epoch held be bh idle phase fail error bytes ti fault lost invalid exhausted next rejected frames identity code faults'.split()
MAX=(1<<64)-1
CASES=list(range(1,12))
LINE_CASES={1,4,5,6,7,8,9,11}
DONE={1:0,4:0,5:1,6:1,7:4,8:1,9:0}
RESULT={1:0,2:1,3:4,4:0,5:1,6:1,7:5,8:6,9:0,10:1}
FRAMES={1:[(0,0,0),(1,0,0),(0,0,2)],2:[(0,0,0)],3:[(0,0,0),(0,1,2)],
        4:[(0,2,2),(0,0,0),(1,0,0)],5:[(0,0,0),(1,0,0)],6:[(0,0,0),(1,0,0)],
        7:[(0,0,0),(1,0,0)],8:[(0,0,0),(1,0,0),(0,0,2)],9:[(0,0,0),(1,0,0)],
        10:[(0,0,0)],11:[(0,0,0),(1,0,0)]}

def require(ok,message):
    if not ok:raise ValueError(message)

def fields(line):
    pairs=re.findall(r'(\w+)=([^\s]+)',line)
    require(len(pairs)==len(dict(pairs)), 'duplicate field')
    return dict(pairs)

def payload(job,seed):
    return int.from_bytes(bytes((job*37+k*19+seed*7)&255 for k in range(64)), 'little')

def review(text,seed):
    require(not re.search(r'^\s*(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    rows=[fields(x) for x in text.splitlines() if x.startswith('OWN_TRACE ')]
    begin=[fields(x) for x in text.splitlines() if x.startswith('OWN_BEGIN ')]
    cases=[fields(x) for x in text.splitlines() if x.startswith('OWN_CASE ')]
    ends=[fields(x) for x in text.splitlines() if x.startswith('OWN_PASS ')]
    require(begin==[dict(seed=str(seed),first='00000000ffffffff')], 'one exact BEGIN')
    require(cases==[dict(id=str(j),seed=str(seed)) for j in CASES], 'ordered fixture cases')
    require(len(ends)==1,'one PASS')
    require(rows and [int(x['cycle']) for x in rows]==list(range(1,len(rows)+1)), 'complete trace from cycle1')
    counts=Counter({k:0 for k in ['arms','lines','commands','data','done','acks','frames','rejects','completions','publications']})
    bycase={j:Counter() for j in CASES};stalls=Counter();metrics=[];blocked=Counter()
    ctx=None;epoch=0;active=False;seen=False;lost=False;cookie=0xffffffff
    hold={};aw=None;wd=None;pending=None;frame_index=Counter();prior_b=False
    rejects=0;identity=0;resets=0;reset_seen=False;observed_cases=[];previous=None
    final=None
    for raw in rows:
        require(set(raw)==set(FIELDS),'trace schema')
        cycle=int(raw['cycle']);job=int(raw['case'])
        if raw['rst']=='0':
            if not reset_seen:resets+=1
            reset_seen=True
            require(all(raw[k]=='0' for k in ['ar','kr','rc','awr','wr','cv','dv','ov']), 'cold reset quiescence')
            ctx=None;epoch=0;active=False;seen=False;lost=False;cookie=0xffffffff
            hold={};aw=None;wd=None;pending=None;prior_b=False;rejects=identity=0;previous=None
            continue
        reset_seen=False
        require(all(re.fullmatch('[0-9a-f]+',v) for v in raw.values()),'known active ports')
        s={k:int(v,10 if k in ['cycle','case'] else 16) for k,v in raw.items()}
        if job and job not in observed_cases:observed_cases.append(job)
        require(s['epoch']==s['be']==epoch, 'joint retained epoch')
        require(s['held']==bool(ctx) and s['bh']==active, 'collector retained context versus active pool lease')
        require(s['next']==cookie and not s['exhausted'], 'full64 cookie allocation')
        invalid=lost or s['restart'] or (seen and not s['admit'])
        require(s['lost']==lost and s['invalid']==bool(invalid), 'sticky session lifetime')
        require(s['fault']==s['rejected']==0, 'no synthetic protocol fault')
        require(s['err']==(s['os'] not in (0,1)), 'explicit DONE status translation')
        require(s['pr']==(0 if invalid else 3), 'completion drain policy')
        require(s['ar']==(s['car'] and s['bar']), 'joint ready')
        require(s['rc']==(s['kv'] and s['kr']), 'single atomic result consumption')
        barrier=s['stopped'] and s['isolated'] and s['qe']==epoch
        transport=s['ti'] and not s['awv'] and not s['wv']
        ack_identity=s['ke']==s['re'] and s['kt']==s['rt']
        expected_ready=s['rv'] and ack_identity and barrier and transport and s['rr'] and not invalid
        require(s['kr']==bool(expected_ready), 'ACK full identity/live/drain/barrier predicate')
        if s['kv'] and not s['kr']:
            for reason,value in [('identity',ack_identity),('barrier',barrier),('transport',transport),('retirement',s['rr']),('session',not invalid)]:
                if not value:blocked[reason]+=1
        channels={'B':('bv','br',['bi','bs']), 'line':('lv','lr',['le','lt','la','ld']),
                  'command':('cv','cr',['cc','ca','ct','cl']), 'data':('dv','dr',['data']),
                  'DONE':('ov','ore',['oe','ot','os','ops']), 'result':('rv','rc',['re','rt','rs'])}
        for name,(v,r,p) in channels.items():
            value=tuple(s[k] for k in p)
            if name in hold:require(s[v] and hold.pop(name)==value,'stable held '+name)
            if s[v] and not s[r]:hold[name]=value;stalls[name]+=1
        def event(name):counts[name]+=1;bycase[job][name]+=1
        if s['ar']:
            require(not active and not s['stop'] and not invalid and s['admit'] and s['drained'] and barrier and transport and not s['rv'], 'ARM admission proof')
            require(s['ae']>epoch and s['aa']<1<<37 and s['aa']%64==s['aw']%64==0,'descriptor range/epoch')
        if s['av'] and s['ar']:
            require(job in CASES and bycase[job]['arms']==0, 'one ARM per fixture case')
            require(s['ae']==(epoch+1 if epoch else 0x100000001) and s['at']==0x800000010000abcd,'full64 fixture identity')
            require(s['ac']==job%2 and s['aa']==(0x1fffffffc0 if job%2 else 0x100000040) and s['aw']==0x1000fffc0,'explicit channel and high local address')
            ctx=dict(job=job,epoch=s['ae'],tag=s['at'],address=s['aa'],channel=s['ac'],
                     line=None,command=None,data=None,cpl=None,done=None,result=None,acked=False,cookie=None)
            active=True;epoch=s['ae'];event('arms')
        if s['lv']:
            require(active and ctx and job in LINE_CASES and ctx['line'] is None,'one LINE under lease')
            require((s['le'],s['lt'],s['la'],s['ld'])==(ctx['epoch'],ctx['tag'],ctx['address'],payload(job,seed)), 'assembled payload/identity')
            require(s['bytes']==MAX, 'complete byte coverage before LINE')
            if s['lr']:ctx['line']=cycle;ctx['cookie']=cookie;cookie+=1;event('lines')
        if s['cv']:
            require(active and ctx and ctx['line'] is not None and ctx['command'] is None,'unique command after LINE')
            require((s['cc'],s['ca'],s['ct'],s['cl'])==(ctx['channel'],ctx['address'],ctx['cookie'],0),'registered channel/address/cookie')
            if s['cr']:ctx['command']=cycle;event('commands')
        if s['dv']:
            require(active and ctx and ctx['command'] is not None and ctx['command']<cycle and ctx['data'] is None,'unique data after command')
            require(s['data']==payload(job,seed),'all 64 data bytes')
            if s['dr']:ctx['data']=cycle;event('data')
        takes=s['pv']&s['pr']
        if takes:
            require(active and ctx and ctx['data'] is not None and ctx['data']<cycle and ctx['cpl'] is None,'owned response after data')
            require(takes==1<<ctx['channel'] and s['pt']==ctx['cookie']<<(64*ctx['channel']), 'full cookie/channel completion')
            expected=3 if job==7 else 0
            require(s['ps']==expected<<(2*ctx['channel']), 'fixture completion status')
            ctx['cpl']=cycle;event('completions')
        if s['ov']:
            require(active and ctx and ctx['cpl'] is not None and ctx['cpl']<cycle and ctx['done'] is None, 'DONE requires actual completion')
            require((s['oe'],s['ot'],s['os'],s['ops'])==(ctx['epoch'],ctx['tag'],DONE[job],3 if job==7 else 0),'DONE identity/status')
            require(not invalid and s['drained'], 'drained live DONE in this fixture')
            if s['ore']:ctx['done']=cycle;event('done')
        if s['rv']:
            require(ctx and (active or ctx['acked']), 'retained result context')
            require((s['re'],s['rt'],s['rs'])==(ctx['epoch'],ctx['tag'],RESULT[job]),'visible RESULT full identity/status')
            if ctx['result'] is None:
                require(barrier and transport, 'RESULT external epoch barrier and drained AXI')
                require(ctx['done'] is not None or job in (2,3,10), 'no result before destination completion')
                ctx['result']=cycle;event('publications')
        if s['kr']:
            require(active and ctx and ctx['result'] is not None and not invalid and s['drained'] and not s['pv'], 'ACK cannot discard active/invalid destination')
            require(ctx['done'] is not None or job in (2,3,10), 'ACK completion or canceled partial line')
        if s['rc']:
            require(active and ctx and not ctx['acked'] and s['ke']==ctx['epoch'] and s['kt']==ctx['tag'],'one ACK for full context')
            ctx['acked']=True;active=False;event('acks')
            metrics.append(dict(job=job,line_to_ack_cycles=cycle-ctx['line'] if ctx['line'] else None,
                                command_to_ack_cycles=cycle-ctx['command'] if ctx['command'] else None))
        # Independently pair AXI AW/W and check the complete 64-byte frame.
        if s['awv'] and s['awr']:
            require(aw is None and pending is None,'one outstanding AW')
            require((s['awa'],s['awi'],s['awl'],s['aws'],s['awb'],s['awk'])==(0x1000fffc0,job%16,0,6,1,0),'frame AW')
            aw=job
        if s['wv'] and s['wr']:
            require(wd is None and pending is None,'one outstanding W')
            require(s['ws']==MAX and s['wl']==1,'full framed W')
            wd=s['wd']
        if aw is not None and wd is not None:
            require(aw==job and ctx is not None,'AW/W lease pairing')
            idx=frame_index[job];require(idx<len(FRAMES[job]),'no extra frame')
            half,variant,resp=FRAMES[job][idx]
            data=payload(job,seed)^(1 if variant==1 else 0)
            e=ctx['epoch']^((1<<32) if variant==2 else 0)
            expected=0x313046555052544e | e<<64 | ctx['tag']<<128 | (half*32)<<192 | ((data>>(half*256))&((1<<256)-1))<<256
            require(wd==expected,'exact F01 header and all payload bytes')
            pending=(resp,variant,job);frame_index[job]+=1;aw=wd=None
        if s['bv']:
            require(pending is not None and (s['bi'],s['bs'])==(pending[2]%16,pending[0]),'B matches staged frame')
            if not prior_b:
                if pending[0]:rejects+=1
                if pending[1]==2:identity+=1
            if s['br']:
                event('frames')
                if pending[0]:event('rejects')
                pending=None
        require((s['frames'],s['identity'])==(rejects,identity),'frame reject ledger')
        prior_b=bool(s['bv'] and not s['br'])
        seen=seen or bool(s['admit']);lost=bool(invalid);previous=s;final=s
    require(observed_cases==CASES and resets==3, 'complete case/reset sequence')
    require(frame_index==Counter({j:len(v) for j,v in FRAMES.items()}),'all fixture frames present')
    expected=dict(arms=11,lines=8,commands=8,data=8,done=7,acks=8,frames=23,rejects=4,completions=7,publications=10)
    require(dict(counts)==expected,'complete transaction inventory')
    for j,c in bycase.items():
        require(c['arms']==1 and c['lines']==int(j in LINE_CASES) and c['acks']==int(j<=8),'per-case ownership inventory')
        require(c['commands']==c['data']==int(j in LINE_CASES) and c['done']==int(j in DONE),'per-case transfer inventory')
    require(ends[0]==dict(seed=str(seed),cycles=str(len(rows)),**{k:str(expected[k]) for k in ['arms','lines','commands','data','done','acks','frames','rejects']}),'PASS counters match observed events')
    require(all(blocked[x]>0 for x in ['identity','barrier','transport','retirement','session']),'ACK negative cases exercised')
    require(all(stalls[x]>0 for x in ['B','command','data','result']),'intentional backpressure exercised')
    require(final['invalid']==final['bh']==1 and not final['rv'] and not final['ov'],'final exposed-loss lease retained without completion')
    require(aw is None and wd is None and pending is None,'final ingress drain')
    return dict(seed=seed,cycles=len(rows),counts=dict(counts),per_case={j:dict(c) for j,c in bycase.items()},
                blocked_ack_cycles=dict(blocked),stalls=dict(stalls),metrics=metrics,reset_intervals=resets,
                scope='Trusted descriptor/ACK, one exclusive pool owner; protocol peers, no MIG/PCIe/CDC/physical timing.')
