"""Carry fixture oracle, derived from the frozen standalone ownership checker.

Reconstruct contexts from accepted ARM/line/command/data/completion/retirement
ports, independently of the DUT's phase encoding. Check every active cycle,
including pauses, unsolicited returns and faulted-session handshakes.
"""
import re
from collections import Counter

FIELDS='cycle case rst stop admit restart pfault drained av ar ae at aa ac rv rr re rt lv lr le lt la ld cv cr cc ca ct cl dv dr data pv pr pt ps ov ore oe ot os ops epoch held fault lost invalid exhausted idle phase next rejected'.split()
MAX=(1<<64)-1


def require(ok,message):
    if not ok:raise ValueError(message)


def fields(line):
    pairs=re.findall(r'(\w+)=([^\s]+)',line)
    require(len(pairs)==len(dict(pairs)),'duplicate field')
    return dict(pairs)


def review(text,seed):
    require(not re.search(r'^\s*(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    rows=[fields(l) for l in text.splitlines() if l.startswith('POOL_BRIDGE_TRACE ')]
    begins=[fields(l) for l in text.splitlines() if l.startswith('POOL_BRIDGE_BEGIN ')]
    ends=[fields(l) for l in text.splitlines() if l.startswith('POOL_BRIDGE_PASS ')]
    case_rows=[fields(l) for l in text.splitlines() if l.startswith('POOL_BRIDGE_CASE ')]
    first=0xffffffff
    require(len(begins)==len(ends)==1 and begins[0]==dict(seed=str(seed),first=f'{first:016x}'),'one exact fixture header')
    require([int(r['id']) for r in case_rows]==[102],'complete ordered case inventory')
    require(all(set(r)=={'id','seed','ps'} and int(r['seed'])==seed for r in case_rows),'case marker schema')
    require(rows and [int(r['cycle']) for r in rows]==list(range(1,len(rows)+1)),'unbroken cycle trace')
    counts=Counter({k:0 for k in ['arms','lines','commands','data','completions','done','retires']})
    stalls=Counter();case_counts={i:Counter() for i in [102]}
    ctx=None;epoch=0;cookie=first;exhausted=False;fault=False;lost=False;seen=False;pfault=False;rejected=0
    held={};prior=None;case=None;metrics=[];contexts=[];active_rows=0
    for raw in rows:
        require(set(raw)==set(FIELDS),'exact cycle schema')
        cycle=int(raw['cycle']);case=int(raw['case']);reset=int(raw['rst'],2)==0
        if reset:
            require(all(raw[k]=='0' for k in ['ar','rr','lr','cv','dv','ov']),'reset interface quiescence')
            ctx=None;epoch=0;cookie=first;exhausted=False;fault=False;lost=False;seen=False;pfault=False;rejected=0;held={};prior=None
            continue
        require(case in case_counts,'active cycle outside a case')
        require(all(re.fullmatch(r'[0-9a-fA-F]+',v) for v in raw.values()),'unknown active port')
        s={k:int(v,10 if k in ['cycle','case'] else 16) for k,v in raw.items()};active_rows+=1
        bools='rst stop admit restart pfault drained av ar ac rv rr lv lr cv cr cc dv dr ov ore held fault lost invalid exhausted idle'.split()
        require(all(s[k] in [0,1] for k in bools),'Boolean port width')
        require(all(0<=s[k]<=3 for k in ['pv','pr','ops']) and 0<=s['ps']<=15,'pool port widths')
        require(s['epoch']==epoch and s['held']==bool(ctx),'context ownership/epoch')
        require(s['next']==cookie and s['exhausted']==exhausted,'nonrepeating cookie ledger')
        require(s['fault']==fault and s['rejected']==rejected,'foreign completion accounting')
        loss=bool(s['restart'] or (seen and not s['admit']))
        invalid=lost or loss
        require(s['lost']==lost and s['invalid']==invalid,'sticky availability loss')
        require(s['pr']==(0 if invalid else 3),'completion acceptance policy')
        # Every exposed command, data or DONE is immutable until handshake.
        channels={'command':('cv','cr',['cc','ca','ct','cl']),
                  'data':('dv','dr',['data']), 'done':('ov','ore',['oe','ot','os','ops'])}
        for name,(valid,ready,payload) in channels.items():
            value=tuple(s[k] for k in payload)
            if name in held:require(s[valid]==1 and held.pop(name)==value,'unstable '+name)
            if s[valid] and not s[ready]:held[name]=value;stalls[name]+=1
        def event(name):counts[name]+=1;case_counts[case][name]+=1
        if s['ar']:
            require(ctx is None and not s['stop'] and not invalid and s['admit'] and not s['pfault'] and not pfault and not fault and not exhausted and s['pv']==0 and s['drained'],'ARM admission policy')
            require(s['ae']>epoch and s['aa']%(64)==0 and s['aa']<(1<<37),'full-width ARM validation')
        if s['lr']:require(ctx is not None and not ctx['line'] and not invalid,'line acceptance context')
        if s['rr']:
            require(ctx is not None and not invalid and s['drained'] and s['pv']==0 and s['re']==epoch and s['rt']==ctx['tag'],'retirement identity/drain')
            require(ctx['done_consumed'] or (not ctx['line'] and (ctx['stopped'] or s['stop']) and not s['lv']),'retirement before completion or partial STOP')
        # Compute returns against the ledger as it stood before this edge;
        # a response simultaneous with DATA cannot stand for its completion.
        match=None;foreign=0
        for ch in range(2):
            if not (s['pv'] & s['pr'])>>ch&1:continue
            event('completions');tag=(s['pt']>>(64*ch))&MAX;status=(s['ps']>>(2*ch))&3
            expected=ctx is not None and ctx['data_sent'] and ctx['completion'] is None and ctx['cookie']==tag and ctx['channel']==ch
            if expected:require(match is None,'duplicate matching completion');match=(cycle,status)
            else:foreign+=1
        if s['cv']:
            require(ctx is not None and ctx['line'] and ctx['local_error'] is None and not ctx['command'],'unexpected/repeated command')
            require((s['cc'],s['ca'],s['ct'],s['cl'])==(ctx['channel'],ctx['address'],ctx['cookie'],0),'registered command/cookie')
            if s['cr']:ctx['command']=True;ctx['command_cycle']=cycle;event('commands')
        if s['dv']:
            require(ctx is not None and ctx['command'] and ctx['command_cycle']<cycle and not ctx['data_sent'],'data before command or repeated data')
            require(s['data']==ctx['payload'],'all registered data bytes')
            if s['dr']:ctx['data_sent']=True;ctx['data_cycle']=cycle;event('data')
        if s['ov']:
            require(ctx is not None and ctx['line'] and not ctx['done_consumed'],'unsolicited/repeated DONE')
            require((s['oe'],s['ot'])==(epoch,ctx['tag']),'full original DONE identity')
            if ctx['published'] is None:
                if ctx['local_error'] is not None:expected=ctx['local_error'];raw_status=0
                else:
                    require(ctx['command'] and ctx['data_sent'] and ctx['completion'] is not None,'DONE before matching destination completion')
                    require(prior is not None and prior['drained'] and not prior['pv'] and not prior['invalid'],'DONE publication without drain/live session')
                    raw_status=ctx['completion'][1]
                    expected=1 if ctx['stopped'] or prior['stop'] else 4 if raw_status else 5 if fault or pfault or prior['pfault'] else 0
                require((s['os'],s['ops'])==(expected,raw_status),'DONE status mapping')
                ctx['published']=(s['os'],s['ops']);ctx['done_cycle']=cycle
            require((s['os'],s['ops'])==ctx['published'],'published DONE changed')
            if s['ore']:ctx['done_consumed']=True;event('done')
        if s['av'] and s['ar']:
            epoch=s['ae'];ctx=dict(epoch=epoch,tag=s['at'],address=s['aa'],channel=s['ac'],arm_cycle=cycle,
                line=False,command=False,data_sent=False,completion=None,published=None,done_consumed=False,
                stopped=False,local_error=None,cookie=None,payload=None)
            event('arms')
        if s['lv'] and s['lr']:
            require(ctx is not None,'line without ARM');ctx['line']=True;ctx['line_cycle']=cycle;event('lines')
            if (s['le'],s['lt'],s['la'])!=(ctx['epoch'],ctx['tag'],ctx['address']):ctx['local_error']=2;fault=True
            elif fault or foreign or s['pfault'] or pfault:ctx['local_error']=5
            else:
                job=ctx['epoch']-0x100000001
                payload=int.from_bytes(bytes((job*37+i*23+seed*11)&255 for i in range(64)),'little')
                require(s['ld']==payload,'independent payload fixture')
                ctx['payload']=payload;ctx['cookie']=cookie
                if cookie==MAX:exhausted=True
                else:cookie+=1
        if match is not None:ctx['completion']=match
        if ctx is not None and s['stop']:ctx['stopped']=True
        if s['rv'] and s['rr']:
            event('retires');contexts.append(dict(case=case,epoch=ctx['epoch'],cookie=ctx['cookie'],channel=ctx['channel'],status=ctx['published'],partial_stop=not ctx['line']))
            if ctx['published'] is not None and ctx['completion'] is not None:
                metrics.append(dict(case=case,epoch=ctx['epoch'],command_to_data_cycles=ctx['data_cycle']-ctx['command_cycle'],
                    completion_to_done_cycles=ctx['done_cycle']-ctx['completion'][0],line_to_retirement_cycles=cycle-ctx['line_cycle']))
            ctx=None
        if foreign:fault=True;rejected=min(0xffffffff,rejected+foreign)
        pfault=pfault or bool(s['pfault']);lost=invalid;seen=seen or bool(s['admit']);prior=s
    require(ctx is None and not held,'terminal live transfer/context')
    require(all(case_counts[k]['arms']>0 for k in case_counts if k!=40),'all active cases exercised')
    require(counts==dict(arms=3,lines=3,commands=3,data=3,completions=3,done=3,retires=3),'carry conservation')
    require(not exhausted and cookie==0x100000002 and [c['cookie'] for c in contexts]==[0xffffffff,0x100000000,0x100000001],'64-bit carry across low word')
    end=ends[0]
    require(set(end)=={'seed','first','cases','cycles',*counts},'summary schema')
    require(int(end['seed'])==seed and int(end['first'],16)==first and int(end['cases'])==len(case_rows) and int(end['cycles'])==len(rows),'summary fixture identity')
    require(all(int(end[k])==v for k,v in counts.items()),'independent handshake summary')
    require(all(stalls[k]>0 for k in ['command','data','done']),'backpressure coverage')
    return dict(status='PASS_FRAGMENT_POOL_BRIDGE_CARRY_PROTOCOL',qualified=True,seed=seed,profile="carry32",
        cases=len(case_rows),cycles=len(rows),active_cycles=active_rows,handshakes=dict(counts),stalls=dict(stalls),
        contexts=contexts,latencies=metrics,scope='One reservation, exclusive two-channel pool interface; both peers are protocol drivers. Cold coordinated reset only. No pool RTL, CDC, MIG, PCIe, physical timing or throughput qualification.')
