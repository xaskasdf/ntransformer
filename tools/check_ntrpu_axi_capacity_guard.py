"""Reconstruct both AXI boundaries of the capacity guard from public traces."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

CAPACITY=1 << 37
MASK64=(1 << 64)-1


def specifications():
    addresses=[0,2**32,CAPACITY-64,CAPACITY-4096,CAPACITY,2**48+65536,
               2**64-64,4032,65,131072,196608,262144,327680,393216,
               CAPACITY-4096,2**32+4096,2**32+4032,CAPACITY-128,458752,524288]
    lengths=[1,4,1,64,1,4,2,2,1,4,4,4,1,256,1,64,1,2,4,8]
    rows={}
    for ident,(addr,beats) in enumerate(zip(addresses,lengths),1):
        size=5 if ident==10 else 6
        burst=0 if ident==11 else 2 if ident==12 else 1
        locked=int(ident==13)
        valid=size==6 and burst==1 and not locked and addr%64==0 and addr%4096+beats*64<=4096 and addr+beats*64<=CAPACITY
        rows[ident]=dict(addr=addr,beats=beats,size=size,burst=burst,lock=locked,allow=int(valid))
    return rows


def word(address,tag,seed):
    address &= MASK64
    return int.from_bytes(bytes((address//64+lane*13+tag*29+seed*11)%256 for lane in range(64)),'little')


def strobe(ident,beat):
    return [MASK64,0,0x5555555555555555,0x8000000000000001][(ident+beat)%4]


def pack(fields):
    result=0
    for value,width in fields:
        assert 0<=value<2**width
        result=(result<<width)|value
    return result


def review(text,seed,require_hold=True,require_orders=True):
    result=dict(status='FAIL',errors=[],seed=seed)
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        need(not re.search(r'(?mi)^\s*(ERROR:|Fatal:|CRITICAL WARNING:|FAIL\b)',text),'simulation error')
        need('NTRPU_CAP_INCOMPLETE' not in text,'incomplete simulation')
        rows=[];by=defaultdict(list)
        for line_number,line in enumerate(text.splitlines()):
            if not line.startswith('NTRPU_CAP_'):continue
            marker=line.split()[0].removeprefix('NTRPU_CAP_')
            r={k:int(v,16 if k in {'addr','data','strb','sample'} else 10) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
            rows.append((line_number,marker,r));by[marker].append(r)
        need(by['CONFIG']==[dict(seed=seed,period_ps=5000,memory_addr_width=37)],'configuration')
        specs=specifications();ids=list(specs)
        need([r['case'] for r in by['BEGIN']]==ids and [r['case'] for r in by['END']]==ids,'ordered twenty cases')
        need(len(by['PASS'])==1,'one completion')
        summary=by['PASS'][0]
        need(all(summary[k]==v for k,v in dict(cases=20,allowed=10,rejected=10,sw=429,mw=150,sr=429,mr=150).items()),'whole-bank totals')
        need(summary['closed_cycles']>20,'closed admission coverage')
        need(len(by['ADMIT'])==2 and [(r['case'],r['value']) for r in by['ADMIT']]==[(1,1),(20,0)],'admission schedule')
        open_at,close_at=by['ADMIT'];release=by['TARGET_RELEASE']
        need(len(release)==1 and release[0]['case']==20 and release[0]['cycle']-close_at['cycle']>=8,'target stall across close')
        for r in by['ADMIT']+release:
            need(r['ps']==r['cycle']*5000,'admission/target changes on negative edge')
        per=defaultdict(lambda:defaultdict(list));events=defaultdict(list);holds=Counter()
        current=None;start_cycle=None
        for line,kind,r in rows:
            if kind=='BEGIN':
                current=r['case'];start_cycle=r['cycle'];spec=specs[current]
                need(all(r[k]==v for k,v in spec.items()),'independent descriptor '+str(current))
                need(r['ps']==r['cycle']*5000 and r['admit']==int(current!=1),'case starts on a negative edge')
            elif kind=='END':
                need(r['case']==current and r['cycle']>start_cycle and r['idle']==1 and r['admit']==int(current!=20),'drained case')
                spec=specs[current]
                need(r['sw']==r['sr']==spec['beats'] and r['mw']==r['mr']==spec['beats']*spec['allow'],'case beat totals')
                current=None
            elif kind in {'ADDR','W','R','B','HOLD'}:
                need(current is not None and r['case']==current and r['cycle']>=start_cycle,'event belongs to current case')
                spec=specs[current];beats=spec['beats'];allowed=spec['allow']
                if kind=='HOLD':
                    ch=r['channel'];need(0<=ch<10,'hold channel')
                    sample=r['sample'];holds[ch]+=1
                    events[current,ch].append((r['cycle'],False,sample))
                    continue
                side=r['side'];need(side in (0,1),'boundary identity')
                need(r['ps']==r['cycle']*5000+2500,'handshake clock edge')
                if side:need(allowed,'rejected access cannot reach memory')
                if kind=='ADDR':
                    wr=r['write'];need(wr in (0,1),'address direction')
                    expected=dict(addr=spec['addr'],beats=beats,size=spec['size'],burst=spec['burst'],lock=spec['lock'],
                                  aid=current*(5 if wr else 3)%16,cache=(current+wr)%16,prot=(current+wr)%8)
                    need(all(r[k]==v for k,v in expected.items()),'address/ID/attribute preservation')
                    key=('AW' if wr else 'AR',side);ch=(4 if wr else 3) if side else wr
                    sample=pack([(r['addr'],37 if side else 64),(beats-1,8),(r['size'],3),(r['burst'],2),(r['aid'],4),(r['cache'],4),(r['prot'],3),(r['lock'],1)])
                    if side==0:
                        need(open_at['ps']<r['ps']<close_at['ps'],'new descriptors require admission')
                elif kind in {'W','R'}:
                    key=(kind,side);beat=len(per[current][key]);need(r['beat']==beat and beat<beats,'ordered beat count')
                    need(r['last']==int(beat==beats-1),'exact LAST')
                    expected=word(spec['addr']+64*beat,current+(32 if kind=='W' else 0),seed)
                    if kind=='R' and not allowed:expected=0
                    need(r['data']==expected,'independent '+kind+' payload')
                    if kind=='W':
                        need(r['strb']==strobe(current,beat),'write strobe preservation')
                        ch=5 if side else 2;sample=pack([(r['data'],512),(r['strb'],64),(r['last'],1)])
                    else:
                        need(r['aid']==current*3%16 and r['resp']==(3 if not allowed else 2 if current==19 and beat==2 else 0),'RID/read response')
                        ch=6 if side else 8;sample=pack([(r['data'],512),(r['resp'],2),(r['aid'],4),(r['last'],1)])
                else:
                    key=('B',side);ch=7 if side else 9
                    need(r['aid']==current*5%16 and r['resp']==(3 if not allowed else 2 if current==19 else 0),'BID/write response')
                    sample=pack([(r['resp'],2),(r['aid'],4)])
                per[current][key].append(r)
                events[current,ch].append((r['cycle'],True,sample))
        need(current is None,'last case completed')
        concurrent=early_w=late_w=0
        for ident,spec in specs.items():
            p=per[ident];beats=spec['beats'];allowed=spec['allow']
            for side in (0,1):
                factor=1 if side==0 else allowed
                for key,count in [('AW',1),('AR',1),('W',beats),('R',beats),('B',1)]:
                    need(len(p[key,side])==count*factor,'exact channel coverage '+str((ident,key,side)))
            sa,saw,sb=p['AR',0][0],p['AW',0][0],p['B',0][0]
            sw,sr=p['W',0],p['R',0]
            need(sr[0]['cycle']>sa['cycle'] and sb['cycle']>max(saw['cycle'],sw[-1]['cycle']),'source response ordering')
            concurrent+=int(max(sa['cycle'],saw['cycle'])<min(sb['cycle'],sr[-1]['cycle']))
            if allowed:
                ma,maw,mb=p['AR',1][0],p['AW',1][0],p['B',1][0]
                mw,mr=p['W',1],p['R',1]
                need(ma['cycle']>sa['cycle'] and maw['cycle']>saw['cycle'],'registered address capture')
                need(mr[0]['cycle']>ma['cycle'] and mb['cycle']>max(maw['cycle'],mw[-1]['cycle']),'target response ordering')
                need(sb['cycle']>mb['cycle'],'registered B response')
                for source,target in zip(sw,mw):need(target['cycle']>source['cycle'],'registered W queue')
                for target,source in zip(mr,sr):need(source['cycle']>target['cycle'],'registered R queue')
                early_w+=int(mw[0]['cycle']<maw['cycle']);late_w+=int(mw[0]['cycle']>maw['cycle'])
            else:
                need(not any(p[k,1] for k in ['AR','AW','W','R','B']),'no downstream activity for rejects')
        need(concurrent==20,'independent read/write progress')
        need(late_w>0,'AW before downstream W exercised')
        if require_orders:
            need(early_w>0 and per[2]['W',1][1]['cycle']<per[2]['AW',1][0]['cycle'],
                 'two downstream W beats accepted before AW')
            need(any(r['channel']==2 and r['cycle']<per[r['case']]['AW',0][0]['cycle']
                     for r in by['HOLD']), 'upstream W offered before AW')
        for kind in ['AR','AW']:
            need(per[20][kind,0][0]['ps']<close_at['ps']<release[0]['ps']<per[20][kind,1][0]['ps'],'pending address retained across admission close')
        need(by['END'][-1]['ps']>max(per[20]['B',0][0]['ps'],per[20]['R',0][-1]['ps']),'closed drain completes')
        stall_summary=by['STALL']
        need([r['channel'] for r in stall_summary]==list(range(10)),'ten stall observations')
        need(all(r['cycles']>0 for r in stall_summary if r['channel']!=7),'stall coverage')
        if require_hold:
            need(bool(by['HOLD']),'per-cycle held payload evidence')
            need(all(holds[r['channel']]==r['cycles'] for r in stall_summary),'complete held-cycle counts')
            for key,stream in events.items():
                stream.sort()
                need(len({e[0] for e in stream})==len(stream),'one event per channel and cycle')
                for i,(cycle,accepted,value) in enumerate(stream):
                    if not accepted:
                        need(i+1<len(stream) and stream[i+1][0]==cycle+1 and stream[i+1][2]==value,'stable VALID/payload until handshake '+str(key))
        result.update(cases=20,allowed=10,rejected=10,source_beats_per_direction=429,target_beats_per_direction=150,
                      concurrent_cases=concurrent,downstream_w_before_aw_cases=early_w,downstream_aw_before_w_cases=late_w,
                      held_cycles=sum(holds.values()),stalls=stall_summary,complete_hold_trace=require_hold,
                      both_aw_w_orders=require_orders,
                      closed_cycles=summary['closed_cycles'],end_cycle=by['END'][-1]['cycle'],
                      scope='Standalone registered64-to37bit AXI guard, deterministic target fixture; no MIG, buffer authorization, runtime reset recovery or board timing.')
    except (ValueError,KeyError,IndexError,OverflowError,AssertionError) as exc:
        result['errors'].append(str(exc))
    result['status']='FAIL' if result['errors'] else 'PASS'
    return result


def mutations(text,require_hold=True):
    out={}
    markers=['CONFIG','PASS','BEGIN','END','ADDR','W','R','B','ADMIT','TARGET_RELEASE','STALL']+(['HOLD'] if require_hold else [])
    for marker in markers:
        out['missing_'+marker]=re.sub(r'^NTRPU_CAP_'+marker+r' .+\n','',text,count=1,flags=re.M)
    for marker,key in [('ADDR','addr'),('ADDR','aid'),('ADDR','cache'),('ADDR','beats'),('W','strb'),('W','data'),
                       ('W','last'),('R','data'),('R','resp'),('B','resp'),('B','aid'),('R','ps'),('END','idle')]+([('HOLD','sample'),('HOLD','cycle')] if require_hold else []):
        pattern=r'(^NTRPU_CAP_'+marker+r' .*?\b'+key+r'=)([0-9a-f]+)'
        base=16 if key in {'addr','data','strb','sample'} else 10
        def change(m,base=base):
            value=int(m[2],base)^1
            return m[1]+(format(value,'x') if base==16 else str(value))
        out['corrupt_'+marker+'_'+key]=re.sub(pattern,change,text,count=1,flags=re.M)
    out['fatal']=text+'\nFatal: injected protocol failure\n'
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('log',type=Path);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    result=review(args.log.read_text(encoding='utf-8'),args.seed)
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
