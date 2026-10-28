"""Replay the registered-head variant, including exact FIFO-to-cache transfer and real payload."""
import argparse
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import re


def need(ok,message):
    if not ok:raise ValueError(message)


def pattern(tag,beat):
    return sum((((tag*0x9e3779b9)^(beat*0x10203)^(lane*0x01010101))&0xffffffff)<<(lane*32) for lane in range(16))


def workload(case):
    if case==1:return [(j%2,j+1,(1,8,64)[j%3]) for j in range(24)]
    if case==2:return [(0,100+j,64) for j in range(8)]+[(1,120+j,64) for j in range(12)]
    if case==3:return [(0,200+j,1) for j in range(24)]+[(1,230+j,8) for j in range(8)]
    return {4:[(0,260,8),(1,261,8)],5:[(0,300,64),(1,301,8)],
            6:[(0,400,1),(1,401,64),(0,402,8)],7:[(0,450,64),(1,451,8)],8:[(0,480,1),(0,481,1)]}[case]


def descriptor(tag,n):
    return (0x76543210<<200)|(tag<<136)|((0x100000000+tag*4096)<<72)|((0x800000000+tag*4096)<<8)|(n-1)


def replay(text):
    jobs={};case=None;lease=None;barrier=None;next_event={};submitted=[];queued=[deque(),deque()]
    owners={};generations=[0]*8;sample=None;sticky=False;cases=[];coverage=Counter();last=-1;last_sample=None;returned=set()
    def finish():
        expected=workload(case)
        need(submitted==expected,'exact submitted workload')
        quarantined={301} if case==5 else {402} if case==6 else set()
        group=[jobs[t] for _,t,_ in expected]
        need({j['tag'] for j in group if 'res' not in j}==quarantined,'explicit quarantine only')
        need(all('cpl' in j and len(j['input'])==len(j['output'])==j['n'] and 'done' in j for j in group if j['tag'] not in quarantined),'all bytes and completions')
        need(not owners and lease is None and barrier is None,'case bank/lease/barrier drain')
        need([list(q) for q in queued]==[[t for ch,t,n in expected if ch==p and t in quarantined] for p in (0,1)],'remaining queued jobs')
        need(sticky==(case in (5,6)),'case fault expectation')
        cases.append(dict(case=case,submitted=len(expected),completed=len(expected)-len(quarantined),quarantined=sorted(quarantined),
                          bytes=sum(j['n']*64 for j in group if j['tag'] not in quarantined)))
    for line in text.splitlines():
        if not line.startswith('RS '):continue
        f=line.split();kind=f[1]
        if kind=='CASE_PASS':continue
        if kind=='SUITE_PASS':continue
        if kind=='CASE':
            if case is not None:finish()
            case=int(f[3]);need(case==len(cases)+1,'case order')
            lease=None;barrier=None;queued=[deque(),deque()];submitted=[];owners={};generations=[0]*8;sticky=False;last_sample=None
            next_event={p:{k:deque() for k in ('aw','w','b','cpl')} for p in (0,1)}
            continue
        now=int(f[2]);need(now>=last,'trace chronology');last=now
        if kind=='S':
            names='cycle case stop external_fault fault drained banks queued can take ready lease fv fr pv pr dv dr bv br ov ore hold_w hold_b hold_cpl hold_payload'.split()
            need(len(f[2:])==len(names),'sample schema')
            sample={k:int(v,10 if k in ('cycle','case') else 16 if k in ('banks','queued') else 2) for k,v in zip(names,f[2:])}
            s=sample;need(s['case']==case and (last_sample is None or now==last_sample+1),'every active cycle sampled');last_sample=now
            need(not sticky or s['fault'],'fault remains latched');sticky=bool(s['fault'])
            need(s['queued']==len(queued[0])+(len(queued[1])<<4),'frontend queue counts')
            need(s['lease']==(lease is not None),'single reserved lease')
            need(s['banks']==sum(1<<b for b in owners),'physical bank ownership')
            returned=set()
            if s['take']:
                need(s['take'] in (1,2) and s['take']&s['can'] and s['ready'] and lease is None and not s['fault'],'atomic resource acceptance')
            if s['drained']:
                need(not any(queued) and lease is None and not owners and barrier is None and
                     all('cpl' in j for j in jobs.values() if j['case']==case),'no premature drained')
            if case==2 and s['hold_w'] and s['can']&2 and not s['can']&1 and queued[0]:coverage['peer_with_full_banks']+=1
            live0=sum('res' in j and 'cpl' not in j and j['ch']==0 for j in jobs.values())
            if case==3 and live0==16 and not s['banks'] and not(s['can']&1):coverage['descriptor_limit_with_free_banks']+=1
            if s['stop'] and s['take']:coverage['stop_drains_queue']+=1
            if s['fault'] and s['pv'] and s['pr']:coverage['payload_after_fault']+=1
            if s['hold_payload'] and s['lease'] and s['banks']:coverage['bank_held_without_payload']+=1
            if s['fv'] and not s['fr']:coverage['fetch_stall']+=1
            if s['ov'] and not s['ore']:coverage['fence_result_stall']+=1
            continue
        need(sample is not None and sample['cycle']==now,'event has cycle sample')
        if kind=='SUB':
            ch,t=int(f[3]),int(f[4]);d=int(f[5],16);n=(d&255)+1
            need(t not in jobs and d==descriptor(t,n),'unique exact descriptor')
            need(not sample['stop'] and not sample['fault'] and barrier is None and not sample['bv'],'closed publication')
            jobs[t]=dict(tag=t,ch=ch,n=n,case=case,sub=now,input=[],output=[]);submitted.append((ch,t,n));queued[ch].append(t)
        elif kind=='RETURN':
            ch,bank,gen=map(int,f[3:6]);need(bank in owners and bank not in returned,'owned bank returns once')
            j=jobs[owners.pop(bank)]
            need(ch==j['ch'] and gen==j['generation'] and len(j['input'])==j['n'] and 'done' in j,'return owner/generation after complete fill')
            returned.add(bank);j['returned']=now
        elif kind=='RES':
            ch,t=int(f[3]),int(f[4]);address=int(f[5],16);n,bank,gen=map(int,f[6:9]);j=jobs[t]
            need(lease is None and queued[ch] and queued[ch].popleft()==t and 'res' not in j,'reservation FIFO/exactly once')
            need(sample['take']==1<<ch and sample['ready'] and sample['can']>>ch&1,'reservation witness')
            need(j['ch']==ch and j['n']==n and address==0x100000000+t*4096,'real command fields')
            need(0<=bank<8 and (bank>=4 or bank//2==ch) and bank not in owners,'eligible free physical bank')
            if bank in returned:coverage['same_edge_bank_reuse']+=1
            need(gen==generations[bank]+1,'bank generation advances');generations[bank]=gen;owners[bank]=t
            j.update(res=now,bank=bank,generation=gen);lease=t
            for q in next_event[ch].values():q.append(t)
        elif kind=='ECHO':
            ch,t=int(f[3]),int(f[4]);j=jobs[t]
            need(t==lease and ch==j['ch'] and 'echo' not in j and now>j['res'],'single command echo, not a second allocation');j['echo']=now
        elif kind=='FETCH':
            ch,t=int(f[3]),int(f[4]);j=jobs[t]
            need(t==lease and ch==j['ch'] and 'echo' in j and now>j['echo'] and 'fetch' not in j and int(f[5],16)==descriptor(t,j['n']),'fetch after reservation/echo');j['fetch']=now
        elif kind=='IN':
            t,beat=int(f[3]),int(f[4]);j=jobs[t]
            need(t==lease and 'fetch' in j and now>j['fetch'] and beat==len(j['input'])<j['n'] and int(f[5],16)==pattern(t,beat),'input byte conservation/order')
            need(sample['pv'] and sample['pr'],'actual input handshake');j['input'].append(now)
        elif kind=='DONE':
            t,n=int(f[3]),int(f[4]);j=jobs[t]
            need(t==lease and len(j['input'])==n==j['n'] and now>j['input'][-1] and 'done' not in j,'counted completion after actual input');j['done']=now;lease=None
        elif kind in ('AW','W','B','CPL'):
            ch,t=int(f[3]),int(f[4]);j=jobs[t];q=next_event[ch][kind.lower()]
            need(q and q[0]==t and ch==j['ch'],'destination AXI order')
            if kind=='AW':
                need(len(j['input'])==j['n'] and now>j['input'][-1] and int(f[5],16)==0x100000000+t*4096 and int(f[6])==j['n'],'AW full-frame metadata');j['aw']=now;q.popleft()
            elif kind=='W':
                beat,lastbit=int(f[5]),int(f[6]);need(beat==len(j['output'])<j['n'] and lastbit==(beat==j['n']-1) and int(f[7],16)==pattern(t,beat),'output byte conservation/last')
                need(len(j['input'])==j['n'] and now>j['input'][-1],'output after fill');j['output'].append(now)
                if 'aw' not in j:coverage['w_before_aw']+=1
                if lastbit:q.popleft()
            elif kind=='B':
                status=int(f[5]);need('aw' in j and len(j['output'])==j['n'] and now>max(j['aw'],j['output'][-1]),'B after AW and WLAST')
                need(status==(2 if t==400 else 0),'expected B status');j.update(b=now,status=status);q.popleft()
            else:
                need('b' in j and now>j['b'] and int(f[5])==j['status'],'completion causality/status');j['cpl']=now;q.popleft()
        elif kind=='FENCE':
            need(barrier is None and sample['bv'] and sample['br'],'fence accepted once');barrier=int(f[3])
        elif kind=='FENCED':
            need(barrier==int(f[3]) and lease is None and not any(queued) and not owners and
                 all('cpl' in j for j in jobs.values() if j['case']==case),'fence waits for payload and results');barrier=None
        elif kind=='DRAIN':
            need(all('cpl' in j for j in jobs.values() if j['case']==case) and not owners and lease is None,'explicit drain')
        else:raise ValueError('unknown event '+kind)
    finish();need(len(cases)==8 and 'RS SUITE_PASS cases=8 ' in text,'complete suite')
    need(all(coverage[k]>0 for k in ['peer_with_full_banks','descriptor_limit_with_free_banks','stop_drains_queue',
        'payload_after_fault','bank_held_without_payload','fetch_stall','fence_result_stall','w_before_aw','same_edge_bank_reuse']),'required stress coverage')
    return dict(cases=cases,submitted=len(jobs),completed=sum(x['completed'] for x in cases),
                bytes=sum(x['bytes'] for x in cases),coverage=dict(coverage))


def negatives(text):
    output=[];lines=text.splitlines()
    for name,prefix in [('write_corruption','RS W '),('input_corruption','RS IN '),('lost_reservation','RS RES '),
                        ('missing_echo','RS ECHO '),('short_done','RS DONE '),('false_offer','RS S '),
                        ('stale_return','RS RETURN '),('missing_return','RS RETURN ')]:
        changed=list(lines)
        i=next(i for i,l in enumerate(lines) if l.startswith(prefix) and (name!='false_offer' or int(l.split()[11],2)!=0))
        f=changed[i].split()
        if name in ('write_corruption','input_corruption'):f[-1]=format(int(f[-1],16)^1,'0128x')
        elif name=='short_done':f[-1]=str(int(f[-1])-1)
        elif name=='stale_return':f[-1]=str(int(f[-1])-1)
        elif name=='false_offer':f[12]='0'
        else:changed.pop(i)
        if name not in ('lost_reservation','missing_echo','missing_return'):changed[i]=' '.join(f)
        try:replay('\n'.join(changed))
        except (ValueError,KeyError) as e:output.append(dict(mutation=name,rejection=str(e)))
        else:raise ValueError('missed negative '+name)
    return output


def check_heads(text):
    queues=[deque(),deque()];expected=[None,None];words=[0,0];good=[0,0]
    samples=0;stalled=0;selections=0;cycle=None;take=0;seen=None;cache_seen=False
    for line in text.splitlines():
        f=line.split()
        if line.startswith('RS CASE '):
            queues=[deque(),deque()];expected=[None,None];words=[0,0];good=[0,0];cache_seen=False
        elif line.startswith('RS S '):
            cycle=int(f[2]);take=int(f[11],2);cache_seen=False
        elif line.startswith('RH HEAD '):
            need(int(f[2])==cycle and not cache_seen,'one cache snapshot per active cycle')
            mask,bits=int(f[3],2),int(f[4],2);seen=[int(f[5],16),int(f[6],16)];cache_seen=True
            need(mask==sum((x is not None)<<p for p,x in enumerate(expected)),'registered valid latency')
            need(seen==words and bits==good[0]+(good[1]<<1),'registered metadata/validation exact')
            samples+=1
            for p in (0,1):
                if expected[p] is not None:
                    need(queues[p] and expected[p]==queues[p][0] and seen[p]==expected[p],'cache is FIFO head')
                    if not(take>>p&1):stalled+=1
                if take>>p&1:
                    need(expected[p] is not None and good[p],'take requires live validated head');expected[p]=None;selections+=1
                elif expected[p] is None and queues[p]:
                    d=queues[p][0];expected[p]=d;words[p]=d
                    address=(d>>72)&((1<<64)-1);n=(d&255)+1
                    good[p]=int(address%64==0 and n<=64 and (address%4096)+64*n<=4096)
        elif line.startswith('RS SUB '):
            need(cache_seen,'sample before events');queues[int(f[3])].append(int(f[5],16))
        elif line.startswith('RS RES '):
            p=int(f[3]);need(cache_seen and queues[p] and (queues[p][0]>>136)&((1<<64)-1)==int(f[4]),'cached allocation identity');queues[p].popleft()
    need(samples==sum(l.startswith('RS S ') for l in text.splitlines()) and selections==85 and stalled>0,'complete cache coverage')
    return dict(samples=samples,selected=selections,retained_head_cycles=stalled)


def head_negatives(text):
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('RH HEAD ') and int(l.split()[3],2))
    output=[]
    for kind in ('stale_descriptor','invalid_validation','withdrawn_valid'):
        changed=list(lines);f=changed[i].split();mask=int(f[3],2);p=0 if mask&1 else 1
        if kind=='stale_descriptor':f[5+p]=format(int(f[5+p],16)^256,'058x')
        elif kind=='invalid_validation':f[4]=format(int(f[4],2)^(1<<p),'02b')
        else:f[3]=format(mask^(1<<p),'02b')
        changed[i]=' '.join(f)
        try:check_heads('\n'.join(changed))
        except ValueError as e:output.append(dict(mutation=kind,rejection=str(e)))
        else:raise ValueError('missed cache negative '+kind)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    ap.add_argument('--allow-incomplete-physical',action='store_true',help='Review completed RTL evidence while explicitly retaining a failed/timed-out physical status')
    args=ap.parse_args()
    record=json.loads((args.run/'output/result.json').read_text());text=(args.run/'output/suite/simulation.log').read_text()
    complete=record['status']=='EXPORTED_PENDING_REVIEW' and not record['timed_out'] and record['returncode']==0
    partial=args.allow_incomplete_physical and record['status']=='FAILED_OR_TIMEOUT' and (args.run/'output/suite/registered_dispatch_synth.dcp').is_file()
    need(complete or partial,'runner success or explicit completed-RTL review after physical failure')
    need(record['sources']==record['sources_after'],'sources preserved')
    need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL )',text),'simulation diagnostics')
    result=replay(text);result['negative_controls']=negatives(text)
    result['registered_heads']=check_heads(text);result['head_negative_controls']=head_negatives(text)
    result.update(status='REGISTERED_HEAD_PAYLOAD_PASS_TIMING_REVIEW_REQUIRED',seconds=record['seconds'],
        simulation_origin=record.get('resume',{}).get('suite',str(args.run/'output/suite')),
        physical_run_status=record['status'],physical_timed_out=record['timed_out'],
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Behavioral AXI target and source; no real DMA, MIG, PCIe/P2P or HBM.',
                    'Single clock/reset epoch, no CDC or warm-recovery qualification.',
                    'Physical timing qualification is recorded separately from this protocol review.',
                    'Fault cases retain queued descriptors; test resets start independent epochs.'])
    with (args.run/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
