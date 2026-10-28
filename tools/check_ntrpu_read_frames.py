"""Independent command/frame/fragment replay and bounded steady-window analysis."""
import argparse,hashlib,json,re
from collections import Counter,defaultdict,deque
from pathlib import Path
from check_ntrpu_read_window import sha,pattern,accounting

def replay(lines,native=0,routed=False):
    cfg=[s for s in lines if s.startswith('RF CONFIG ')];assert len(cfg)==1
    cfg={k:int(v) for k,v in (p.split('=') for p in cfg[0].split()[2:])}
    frames=cfg['frames'];assert frames in (1,2) and cfg['native']==native
    case_count=12 if not native and frames==2 else 4
    starts=[int(s.split()[2]) for s in lines if s.startswith('RR CASE ')]
    ends=[int(s.split()[2]) for s in lines if s.startswith('RR PASS ')]
    assert starts==ends==list(range(1,case_count+1))
    final=[s for s in lines if s.startswith('RF ALL_PASS ')];assert len(final)==1
    totals={k:int(v) for k,v in (p.split('=') for p in final[0].split()[2:])}
    assert totals['cases']==case_count
    boots=[s.split() for s in lines if s.startswith('RR BOOT ')]
    assert boots and all(s[4:6]==['1','1'] for s in boots)
    assert [int(s[3]) for s in boots]==list(range(1,len(boots)+1))
    assert len(boots)==(1 if native else (10 if frames==2 else 2))
    groups=defaultdict(list);events=[];case=0
    for line in lines:
        p=line.split()
        if len(p)<2 or p[0] not in ('RR','RF'):continue
        if p[0]=='RR' and p[1]=='CASE':case=int(p[2])
        if p[1] not in ('WCMD','WI','WCPL','CMD','AR','NR','DATA','CPL','STATE','FILL','SEAL','OVERLAP','QUARANTINE'):continue
        if p[0]=='RR' and p[1]=='STATE':continue
        ts,epoch,lane=map(int,p[2:5])
        # QUARANTINE carries a tag in the lane field.
        if p[1]=='QUARANTINE':lane=0;p=[*p[:4],'0',p[4]]
        event=(p[0],p[1],ts,epoch,lane,p[5:],case);events.append(event)
        groups[ts,epoch,lane].append(event)
    sampled_lanes=defaultdict(set)
    for e in events:
        if e[1]=='STATE':sampled_lanes[e[2],e[3]].add(e[4])
    assert all(lanes=={0,1} for lanes in sampled_lanes.values())
    if routed:assert not sampled_lanes and 'RF ROUTED_PUBLIC_ONLY' in lines
    occupancy={};peak=0;samples=0
    for key,group in sorted(groups.items()):
        ts,epoch,lane=key;slot=(epoch,lane)
        states=[e for e in group if e[1]=='STATE']
        changes=sum(e[1]=='CMD' for e in group)-sum(e[1]=='CPL' for e in group)
        if changes or states:
            assert len(states)==(0 if routed else 1)
            count=occupancy.get(slot,0)
            if not routed:assert int(states[0][5][0])==count
            peak=max(peak,count);samples+=1;count+=changes;assert 0<=count<=frames;occupancy[slot]=count
            peak=max(peak,count)
    assert (samples>0 if routed else samples>100) and peak==frames
    tracker=None if routed else accounting(lines,8)
    memory={};writes={};wactive={};commands={};active=defaultdict(deque);fragments=defaultdict(deque)
    counts=Counter();quarantined=set();overlap=0
    for family,kind,ts,epoch,lane,p,case in sorted(events,key=lambda e:e[2]):
        key=(epoch,lane)
        if kind=='WCMD':
            tag,n,addr=int(p[0]),int(p[1]),int(p[2],16);ck=(epoch,tag)
            assert key not in wactive and ck not in writes and 1<=n<=64 and tag%2==lane
            assert addr==(lane<<33)+0x10000+(tag//2)*4096
            writes[ck]=dict(n=n,addr=addr,count=0,done=False);wactive[key]=ck;counts['writes']+=1
        elif kind=='WI':
            ck=wactive[key];w=writes[ck];value=int(p[0],16)
            assert w['count']<w['n'] and value==pattern(ck[1],w['count'])
            for half in (0,1):memory[epoch,w['addr']+64*w['count']+32*half]=(value>>(256*half))&((1<<256)-1)
            w['count']+=1
        elif kind=='WCPL':
            ck=wactive.pop(key);assert ck==(epoch,int(p[0]));w=writes[ck]
            assert w['count']==w['n'];w['done']=True
        elif kind=='CMD':
            tag,n,addr=int(p[0]),int(p[1]),int(p[2],16);ck=(epoch,tag)
            assert ck not in commands and len(active[key])<frames and ck in writes and writes[ck]['done']
            assert (n,addr)==(writes[ck]['n'],writes[ck]['addr']) and tag%2==lane
            commands[ck]=dict(lane=lane,n=n,addr=addr,issued=0,received=0,filled=0,delivered=0,error=0,protocol=False,
                              sealed=False,complete=False,case=case,accepted_ps=ts,data_times=[])
            active[key].append(ck);counts['commands']+=1
        elif kind=='AR':
            addr,n=int(p[0],16),int(p[1]);owners=[ck for ck in active[key] if commands[ck]['issued']<2*commands[ck]['n']]
            assert owners;ck=owners[0];c=commands[ck]
            assert 1<=n<=16 and addr==c['addr']+32*c['issued'] and c['issued']+n<=2*c['n']
            assert len(fragments[key])<8
            fragments[key].append(dict(owner=ck,addr=addr,n=n,count=0));c['issued']+=n;counts['native_addresses']+=1
        elif kind=='NR':
            assert fragments[key];f=fragments[key][0];ck=f['owner'];c=commands[ck]
            rid,resp,last,value=int(p[0]),int(p[1]),int(p[2]),int(p[3],16)
            assert f['count']<f['n'] and value==memory[epoch,f['addr']+32*f['count']]
            malformed=rid!=0 or resp==1 or last!=int(f['count']+1==f['n'])
            if malformed:
                mode=case-4;assert not native and lane==0 and ck[1]==402+mode*4
                assert ((mode==3 and rid==32) or (mode in (4,5) and last!=int(f['count']+1==f['n'])) or (mode==6 and resp==1))
                c['protocol']=True;counts['malformed_beats']+=1
            if resp in (2,3):
                assert not native and case in (5,6) and lane==0 and ck[1]==402+(case-4)*4
                c['error']=2
            f['count']+=1;c['received']+=1;counts['native_reads']+=1
            if f['count']==f['n']:fragments[key].popleft()
        elif kind=='FILL':
            tag,index,resp,last,value=int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4],16)
            c=commands[epoch,tag];assert (epoch,tag) in active[key] and not c['sealed'] and not c['protocol']
            assert index==c['filled']<c['n'] and c['received']>=2*(index+1)
            assert value==pattern(tag,index) and last==int(index+1==c['n']) and resp in (0,2,3)
            if resp:assert c['error']==2
            c['filled']+=1;counts['filled_beats']+=1
        elif kind=='SEAL':
            tag,status=map(int,p);c=commands[epoch,tag]
            assert (epoch,tag) in active[key] and not c['sealed'] and not c['protocol']
            assert c['filled']==c['n'] and c['received']==2*c['n'] and status==c['error']
            c['sealed']=True;counts['sealed_frames']+=1
        elif kind=='DATA':
            tag,index,last,value=int(p[0]),int(p[1]),int(p[2]),int(p[3],16);ck=(epoch,tag)
            assert active[key] and active[key][0]==ck;c=commands[ck]
            assert (c['received']==2*c['n'] if routed else c['sealed']) and not c['error'] and not c['protocol'] and index==c['delivered']<c['n']
            assert value==pattern(tag,index) and last==int(index+1==c['n'])
            c['delivered']+=1;c['data_times'].append(ts);counts['output_beats']+=1
        elif kind=='CPL':
            tag,status,nbytes=map(int,p);ck=(epoch,tag)
            assert active[key] and active[key].popleft()==ck;c=commands[ck]
            assert not c['complete'] and (c['received']==2*c['n'] if routed else c['sealed']) and not c['protocol'] and status==c['error']
            assert c['delivered']==(0 if status else c['n']) and nbytes==c['delivered']*64
            c['complete']=True;c['completed_ps']=ts;counts['completions']+=1
        elif kind=='QUARANTINE':
            ck=(epoch,int(p[0]));c=commands[ck];assert c['protocol'] and not c['complete'] and not c['sealed'] and c['delivered']==0
            quarantined.add(ck)
        elif kind=='OVERLAP':
            fill,publish=map(int,p);assert frames==2 and fill!=publish
            group=groups[ts,epoch,lane]
            assert any(e[1]=='FILL' and int(e[5][0])==fill for e in group)
            assert any(e[1]=='DATA' and int(e[5][0])==publish for e in group)
            overlap+=1
    for name in ('writes','commands','native_reads','output_beats','completions'):assert counts[name]==totals[name]
    assert not wactive and {ck for q in active.values() for ck in q}==quarantined
    assert len(quarantined)==(4 if case_count==12 else 0)
    assert all(commands[ck]['complete'] or ck in quarantined for ck in commands)
    assert all(commands[f['owner']]['protocol'] for q in fragments.values() for f in q)
    assert sum(s.startswith('RF FULL ') for s in lines)==sum(s.startswith('RF CPL_HELD ') for s in lines)==sum(s.startswith('RF STREAM_BEGIN ') for s in lines)==1
    if routed:assert overlap==0 and not counts['filled_beats'] and not counts['sealed_frames']
    elif frames==1:assert overlap==0
    elif not cfg['common']:assert overlap>0
    stream=[(ck,c) for ck,c in commands.items() if c['case']==3]
    assert {ck[1] for ck,c in stream}==set(range(300,324))
    by_tag={ck[1]:c for ck,c in stream};windows=[]
    for lane in (0,1):
        start=by_tag[302+lane]['completed_ps'];end=by_tag[318+lane]['completed_ps']
        assert end>start
        payload=[ts for t in range(304+lane,320+lane,2) for ts in by_tag[t]['data_times']]
        assert len(payload)==8*64 and all(start<ts<end for ts in payload)
        windows.append(dict(lane=lane,start_ps=start,end_ps=end,bytes=32768,gb_s=32768*1000/(end-start)))
    span=max(v['end_ps'] for v in windows)-min(v['start_ps'] for v in windows)
    canonical=[(epoch,tag,*[c[k] for k in ('lane','n','addr','error','protocol','complete','delivered')]) for (epoch,tag),c in sorted(commands.items())]
    return dict(frames=frames,native_profile=native,common=cfg['common'],cases=case_count,**counts,
                transaction_digest=hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),internal_trace_checked=not routed,
                tracker=tracker,frame_peak=peak,frame_samples=samples,overlap_cycles=overlap,quarantined_commands=len(quarantined),
                boot_ps=int(boots[0][2]),steady_window=dict(warmup_frames_per_channel=2,measured_frames_per_channel=8,
                trailing_frames_per_channel=2,lanes=windows,aggregate_gb_s=65536*1000/span,span_ns=span/1000),
                board_throughput_measured=False)

def controls(lines,native=0,routed=False):
    passed=[]
    for prefix,column,base in [('RR WI ',5,16),('RR CMD ',6,10),('RR AR ',5,16),('RR NR ',8,16),
                               ('RR DATA ',8,16),('RR CPL ',6,10),('RF STATE ',5,10),('RF FILL ',9,16),('RF SEAL ',6,10)]:
        if routed and prefix.startswith('RF '):continue
        copy=list(lines);i=next(i for i,s in enumerate(copy) if s.startswith(prefix));p=copy[i].split()
        p[column]=format(int(p[column],base)^1,'x' if base==16 else 'd');copy[i]=' '.join(p)
        try:replay(copy,native,routed)
        except (AssertionError,KeyError,IndexError):passed.append('corrupt_'+prefix.strip().replace(' ','_'))
        else:raise AssertionError('Missed corruption '+prefix)
    for prefix in ['RR WCMD ','RR WI ','RR WCPL ','RR CMD ','RR AR ','RR NR ','RR DATA ','RR CPL ',
                   'RF STATE ','RF FILL ','RF SEAL ','RF FULL ','RF CPL_HELD ','RF STREAM_BEGIN ','RF ALL_PASS ']:
        if routed and prefix in ('RF STATE ','RF FILL ','RF SEAL '):continue
        copy=list(lines);del copy[next(i for i,s in enumerate(copy) if s.startswith(prefix))]
        try:replay(copy,native,routed)
        except (AssertionError,KeyError,IndexError):passed.append('missing_'+prefix.strip().replace(' ','_'))
        else:raise AssertionError('Missed deletion '+prefix)
    return passed

def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='read_frames'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    lines=(out/'suite/simulation.log').read_text().splitlines()
    assert not any(re.match(r'\s*(ERROR:|Fatal:|FAIL )',s) for s in lines)
    result=replay(lines,r['native_profile']);negative=controls(lines,r['native_profile'])
    return dict(status='READ_FRAMES_PREFLIGHT_REVIEWED',result_sha256=sha(out/'result.json'),seconds=r['seconds'],
                **result,negative_controls=negative,native_hbm_controller=False,physical_timing_qualified=False,board_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
