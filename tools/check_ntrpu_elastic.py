"""Public transaction replay with independent AW/W progress and delayed B errors."""
from collections import defaultdict, deque
import re
from check_ntrpu_peer_multibuffer import pattern, require


def replay_external(text, profile, seed):
    channels,reserve,maximum,depth,pending=profile
    jobs={}; fill=None
    aw=[deque() for _ in range(channels)]
    w=[deque() for _ in range(channels)]
    b=[deque() for _ in range(channels)]
    cpl=[deque() for _ in range(channels)]
    events=defaultdict(list)
    for line in text.splitlines():
        if line.startswith('EXT ') or line.startswith(('TRACE IN ','TRACE CPL ','TRACE DRAIN ','TRACE RESET ')):
            fields=line.split()[1:]; events[int(fields[1])].append(fields)
    nbytes=errors=foreign=parallel=before_aw=0
    rank={'AW':0,'W':1,'B':2,'CPL':3,'IN':4,'CMD':5,'DRAIN':6,'RESET':7}
    for cycle,group in sorted(events.items()):
        writes=set()
        for e in sorted(group,key=lambda e:rank[e[0]]):
            kind=e[0]
            if kind=='CMD':
                port,tag=int(e[2]),int(e[3]); addr=int(e[4],16); beats=int(e[5])
                require(0<=port<channels and tag not in jobs and fill is None,'invalid command identity/fill')
                require(len(cpl[port])<depth,'descriptor overflow')
                require(addr==0x100000000+(tag<<12) and 1<=beats<=maximum,'command geometry')
                jobs[tag]=dict(port=port,addr=addr,beats=beats,input=[],written=[])
                fill=tag
                for qs in (aw,w,b,cpl): qs[port].append(tag)
            elif kind=='IN':
                require(fill is not None,'input without command')
                j=jobs[fill]; data=int(e[2],16)
                require(data==pattern(fill,len(j['input'])),'input bytes')
                j['input'].append(data)
                if len(j['input'])==j['beats']: j['filled']=cycle; fill=None
            elif kind=='AW':
                port=int(e[2]); require(aw[port],'unexpected AW')
                tag=aw[port].popleft(); j=jobs[tag]
                require(j.get('filled',cycle)>=0 and j.get('filled',cycle)<cycle,'AW before fill')
                require(int(e[3],16)==j['addr'] and int(e[4])==j['beats'],'AW order/length')
                j['aw']=cycle
            elif kind=='W':
                port=int(e[2]); require(w[port],'unexpected W')
                tag=w[port][0]; j=jobs[tag]; data=int(e[4],16); beat=len(j['written'])
                require('filled' in j and j['filled']<cycle,'W before fill')
                require(beat<j['beats'] and data==pattern(tag,beat),'W payload/order')
                require(int(e[3])==int(beat==j['beats']-1),'WLAST mismatch')
                before_aw+= 'aw' not in j
                j['written'].append(data); nbytes+=64; writes.add(port)
                if int(e[3]): j['last_w']=cycle; w[port].popleft()
            elif kind=='B':
                port,bid,resp=map(int,e[2:]); require(b[port],'unexpected B')
                j=jobs[b[port][0]]
                require('aw' in j and 'last_w' in j and max(j['aw'],j['last_w'])<cycle,'B before AW/WLAST')
                if bid: foreign+=1
                else:
                    b[port].popleft(); j['b']=cycle; j['status']=resp; errors+=resp!=0
            elif kind=='CPL':
                port,tag,status=map(int,e[2:]); require(cpl[port] and cpl[port][0]==tag,'completion order')
                j=jobs[tag]
                require('b' in j and j['b']<cycle and status==j['status'],'completion before B/status')
                require('completed' not in j,'duplicate completion')
                j['completed']=cycle; cpl[port].popleft()
            else:
                require(fill is None and not any(aw+w+b+cpl),'premature drain/reset')
        parallel+=len(writes)>1
    require(jobs and all('completed' in j for j in jobs.values()),'missing completion')
    require(errors and foreign and before_aw,'missing error/skew coverage')
    if channels>1: require(parallel>0,'missing external parallel transfers')
    expected={'traffic_and_isolation','parallel_ports','aw_pause_and_stop','stop_during_fill',
              'result_credit','error_response','foreign_response','invalid_descriptor'}
    require(set(re.findall(r'^CASE (.+)$',text,re.M))==expected,'missing core cases')
    require(set(re.findall(r'^ELASTIC_CASE (.+)$',text,re.M))==
            {'data_before_address','bank_return_with_queued_data'},'missing elastic cases')
    require(len(re.findall(rf'^PASS seed={seed} cycles=\d+ peak_pending=\d+$',text,re.M))==1,'missing PASS')
    require(not re.search(r'FATAL:|Fatal:|ERROR:|FAIL ',text),'simulator error')
    summary=re.findall(r'^ELASTIC_SCOREBOARD (.+)$',text,re.M)
    require(len(summary)==1,'missing scoreboard')
    sv={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',summary[0])}
    for key,val in dict(jobs=len(jobs),bytes=nbytes,completed=len(jobs),w_before_aw=before_aw,
                        parallel=parallel,foreign_b=foreign,error_b=errors).items():
        require(sv[key]==val,'scoreboard/replay mismatch: '+key)
    return dict(status='ELASTIC_PUBLIC_TRANSACTION_REPLAY_PASS',**sv)
