"""Replay the complete memory-boundary path, including rejects and reset aborts."""
import argparse
from collections import deque,Counter
import hashlib
import json
from pathlib import Path
import re


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def payload(tag,beat):
    return sum((((tag*0x9e3779b9)^(beat*0x10203)^(lane*0x01010101))&0xffffffff)<<(32*lane) for lane in range(16))


def replay(text,profile):
    jobs={};fill=None;current_case=None;cases=set()
    aw=[deque(),deque()];w=[deque(),deque()];b=[deque(),deque()];cpl=[deque(),deque()]
    last_time=-1;written=aborted=resets=0;high=limits=rejected=errors=early_w=0
    rate=None
    for line in text.splitlines():
        if line.startswith('MEM CASE '):current_case=line[9:];cases.add(current_case);continue
        if line.startswith('MEM RATE '):
            require(rate is None,'duplicate rate')
            rate={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
            continue
        if not line.startswith('MEM ') or line.startswith('MEM CONFIG '):continue
        e=line.split();kind=e[1];now=int(e[2])
        require(now>=last_time,'nonmonotonic time');last_time=now
        if kind=='CMD':
            p,tag=int(e[3]),int(e[4]);addr=int(e[5],16);length,status=map(int,e[6:])
            require(p in (0,1) and tag not in jobs and fill is None,'command identity/fill')
            require(1<=length<=64 and addr%64==0 and addr%4096+64*length<=4096,'input geometry')
            require(len(cpl[p])<16,'descriptor overflow')
            valid=addr+64*length<=1<<37
            expected=3 if not valid else 2 if current_case=='response_error' else 0
            require(status==expected,'test response expectation')
            jobs[tag]=dict(port=p,addr=addr,length=length,status=status,case=current_case,
                           valid=valid,input=[],output=[])
            fill=tag;cpl[p].append(tag)
            if valid:
                for queues in (aw,w,b):queues[p].append(tag)
                high+=addr>=1<<32;limits+=addr+64*length==1<<37
            else:rejected+=1
        elif kind=='IN':
            tag,beat=int(e[3]),int(e[4]);data=int(e[5],16)
            require(tag==fill,'input frame identity');j=jobs[tag]
            require(beat==len(j['input']) and data==payload(tag,beat),'input byte/order')
            j['input'].append(data)
            if len(j['input'])==j['length']:j['filled']=now;fill=None
        elif kind in ('AW','W','B'):
            p,tag=int(e[3]),int(e[4]);require(tag in jobs,'unknown memory tag');j=jobs[tag]
            require(j['port']==p and j['valid'] and 'aborted' not in j,'unauthorized memory access')
            if kind=='AW':
                require(aw[p] and aw[p][0]==tag,'AW order');aw[p].popleft()
                require(int(e[5],16)==j['addr'] and int(e[6])==j['length'],'AW alias/length')
                require(j.get('filled',now)<now,'AW before complete frame');j['aw']=now
            elif kind=='W':
                require(w[p] and w[p][0]==tag,'W order')
                beat,last=int(e[5]),int(e[6]);data=int(e[7],16)
                require(j.get('filled',now)<now,'W before complete frame')
                require(beat==len(j['output']) and beat<j['length'] and data==payload(tag,beat),'payload mismatch')
                require(last==int(beat==j['length']-1),'WLAST')
                j['output'].append(data);written+=64;early_w+='aw' not in j
                if last:j['last_w']=now;w[p].popleft()
            else:
                require(b[p] and b[p][0]==tag,'B order');b[p].popleft()
                require('aw' in j and 'last_w' in j and max(j['aw'],j['last_w'])<now,'B before write')
                require(int(e[5])==j['status'],'B response');j['b']=now;errors+=j['status']!=0
        elif kind=='CPL':
            p,tag,status=map(int,e[3:]);require(cpl[p] and cpl[p][0]==tag,'completion order')
            cpl[p].popleft();j=jobs[tag]
            require('aborted' not in j and 'completed' not in j,'stale/duplicate completion')
            require(status==j['status'] and j.get('filled',now)<now,'completion status/input')
            if j['valid']:require(j.get('b',now)<now and len(j['output'])==j['length'],'premature completion')
            else:require('aw' not in j and not j['output'] and 'b' not in j,'rejected address leaked')
            j['completed']=now
        elif kind=='DRAIN':
            require(fill is None and not any(aw+w+b+cpl),'premature drain')
        elif kind=='RESET':
            resets+=1
            for j in jobs.values():
                if 'completed' not in j and 'aborted' not in j:
                    require(j['case']=='reset_busy','unexpected reset abandonment')
                    j['aborted']=now;aborted+=1
            aw=[deque(),deque()];w=[deque(),deque()];b=[deque(),deque()];cpl=[deque(),deque()];fill=None
        else:raise RuntimeError('unknown event '+kind)
    required={'bulk','isolation','stop_fill','stop_after_aw','address_pause','clock_pause',
              'result_credit','address_limits','response_error','capacity_reject','reset_busy'}
    require(cases==required,'scenario coverage')
    require(jobs and all('completed' in j or 'aborted' in j for j in jobs.values()),'unfinished work')
    require(not any(aw+w+b+cpl) and fill is None,'unfinished final channels')
    require(aborted==1 and rejected==3 and errors==1 and early_w>0 and high>0 and limits==1,'boundary coverage')
    summary=re.findall(r'^MEMORY_SUITE_PASS (.+)$',text,re.M);require(len(summary)==1,'missing PASS')
    values={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',summary[0])}
    expected=dict(profile=profile,accepted=len(jobs),completed=len(jobs)-aborted,aborted=aborted,bytes=written,resets=resets)
    require(all(values.get(k)==v for k,v in expected.items()),'scoreboard/replay disagreement')
    require(not re.search(r'^\s*(Fatal:|FATAL:|ERROR:|FAIL )',text,re.M),'positive simulation error')
    require(rate and rate['bytes']==32*4096 and rate['elapsed_ps']>0,'bulk measurement')
    return dict(status='MEMORY_BOUNDARY_REPLAY_PASS',**expected,range_rejects=rejected,
                above_4gib=high,capacity_end_cases=limits,w_before_aw_beats=early_w,
                case_counts=dict(Counter(j['case'] for j in jobs.values())),
                bulk=dict(**rate,GB_per_second=round(rate['bytes']*1000/rate['elapsed_ps'],6)))


def review(root):
    output=root/'output';record=json.loads((output/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['returncode']==0 and not record['timed_out'],'incomplete suite')
    require(record['sources']==record['sources_after'],'mutated inputs')
    for n,h in record['sources'].items():require(digest(root/'inputs'/n)==h,'input hash '+n)
    for n,h in json.loads((output/'FILES.json').read_text()).items():require(digest(output/n)==h,'output hash '+n)
    console=(output/'console.log').read_text()
    require('NTRPU_MEMORY_BOUNDARY_EXPORTED' in console,'missing export')
    require(not re.search(r'^CRITICAL WARNING:|^ERROR:',console,re.M),'tool error/critical warning')
    configured=(output/'suite/cdc_configured.rpt').read_text()
    for name,value in [('PROTOCOL','AXI4'),('DATA_WIDTH','512'),('ADDR_WIDTH','64'),('ID_WIDTH','4'),('ACLK_ASYNC','1'),('SYNCHRONIZATION_STAGES','3')]:
        require(re.search(r'CONFIG\.'+name+r'\s+\w+\s+(?:true|false)\s+'+value+r'\s*$',configured,re.M),'CDC parameter '+name)
    results=[replay((output/f'suite/profile-{p}-negative-0.log').read_text(),p) for p in range(3)]
    negatives=[]
    for number,reason in [(1,'MEMORY_PAYLOAD_MISMATCH'),(2,'MEMORY_COMPLETION_STATUS_ORDER'),(3,'MEMORY_UNAUTHORIZED_AW')]:
        text=(output/f'suite/profile-0-negative-{number}.log').read_text()
        require(reason in text and re.search(r'Fatal:',text,re.I) and 'MEMORY_SUITE_PASS' not in text,'negative control '+str(number))
        negatives.append(dict(number=number,reason=reason))
    return dict(status='MEMORY_BOUNDARY_FUNCTIONAL_PASS_NOT_BOARD_INTEGRATED',profiles=results,
                negative_controls=negatives,seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
                warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
                limits=['Actual AMD asynchronous clock-converter IP; behavioral AXI memory targets',
                        'No PCIe endpoint/MIG/HBM controller in this suite','No placement/routing/physical CDC or MTBF qualification',
                        'Capacity guard serializes each destination to one write pending B',
                        'Coordinated reset aborts pending jobs; partial writes are not rolled back',
                        'No partial/uncoordinated reset or autonomous clock-loss recovery'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=review(args.run)
    with (args.run/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
