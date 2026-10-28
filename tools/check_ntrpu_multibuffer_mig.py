"""Replay cold multibuffer writes/readback and qualify every DDR model warning."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_multibuffer_periodic import review as periodic_review


def need(ok, message):
    if not ok: raise ValueError(message)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern(tag, beat):
    return int.from_bytes(bytes((tag*37+beat*13+lane*19)&255 for lane in range(64)), 'little')


def replay(text):
    need(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|Fatal:|FAIL )|\bModel:ERROR:',text), 'Simulator/model error')
    locations=[0,0x100000040,0x800000000,0x1000000000,0x1800000000,0x1fffffffc0,
               0x100001000,0x1ffffffe00,0x100002000,0x100003000,0x100004000,0x100005000,
               0x100006000,0x100007000,0x100008000,0x2000000000]
    lengths=[1]*6+[64,8]+[1]*4+[8,8,1,1]
    cases=[];case=None;state={};jobs={};fill=None
    aw=deque();w=deque();b=deque();cpl=deque();golden={};reading=None
    completed=written=read_count=peak=pending=0;last_time=-1;saturated=False
    reads_by_case=Counter();stops=[];summary=None;drains=0
    for line in text.splitlines():
        if not line.startswith('COLD '):continue
        fields=line.split();kind=fields[1]
        if kind=='CASE':case=fields[2];cases.append(case);continue
        if kind=='PASS':
            need(summary is None,'Duplicate PASS')
            summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)};continue
        now=int(fields[2]);need(now>=last_time,'Time order');last_time=now
        if kind=='STATE':state=dict(x.split('=') for x in fields[3:]);continue
        if kind=='CMD':
            tag=int(fields[3]);address=int(fields[4],16);length,status=map(int,fields[5:])
            need(tag==len(jobs)+1 and tag<=16 and fill is None,'Command identity')
            need(address==locations[tag-1] and length==lengths[tag-1] and status==(3 if tag==16 else 0),'Command vector/range')
            need(state.get('calib')=='1' and state.get('ui_rst')=='0' and state.get('admit')=='1'
                 and state.get('running')=='1' and state.get('restart')=='0','Admission before calibration')
            jobs[tag]=dict(address=address,length=length,status=status,inputs=[],outputs=[])
            cpl.append(tag);fill=tag
            if status==0:
                for queue in (aw,w,b):queue.append(tag)
        elif kind=='IN':
            tag,beat=map(int,fields[3:5]);value=int(fields[5],16)
            need(tag==fill and beat==len(jobs[tag]['inputs']) and value==pattern(tag,beat),'Input frame')
            jobs[tag]['inputs'].append(value)
            if len(jobs[tag]['inputs'])==jobs[tag]['length']:jobs[tag]['filled']=now;fill=None
        elif kind in ('AW','W','B'):
            tag=int(fields[3]);need(tag in jobs and jobs[tag]['status']==0,'Rejected/unknown job reached MIG')
            job=jobs[tag];need(job.get('filled',now)<now,'MIG before full input frame')
            if kind=='AW':
                need(aw and aw.popleft()==tag and int(fields[4],16)==job['address'] and int(fields[5])==job['length'],'MIG address/order')
                job['aw']=now;pending+=1;peak=max(peak,pending);need(pending<=4,'Credit overflow')
            elif kind=='W':
                beat,last=map(int,fields[4:6]);value=int(fields[6],16)
                need(w and w[0]==tag and beat==len(job['outputs']) and value==pattern(tag,beat),'MIG payload/order')
                need(last==int(beat==job['length']-1),'WLAST')
                job['outputs'].append(value);written+=64
                if last:job['last_w']=now;w.popleft()
            else:
                need(b and b.popleft()==tag and int(fields[4])==0,'MIG B identity/status')
                need(job.get('aw',now)<now and job.get('last_w',now)<now,'MIG B before AW/W')
                job['b']=now;pending-=1;need(pending>=0,'B without credit')
                for beat,value in enumerate(job['outputs']):golden[job['address']+beat*64]=value
        elif kind=='CPL':
            tag,status=map(int,fields[3:]);need(cpl and cpl.popleft()==tag,'Completion order')
            job=jobs[tag];need(status==job['status'] and job.get('filled',now)<now,'Completion status')
            if status==0:need(job.get('b',now)<now,'Completion before MIG B')
            else:need(not job['outputs'] and 'aw' not in job and 'b' not in job,'Reject leaked')
            completed+=1
        elif kind=='DRAIN':
            need(fill is None and not any((aw,w,b,cpl)) and pending==0,'Premature drain');drains+=1
        elif kind=='AR':
            address=int(fields[3],16)
            need(not any((aw,w,b,cpl)) and fill is None and reading is None and address in golden,'Read before committed writes')
            slot=reads_by_case[case]
            need(slot<len(golden) and address==list(golden)[slot],'Sentinel sweep order')
            reading=(address,now)
        elif kind=='R':
            address=int(fields[3],16);value=int(fields[4],16)
            need(reading is not None and address==reading[0] and now>reading[1] and value==golden[address],'Readback data/address')
            reading=None;read_count+=1;reads_by_case[case]+=1
        elif kind=='SATURATED':
            need(not saturated and fields[3]=='pending=4' and pending==4,'Missing four-credit saturation');saturated=True
        elif kind=='STOP':
            reason=fields[3];stops.append(reason)
            need(case in ('four_credits','stop_fill','stop_after_aw'),'Stop case')
            if reason=='b_held':need(len(jobs)==12 and pending==4,'Stop before saturation')
            elif reason=='fill':need(fill==13 and len(jobs[13]['inputs'])==2,'Stop during fill coverage')
            elif reason=='aw_exposed':need(14 in jobs and 'aw' in jobs[14] and 'last_w' not in jobs[14],'Stop after AW coverage')
            else:raise ValueError('Unknown stop reason')
        else:raise ValueError('Unknown event '+kind)
    need(cases==['cold_admission','sparse_ranks','burst_boundaries','four_credits','stop_fill',
                 'stop_after_aw','result_backpressure','range_reject'],'Case coverage')
    need(stops==['b_held','fill','aw_exposed'] and saturated and peak==4,'Stop/credit coverage')
    need(len(jobs)==completed==16 and written==6336 and read_count==279 and len(golden)==98,'Transfer totals')
    need(reads_by_case==dict(sparse_ranks=6,burst_boundaries=77,result_backpressure=98,range_reject=98),'Sentinel coverage')
    need(not any((aw,w,b,cpl)) and reading is None and fill is None and pending==0,'Unfinished transaction')
    need(summary and all(summary.get(k)==v for k,v in dict(commands=16,completions=16,writes=99,reads=279,locations=98,peak=4).items()),'Summary mismatch')
    need(summary['blocked']>20 and summary['ps']>=last_time,'Startup/time coverage')
    rank_counts=Counter()
    for rank,count in re.findall(r'^COLD_ACT .*?rank=(\d+) count=(\d+)',text,re.M):
        rank=int(rank);rank_counts[rank]+=1;need(rank_counts[rank]==int(count),'Rank ACT counter')
    need(set(rank_counts)=={0,1,2,3},'Rank activity')
    return dict(status='COLD_MIG_BYTE_REPLAY_PASS',summary=summary,write_bytes=written,read_bytes=read_count*64,
                source_bytes=sum(lengths)*64,range_rejects=1,rank_activations=dict(rank_counts),drains=drains)


def corruptions(text):
    results=[]
    for kind in ('write_payload','read_payload','address_alias','missing_b','status','duplicate_cpl'):
        lines=text.splitlines()
        prefix={'write_payload':'COLD W ','read_payload':'COLD R ','address_alias':'COLD AW ',
                'missing_b':'COLD B ','status':'COLD CPL ','duplicate_cpl':'COLD CPL '}[kind]
        n=next(i for i,l in enumerate(lines) if l.startswith(prefix) and (kind!='address_alias' or int(l.split()[4],16)>0xffffffff))
        fields=lines[n].split()
        if kind in ('write_payload','read_payload'):fields[-1]=format(int(fields[-1],16)^1,'0128x');lines[n]=' '.join(fields)
        elif kind=='address_alias':fields[4]=format(int(fields[4],16)&0xffffffff,'x');lines[n]=' '.join(fields)
        elif kind=='missing_b':lines.pop(n)
        elif kind=='status':fields[-1]='2';lines[n]=' '.join(fields)
        else:lines.insert(n,lines[n])
        try:replay('\n'.join(lines))
        except ValueError as error:results.append(dict(mutation=kind,rejection=str(error)))
        else:raise ValueError('Corruption escaped: '+kind)
    return results


def review(root):
    output=root/'output';record=json.loads((output/'result.json').read_text())
    need(record['status']=='EXPORTED_PENDING_REVIEW' and record['example_unchanged'] and not record['timed_out'],'Incomplete run/modified input')
    need(record['sources']==record['sources_after'],'Changed snapshot')
    for name,value in record['sources'].items():need(digest(root/'inputs'/name)==value,'Source hash: '+name)
    for name,value in json.loads((output/'FILES.json').read_text()).items():need(digest(output/name)==value,'Output hash: '+name)
    console=(output/'console.log').read_text();need('NTRPU_MULTIBUFFER_MIG_EXPORTED' in console,'No export')
    unit=(output/'suite/startup.log').read_text()
    cases=re.findall(r'^COLD_START_CASE_PASS case=(\d+) entries=(\d+) locked=(\d+)$',unit,re.M)
    need(sorted(cases)==[(str(g),'1',str(int(1<=g<=5))) for g in range(8)] and 'COLD_START_SUITE_PASS cases=8' in unit,'Startup coverage')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',unit,re.M),'Startup diagnostic')
    text=(output/'suite/memory.log').read_text()
    result=replay(text);result['trace_corruptions']=corruptions(text)
    result['periodic']=periodic_review(text)
    result.update(status='SINGLE_CONTROLLER_COLD_MIG_PASS_NOT_BOARD_ADOPTED',startup_cases=8,
                  seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
                  warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
                  critical_warnings=re.findall(r'^CRITICAL WARNING:.*',console,re.M),
                  checker_sha256=digest(Path(__file__)),
                  limits=['One real generated MIG in BFM mode, destination 1 inactive',
                          'Generic geometry, not exact DIMM, whole capacity, physical timing or HBM',
                          'Cold start and orderly drain; runtime reset/loss tested on adapter only',
                          'Restart after first RUN requires reconfiguration; no warm recovery claim'])
    need(not result['critical_warnings'],'Critical tool warning')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=review(args.run)
    with (args.run/'REVIEW.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
