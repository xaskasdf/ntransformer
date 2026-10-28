"""Replay overlap experiment traces independently of the HDL scoreboards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import replay, negative_controls, pattern, require
from check_ntrpu_transport_study import performance, mutations


def credit(text):
    takes={i:[] for i in range(1,6)};passed=[];prepared_without_offer=busy_prepared=0
    samples=0
    for line in text.splitlines():
        f=line.split()
        if line.startswith('OC S '):
            _,_,cycle,case,offer,valid,ch,take,queued,desc,state,fault=f
            case=int(case);offer=int(offer,2);take=int(take,2);ch=int(ch,2);state=int(state)
            valid=int(valid,2);fault=int(fault,2);samples+=1
            require(take!=3 and take & offer==take,'Allocation without live offer')
            if valid:
                expected=(0x12345678<<200)|((10+ch)<<136)|((0x100000000+(10+ch)*4096)<<72)|(0x800000000<<8)|7
                require(int(desc,16)==expected,'Prepared identity changed')
                prepared_without_offer+=offer==0;busy_prepared+=state==3
            if take:
                require(valid and not fault and state==0 and take==1<<ch,'Invalid commit state')
                require(ch not in takes[case],'Duplicate allocation');takes[case].append(ch)
        elif line.startswith('OC CASE_PASS '):passed.append(int(f[2]))
    require(passed==[1,2,3,4,5] and text.count('OC SUITE_PASS cases=5')==1,'Credit suite incomplete')
    require(takes=={1:[1,0],2:[0,1],3:[],4:[0,1],5:[0,1]},'Credit allocation sequence')
    require(prepared_without_offer>=8 and busy_prepared>0,'Missing preparation coverage')
    return dict(cases=5,samples=samples,allocations=sum(map(len,takes.values())),
                prepared_without_offer=prepared_without_offer,busy_prepared=busy_prepared)


def extra(text):
    case=0;jobs={};passed=[];windows=[];bytes_checked=0;last_fault=False
    active=None;budget=0;fences=[];last_fault_beats=[];last_beats=[]
    for line in text.splitlines():
        f=line.split()
        if line.startswith('TE CASE '):
            case=int(f[3]);require(case==len(passed)+1,'Extra case sequence')
            jobs={};active=None;budget=0;order=[[],[]];counts={k:[0,0] for k in ('FETCH','AW','W','B','CORE_CPL','CPL')}
        elif line.startswith('TE S '):
            last_fault=bool(int(f[5]));require(int(f[14])==budget,'Extra sampled budget')
            if int(f[6]):require(all('CPL' in j for j in jobs.values()) and budget==0,'False extra drain')
        elif line.startswith('OV WINDOW '):windows.append((case,int(f[2]),f[3]))
        elif line.startswith('OV LAST '):
            last_beats.append((int(f[3]),int(f[4])))
            if int(f[3])==3 and int(f[5]):last_fault_beats.append(int(f[4]))
        elif line.startswith('TE FENCED '):
            require(case==2 and int(f[3])==202 and all('CPL' in j for j in jobs.values()),'Fence overtook external completion')
            fences.append(202)
        elif line.startswith('TE CASE_PASS '):
            expected=[1,1,1,1] if case==2 else [1,1,1,0]
            require(list(map(int,f[3:]))==expected and set(jobs)=={1000,1001},'Extra workload totals')
            require(budget==0 and active is None and 'CPL' in jobs[1000],'Committed work lost')
            require(('CPL' in jobs[1001])==(case==2),'Incorrect quarantine/drain')
            if case!=2:require('RES' not in jobs[1001],'Prepared job wrongly allocated')
            passed.append(case)
        elif line.startswith('TE ') and f[1] in ('SUB','ADMIT','RES','FETCH','IN','AW','W','B','CORE_CPL','CPL'):
            kind=f[1];cyc=int(f[2])
            if kind=='IN':
                tag,beat,data=int(f[3]),int(f[4]),int(f[5],16)
                require(active==tag and budget>0 and jobs[tag]['in']==beat and data==pattern(tag,beat),'Extra input bytes/owner')
                jobs[tag]['in']+=1;budget-=1
                if budget==0:active=None
                continue
            ch,tag=int(f[3]),int(f[4]);require(ch in (0,1) and tag==1000+ch,'Extra job identity')
            if kind=='SUB':
                desc=(0x76543210<<200)|(tag<<136)|((0x100000000+tag*4096)<<72)|((0x800000000+tag*4096)<<8)|7
                require(tag not in jobs and int(f[5],16)==desc,'Extra descriptor')
                jobs[tag]=dict(SUB=cyc,desc=desc,**{'in':0,'out':0});order[ch].append(tag);continue
            require(tag in jobs,'Extra unknown job');j=jobs[tag]
            if kind=='W':
                beat,last,data=int(f[5]),int(f[6]),int(f[7],16)
                require('FETCH' in j and beat==j['out']<j['in'] and data==pattern(tag,beat) and last==(beat==7),'Extra output bytes/WLAST')
                j['out']+=1;bytes_checked+=64;counts[kind][ch]+=last;continue
            predecessor={'ADMIT':'SUB','RES':'ADMIT','FETCH':'RES','AW':'FETCH','B':'AW','CORE_CPL':'B','CPL':'CORE_CPL'}[kind]
            require(predecessor in j and kind not in j and cyc>=j[predecessor],'Extra causality/duplicate event')
            if kind in counts:
                require(counts[kind][ch]==0,'Extra duplicate channel event');counts[kind][ch]+=1
            if kind=='RES':require(not last_fault,'Allocation after fault')
            if kind=='FETCH':
                require(active is None and budget==0 and int(f[5],16)==j['desc'],'Extra fetch ownership')
                active=tag;budget=8
            if kind=='AW':require(int(f[5],16)==0x100000000+tag*4096 and int(f[6])==8,'Extra AXI metadata')
            if kind=='B':require(j['out']==8 and f[5]=='0','Extra response before full write')
            if kind=='CPL':require(f[5]=='0','Extra completion status')
            j[kind]=cyc
    require(passed==[1,2,3] and text.count('OV EXTRA_PASS cases=3')==1,'Extra suite incomplete')
    require(windows==[(1,1,'active_candidate_fault'),(2,2,'active_candidate_stop_fence'),(3,3,'last_engine_beat_fault')],'Missing directed window')
    require(last_fault_beats==[1000] and fences==[202] and bytes_checked==2048,'Directed boundary coverage')
    require(last_beats==[(1,1000),(2,1000),(2,1001),(3,1000)],'Missing or duplicate last-beat event')
    return dict(cases=3,bytes=bytes_checked,completed_jobs=4,quarantined_jobs=2,last_beat_fault_verified=True)


def negative(text,checker,prefixes):
    rejected=[]
    for prefix in prefixes:
        lines=text.splitlines();idx=next(i for i,s in enumerate(lines) if s.startswith(prefix));del lines[idx]
        try:checker('\n'.join(lines))
        except (ValueError,IndexError,KeyError):rejected.append('missing_'+prefix.strip())
        else:raise ValueError('Undetected mutation '+prefix)
    return rejected


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--review',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True)
    a=ap.parse_args();out=a.output;record=json.loads((out/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['sources']==record['sources_after'],'Run/source failure')
    files=json.loads((out/'FILES.json').read_text())
    require(all(hashlib.sha256((out/n).read_bytes()).hexdigest()==h for n,h in files.items()),'Evidence hashes')
    console=(out/'console.log').read_text();variant=record['variant']
    require('NTRPU_OVERLAP_COMPLETE '+variant in console and not re.search(r'^ERROR:',console,re.M),'Tool completion')
    texts={n:(out/'suite'/(n+'.log')).read_text() for n in ('credit','protocol','extra','perf')}
    require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )',s,re.M) for s in texts.values()),'Simulation error')
    te=[s for s in texts['protocol'].splitlines() if s.startswith('TE ')]
    perf=performance(texts['perf']);baseline_text=a.baseline.read_text();baseline=performance(baseline_text)
    baseline_record=json.loads((a.baseline.parent.parent/'result.json').read_text())
    require(baseline_record['status']=='EXPORTED_PENDING_REVIEW' and
            baseline_record['sources']==baseline_record['sources_after'],'Baseline provenance')
    require(all(record['sources'][n]==h for n,h in baseline_record['sources'].items()),'Baseline reference sources changed')
    result=dict(status='OVERLAP_RTL_PASS',variant=variant,seconds=record['seconds'],physical_qualified=False,
                credit=credit(texts['credit']),protocol=replay(te),extra=extra(texts['extra']),performance=perf)
    result['negative_controls']=dict(protocol=negative_controls(te),performance=mutations(texts['perf'],performance),
        credit=negative(texts['credit'],credit,['OC CASE_PASS ','OC SUITE_PASS ']),
        extra=negative(texts['extra'],extra,['TE W ','TE RES ','OV WINDOW ','OV LAST ']))
    observed=Counter();last_count=0
    for line in texts['perf'].splitlines():
        f=line.split()
        if line.startswith('OV AHEAD '):
            require(f[4]!=f[5],'Prepared metadata aliases active lease');observed['busy_prepared']+=1
        elif line.startswith('OV ECHO_FETCH '):observed['simultaneous_echo_fetch']+=1
        elif line.startswith('OV LAST '):
            require(int(f[6])==(variant=='compact'),'Incorrect frame completion edge');last_count+=1
    require(observed['busy_prepared']>0 and last_count==1920,'Missing overlap coverage')
    require(observed['simultaneous_echo_fetch']==(1920 if variant=='compact' else 0),'Echo/fetch coverage')
    result['overlap_coverage']=dict(observed,last_beats=last_count)
    result['comparison']=[dict(profile=x['profile'],block_bytes=x['block_bytes'],baseline_cycles=b['cycles'],
        variant_cycles=x['cycles'],batch_gain_percent=round((b['cycles']/x['cycles']-1)*100,3),
        baseline_fetch_interval=b['fetch_interval_cycles'],variant_fetch_interval=x['fetch_interval_cycles'])
        for b,x in zip(baseline['rows'],perf['rows'])]
    result['baseline']=dict(path=str(a.baseline),sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest())
    result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','variant','seconds','overlap_coverage')}))


if __name__=='__main__':main()
