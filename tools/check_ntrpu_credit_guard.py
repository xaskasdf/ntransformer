"""Review direct write-guard safety traces and bounded physical comparisons."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def require(ok,message):
    if not ok:raise RuntimeError(message)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def payload(tag,beat):
    return sum((((tag*0x9e3779b9)^(beat*0x10203)^(lane*0x01010101))&0xffffffff)<<(lane*32) for lane in range(16))


def replay(text,depth):
    rejected={1,3,4,5,6,7,8,9,15,16}
    jobs={};valid=[n for n in range(36) if n not in rejected]
    counters=dict(AW=0,W=0,B=0,CPL=0);written=0;peak=0;last_time=-1
    for line in text.splitlines():
        if not line.startswith('UNIT '):continue
        f=line.split();kind=f[1];cycle,tag=map(int,f[2:4])
        require(cycle>=last_time,'nonmonotonic trace');last_time=cycle
        if kind=='CMD':
            require(tag==len(jobs),'input command order')
            count=1+tag%7
            if tag in (2,4):count=1
            elif tag==3:count=2
            elif tag==15:count=65
            response=3 if tag in rejected else 2 if tag==11 else 0
            require(int(f[5])==count and int(f[6])==response,'test vector mismatch')
            address={1:0x2000000000,2:0x1fffffffc0,3:0x100000fc0,4:0xffffffffffffffc0,
                     9:0x100009001,16:0x2100000000}.get(tag,0x100000000+(tag<<12))
            require(int(f[4],16)==address,'input address vector mismatch')
            jobs[tag]=dict(length=count,response=response,output=[])
            peak=max(peak,len(jobs)-counters['CPL'])
            require(peak<=depth,'descriptor overflow')
            continue
        require(tag in jobs,'unknown job');job=jobs[tag]
        if kind in ('AW','W','B'):
            require(counters[kind]<len(valid) and valid[counters[kind]]==tag,'memory access order/reject leak')
            if kind=='AW':job['aw']=cycle;counters[kind]+=1
            elif kind=='W':
                beat=int(f[4]);strobe=int(f[5],16);data=int(f[6],16)
                expected_strobe=0xffffffffffffffff if (tag+beat)%2==0 else 0xff00ff00ff00ff00
                require(beat==len(job['output']) and beat<job['length'] and data==payload(tag,beat) and strobe==expected_strobe,'output bytes/strobes')
                job['output'].append(data);written+=64
                if len(job['output'])==job['length']:job['w']=cycle;counters[kind]+=1
            else:
                require('aw' in job and 'w' in job and max(job['aw'],job['w'])<cycle,'early memory B')
                require(int(f[4])==job['response'],'memory status');job['b']=cycle;counters[kind]+=1
        elif kind=='CPL':
            require(tag==counters['CPL'] and int(f[4])==job['response'],'completion order/status')
            if tag in rejected:require(not job['output'] and 'aw' not in job and 'b' not in job,'reject leak')
            else:require(job.get('b',cycle)<cycle,'premature completion')
            counters[kind]+=1
        else:raise RuntimeError('unknown event '+kind)
    require(len(jobs)==36 and counters==dict(AW=26,W=26,B=26,CPL=36),'incomplete trace')
    require(peak==depth,'credit depth not exercised')
    m=re.findall(r'^WRITE_GUARD_SUITE_PASS (.+)$',text,re.M);require(len(m)==1,'missing pass')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',m[0])}
    require(all(summary.get(k)==v for k,v in dict(depth=depth,commands=36,accepted_memory=26,bytes=written,peak=peak).items()),'summary mismatch')
    require(all(summary[k]>0 for k in ('early_w','credit_stall','admit_stall','b_stall')),'missing stress coverage')
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'unit error')
    return dict(status='DIRECT_GUARD_REPLAY_PASS',**summary,rejected=10)


def negative_traces(text,depth):
    lines=text.splitlines();results=[]
    for kind in ('payload','strobe','address','status','duplicate','missing_b'):
        changed=lines.copy()
        if kind in ('payload','strobe'):
            n=next(i for i,l in enumerate(changed) if l.startswith('UNIT W '))
            fields=changed[n].split();column=6 if kind=='payload' else 5
            fields[column]=format(int(fields[column],16)^1,'x');changed[n]=' '.join(fields)
            reason='output bytes/strobes'
        elif kind=='address':
            n=next(i for i,l in enumerate(changed) if l.startswith('UNIT CMD '))
            fields=changed[n].split();fields[4]=format(int(fields[4],16)&0xffffffff,'x');changed[n]=' '.join(fields)
            reason='input address vector mismatch'
        elif kind=='missing_b':
            n=next(i for i,l in enumerate(changed) if l.startswith('UNIT B '))
            changed.pop(n);reason='premature completion'
        else:
            n=next(i for i,l in enumerate(changed) if l.startswith('UNIT CPL '))
            if kind=='duplicate':changed.insert(n,changed[n])
            else:
                fields=changed[n].split();fields[-1]='3';changed[n]=' '.join(fields)
            reason='completion order/status'
        try:replay('\n'.join(changed),depth)
        except RuntimeError as error:
            require(str(error)==reason,'unexpected unit trace rejection '+str(error))
            results.append(dict(mutation=kind,rejection=reason))
        else:raise RuntimeError('Unit corruption escaped: '+kind)
    return results


def review(root):
    output=root/'output';record=json.loads((output/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['returncode']==0 and not record['timed_out'],'incomplete run')
    require(record['sources']==record['sources_after'],'source mutation')
    for name,h in record['sources'].items():require(digest(root/'inputs'/name)==h,'input hash '+name)
    for name,h in json.loads((output/'FILES.json').read_text()).items():require(digest(output/name)==h,'output hash '+name)
    console=(output/'console.log').read_text()
    require('NTRPU_CREDIT_GUARD_EXPORTED' in console,'missing export')
    require(not re.search(r'^ERROR:|^CRITICAL WARNING:',console,re.M),'tool error/critical warning')
    require((output/'suite/fixture.xdc').read_text().strip()=='create_clock -period 5.000 -name core_clk [get_ports clk]','unexpected fixture constraints')
    warning_meanings={
        'XSIM 43-3431':'LIBRARY_PATH from the Vivado environment affects the compiler',
        'VRFC 10-3532':'DEPTH generic is also offered to glbl; actual testbench parameter is checked in traces',
        'Synth 8-13374':'Unreachable constant-propagated instances removed; routed resource totals are reviewed',
        'Synth 8-4767':'Small descriptor memories map to registers',
        'Netlist 29-101':'Large primitive population in the register fixture; not a board floorplan',
        'Vivado 12-180':'No black-box cells matched the explicit query',
        'DRC 23-814':'OOC excludes some connectivity checks; zero errors is limited to checks actually run',
        'Route 35-198':'Fixture ports lack physical partition-pin locations; external IO timing is unqualified',
        'Route 35-197':'Clock input has no external HD.CLK_SRC; physical upstream clock checks are unqualified'}
    warning_counts=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M)))
    require(set(warning_counts)<=set(warning_meanings),'unreviewed warnings '+str(set(warning_counts)-set(warning_meanings)))
    units=[]
    for d in (1,2,4):
        text=(output/f'suite/unit-{d}.log').read_text();result=replay(text,d)
        result['trace_negative_controls']=negative_traces(text,d);units.append(result)
    negatives=[]
    for n,pattern in [(1,r'WRITE_GUARD_(AW|DATA)_MISMATCH'),(2,'WRITE_GUARD_DATA_MISMATCH'),(3,'WRITE_GUARD_RESPONSE')]:
        text=(output/f'suite/unit-negative-{n}.log').read_text()
        require(re.search('Fatal: '+pattern,text) and 'WRITE_GUARD_SUITE_PASS' not in text,'negative escaped')
        negatives.append(n)
    physical=[]
    for slots in (0,1,2,4):
        folder=output/f'suite/physical-{slots}'
        metrics=dict(line.split('=',1) for line in (folder/'metrics.txt').read_text().splitlines())
        require(int(metrics['DRC_ERRORS'])==0,'physical DRC error')
        util=(folder/'guard_utilization.rpt').read_text()
        resources={}
        for label in ('CLB LUTs','LUT as Memory','CLB Registers','Register as Latch','Block RAM Tile','CARRY8','DSPs'):
            m=re.search(r'^\|\s*'+re.escape(label)+r'\*?\s*\|\s*([\d.,]+)',util,re.M)
            require(m is not None,'missing resource '+label)
            resources[label]=float(m.group(1).replace(',',''))
        require(resources['Register as Latch']==0,'inferred latches')
        timing=(folder/'timing.rpt').read_text()
        checks={name:int(count) for name,count in re.findall(r'checking (\w+) \((\d+)\)',timing)}
        for name in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops'):
            require(checks.get(name)==0,'timing coverage '+name)
        match=re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
        require(match and len(match[1].split())==12,'timing table')
        values=list(map(float,match[1].split()))
        require(values[0]==float(metrics['SETUP_NS']) and values[4]==float(metrics['HOLD_NS']),'slack table mismatch')
        routing={};route=(folder/'route_status.rpt').read_text()
        for name in ('# of routable nets','# of fully routed nets','# of nets with routing errors'):
            found=re.search(re.escape(name)+r'\.+\s*:\s*(\d+)',route)
            require(found is not None,'route field '+name);routing[name]=int(found[1])
        require(routing['# of routable nets']==routing['# of fully routed nets'] and routing['# of nets with routing errors']==0,'incomplete routing')
        physical.append(dict(slots=slots,setup_ns=values[0],hold_ns=values[4],pulse_ns=values[8],
            setup_failing_endpoints=values[2],hold_failing_endpoints=values[6],pulse_failing_endpoints=values[10],
            resources=resources,timing_checks=checks,routing=routing))
    closed=all(p['setup_ns']>=0 and p['hold_ns']>=0 and p['pulse_ns']>=0 for p in physical)
    return dict(status='GUARD_FUNCTIONAL_AND_OOC_TIMING_PASS' if closed else 'GUARD_FUNCTIONAL_PASS_TIMING_OPEN',
                units=units,negative_controls=negatives,physical=physical,seconds=record['seconds'],
                cpus=record['cpus'],nice=record['nice'],
                warnings=warning_counts,warning_meanings=warning_meanings,
                limits=['Guard only; common registered OOC fixture at 200 MHz',
                        'Top-level fixture IO delays and pin assignments are not board timing constraints',
                        'No memory CDC, PCIe, MIG/HBM or whole-composition physical closure'],
                checker_sha256=digest(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=review(args.run)
    with (args.run/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
