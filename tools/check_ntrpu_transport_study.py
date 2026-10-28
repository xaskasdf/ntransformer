"""Independent performance accounting, physical gates and routed-port comparison."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from check_ntrpu_candidate_transport import pattern, replay, negative_controls, require


def distribution(values):
    a=sorted(values)
    return dict(min=min(a),median=statistics.median(a),mean=round(statistics.mean(a),3),
                p95=a[math.ceil(.95*len(a))-1],max=max(a))


def performance(text):
    rows=[];case=0;jobs={};config=None;last_cycle=-1;samples=0;bytes_checked=0
    for line in text.splitlines():
        f=line.split()
        if line.startswith('TE CASE '):
            require(case==len(rows),'Incomplete performance case')
            case=int(f[3]);require(case==len(rows)+1,'Case order')
            jobs={};order=[[],[]];counts={k:[0,0] for k in ('FETCH','AW','W','B','CPL')}
            config=None;active=None;budget=0;writes=[];fetch_cycles=[];release=None;peak_banks=peak_queued=0
        elif line.startswith('PF CONFIG '):
            require(int(f[2])==case and config is None,'Duplicate config')
            config=tuple(map(int,f[3:]));profile,beats,count=config
            require(profile==(case-1)//3 and beats==(1,8,64)[(case-1)%3] and count==64,'Workload configuration')
        elif line.startswith('PF RELEASE '):
            require(config[0]==4 and release is None and counts['CPL']==[0,64],'Blocked lane did not preserve peer progress')
            release=int(f[2])
        elif line.startswith('TE S '):
            cycle=int(f[2]);require(cycle>last_cycle and int(f[3])==case,'Sample sequence');last_cycle=cycle;samples+=1
            require(f[4:6]==['0','0'],'Unexpected stop/fault')
            require(int(f[14])==budget,'Sample payload budget')
            queued=int(f[7],16);banks=int(f[8],16).bit_count()
            require(max(queued&15,queued>>4)<=10,'Queue overflow')
            peak_banks=max(peak_banks,banks);peak_queued=max(peak_queued,queued&15,queued>>4)
            if int(f[6]):require(all(j['done'] for j in jobs.values()) and budget==0,'False drain')
        elif line.startswith('TE CASE_PASS '):
            require(config and int(f[2])==case and list(map(int,f[3:]))==[64,64,64,64],'Workload totals')
            require(len(jobs)==128 and all(j['done'] for j in jobs.values()) and budget==0,'Lost job')
            require((release is not None)==(profile==4),'Missing blocked-peer case')
            total=128*beats*64;require(len(writes)*64==total,'Exact byte count');bytes_checked+=total
            begin=min(j['SUB'] for j in jobs.values());end=max(j['CPL'] for j in jobs.values())
            trim=len(writes)//8;window=sorted(writes)[trim:len(writes)-trim]
            span=window[-1]-window[0]+1
            latency={name:distribution([j[b]-j[a]+inclusive for j in jobs.values()])
                     for name,a,b,inclusive in [('submit_to_admit','SUB','ADMIT',0),('admit_to_reserve','ADMIT','RES',0),
                       ('reserve_to_fetch','RES','FETCH',0),('fetch_to_first_input','FETCH','first_in',0),
                       ('fill_span','first_in','last_in',1),('last_input_to_first_write','last_in','first_w',0),
                       ('last_write_to_response','last_w','B',0),('response_to_completion','B','CPL',0),
                       ('submit_to_completion','SUB','CPL',0)]}
            row=dict(case=case,profile=('unpaused','fetch_1_in_8','payload_1_in_2','write_1_in_2','lane0_blocked')[profile],
                     block_bytes=beats*64,jobs=128,bytes=total,cycles=end-begin+1,
                     batch_bytes_per_cycle=round(total/(end-begin+1),6),
                     batch_gbps_at_200mhz=round(total/(end-begin+1)/5,4),
                     trimmed_write_bytes_per_cycle=round(len(window)*64/span,6),
                     trimmed_write_gbps_at_200mhz=round(len(window)*64/span/5,4),
                     peak_banks=peak_banks,peak_queued_per_lane=peak_queued,latency_cycles=latency,
                     fetch_interval_cycles=distribution([b-a for a,b in zip(fetch_cycles,fetch_cycles[1:])]))
            if release is not None:
                first=min(j['SUB'] for j in jobs.values() if j['ch']==1)
                last=max(j['CPL'] for j in jobs.values() if j['ch']==1)
                require(all(j['first_w']>release for j in jobs.values() if j['ch']==0),'Blocked lane wrote before release')
                row['peer_while_blocked']=dict(bytes=64*beats*64,cycles=last-first+1,gbps_at_200mhz=round(64*beats*64/(last-first+1)/5,4))
            rows.append(row)
        elif line.startswith('TE ') and f[1] in ('SUB','ADMIT','RES','FETCH','IN','AW','W','B','CORE_CPL','CPL'):
            kind=f[1];cyc=int(f[2]);require(cyc==last_cycle,'Unpaired event sample')
            if kind=='IN':
                tag,beat,data=int(f[3]),int(f[4]),int(f[5],16)
                require(active==tag and budget>0 and jobs[tag]['in']==beat and data==pattern(tag,beat),'Input ownership/bytes')
                j=jobs[tag];j.setdefault('first_in',cyc);j['last_in']=cyc;j['in']+=1;budget-=1
                if budget==0:active=None
                continue
            ch,tag=int(f[3]),int(f[4]);require(ch in (0,1),'Lane')
            if kind=='SUB':
                require(config and tag not in jobs and tag==1+ch+2*len(order[ch]),'Duplicate or unordered publication')
                desc=int(f[5],16);expected=(0x76543210<<200)|(tag<<136)|((0x100000000+tag*4096)<<72)|((0x800000000+tag*4096)<<8)|(beats-1)
                require(desc==expected,'Descriptor metadata')
                jobs[tag]=dict(ch=ch,desc=desc,SUB=cyc,done=False,**{'in':0,'out':0});order[ch].append(tag)
                continue
            require(tag in jobs and jobs[tag]['ch']==ch,'Unknown job');j=jobs[tag]
            if kind=='W':
                beat,last,data=int(f[5]),int(f[6]),int(f[7],16)
                require(order[ch][counts[kind][ch]]==tag and 'FETCH' in j and beat==j['out']<j['in'],'Write order/input causality')
                require(data==pattern(tag,beat) and last==(beat==beats-1),'Write data/WLAST')
                j.setdefault('first_w',cyc);j['last_w']=cyc;j['out']+=1;counts[kind][ch]+=last;writes.append(cyc)
                continue
            predecessor={'ADMIT':'SUB','RES':'ADMIT','FETCH':'RES','AW':'FETCH','B':'AW','CORE_CPL':'B','CPL':'CORE_CPL'}[kind]
            require(predecessor in j and kind not in j and cyc>=j[predecessor],'Job causality/duplicate event')
            if kind in counts:
                require(order[ch][counts[kind][ch]]==tag,'Per-lane ordering');counts[kind][ch]+=1
            if kind=='FETCH':
                require(active is None and budget==0 and int(f[5],16)==j['desc'],'Fetch ownership')
                active=tag;budget=beats
                fetch_cycles.append(cyc)
            elif kind=='AW':require(int(f[5],16)==0x100000000+tag*4096 and int(f[6])==beats,'AXI address')
            elif kind=='B':require(j['out']==beats and f[5]=='0','Write/response')
            elif kind=='CPL':require(f[5]=='0' and not j['done'],'Completion status');j['done']=True
            j[kind]=cyc
    require(len(rows)==15 and len(re.findall(r'^PF ALL_PASS cases=15 ',text,re.M))==1,'Incomplete performance suite')
    return dict(status='PERFORMANCE_RTL_PASS',samples=samples,bytes=bytes_checked,rows=rows,
                timing_basis='5 ns simulation cycles; numerical GB/s conversion, not board throughput',
                window='Batch: first SUB through last CPL inclusive; trimmed W window removes first/last 1/8 of beat events')


def physical(out,stage):
    s=out/'suite';fields=dict(line.split(' ',1) for line in (s/'metrics.txt').read_text().splitlines())
    require(fields['CLOCK_PERIOD']=='5.000' and fields['CLOCK_LOCATION']=='BUFGCE_X0Y60','Physical clock context')
    require(all(fields[k]==v for k,v in {'CLOCKS':'1','BUFGCE':'1','LATCHES':'0','BLACKBOXES':'0'}.items()),'Physical structure')
    timing=(s/'timing.rpt').read_text();checks={k:int(v) for k,v in re.findall(r'checking (\w+) \((\d+)\)',timing)}
    match=re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
    require(match and len(match[1].split())==12,'Timing summary');values=list(map(float,match[1].split()))
    xdc=(s/'constraints_after.xdc').read_text()
    require(not re.search(r'^\s*(set_false_path|set_multicycle_path|set_clock_groups|set_max_delay|set_min_delay|set_disable_timing|create_waiver)\b',xdc,re.M),'Timing exceptions forbidden')
    require(not re.search(r'^\s*set_property\s+SEVERITY\b',xdc,re.M),'DRC override forbidden')
    if stage!='synth':
        before=(s/'constraints_before.xdc').read_text()
        budgets=lambda text:[x.strip() for x in text.splitlines() if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s',x)]
        require(budgets(before)==budgets(xdc),'Budgets changed')
    with (s/'drc_violations.tsv').open(newline='') as f:drc=list(csv.DictReader(f,delimiter='\t'))
    gates=dict(structure=True,setup_hold_pulse=all(values[i]>=0 for i in (0,4,8)) and all(values[i]==0 for i in (2,6,10)),
               internal_coverage=all(k in checks and checks[k]==0 for k in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops')),
               no_drc_errors=not any(x['severity'] in ('Error','Critical Warning') for x in drc))
    result=dict(status=stage.upper()+'_DIAGNOSTIC',gates=gates,internal_setup_ns=float(fields['INTERNAL_SETUP']),
                internal_hold_ns=float(fields['INTERNAL_HOLD']),setup_ns=values[0],hold_ns=values[4],pulse_ns=values[8],timing_checks=checks,drc=drc)
    if stage=='route':
        require((s/'route_command.txt').read_text().startswith('RETURN_CODE 0\n'),'Router returned failure')
        text=(s/'route_status.rpt').read_text();routes={}
        for label in ('# of routable nets','# of fully routed nets','# of nets with routing errors'):
            m=re.search(re.escape(label)+r'\.+\s*:\s*(\d+)',text);require(m,'Route counter absent');routes[label]=int(m[1])
        gates['routing']=routes['# of routable nets']>0 and routes['# of routable nets']==routes['# of fully routed nets'] and routes['# of nets with routing errors']==0
        with (s/'boundary_nets.tsv').open(newline='') as f:boundary=list(csv.DictReader(f,delimiter='\t'))
        gates['boundary_warnings_external_only']=all(r['ports'] and not r['loads'] for r in boundary)
        gates['no_critical_console']=not re.search(r'^CRITICAL WARNING:',(out/'console.log').read_text(),re.M)
        result.update(status='ROUTED_INTERNAL_TIMING_PASS' if all(gates.values()) else 'ROUTED_INTERNAL_NOT_QUALIFIED',routing=routes,boundary_warning_nets=len(boundary))
    result['limitations']=['External ports lack board/interface delays; no PCIe/HBM timing qualification.',
                           'Single clock and coordinated reset epoch; no CDC or warm recovery.']
    return result


def netlist(text):
    original=[s for s in text.splitlines() if s.startswith('TE ')]
    ref=replay(original);expected=defaultdict(list);actual=defaultdict(list);samples=Counter();matches=Counter();case=0;count=0
    for line in text.splitlines():
        f=line.split()
        if line.startswith('TE S '):case=int(f[3]);samples[case]+=1
        elif line.startswith('TE FETCH '):expected[case,'FETCH',int(f[3])].append((int(f[5],16),))
        elif line.startswith('TE W '):expected[case,'W',int(f[3])].append((int(f[7],16),(1<<64)-1,int(f[6])))
        elif line.startswith('TE CPL '):expected[case,'CPL',int(f[3])].append((int(f[4]),int(f[5])))
        elif line.startswith('TN MATCH '):
            count+=1;require(int(f[3])==count,'Missing routed sample');matches[int(f[2])]+=1
        elif line.startswith('TN FETCH '):actual[int(f[2]),'FETCH',int(f[3])].append((int(f[4],16),))
        elif line.startswith('TN W '):actual[int(f[2]),'W',int(f[3])].append((int(f[4],16),int(f[5],16),int(f[6])))
        elif line.startswith('TN CPL '):actual[int(f[2]),'CPL',int(f[3])].append((int(f[4]),int(f[5])))
    require(samples==matches and count>0 and re.findall(r'TN ALL_PASS checks=(\d+)',text)==[str(count)],'Routed sample coverage')
    require(expected==actual,'Routed data/metadata/status mismatch')
    ref.update(status='ROUTED_FUNCTIONAL_EQUIVALENCE_PASS',routed_samples=count,output_ports=28,sdf=False,
               negative_controls=negative_controls(original))
    return ref


def mutations(text,checker):
    rejected=[]
    prefixes=('TE W ','TE IN ','TE ADMIT ','TE CPL ') if checker is performance else ('TN W ','TN MATCH ','TN CPL ')
    for prefix in prefixes:
        lines=text.splitlines();at=next(i for i,s in enumerate(lines) if s.startswith(prefix));del lines[at]
        try:checker('\n'.join(lines))
        except (ValueError,IndexError,KeyError):rejected.append('missing_'+prefix.strip())
        else:raise ValueError('Undetected trace mutation '+prefix)
    return rejected


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();out=a.output;record=json.loads((out/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['sources']==record['sources_after'] and record['inputs']==record['inputs_after'],'Run/provenance failed')
    files=json.loads((out/'FILES.json').read_text())
    require(all(hashlib.sha256((out/n).read_bytes()).hexdigest()==h for n,h in files.items()),'Evidence hashes')
    stage=record['stage'];console=(out/'console.log').read_text()
    require('NTRPU_TRANSPORT_STUDY_COMPLETE '+stage in console and not re.search(r'^ERROR:',console,re.M),'Clean tool completion')
    if stage in ('perf','netlist'):
        text=(out/'suite/simulation.log').read_text();require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'Simulation error')
        checker=performance if stage=='perf' else netlist;result=checker(text);result['trace_negative_controls']=mutations(text,checker)
        if stage=='netlist':
            comp=json.loads((out/'suite/generated/COMPARISON.json').read_text())
            require(comp['latency_edges']==0 and comp['startup_ns']==150 and comp['reversible_renaming'] and not comp['sdf'] and not comp['force'],'Comparison scope')
            require(comp['netlist_sha256']==record['inputs']['physical_funcsim.v']==record['prior']['sha256'],'Netlist provenance')
    else:result=physical(out,stage)
    result.update(stage=stage,seconds=record['seconds'],checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','cases','sources','timing_checks','drc','negative_controls')}))


if __name__=='__main__':main()
