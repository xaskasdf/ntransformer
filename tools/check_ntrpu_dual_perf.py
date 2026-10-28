"""Replay finite routed R4 workloads and measure public-interface utilization."""
import argparse
from collections import Counter, deque
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

from check_ntrpu_composition import common, digest
from check_ntrpu_multibuffer_mig import need, pattern
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING, WARNING_DELAY, BEAT_PS, records

NAMES = ['balanced_64', 'balanced_512', 'balanced_4096', 'skew_3to1_4096',
         'w_pause0_4096', 'b_pause0_4096']


def vectors():
    rows = [(-1, 0, 0, 1), (-1, 1, 0, 1)]
    for s in range(6):
        n = 1 if s == 0 else 8 if s == 1 else 64
        rows += [(s, int(k % 4 == 3) if s == 3 else k % 2,
                  0x100000000+s*0x100000+k*4096, n) for k in range(16)]
    return rows


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)]


def longest(flags):
    peak = run = 0
    for flag in flags:
        run = run+1 if flag else 0
        peak = max(peak, run)
    return peak


def replay(text):
    need(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|Fatal:|FAIL )|\bModel:ERROR:', text), 'Tool/model failure')
    expected = vectors();jobs = {};offers = {};begins = {};ends = {};verified = {};releases = {};samples = {s:[] for s in range(6)}
    queues = [{k:deque() for k in ['AW','W','B','CPL']} for _ in range(2)]
    read_queue = [deque(),deque()];fill = None;state = {};clocks = {};summary = None
    writes = [0,0];reads = [0,0];pending = [0,0];peak = [0,0];last = -1
    bank_changes=[(0,0)]
    for line in text.splitlines():
        if not line.startswith(('DUAL ','PERF ')):continue
        f = line.split();kind = f[1]
        if line.startswith('DUAL PERF PASS '):
            need(summary is None,'Unique terminal summary')
            summary = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)};continue
        if kind == 'CASE':continue
        now = int(f[2]);need(now >= last,'Chronology');last = now
        if f[0] == 'PERF':
            s = int(f[3])
            if kind == 'OFFER':
                tag,ch = map(int,f[4:6]);need(tag not in offers and tag<=98,'Offer identity')
                need((s,ch)==expected[tag-1][:2],'Offered destination');offers[tag] = now
            elif kind == 'BEGIN':
                need(s not in begins and s in range(6),'Unique scenario start')
                fields = dict(x.split('=') for x in f[4:]);n = 1 if s==0 else 8 if s==1 else 64
                need(fields==dict(first=str(3+s*16),jobs='16',beats=str(n),w_hold='01' if s==4 else '00',b_hold='01' if s==5 else '00'),'Scenario configuration')
                begins[s] = now
            elif kind == 'END':
                need(s in begins and s not in ends and now>begins[s],'Scenario end')
                need(dict(x.split('=') for x in f[4:])==dict(accepted=str(18+s*16),completed=str(18+s*16)),'Scenario totals')
                need(all(not any(q.values()) for q in queues) and fill is None and pending==[0,0],'End drain')
                ends[s] = now
            elif kind == 'RELEASE':
                need(s in (4,5) and s not in releases and f[4]=='port=0','Controlled pause release')
                need(4000000<=now-begins[s]<=4005000,'800 source-cycle pause');releases[s]=now
            elif kind == 'S':
                need(s in begins and s not in ends and len(f)==12,'Sample within measurement')
                mask,p = int(f[4],16),int(f[5],16);bits=[int(v,2) for v in f[6:]]
                need(mask<256 and p<1024 and (p&31)<=4 and (p>>5)<=4 and all(v in (0,1) for v in bits[:4]) and all(v<4 for v in bits[4:]),'Known bounded sample')
                samples[s].append((now,mask,p,*bits))
            elif kind == 'AR':
                ch=int(f[4]);addr=int(f[5],16);length=int(f[6])
                need(ch in clocks and now%5000==clocks[ch] and not read_queue[ch],'Read burst clock/serialization')
                candidates=[(t,j) for t,j in jobs.items() if j['ch']==ch and j['addr']==addr and j['scenario']==s]
                need(len(candidates)==1,'Read burst address owner');tag,j=candidates[0]
                need(length==j['n'] and 'cpl' in j and j['cpl']<now and not j['read'],'Read after completion, exact length')
                read_queue[ch].append((tag,now))
            elif kind == 'RD':
                ch,tag,beat=map(int,f[4:7]);value=int(f[7],16);is_last=int(f[8])
                need(ch in clocks and now%5000==clocks[ch] and read_queue[ch] and read_queue[ch][0][0]==tag,'Read identity/clock')
                j=jobs[tag];need(j['scenario']==s and beat==len(j['read']) and value==pattern(tag,beat) and is_last==int(beat==j['n']-1),'Read payload/length')
                need(now>(j['read'][-1] if j['read'] else read_queue[ch][0][1]),'Read causal ordering');j['read'].append(now);reads[ch]+=1
                if is_last:read_queue[ch].popleft()
            elif kind == 'VERIFIED':
                need(s in ends and s not in verified and now>ends[s],'Readback follows measurement')
                need(all(len(j['read'])==j['n'] for j in jobs.values() if j['scenario']==s),'All case bytes read')
                verified[s]=now
            else:raise ValueError('Unknown PERF event '+kind)
            continue
        if kind == 'STATE':state=dict(x.split('=') for x in f[3:])
        elif kind == 'CLOCK':
            v=dict(x.split('=') for x in f[3:]);ch=int(v['port'])
            need(ch not in clocks and int(v['period'])==5000 and int(v['phase'])==now%5000,'UI clock');clocks[ch]=int(v['phase'])
        elif kind == 'CMD':
            ch,tag=map(int,f[3:5]);addr=int(f[5],16);n,status=map(int,f[6:8])
            need(tag==len(jobs)+1 and tag in offers and offers[tag]<=now and fill is None,'Command sequence')
            need(now%5000==3200,'Command source clock')
            s=expected[tag-1][0];need((s,ch,addr,n)==expected[tag-1] and status==0,'Exact workload')
            need(state.get('calib')=='11' and state.get('ui_rst')=='00' and state.get('admit')==state.get('running')=='1','Admission')
            jobs[tag]=dict(scenario=s,ch=ch,addr=addr,n=n,cmd=now,inputs=[],outputs=[],read=[])
            fill=tag
            for q in queues[ch].values():q.append(tag)
        elif kind == 'IN':
            tag,beat=map(int,f[3:5]);j=jobs[tag]
            need(tag==fill and beat==len(j['inputs']) and int(f[5],16)==pattern(tag,beat),'Input payload')
            need(now%5000==3200 and now>(j['inputs'][-1] if j['inputs'] else j['cmd']),'Input source clock/order')
            j['inputs'].append(now)
            if len(j['inputs'])==j['n']:fill=None
        elif kind in ('AW','W','B','CPL'):
            ch,tag=map(int,f[3:5]);j=jobs[tag];q=queues[ch][kind]
            need(j['ch']==ch and q and q[0]==tag,'Destination queue order')
            if kind!='CPL':need(ch in clocks and now%5000==clocks[ch],'Memory handshake clock')
            if kind=='AW':
                need(int(f[5],16)==j['addr'] and int(f[6])==j['n'] and len(j['inputs'])==j['n'] and j['inputs'][-1]<now,'AW fields/full frame')
                j['aw']=now;pending[ch]+=1;peak[ch]=max(peak[ch],pending[ch]);need(pending[ch]<=4,'MIG credits');q.popleft()
            elif kind=='W':
                beat,last_beat=map(int,f[5:7]);need(beat==len(j['outputs']) and int(f[7],16)==pattern(tag,beat) and last_beat==int(beat==j['n']-1),'Write payload/last')
                need(len(j['inputs'])==j['n'] and now>j['inputs'][-1],'Write after fill')
                need(not j['outputs'] or now>j['outputs'][-1],'Distinct write clock edges')
                j['outputs'].append(now);writes[ch]+=1
                if last_beat:q.popleft()
            elif kind=='B':
                need(int(f[5])==0 and 'aw' in j and len(j['outputs'])==j['n'] and now>max(j['aw'],j['outputs'][-1]),'B causality/status')
                j['b']=now;pending[ch]-=1;need(pending[ch]>=0,'Pending balance');q.popleft()
            else:
                need(int(f[5])==0 and 'b' in j and now>j['b'] and 'cpl' not in j,'Completion causality')
                need(now%5000==3200,'Completion source clock')
                j['cpl']=now;q.popleft()
        elif kind=='DRAIN':need(fill is None and all(not any(q.values()) for q in queues) and pending==[0,0],'Full drain')
        elif kind=='BANKS':
            mask=int(f[3].split('=')[1],16)
            need(now%5000==3200 and mask<256 and mask!=bank_changes[-1][1],'Bank transition')
            bank_changes.append((now,mask))
        elif kind=='R':pass  # Tagged burst reads are reviewed independently.
        else:raise ValueError('Unknown DUAL event '+kind)
    need(clocks=={0:998,1:1729} and len(jobs)==98 and len(offers)==98 and set(verified)==set(begins)==set(ends)==set(range(6)),'Complete workload coverage')
    need(set(releases)=={4,5} and not any(read_queue) and all(not any(q.values()) for q in queues),'Final queues')
    need(all(len(j['inputs'])==len(j['outputs'])==len(j['read'])==j['n'] and 'cpl' in j for j in jobs.values()),'All bytes completed/read')
    totals=dict(commands=98,completions=98,writes0=writes[0],writes1=writes[1],reads0=reads[0],reads1=reads[1],scenarios=6)
    need(summary and all(summary.get(k)==v for k,v in totals.items()) and summary['ps']>=last,'Summary matches replay')
    result=[]
    for s,name in enumerate(NAMES):
        rows=samples[s];start,end=begins[s],ends[s];duration=end-start;group={t:j for t,j in jobs.items() if j['scenario']==s}
        need([r[0] for r in rows]==list(range(start+2500,end,5000)) and len(rows)==duration//5000,'Every source cycle sampled')
        for row in rows:
            need(row[1]==next(mask for t,mask in reversed(bank_changes) if t<=row[0]),'Cycle sample matches bank transition monitor')
        need(all(start<=offers[t]<=j['cmd']<j['cpl']<end for t,j in group.items()),'All jobs within window')
        if s:need(verified[s-1]<start,'Sequential verified scenarios')
        commands={j['cmd'] for j in group.values()};data_times={t for j in group.values() for t in j['inputs']}
        need(len(commands)==16 and len(data_times)==sum(j['n'] for j in group.values()),'One source handshake per edge')
        need({r[0] for r in rows if r[3] and r[4]}==commands and {r[0] for r in rows if r[5] and r[6]}==data_times,'Sample/transaction handshake agreement')
        for ch in range(2):
            need({r[0] for r in rows if (r[7]&r[8])>>ch&1}=={j['cpl'] for j in group.values() if j['ch']==ch},'Sample/completion agreement')
        byte_counts=[sum(j['n']*64 for j in group.values() if j['ch']==ch) for ch in range(2)]
        cmd_stalls=[bool(r[3] and not r[4]) for r in rows];data_stalls=[bool(r[5] and not r[6]) for r in rows]
        lat=[(j['cpl']-j['cmd'])/1000 for j in group.values()]
        offered_lat=[(j['cpl']-offers[t])/1000 for t,j in group.items()]
        occupancy=Counter(r[1].bit_count() for r in rows);reserve=Counter((r[1]>>4).bit_count() for r in rows)
        per_port=[]
        for ch in range(2):
            port={t:j for t,j in group.items() if j['ch']==ch}
            waits=[(j['cmd']-offers[t])/1000 for t,j in port.items()]
            per_port.append(dict(port=ch,bytes=byte_counts[ch],goodput_GBps=byte_counts[ch]*1000/duration,
                command_wait_max_ns=max(waits),pending_source_peak=max((r[2]>>(5*ch))&31 for r in rows),
                descriptor_peak=max(sum(j['cmd']<r[0]<=j['cpl'] for j in port.values()) for r in rows),
                dedicated_bank_peak=max(((r[1]>>(2*ch))&3).bit_count() for r in rows),
                completed_during_pause=sum(j['cpl']<releases[s] for j in port.values()) if s>=4 else None))
        if s>=4:
            need(per_port[1]['completed_during_pause']>0,'Unpaused peer progress')
            if s==4:need(all(t>=releases[s] for j in group.values() if j['ch']==0 for t in j['outputs']),'W pause enforced')
            else:need(all(j['b']>=releases[s] for j in group.values() if j['ch']==0),'B pause enforced')
        result.append(dict(name=name,id=s,jobs=16,bytes=sum(byte_counts),begin_ps=start,end_ps=end,duration_ns=duration/1000,
            goodput_GBps=sum(byte_counts)*1000/duration,input_bus_utilization=len(data_times)/len(rows),
            source_cycles=len(rows),input_beats=len(data_times),input_idle_cycles=sum(not r[5] for r in rows),
            input_stall_cycles=sum(data_stalls),command_stall_cycles=sum(cmd_stalls),
            longest_command_stall_ns=longest(cmd_stalls)*5,longest_data_stall_ns=longest(data_stalls)*5,
            bank_peak=max(occupancy),bank_mean=sum(n*c for n,c in occupancy.items())/len(rows),
            reserve_peak=max(reserve),bank_occupancy_cycles=dict(occupancy),reserve_occupancy_cycles=dict(reserve),
            completion_latency_ns={f'p{p}':percentile(lat,p/100) for p in [50,95,99]},ports=per_port,
            offer_to_completion_latency_ns={f'p{p}':percentile(offered_lat,p/100) for p in [50,95,99]},
            pause_duration_ns=(releases[s]-start)/1000 if s>=4 else 0))
    return dict(status='FINITE_R4_PERFORMANCE_BYTE_REPLAY_PASS',scenarios=result,write_beats=writes,read_beats=reads,
                total_write_bytes=sum(writes)*64,total_read_bytes=sum(reads)*64,memory_credit_peak=peak)


# Same physical periodic witness; transaction totals come from this workload.
def periodic(text, writes, reads):
    parts = [[],[]]; candidates = {}
    for ch,record in re.findall(r'^DUAL_PER ([01]) (.*)$',text,re.M): parts[int(ch)].append(record)
    need(len(re.findall(r'^DUAL_PER ',text,re.M))==sum(map(len,parts)),'Known periodic port tags')
    for ch,lines in enumerate(parts):
        for pin in records('\n'.join(lines),'NTRPU_DDR_PIN_CMD', ['ps','slot','rank','act_n','cke'], ['cmd','adr','bg','bank']):
            if pin['act_n']==1 and (pin['adr']>>14)&7==5:
                for beat in range(8):candidates.setdefault(pin['ps']+WARNING_DELAY+beat*BEAT_PS,set()).add(ch)
    warning_counts = [0,0]
    for line in text.splitlines():
        if re.search(r':\s*WARNING\s*:',line):
            match=WARNING.fullmatch(line);need(match is not None,'Unknown model warning')
            owners=candidates.get(int(match[7]),set());need(len(owners)==1,'Unique physical owner of model warning')
            ch=next(iter(owners));parts[ch].append(line);warning_counts[ch]+=1
    finish=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    times={int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in finish}
    need(len(times)==1,'One global finish time');finish_ps=times.pop()
    result=[]
    for ch,lines in enumerate(parts):
        local='\n'.join(lines); witness=review_periodic(local)
        rows=records(local,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
        need(len(rows)==1,'One final periodic drain per controller');d=rows[0]
        phase=(witness['configuration']['ps']+2500)%5000
        first=d['begin_ps']+(phase-d['begin_ps'])%5000
        if first==d['begin_ps']:first+=5000
        need(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'Consecutive local drain edges')
        need(0<=finish_ps-d['end_ps']<1000000,'Global finish within bounded peer drain window')
        counts=witness['counters']
        need(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'Drain counters')
        returns=records(local,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
        need(returns[-1]['ps']<d['end_ps'] and all(x['return_ps']<d['end_ps'] for x in witness['lifecycle']),'All returns before local drain')
        need(counts['user_cas']==reads[ch] and witness['physical_writes']==writes[ch],'Per-controller physical transfers')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='DUAL_PERIODIC_WITNESS_PASS',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)


def negative_controls(text):
    out=[]
    lines=text.splitlines()
    cases=[('write_byte','DUAL W '),('read_byte','PERF RD '),('missing_b','DUAL B '),
           ('missing_sample','PERF S '),('wrong_clock','PERF RD '),('missing_case','PERF VERIFIED '),
           ('pause_release','PERF RELEASE '),('wrong_occupancy','PERF S ')]
    for label,prefix in cases:
        changed=list(lines);i=next(i for i,l in enumerate(lines) if l.startswith(prefix));f=changed[i].split()
        if label=='write_byte':f[-1]=format(int(f[-1],16)^1,'0128x')
        elif label=='read_byte':f[-2]=format(int(f[-2],16)^1,'0128x')
        elif label=='wrong_clock':f[2]=str(int(f[2])+1)
        elif label=='pause_release':f[2]=str(int(f[2])+10000)
        elif label=='wrong_occupancy':f[4]=format(int(f[4],16)^1,'02x')
        else:changed.pop(i)
        if label not in ('missing_b','missing_sample','missing_case'):changed[i]=' '.join(f)
        try:replay('\n'.join(changed))
        except (ValueError,KeyError,IndexError) as error:out.append(dict(mutation=label,rejection=str(error)))
        else:raise ValueError('Missed negative control '+label)
    return out


def review(root,physical):
    record,console=common(root,'NTRPU_DUAL_PERF_EXPORTED')
    need(record['example_unchanged'],'MIG seed preserved')
    need(digest(root/'inputs/evidence/routed_core_funcsim.v')==digest(physical/'output/suite/routed_core_funcsim.v'),'Exact routed core provenance')
    need((root/'inputs/fpga/rtl/ntrpu_dual_mig_cold_start.sv').read_text()==
         (physical/'inputs/fpga/rtl/ntrpu_dual_mig_cold_start.sv').read_text(),'Integrated source matches routed candidate modulo line endings')
    unit=(root/'output/suite/startup.log').read_text()
    cases=re.findall(r'DUAL_START_CASE_PASS case=(\d+) entries=(\d+) locked=(\d+)',unit)
    need(sorted(tuple(map(int,c)) for c in cases)==[(g,1,int(1<=g<=8 or g==12)) for g in range(13)] and
         'DUAL_START_SUITE_PASS cases=13' in unit and not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',unit,re.M),'13 startup cases')
    text=(root/'output/suite/memory.log').read_text();result=replay(text)
    result['periodic']=periodic(text,result['write_beats'],result['read_beats'])
    result['negative_controls']=negative_controls(text)
    result.update(status='ROUTED_R4_FINITE_PERFORMANCE_PASS',seconds=record['seconds'],startup_cases=13,
        routed_core_sha256=digest(root/'inputs/evidence/routed_core_funcsim.v'),cpus=record['cpus'],nice=record['nice'],
        limitations=['Six finite 16-frame cases, single deterministic run; not steady-state throughput or statistical tail estimates.',
                    'PCIe/P2P, HBM, board/PHY implementation and SDF timing are outside this test.',
                    'Goodput window includes frame fill, command overhead and final drain; excludes calibration and readback.',
                    'Descriptors are inferred from accepted commands minus completions; public pending_counts counts source outstanding writes.',
                    'Bank occupancy is sampled each source cycle; reserve ownership by destination is not directly observed.',
                    'One shared source sends frames serially; a blocked command can delay later frames for the other destination.'])
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--physical',type=Path,required=True)
    args=ap.parse_args();result=review(args.run,args.physical);result['checker_sha256']=digest(Path(__file__))
    with (args.run/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','scenarios','total_write_bytes','total_read_bytes','seconds']},indent=2))
