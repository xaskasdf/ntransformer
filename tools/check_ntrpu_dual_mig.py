"""Replay two-controller cold MIG transactions and passive DDR witnesses."""
import argparse
from collections import Counter, deque
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_multibuffer_mig import need, pattern
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING, WARNING_DELAY, BEAT_PS, records


def vectors():
    out = [(0,0,1),(1,0,1)]
    sparse = [0x100000040,0x800000000,0x1000000000,0x1800000000,0x1fffffffc0]
    out += [(ch,a,1) for ch in range(2) for a in sparse]
    out += [(ch,a,n) for ch in range(2) for a,n in [(0x100001000,64),(0x1ffffffe00,8)]]
    out += [(ch,0x100002000+k*4096,1) for k in range(4) for ch in range(2)]
    out += [(ch,0x100006000,4) for ch in range(2)]
    out += [(k%2,0x200001000+(k//2)*4096,4) for k in range(24)]
    out += [(0,0x300000000,8),(1,0x300000000,8),(1,0x400000000,8),
            (0,0x500000000,1),(1,0x500000000,1),(1,0x2000000000,1)]
    return out


def replay(text):
    need(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|Fatal:|FAIL )|\bModel:ERROR:',text), 'Simulator/model error')
    expected = vectors(); jobs = {}; fill = None; completed = set()
    queues = [{kind:deque() for kind in ('AW','W','B','CPL')} for _ in range(2)]
    golden = [{},{}]; reads = [0,0]; writes = [0,0]; pending = [0,0]; peak = [0,0]
    reading = [None,None]; sweep = [deque(),deque()]; sweeps = Counter()
    case = None; cases = []; state = {}; clocks = {}; stops = []
    summary = None; last_time = -1; banks = max_banks = drains = 0
    saturated = progress = reserve = False
    for line in text.splitlines():
        if not line.startswith('DUAL '): continue
        fields = line.split(); kind = fields[1]
        if kind == 'CASE': case = fields[2]; cases.append(case); continue
        if kind == 'PASS':
            need(summary is None, 'Duplicate PASS')
            summary = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}; continue
        now = int(fields[2]); need(now >= last_time, 'Trace chronology'); last_time = now
        if kind == 'STATE': state = dict(x.split('=') for x in fields[3:]); continue
        if kind == 'CLOCK':
            values = dict(x.split('=') for x in fields[3:]); ch = int(values['port'])
            need(ch in (0,1) and ch not in clocks and int(values['period'])==5000, 'Clock configuration')
            clocks[ch] = int(values['phase']); need(clocks[ch]==now%5000,'Clock phase'); continue
        if kind in ('AW','W','B','AR','R'):
            ch=int(fields[3]);need(ch in clocks and now%5000==clocks[ch],'Handshake on actual destination UI edge')
        if kind == 'BANKS':
            banks = int(fields[3].split('=')[1],16); need(banks<256,'Bank mask width')
            max_banks = max(max_banks,banks.bit_count()); continue
        if kind == 'CMD':
            ch,tag = map(int,fields[3:5]); address = int(fields[5],16); length,status = map(int,fields[6:])
            need(tag==len(jobs)+1 and tag<=56 and fill is None,'Command identity')
            need((ch,address,length)==expected[tag-1] and status==(3 if tag==56 else 0),'Command vector/range')
            need(state.get('calib')=='11' and state.get('ui_rst')=='00' and state.get('resets')=='11'
                 and state.get('admit')=='1' and state.get('running')=='1','Shared admission barrier')
            jobs[tag] = dict(ch=ch,address=address,length=length,status=status,inputs=[],outputs=[])
            fill = tag; queues[ch]['CPL'].append(tag)
            if status==0:
                for name in ('AW','W','B'): queues[ch][name].append(tag)
        elif kind == 'IN':
            tag,beat = map(int,fields[3:5]); value = int(fields[5],16)
            need(tag==fill and beat==len(jobs[tag]['inputs']) and value==pattern(tag,beat),'Source frame')
            jobs[tag]['inputs'].append(value)
            if len(jobs[tag]['inputs'])==jobs[tag]['length']: jobs[tag]['filled']=now; fill=None
        elif kind in ('AW','W','B','CPL'):
            ch,tag = map(int,fields[3:5]); need(tag in jobs and ch==jobs[tag]['ch'],'Destination identity')
            job = jobs[tag]; queue = queues[ch][kind]
            need(queue and queue[0]==tag,'Per-destination '+kind+' order')
            need(job.get('filled',now)<now,'Transaction before completed source frame')
            if kind=='AW':
                need(int(fields[5],16)==job['address'] and int(fields[6])==job['length'],'MIG address/length')
                queue.popleft(); job['aw']=now; pending[ch]+=1; peak[ch]=max(peak[ch],pending[ch])
                need(pending[ch]<=4,'Credit overflow')
            elif kind=='W':
                beat,last = map(int,fields[5:7]); value = int(fields[7],16)
                need(beat==len(job['outputs']) and value==pattern(tag,beat),'MIG write payload')
                need(last==int(beat==job['length']-1),'WLAST')
                job['outputs'].append(value); writes[ch]+=1
                if last: queue.popleft(); job['last_w']=now
            elif kind=='B':
                need(int(fields[5])==0 and job.get('aw',now)<now and job.get('last_w',now)<now,'B status/causality')
                queue.popleft(); job['b']=now; pending[ch]-=1; need(pending[ch]>=0,'B without outstanding AW')
                for beat,value in enumerate(job['outputs']): golden[ch][job['address']+beat*64]=value
            else:
                need(int(fields[5])==job['status'] and tag not in completed,'Completion status/uniqueness')
                if job['status']==0: need(job.get('b',now)<now,'Completion before MIG B')
                else: need(not job['outputs'] and 'aw' not in job and 'b' not in job,'Range reject reached MIG')
                completed.add(tag); queue.popleft()
        elif kind=='SWEEP':
            values = {k:int(v) for k,v in (x.split('=') for x in fields[3:])}; ch=values['port']
            need(not sweep[ch] and reading[ch] is None and not any(queues[ch].values()),'Readback before local drain')
            addresses = list(golden[ch]); first,count = values['first'],values['count']
            need(first>=0 and count>0 and first+count<=len(addresses),'Sweep bounds')
            expected_sweeps = {'sparse_ranks':(0,6),'bursts':(0,77),'independent_progress':(81,4),
                'reserve_pool':(85,48),'completion_backpressure':(0,142+8*ch),'range_reject':(0,1)}
            need(case in expected_sweeps and (first,count)==expected_sweeps[case] and
                 (case!='independent_progress' or ch==1),'Sweep coverage vector')
            need(sweeps[(case,ch)]==0,'Duplicate sweep'); sweeps[(case,ch)]+=1
            sweep[ch].extend(addresses[first:first+count])
        elif kind=='AR':
            ch=int(fields[3]); address=int(fields[4],16)
            need(reading[ch] is None and sweep[ch] and address==sweep[ch].popleft(),'Read address/order')
            reading[ch]=(address,now)
        elif kind=='R':
            ch=int(fields[3]); address=int(fields[4],16); value=int(fields[5],16)
            need(reading[ch] is not None and address==reading[ch][0] and now>reading[ch][1]
                 and value==golden[ch][address],'Readback payload/destination')
            reading[ch]=None; reads[ch]+=1
        elif kind=='DRAIN':
            need(fill is None and all(not any(q.values()) for q in queues) and pending==[0,0] and banks==0,'Premature global drain')
            drains+=1
        elif kind=='SATURATED':
            need(case=='credits' and not saturated and pending==[4,4] and len(jobs)==24,'Two-controller saturation')
            saturated=True
        elif kind=='PROGRESS':
            need(case=='independent_progress' and not progress and 26 in completed and 25 not in completed and
                 'aw' in jobs[25] and not jobs[25]['outputs'],'Independent progress while other W stalled')
            progress=True
        elif kind=='RESERVE':
            need(case=='reserve_pool' and not reserve and banks==255 and pending==[4,4],'Shared reserve pool saturation')
            reserve=True
        elif kind=='STOP':
            reason=fields[3]; stops.append(reason)
            if reason=='b_held':need(pending==[4,4] and saturated,'Stop with B pending')
            elif reason=='fill':need(fill in (51,52) and len(jobs[fill]['inputs'])==2,'Stop during frame')
            elif reason=='aw_exposed':need('aw' in jobs[53] and not jobs[53]['outputs'],'Stop after exposed AW')
            else:raise ValueError('Unknown stop')
        else: raise ValueError('Unknown trace event '+kind)
    need(cases==['cold_barrier','sparse_ranks','bursts','credits','independent_progress','reserve_pool',
                 'stop_fill','stop_after_aw','completion_backpressure','range_reject'],'Case coverage')
    need(len(jobs)==len(completed)==56 and writes==[143,151] and reads==[274,286], 'Transaction totals')
    need([len(x) for x in golden]==[142,150] and peak==[4,4] and max_banks==8,'Memory/pool totals')
    need(saturated and progress and reserve and stops==['b_held','fill','fill','aw_exposed'],'Stall/stop coverage')
    need(fill is None and not any(reading) and not any(sweep) and all(not any(q.values()) for q in queues),'Final drain')
    need(clocks=={0:998,1:1729},'Separate UI clock phases')
    totals=dict(commands=56,completions=56,writes0=143,writes1=151,reads0=274,reads1=286,
                locations0=142,locations1=150,peak0=4,peak1=4,banks=8)
    need(summary and all(summary.get(k)==v for k,v in totals.items()) and summary['one_ready']>500 and
         summary['blocked']>summary['one_ready'] and summary['ps']>=last_time,'Final summary')
    ranks = [Counter(),Counter()]
    for ch,rank,count in re.findall(r'^DUAL_ACT port=(\d+) ps=\d+ rank=(\d+) count=(\d+)$',text,re.M):
        ch,rank,count=map(int,(ch,rank,count));need(ch in (0,1) and rank in range(4),'Rank identity')
        ranks[ch][rank]+=1;need(ranks[ch][rank]==count,'Rank counter')
    need(all(set(r)==set(range(4)) for r in ranks),'Four ranks per controller')
    return dict(status='DUAL_MIG_BYTE_REPLAY_PASS',summary=summary,write_bytes=sum(writes)*64,
                read_bytes=sum(reads)*64,source_bytes=sum(v[2] for v in expected)*64,
                ranks=[dict(r) for r in ranks],drains=drains,clocks=clocks)


def periodic(text):
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
        need(counts['user_cas']==[274,286][ch] and witness['physical_writes']==[143,151][ch],'Per-controller physical transfers')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='DUAL_PERIODIC_WITNESS_PASS',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)


def negative_controls(text):
    out=[]
    def reject(name,mutated,checker):
        need(mutated!=text,'Effective mutation '+name)
        try:checker(mutated)
        except ValueError as error:out.append(dict(mutation=name,rejection=str(error)))
        else:raise ValueError('Corruption escaped: '+name)
    lines=text.splitlines()
    for kind,prefix in [('write_payload','DUAL W '),('read_payload','DUAL R '),('address_alias','DUAL AW '),
                        ('missing_b','DUAL B '),('duplicate_cpl','DUAL CPL '),('wrong_destination','DUAL CPL '),
                        ('missing_reserve','DUAL RESERVE '),('missing_progress','DUAL PROGRESS '),
                        ('off_ui_edge','DUAL AR ')]:
        changed=list(lines);index=next(i for i,l in enumerate(lines) if l.startswith(prefix) and
                                     (kind!='address_alias' or int(l.split()[5],16)>0xffffffff))
        fields=changed[index].split()
        if kind in ('write_payload','read_payload'):fields[-1]=format(int(fields[-1],16)^1,'0128x');changed[index]=' '.join(fields)
        elif kind=='address_alias':fields[5]=format(int(fields[5],16)&0xffffffff,'x');changed[index]=' '.join(fields)
        elif kind=='duplicate_cpl':changed.insert(index,changed[index])
        elif kind=='wrong_destination':fields[3]=str(1-int(fields[3]));changed[index]=' '.join(fields)
        elif kind=='off_ui_edge':fields[2]=str(int(fields[2])+1);changed[index]=' '.join(fields)
        else:changed.pop(index)
        reject(kind,'\n'.join(changed),replay)
    for kind,token in [('missing_periodic_cas',' NTRPU_DDR_READ_CAS '),('missing_return',' NTRPU_DDR_READ_RETURN '),
                       ('missing_drain',' NTRPU_DDR_PERIODIC_DRAIN '),('missing_config',' NTRPU_DDR_PERIODIC_CONFIG ')]:
        line=next(l for l in lines if l.startswith('DUAL_PER ') and token in l)
        reject(kind,text.replace(line+'\n','',1),periodic)
    line=next(l for l in lines if ' NTRPU_DDR_READ_RETURN ' in l and 'injected=1' in l)
    reject('periodic_leaks_to_user',text.replace(line,line.replace('native=0','native=1'),1),periodic)
    reject('unknown_model_warning',text+'\nModel:WARNING: unexpected diagnostic\n',periodic)
    reject('unowned_model_warning',text+'\nModel:WARNING: Reading unwritten address: C:0 BG:0 B:0 R:0 C:0 dm:0 @1\n',periodic)
    if any(l.startswith('Model:WARNING:') for l in lines):
        reject('suppressed_model_warnings','\n'.join(l for l in lines if not l.startswith('Model:WARNING:')),periodic)
    return out


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review(run):
    output=run/'output';record=json.loads((output/'result.json').read_text())
    need(record['status']=='EXPORTED_PENDING_REVIEW' and not record['timed_out'] and record['example_unchanged'],'Completed immutable run')
    need(record['sources']==record['sources_after'],'Source snapshot stable')
    for name,value in record['sources'].items():need(digest(run/'inputs'/name)==value,'Source hash '+name)
    for name,value in json.loads((output/'FILES.json').read_text()).items():need(digest(output/name)==value,'Output hash '+name)
    unit=(output/'suite/startup.log').read_text()
    cases=re.findall(r'^DUAL_START_CASE_PASS case=(\d+) entries=(\d+) locked=(\d+)$',unit,re.M)
    need(sorted(tuple(map(int,c)) for c in cases)==[(g,1,int(1<=g<=8 or g==12)) for g in range(13)] and
         'DUAL_START_SUITE_PASS cases=13' in unit,'Startup coverage')
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',unit,re.M),'Startup diagnostics')
    console=(output/'console.log').read_text()
    need('NTRPU_DUAL_MIG_EXPORTED' in console and not re.search(r'^CRITICAL WARNING:',console,re.M),'Tool export/diagnostics')
    trace=(output/'suite/memory.log').read_text()
    result=replay(trace);result['periodic']=periodic(trace);result['negative_controls']=negative_controls(trace)
    result.update(status='DUAL_MIG_COLD_FUNCTIONAL_PASS_NOT_BOARD_ADOPTED',startup_cases=13,
        seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],checker_sha256=digest(Path(__file__)),
        warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
        limitations=['Two instances of generated MIG in BFM mode, same nominal UI rate with separate phase',
                     'One common cold epoch, all-ready policy; no degraded mode or warm recovery',
                     'Generic geometry, sparse locations, not physical DDR/HBM/PCIe qualification',
                     'No synthesis/place/route or physical CDC audit for the new supervisor/integration'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=review(args.run)
    with (args.run/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
