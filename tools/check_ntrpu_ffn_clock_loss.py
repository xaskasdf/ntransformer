"""Review full FFN numerical results and automatic clock-loss recovery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from check_ntrpu_ffn_irq_async import review as ffn_review, mutations as ffn_mutations
from check_ntrpu_ffn_host import records


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def review(text, seed, host_period_ps, *, recovery_each=False, release_retry=False):
    ffn = ffn_review(text, seed, host_period_ps, None, delayed_reset=True, recovery_each=recovery_each)
    errors = list(ffn['errors'])
    fault_ids = [16,18,20,22] if recovery_each else [16,17,18,19]
    case_count = 24 if recovery_each else 21

    def need(ok, why):
        if not ok:
            raise ValueError(why)

    try:
        need(host_period_ps in (4000, 6200), 'host clock profile')
        need(not release_retry or recovery_each,'RELEASE retry schedule')
        fault_rows=[]
        fault_pass=0
        for line in text.splitlines():
            if not line.startswith('NTRPU_CLOCK_FFN_'):continue
            kind=line.split()[0].removeprefix('NTRPU_CLOCK_FFN_')
            raw=fields(line)
            need(all(re.fullmatch(r'\d+',v) for v in raw.values()),'unknown clock-loss field')
            row={k:int(v) for k,v in raw.items()}
            if kind=='PASS':
                need(row==dict(losses=4,device=2,host=2),'clock-loss terminal coverage')
                fault_pass+=1;continue
            need(not fault_pass,'clock-loss trace after PASS')
            need(set(row)=={'id','target','ps','dt','ht','den','hen','rst','hrst','da','ha','run','ext'},'clock-loss record fields')
            fault_rows.append((kind,row))
        need(fault_pass==1 and len(fault_rows)==16,'four complete clock-loss traces')
        pauses={'dt':[],'ht':[]}
        losses=[]
        for target in range(4):
            group=fault_rows[target*4:target*4+4]
            need([k for k,r in group]==['STOP','BLOCKED','RESUME','RECOVERED'],'clock-loss stage order')
            stop,blocked,resume,recovered=[r for k,r in group]
            need(all(r['id']==fault_ids[target] and r['target']==target and r['ext']==0 for k,r in group),'automatic fault identity')
            need(stop['ps']<blocked['ps']<resume['ps']<recovered['ps'],'clock-loss time order')
            need(0<blocked['ps']-stop['ps']<=1000000,'clock-loss detection deadline')
            need(resume['ps']-blocked['ps']>=1000000,'stopped clock hold interval')
            need(recovered['ps']-resume['ps']<4000000,'clock-loss recovery deadline')
            key,period,phase=('dt',5000,0) if (target+int(host_period_ps==6200))%2==0 else ('ht',host_period_ps,700)
            for r in [stop,blocked]:
                need((r['den'],r['hen'])==((0,1) if key=='dt' else (1,0)),'selected clock disabled')
            for r in [resume,recovered]:need((r['den'],r['hen'])==(1,1),'both clocks resumed')
            for r in [stop,resume]:need((r['ps']-1-phase)%period==0,'clock gate changes in low phase')
            for r in [stop,recovered]:need(all(r[k]==1 for k in ['rst','hrst','da','ha','run']),'active clock-loss boundary')
            for r in [blocked,resume]:need(all(r[k]==0 for k in ['rst','hrst','da','ha','run']),'fault boundary closed')
            need(stop[key]==blocked[key]==resume[key],'no selected-clock edges during loss')
            pauses[key].append((stop['ps'],resume['ps']))
            losses.append(dict(id=fault_ids[target],domain=key,detection_observed_ps=blocked['ps']-stop['ps'],recovery_observed_ps=recovered['ps']-resume['ps'],stopped_ps=resume['ps']-stop['ps']))
        def edge_count(t,key,period,phase):
            full=lambda x:0 if x<phase else 1+(x-phase)//period
            return full(t)-sum(full(min(t,end))-full(start) for start,end in pauses.get(key,[]) if start<=t)
        for kind,row in fault_rows:
            for key,period,phase in [('dt',5000,2500),('ht',host_period_ps,700+host_period_ps//2)]:
                need(row[key]==edge_count(row['ps'],key,period,phase),'fault observed edge count '+key)
        for key,names in [('dt',['JOB','DONE','EVENT','SERVICE','RETIRE']),('ht',['EDGE','ACK'])]:
            for name in names:
                for row in records(text,'NTRPU_FFN_IRQ_'+name):
                    need(not any(a<row['time_ps']<z for a,z in pauses[key]),'handshake while clock stopped '+name)
        lines=text.splitlines()
        for target in range(4):
            ident=fault_ids[target]
            start=next(i for i,l in enumerate(lines) if l.startswith(f'NTRPU_CLOCK_FFN_STOP id={ident} '))
            assertion=next(i for i in range(start+1,len(lines)) if lines[i].startswith('NTRPU_FFN_IRQ_RESET_ASSERT '))
            written=[i for i,l in enumerate(lines) if l.startswith(f'NTRPU_FFN_MEMORY_W id={ident} ')]
            need(written and max(written)<assertion,'no physical write after reset assertion')
            target_writes=[i for i in written if f'buffer={target+4} ' in lines[i]]
            need(target_writes and min(target_writes)<start,'clock loss follows the target phase first write')
        recovery_rows=records(text,'NTRPU_FFN_RECOVERY')
        need(len(recovery_rows)==(4 if recovery_each else 0),'post-fault FFN coverage')
        if recovery_each:
            jobs=records(text,'NTRPU_FFN_IRQ_JOB')
            retires=records(text,'NTRPU_FFN_IRQ_RETIRE')
            for phase,row in enumerate(recovery_rows):
                ident=fault_ids[phase]+1
                need(set(row)=={'fault','id','target','ps'} and
                     (row['fault'],row['id'],row['target'])==(ident-1,ident,phase),'post-fault FFN identity')
                job=next(r for r in jobs if r['id']==ident)
                retire=next(r for r in retires if r['id']==ident)
                recovered=fault_rows[phase*4+3][1]['ps']
                need(recovered<job['time_ps']<retire['time_ps']<=row['ps'],'post-fault FFN completes after recovery')
                if phase<3:need(row['ps']<fault_rows[(phase+1)*4][1]['ps'],'post-fault FFN before next loss')
        release_rows=[]
        for line in lines:
            if not line.startswith('NTRPU_FFN_RELEASE_'):continue
            kind=line.split()[0].removeprefix('NTRPU_FFN_RELEASE_')
            raw=fields(line)
            need(all(re.fullmatch(r'\d+',v) for v in raw.values()),'unknown RELEASE field')
            row={k:int(v) for k,v in raw.items()}
            need(set(row)=={'attempt','ps','dt','ht','ext','rst','hrst','da','ha','run','mr'},'RELEASE fields')
            need(all(row[k] in (0,1) for k in ['ext','rst','hrst','da','ha','run','mr']),'RELEASE binary pin')
            need(not any(row[k] for k in ['da','ha','run','mr']),'RELEASE admission before memory readiness')
            for key,period,phase in [('dt',5000,2500),('ht',host_period_ps,700+host_period_ps//2)]:
                need(row[key]==edge_count(row['ps'],key,period,phase),'RELEASE edge count '+key)
            release_rows.append((kind,row))
        need(len(release_rows)==(10 if release_retry else 0),'RELEASE retry coverage')
        if release_retry:
            need([k for k,r in release_rows]==['BEFORE','ASSERT','DEASSERT']*3+['READY'],'RELEASE retry stage order')
            need(all(a[1]['ps']<z[1]['ps'] for a,z in zip(release_rows,release_rows[1:])),'RELEASE retry time order')
            for attempt in range(3):
                before,asserted,deasserted=[r for k,r in release_rows[attempt*3:attempt*3+3]]
                need(all(r['attempt']==attempt+1 for r in [before,asserted,deasserted]),'RELEASE attempt identity')
                need((before['ext'],before['rst'],before['hrst'])==(0,1,1),'retry begins after both resets released')
                need((asserted['ext'],asserted['rst'],asserted['hrst'])==(1,0,0),'retry asserts both resets')
                need((deasserted['ext'],deasserted['rst'],deasserted['hrst'])==(0,0,0),'retry release remains closed')
                need(asserted['ps']-before['ps']==1 and deasserted['ps']-asserted['ps']==78700,'external retry pulse duration')
            ready=release_rows[-1][1]
            need((ready['attempt'],ready['ext'],ready['rst'],ready['hrst'])==(4,0,1,1),'fourth release before memory readiness')
            first=records(text,'NTRPU_FFN_IRQ_BEGIN')[0]
            need(ready['ps']<first['time_ps'],'first FFN after all RELEASE retries')
        releases = Counter()
        reset_assertions = []
        release_cases = []
        asserted = {'rst': 0, 'hrst': 0}
        counts = Counter()
        last = None
        block = None
        passed = False
        supervision = []
        mmio = {'B': [], 'R': []}
        binary = {'ext','rst','hrst','da','ha','run','req','can','held','ready','mh','mr','hh','hr','allow'}
        for line in text.splitlines():
            if not line.startswith('NTRPU_SFFN_'):
                continue
            need(not passed, 'supervision trace after terminal PASS')
            kind = line.split()[0].removeprefix('NTRPU_SFFN_')
            raw = fields(line)
            if kind in mmio:
                row = {k: int(v,16 if k in {'addr','data'} else 10) for k,v in raw.items()}
                need(row['total'] == len(mmio[kind])+1, 'MMIO handshake counter')
                need((row['ps']-700) % host_period_ps == host_period_ps//2, 'MMIO handshake edge')
                need(last and last['ha'] and last['run'] and last['rst'] and last['hrst'], 'MMIO before admission')
                need(not any(a<row['ps']<z for a,z in pauses['ht']),'MMIO while host clock stopped')
                mmio[kind].append(row)
                continue
            need(all(re.fullmatch(r'\d+', v) for v in raw.values()), 'unknown supervision pin')
            row = {k: int(v) for k, v in raw.items()}
            if kind == 'PASS':
                need(row == dict(cases=case_count, resets=5, memory_peer_blocked=1), 'supervised terminal coverage')
                passed = True
                continue
            if kind == 'PEER_BLOCK':
                need(block is None and last is not None, 'duplicate peer-block observation')
                need(last['id'] == 0 and not any(last[k] for k in ['ext','rst','hrst','run','can','mh','allow']), 'unacknowledged memory did not block reset release')
                need(row['ps'] >= 1600000, 'memory-block observation too early')
                block = row
                continue
            need(kind in {'M','OUT'}, 'unexpected supervision record')
            need(set(row) == binary | {'id','ps','mt','dt','ht'}, 'supervision fields')
            need(all(row[k] in (0, 1) for k in binary), 'nonbinary supervision pin')
            need(row['id'] in {0,*fault_ids}, 'unexpected reset/recovery case')
            t = row['ps']
            for key, period, phase in [('dt',5000,2500),('ht',host_period_ps,700+host_period_ps//2),('mt',10000,5000)]:
                expected = edge_count(t,key,period,phase)
                need(row[key] == expected, 'actual clock edge counter '+key)
            if kind == 'M':
                need(t % 10000 == 5001 and not row['run'], 'management observation clock')
            if last:
                need(t >= last['ps'], 'supervision time order')
                for reset, admit, counter, period, phase in [('rst','da','dt',5000,2500),('hrst','ha','ht',host_period_ps,700+host_period_ps//2)]:
                    if last[reset] and not row[reset]:
                        asserted[reset] = row[counter]
                        if reset == 'rst':
                            reset_assertions.append(row['id'])
                    if not last[reset] and row[reset]:
                        need(row[counter]-asserted[reset] >= 64, 'insufficient actual reset edges')
                        need((t-1-phase) % period == 0, 'asynchronous reset release')
                        need(not row['ext'] and not row['req'], 'release before supervisor permission')
                        releases[reset] += 1
                        if reset == 'rst':
                            release_cases.append(row['id'])
                    if not last[admit] and row[admit]:
                        need((t-1-phase) % period == 0, 'asynchronous admission release')
                        releases[admit] += 1
            if row['da'] or row['ha']:
                need(row['run'] and row['rst'] and row['hrst'] and row['ready'] and row['mr'] and row['hr'] and not row['req'] and not row['ext'], 'unsafe admission')
            if row['can']:
                need(row['held'] and row['mh'] and row['hh'] and row['req'] and not row['ext'], 'rearm without peer acknowledgement')
            if not row['allow']:
                need(not row['mh'] and not row['can'] and not row['rst'] and not row['hrst'], 'memory acknowledgement bypassed')
            last = row
            counts[kind] += 1
            supervision.append(row)
        need(passed and block is not None, 'missing peer block or terminal PASS')
        need(releases == {'rst':5+3*release_retry,'hrst':5+3*release_retry,'da':5,'ha':5}, 'release coverage')
        need(reset_assertions == ([0]*3 if release_retry else [])+fault_ids, 'in-flight reset coverage')
        need(release_cases == [0]*(4 if release_retry else 1)+fault_ids, 'recovery order')
        need(counts['M'] > 300 and counts['OUT'] >= 25, 'insufficient supervision observations')
        writes = records(text, 'NTRPU_FFN_HOST_WRITE')
        reads = records(text, 'NTRPU_FFN_IRQ_READ')
        need(len(mmio['B']) == len(writes) and len(mmio['R']) == len(reads), 'MMIO handshake coverage')
        for bus, logged in zip(mmio['B'], writes):
            need({k:bus[k] for k in ['id','addr','data','resp','order']} == logged, 'observed B identity')
        for bus, logged in zip(mmio['R'], reads):
            need(bus['resp'] == 0 and {k:bus[k] for k in ['id','addr','data']} | {'time_ps':bus['ps']} == logged, 'observed R identity')
        return dict(status='FAIL' if errors else 'PASS', errors=errors, ffn=ffn,
                    supervision=dict(releases=dict(releases),observations=dict(counts),peer_block=block,
                                     reset_cases=reset_assertions,release_cases=release_cases,mmio_handshakes={k:len(v) for k,v in mmio.items()}),
                    clock_losses=losses,recovery_each=recovery_each,post_fault_ffns=recovery_rows,release_retry=release_retry,release_retries=release_rows,scope='Actual full FFN and supervisor with four automatic clock-loss recoveries and one initial reset per seed. External memory and host peers are models; host AXI4 DMA idle. Recovery schedule is explicit: recovery_each requires a complete synthetic FFN after every loss; default only after the last. release_retry=true additionally qualifies three initial RELEASE resets with memory readiness withheld. No XDMA/DDR/board qualification.')
    except (ValueError, KeyError, TypeError, StopIteration) as exc:
        errors.append(str(exc))
        return dict(status='FAIL',errors=errors,ffn=ffn)


def mutations(text, *, recovery_each=False, release_retry=False):
    result = ffn_mutations(text, False)
    lines = text.splitlines()
    def change(name, predicate, key=None, value=None, remove=False):
        changed = lines.copy()
        i = next(i for i,l in enumerate(lines) if predicate(l,fields(l)))
        if remove:
            changed.pop(i)
        else:
            changed[i] = re.sub(r'\b'+key+r'=\S+',key+'='+value,changed[i],count=1)
        result[name] = '\n'.join(changed)+'\n'
    change('supervisor_missing_pass',lambda l,r:l.startswith('NTRPU_SFFN_PASS'),remove=True)
    change('supervisor_missing_peer_block',lambda l,r:l.startswith('NTRPU_SFFN_PEER_BLOCK'),remove=True)
    change('supervisor_unsafe_admission',lambda l,r:l.startswith('NTRPU_SFFN_M') and r['run']=='0','da','1')
    change('supervisor_counter',lambda l,r:l.startswith('NTRPU_SFFN_OUT') and r['rst']=='1','dt','0')
    change('supervisor_unknown',lambda l,r:l.startswith('NTRPU_SFFN_M'),'hrst','x')
    change('supervisor_false_ack',lambda l,r:l.startswith('NTRPU_SFFN_M') and r['allow']=='0','can','1')
    change('supervisor_missing_recovery',lambda l,r:l.startswith('NTRPU_SFFN_OUT') and r['id']==('22' if recovery_each else '19') and r['ha']=='1',remove=True)
    change('missing_b_handshake',lambda l,r:l.startswith('NTRPU_SFFN_B '),remove=True)
    change('wrong_b_count',lambda l,r:l.startswith('NTRPU_SFFN_B '),'total','0')
    change('wrong_r_edge',lambda l,r:l.startswith('NTRPU_SFFN_R '),'ps','1')
    change('missing_loss_pass',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_PASS'),remove=True)
    change('missing_clock_stop',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_STOP'),remove=True)
    stopped=next(fields(l) for l in lines if l.startswith('NTRPU_CLOCK_FFN_STOP'))
    change('clock_not_stopped',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_STOP'),'den' if stopped['den']=='0' else 'hen','1')
    change('external_reset_assistance',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_BLOCKED'),'ext','1')
    change('clock_edges_during_loss',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_RESUME'),'dt','0')
    change('fault_recovery_missing',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_RECOVERED'),remove=True)
    change('fault_admission_open',lambda l,r:l.startswith('NTRPU_CLOCK_FFN_BLOCKED'),'da','1')
    if recovery_each:
        change('missing_post_fault_ffn',lambda l,r:l.startswith('NTRPU_FFN_RECOVERY'),remove=True)
        change('wrong_post_fault_owner',lambda l,r:l.startswith('NTRPU_FFN_RECOVERY'),'fault','0')
        change('early_post_fault_done',lambda l,r:l.startswith('NTRPU_FFN_RECOVERY'),'ps','1')
        change('post_fault_wrong_bytes',lambda l,r:l.startswith('NTRPU_FFN_MEMORY_W id=17 ') and r['buffer']=='7','data','0')
    if release_retry:
        change('missing_release_pulse',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_ASSERT'),remove=True)
        change('release_admitted',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_BEFORE'),'ha','1')
        change('release_reset_not_asserted',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_ASSERT'),'hrst','1')
        change('release_wrong_attempt',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_BEFORE'),'attempt','0')
        change('release_wrong_pulse_duration',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_DEASSERT'),'ps','1')
        change('release_ready_missing',lambda l,r:l.startswith('NTRPU_FFN_RELEASE_READY'),remove=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--host-period-ps',type=int,required=True)
    parser.add_argument('--recovery-each',action='store_true')
    parser.add_argument('--release-retry',action='store_true')
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'),args.seed,args.host_period_ps,recovery_each=args.recovery_each,release_retry=args.release_retry)
    print(json.dumps(result,indent=2))
    raise SystemExit(result['status'] != 'PASS')
