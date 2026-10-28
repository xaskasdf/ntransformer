"""Reconstruct descriptor obligations from public events; no DUT state access."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows as raw_vcd_rows

# Faults, H2C published/completed, C2H published/completed, active, closed, accounted.
CASES = {
    1: (0,2,2,1,1,1,1,1), 2: (0,0,0,0,0,1,1,1),
    3: (8,0,0,0,0,1,0,0), 4: (2,1,0,0,0,1,0,0),
    5: (16,1,0,0,0,1,0,0), 6: (2,0,0,0,0,1,0,0),
    7: (256,0,0,0,0,1,1,0), 8: (32,2,2,0,0,1,1,0),
    9: (1,0,0,0,0,1,1,0), 10: (128,1,0,0,0,1,0,0),
    11: (2,0,0,0,0,1,0,0), 12: (2,0,0,0,0,0,0,0),
    13: (64,15,15,0,0,1,0,0), 14: (16,0,0,0,0,1,0,0),
    15: (4,0,0,0,0,0,0,0), 16: (256,0,0,0,0,1,1,0),
}


def settled_rows(text, netlist):
    """Fold only the verified UNISIM #100 output delay into its clock event.

    All TB drives and clocks must remain on the original 2ns event grid.
    This is cycle equivalence, not equality of raw propagation times.
    """
    outputs = {'arm_ready','active','closed','generation','tag','started','faults',
               'published','completed','outstanding','accounted'}
    group_time, group_before, group_after = None, None, None
    for t, before, after in raw_vcd_rows(text):
        if all(v is None for v in after.values()):
            continue
        offset = t % 2000
        require(offset in ((0, 100) if netlist else (0,)), 'event outside cell settling interval')
        if offset:
            require(group_time == t - offset, 'cell output requires preceding event')
            changed = {n.rsplit('/', 1)[-1] for n in after if before[n] != after[n]}
            require(changed <= outputs, 'only DUT outputs may have cell delay')
        if t >= 100 or not netlist:
            require(all(v is not None for v in after.values()), 'unknown after cell initialization')
        base = t - offset
        if group_time is not None and base != group_time:
            yield group_time, group_before, group_after
            group_time = None
        if group_time is None:
            group_time, group_before = base, before
        group_after = after
    if group_time is not None:
        yield group_time, group_before, group_after


def review(sim, vcd, bits, *, netlist):
    require(netlist in (0,1), 'declared netlist mode')
    require(markers(sim,'LEDGER_REPLAY') == [dict(bits=str(bits),netlist=str(netlist))], 'replay variant marker')
    require(bits in (32,4), 'supported counter variant')
    phases = [p for p in CASES if bits==4 or p!=13]
    require(markers(sim,'LEDGER_PASS') == [dict(cases=str(len(phases)),bits=str(bits),
            native_xdma='0',global_drain='0',reset_permission='0')], 'unit scope and variant')
    require(not re.search(r'^\s*(?:ERROR:|CRITICAL WARNING:|Fatal:|Failure:)',sim,re.M), 'simulation errors')
    finish = re.findall(r'\$finish called at time : (\d+) ns',sim)
    require(len(finish)==1 and int(finish[0])==(888 if bits==4 else 784), 'bounded finish')
    cases = markers(sim,'LEDGER_CASE')
    require([int(c['phase']) for c in cases]==phases,'all scenarios in order')
    checkpoints = {int(c['t']):c for c in cases}
    observations, events = [], {'ARM':[],'PUBLISH':[],'DONE':[]}
    m = None
    last_clock = None
    samples, lifetimes, retained_wait = 0, 0, []
    maximum = (1<<bits)-1
    prefix='/tb_ntrpu_xdma_descriptor_ledger/'

    def fresh(g,t):
        return g>0 and t>0 and (not m['ever'] or (g,t)>m['id'])

    def levels(a):
        runs=((a['h2c_status']>>6)&1, (a['c2h_status']>>6)&1)
        dones=((a['h2c_status']>>3)&1, (a['c2h_status']>>3)&1)
        idle=not any(runs) and not (a['h2c_status']&1 or a['c2h_status']&1)
        host=a['host_closed'] and (a['host_generation'],a['host_tag'])==m['id']
        accounted=bool(a['cold_rst_n'] and a['core_ready'] and m['active'] and m['closed'] and
                       not m['fault'] and a['close_request'] and host and not a['publish_valid'] and
                       not any(dones) and idle and m['pub']==m['done'])
        ready=bool(a['cold_rst_n'] and a['core_ready'] and not m['active'] and not m['fault'] and
                   idle and fresh(a['arm_generation'],a['arm_tag']))
        return runs,dones,idle,host,accounted,ready

    for t,b,a in settled_rows(vcd,netlist):
        b={n[len(prefix):]:v for n,v in b.items() if n.startswith(prefix)}
        a={n[len(prefix):]:v for n,v in a.items() if n.startswith(prefix)}
        if not a or all(v is None for v in a.values()):continue
        require(all(v is not None for v in a.values()),'known public signals')
        if m is None:
            m=dict(active=False,closed=False,ever=False,id=(0,0),started=[False,False],
                   fault=0,pub=[0,0],done=[0,0],low=False,previous_run=(0,0))
        edge=b.get('clk')==0 and a['clk']==1
        if edge:
            if last_clock is not None:require(t-last_clock==4000,'continuous 250MHz ledger clock')
            last_clock=t
            if not b['cold_rst_n']:
                m=dict(active=False,closed=False,ever=False,id=(0,0),started=[False,False],
                       fault=0,pub=[0,0],done=[0,0],low=False,previous_run=(0,0))
            else:
                old={k:(v.copy() if isinstance(v,list) else v) for k,v in m.items()}
                runs,dones,idle,host,can_release,can_arm=levels(b)
                rises=[runs[c] and not old['previous_run'][c] for c in range(2)]
                accepted=[bool(old['active'] and b['publish_valid'] and b['publish_channel']==c and
                          (b['publish_generation'],b['publish_tag'])==old['id'] and
                          b['publish_index']==old['pub'][c] and old['pub'][c]<maximum) for c in range(2)]
                if b['arm_valid']:
                    if old['active'] or not fresh(b['arm_generation'],b['arm_tag']):m['fault']|=2
                    elif can_arm:
                        m.update(active=True,closed=False,ever=True,id=(b['arm_generation'],b['arm_tag']),
                                 started=[False,False],pub=[0,0],done=[0,0],low=not b['host_closed'])
                        events['ARM'].append(dict(t=t,phase=b['phase'],generation=b['arm_generation'],tag=b['arm_tag']))
                if old['active']:
                    if not b['core_ready']:m['fault']|=1
                    if not b['host_closed']:m['low']=True
                    if b['close_request'] and b['host_closed'] and old['low'] and not old['closed']:
                        if host:m['closed']=True
                        else:m['fault']|=2
                    if old['closed'] and (not b['close_request'] or not host):m['fault']|=256
                if b['publish_valid']:
                    events['PUBLISH'].append(dict(t=t,phase=b['phase'],channel=b['publish_channel'],
                        index=b['publish_index'],generation=b['publish_generation'],tag=b['publish_tag']))
                    c=b['publish_channel']
                    if not old['active']:m['fault']|=4
                    else:
                        if (b['publish_generation'],b['publish_tag'])!=old['id'] or b['publish_index']!=old['pub'][c]:m['fault']|=2
                        if old['closed']:m['fault']|=32
                        if old['pub'][c]==maximum:m['fault']|=64
                if any(dones):events['DONE'].append(dict(t=t,phase=b['phase'],mask=dones[0]+2*dones[1]))
                for c in range(2):
                    if accepted[c]:m['pub'][c]+=1
                    if rises[c]:
                        if not old['active']:m['fault']|=4
                        elif old['started'][c] or old['closed'] or not (old['pub'][c] or accepted[c]):m['fault']|=16
                        else:m['started'][c]=True
                    if dones[c]:
                        if not old['active'] or not old['started'][c] or rises[c] or not (old['done'][c]<old['pub'][c] or accepted[c]):m['fault']|=8
                        elif old['done'][c]==maximum:m['fault']|=64
                        else:m['done'][c]+=1
                if b['release_valid']:
                    if can_release:m.update(active=False,closed=False,low=False)
                    else:m['fault']|=128
                m['previous_run']=runs
                require((m['fault']|old['fault'])==m['fault'],'fault retention')
        if b.get('cold_rst_n')==0 and a['cold_rst_n']==1:lifetimes+=1
        _,_,_,_,accounted,ready=levels(a)
        require(a['arm_ready']==ready and a['accounted']==accounted,'admission and immediate revocation predicate')
        require((a['active'],a['closed'],a['generation'],a['tag'])==(m['active'],m['closed'],*m['id']),'retained session identity/lifecycle')
        require(a['faults']==m['fault'],'exact retained fault mask')
        require(a['started']==int(m['started'][0])+2*int(m['started'][1]),'Run lifetime association')
        for c in range(2):
            require(0<=m['done'][c]<=m['pub'][c]<=maximum,'bounded obligations without wrap')
            for n,expected in [('published',m['pub'][c]),('completed',m['done'][c]),('outstanding',m['pub'][c]-m['done'][c])]:
                require((a[n]>>(32*c))&0xffffffff==expected,'public '+n+' matches actual descriptor events')
        if a['phase']==1 and a['outstanding'] and not (a['h2c_status']&0x41) and not a['accounted']:
            retained_wait.append(t)
        if t in checkpoints:
            row=checkpoints[t];phase=int(row['phase'])
            actual=(a['faults'],a['published']&0xffffffff,a['completed']&0xffffffff,
                    a['published']>>32,a['completed']>>32,a['active'],a['closed'],a['accounted'])
            require(actual==CASES[phase],'independent scenario outcome '+str(phase))
            for field in ('faults','published','completed','outstanding','active','closed','accounted'):
                require(int(row[field],16 if field in ('faults','published','completed','outstanding') else 2)==a[field],'case log/VCD correspondence')
            observations.append(dict(phase=phase,t=t,faults=a['faults'],published=a['published'],completed=a['completed'],accounted=a['accounted']))
        samples+=1
    require(len(observations)==len(phases),'all scenario checkpoints captured')
    require(len(retained_wait)>=2,'Run/Busy zero must retain unmatched descriptor obligations')
    require(lifetimes==len(phases)-1,'independent cold configurations, plus fresh session without reset')
    for name,expected in events.items():
        actual=[]
        for row in markers(sim,'LEDGER_'+name):
            actual.append({k:int(v,10 if k in ('t','phase') else 2 if k in ('channel','mask') else 16) for k,v in row.items()})
        require(actual==expected,'event log/VCD correspondence '+name)
    return dict(status='PASS_DESCRIPTOR_LEDGER_REPLAY_PUBLIC_CONTRACT',cell_settling_ps=100 if netlist else 0,raw_time_equivalence=False,bits=bits,scenarios=observations,
                sampled_timestamps=samples,configuration_lifetimes=lifetimes,event_counts={k:len(v) for k,v in events.items()},
                finish_ns=int(finish[0]),native_xdma_tested=False,global_drain_qualified=False,operational_reset_qualified=False,
                scope='Public descriptor accounting and closure interface only. Production32 checked without wrap; reduced4 exercises saturation. Driver publication, register stop controller, protected PCIe inventory, CDC/global join and reset remain pending.')
