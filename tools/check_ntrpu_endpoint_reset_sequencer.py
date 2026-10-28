"""Check isolated sequencer phase/ownership evidence and the full timer interval."""
import re


def need(condition, message):
    if not condition:
        raise ValueError(message)


def review(text):
    need(not re.search(r'^\s*(?:Fatal:|ERROR:|FAIL )',text,re.M),'simulator error')
    passes=re.findall(r'^SEQ_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$',text,re.M)
    names=['normal_and_stale_ack','missing_owner_ack','missing_quiesce_ack','held_observation_lost',
           'external_hold_and_repeat_perst','actual_clock_probe_stop_return',
           'loss_during_release_and_retry','tag_exhaustion','full_50ms_counter']
    need([(int(i),name) for i,name,t in passes]==list(enumerate(names)),'nine ordered scenarios')
    need(text.splitlines().count('SEQ_SUITE_PASS cases=9 full_cycles=5000001 period_ps=10000 hold_ps=50000010000 native_endpoint=0')==1,'suite scope and completion')
    rows=[]
    for line in text.splitlines():
        if not line.startswith(('SEQ_TRACE ','SEQ_EVENT ')):continue
        r=dict(re.findall(r'(\w+)=(\S+)',line))
        r={k:(v if k=='name' else int(v,16 if k in {'tag','atag'} else 10)) for k,v in r.items()}
        r['type']=line.split()[0];rows.append(r)
    need(rows and [r['ps'] for r in rows]==sorted(r['ps'] for r in rows),'global observed timeline')
    events=[r for r in rows if r['type']=='SEQ_EVENT'];traces=[r for r in rows if r['type']=='SEQ_TRACE']
    def event(case,name):
        found=[r for r in events if r['case']==case and r['name']==name]
        need(len(found)==1,'unique event '+str(case)+' '+name);return found[0]
    for case in range(9):
        start=event(case,'START');need(start['state']==1 and not start['er'] and not start['lr'],'owner before resets')
        group=[r for r in traces if r['case']==case]
        need(group,'state observations '+str(case))
        issued=[r['tag'] for r in group if r['rv']]
        need(issued==sorted(issued) and len(issued)==len(set(issued)),'strict 64-bit request tags')
        for r in group:
            need(r['blocked']==r['invalid']==1,'persistent invalidation/admission fence')
            if r['er'] or r['lr']:need(r['owner']==1,'no reset before ownership retention')
            if r['state']==6:need(r['er']==0 and r['lr']==1,'endpoint release keeps local reset')
            if r['complete']:need(r['er']==r['lr']==0 and r['owner']==1,'maintenance completion retains ownership')
    need(sum(r.get('name')=='STALE_REJECTED' for r in events)>=9,'stale/wrong ACK coverage')
    need(event(0,'DONE')['tag']>0xffffffff,'no 32-bit tag truncation')
    for case,name,state in [(1,'WAIT_OWNER',1),(2,'WAIT_DRAIN',2)]:
        r=event(case,name);need(r['state']==state and r['er']==r['lr']==0,'missing ACK cannot reset')
    for case,start_name in [(0,'HOLD_START'),(3,'HOLD_RESTART')]:
        a,z=event(case,start_name),event(case,'EP_RELEASE')
        need(a['count']==0 and z['count']==7,'short timer boundaries')
        need(z['cycle']-a['cycle']==7 and z['ps']-a['ps']==70000,'short full intervals')
    r=event(4,'EXT_HELD');need(r['state']==5 and r['er']==r['lr']==1 and r['count']==7,'external hold and saturation')
    r=event(4,'REPEAT_PERST');need(r['state']==2 and r['er']==0 and r['lr']==1,'repeated PERST needs new drain')
    r=event(5,'CLOCK_STOPPED');need(r['alive']==0 and r['state']==6 and r['er']==0 and r['lr']==1,'clock loss readiness wait')
    need(event(5,'DONE')['alive']==1,'actual probe recovery')
    for name in ['LOSS_DURING_RELEASE','RETRY_COMPLETE']:
        r=event(6,name);need(r['state']==2 and r['complete']==r['er']==r['lr']==0,'retry requalifies drain')
    r=event(7,'EXHAUSTED');need(r['state']==9 and r['tag']==0xffffffffffffffff and not r['complete'],'tag exhaustion locks')
    a,last,z=event(8,'HOLD_START'),event(8,'FULL_LAST_HELD'),event(8,'EP_RELEASE')
    need(a['count']==0 and last['count']==5000000 and z['count']==5000001,'production timer boundaries')
    need(last['er']==last['lr']==1 and z['er']==0 and z['lr']==1,'production release order')
    need(last['cycle']-a['cycle']==5000000 and z['cycle']-a['cycle']==5000001,'production cycle count')
    need(last['ps']-a['ps']==50000000000 and z['ps']-a['ps']==50000010000,'production elapsed simulator time')
    full=[r for r in traces if r['case']==8]
    held=next(r for r in full if r['state']==5);released=next(r for r in full if r['state']==6)
    need(released['ps']-held['ps']==50000010000,'state transitions independently span full hold')
    need(event(8,'DONE')['complete']==1,'full-span maintenance completes')
    return dict(status='PASS_ISOLATED_ENDPOINT_SEQUENCE_9_CASES_FULL_TIMER',cases=names,
        full_hold_cycles=5000001,full_hold_ps=50000010000,management_period_ps=10000,
        stale_ack_observations=sum(r.get('name')=='STALE_REJECTED' for r in events),
        state_observations=len(traces),event_observations=len(events),
        native_endpoint_qualified=False,physical_design_qualified=False,
        scope='Management contract and actual XPM clock-probe fixture only; acknowledgements are controlled test inputs. No native full-hold/release, current guard integration or reusable ownership qualification.')
