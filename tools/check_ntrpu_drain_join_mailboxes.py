"""Review the join driven by seven native XPM mailboxes and fixture providers."""
import bisect
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review(sim, vcd):
    require(not re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:)',sim,re.M),'simulation failure')
    rows=markers(sim,'JOIN_ASYNC_CHECK');end=markers(sim,'JOIN_ASYNC_SUITE_PASS')
    require([int(r['label']) for r in rows]==list(range(1,13)),'all asynchronous scenarios')
    require(len(end)==1 and end[0]==dict(observations='12',providers='7',real_xpm='1',native_endpoint='0',external_providers_qualified='0',reset_driven='0'),'scope')
    require('$finish called at time' in sim,'simulation finish')
    for row in rows:
        expected=int(int(row['label']) in {3,6,10})
        require(row['expected']==row['complete']==str(expected),'asynchronous expected result')
    times=[int(r['t']) for r in rows]
    require(times==sorted(set(times)) and 0<times[0]<times[-1]<50000000,'bounded increasing observations')
    require(int(rows[0]['present'],2)==123 and int(rows[0]['request'],2)&100==0,'protected contract absent')
    require(int(rows[1]['run'],2)==63 and int(rows[1]['busy'],2)&64 and not int(rows[1]['ack'],2)&64,'local peer stalled')
    require(rows[4]['revoked']=='1','restored readiness cannot revive round')
    for index in [6,7,8]:require(int(rows[index]['busy'],2)&16,'old mailbox obligation retained')
    require(int(rows[8]['gen'],16)==3 and int(rows[8]['tag'],16)==1,'replacement identity')
    require(int(rows[9]['obsolete4'])>=1,'obsolete completion drained before current certificate')
    require(int(rows[11]['faults'],2)&16,'sticky provider fault')
    prefix='/tb_ntrpu_drain_join_mailboxes/'
    keys=['clk','cold','req','gen','tag','present','ready','fault','run_peer','ack','request','accepted',
          'armed','retained','busy','pg','pt','cg','ct','complete','revoked','replay','ag','at','obsolete']
    observations={int(r['t']):r for r in rows};found=set();lows=set();identity=None;edges=0;positive=0
    timeline=[]
    for time,before,after in vcd_rows(vcd):
        require(all(prefix+k in after for k in keys),'public mailbox signals')
        s={k:after[prefix+k] for k in keys};old={k:before[prefix+k] for k in keys}
        if all(v is None for v in s.values()):
            require(time==0,'only empty initial parser frame');continue
        timeline.append((time,old,s))
        new=(s['pg'],s['pt'])
        if identity!=new or s['cold']==0:lows.clear();identity=new
        if old['clk']==0 and s['clk']==1:
            edges+=1
            if old['request'] is not None and old['ack'] is not None:
                lows.update(i for i in range(7) if old['request']&(1<<i) and not old['ack']&(1<<i))
        require(s['complete'] in {0,1},'known complete output')
        if s['complete']==1:
            require(s['cold']==s['req']==1 and s['ack']==s['present']==s['ready']==127,'all live contracts')
            require(s['fault']==s['retained']==s['revoked']==s['replay']==0,'no qualified fault')
            require(s['gen']==s['pg']==s['cg'] and s['tag']==s['pt']==s['ct'],'current output identity')
            require(lows==set(range(7)),'all challenges fresh')
            for i in range(7):
                require((s['ag']>>(64*i))&((1<<64)-1)==s['gen'] and (s['at']>>(64*i))&((1<<64)-1)==s['tag'],'native mailbox identity')
            positive+=1
    # VCD omits timestamps with no signal change. Use the last settled state
    # for those observations; do not require an artificial log-counter signal.
    sample_times=[t for t,_,_ in timeline]
    for time,row in observations.items():
        index=bisect.bisect_right(sample_times,time)-1
        require(index>=0,'observation after VCD start')
        t,old,s=timeline[index]
        expect={k:int(row[k],16) for k in ['gen','tag','pg','pt']}
        expect.update({k:int(row[k],2) for k in ['complete','req','present','ready','request','ack','accepted','armed','busy','revoked']})
        expect.update(run_peer=int(row['run'],2),retained=int(row['faults'],2))
        require(any(all(state[k]==v for k,v in expect.items()) and (state['obsolete']>>(4*32))&0xffffffff==int(row['obsolete4']) for state in ([old,s] if t==time else [s])),'whole marker/VCD snapshot')
        found.add(time)
    require(found==set(times) and edges>100 and positive>3,'continuous public evidence')
    return dict(status='PASS_JOIN_WITH_SEVEN_XPM_MAILBOXES',observations=12,management_clock_edges=edges,
                final_time_ps=times[-1],obsolete_slot4_completions=int(rows[9]['obsolete4']),
                real_xpm_mailboxes=7,qualified_management_handshakes=True,qualified_external_providers=False,
                qualified_complete_native_drain=False,qualified_reset=False,qualified_physical_design=False,
                clock_trace_scope='Clock enable controls and public handshake stalls traced; individual peer-clock nets are not in this VCD.')
