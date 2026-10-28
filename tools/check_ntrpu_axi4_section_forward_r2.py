"""Independent external accounting for malformed forwarded writes over AMD CDC."""
from collections import Counter
import re
from check_ntrpu_axi4_drain import pattern, mask, require
from check_ntrpu_axi4_pair_r2 import FIFO_WARNING
from check_ntrpu_axi4_section_cancel import parse, FIELDS


def review(text, kind, period):
    require(kind in range(4) and period in {5000,37000}, 'case parameters')
    lines=text.splitlines(); final_fault=66564
    require(lines.count(FIFO_WARNING)==5,'five functional FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|WARNING:|Warning:)',
                         '\n'.join(x for x in lines if x!=FIFO_WARNING),re.M),'runtime diagnostic')
    require(lines.count(f'AXI4_FORWARD_PROFILE kind={kind} device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000')==1,'profile')
    require(lines.count(f'AXI4_FORWARD_SUITE_PASS kind={kind} device_period_ps={period} fault=66564 native_endpoint=0')==1,'suite marker')
    def rows(prefix):return [parse(x) for x in lines if x.startswith('AXI4_FORWARD_'+prefix+' ')]
    events=rows('EVENT');by={e['name']:e for e in events}
    names=['START','HEALTHY_HELD','BAD_START','STIMULUS_DONE','HOST_FAULT_STOP','DEVICE_FAULT_STOP',
           'HOST_FAULT_REPORTED','HOST_FAULT_RESUME','FAULT_REPORTED','FAULT_FRESH_REQUEST','FAULT_REQUEST_WAIT','CLOCKS_RESUME','HELD_AFTER_FAULT',
           'REPLIES_RELEASE','TERMINAL']+(['PROOF_REQUEST','FINISH_PARTIAL'] if kind==3 else [])
    require(Counter(e['name'] for e in events)==Counter(names),'unique complete event inventory')
    require(all(a['t']<=b['t'] for a,b in zip(events,events[1:])),'monotonic event trace')
    for stop in ['HOST_FAULT_STOP','DEVICE_FAULT_STOP']:
        order=['START','HEALTHY_HELD','BAD_START',stop,'FAULT_REPORTED','FAULT_FRESH_REQUEST',
               'FAULT_REQUEST_WAIT','CLOCKS_RESUME','HELD_AFTER_FAULT','REPLIES_RELEASE','TERMINAL']
        require(all(by[a]['t']<=by[b]['t'] for a,b in zip(order,order[1:])),'causal event order')
    require(by['HOST_FAULT_STOP']['t']<by['HOST_FAULT_REPORTED']['t']==by['HOST_FAULT_RESUME']['t']<=by['FAULT_REPORTED']['t'],'host resumes after its independently reported alarm')
    require(by['HOST_FAULT_REPORTED']['fm'] & 1028 == 1028,'host report event binds registered alarms')
    require(by['STIMULUS_DONE']['t']<=by['HELD_AFTER_FAULT']['t'],'stimulus finished before healthy release')
    require(by['HEALTHY_HELD']['state']==0 and by['HEALTHY_HELD']['fm']==0,'RUN healthy setup')
    if kind==3:
        e=by['FINISH_PARTIAL']
        require(by['BAD_START']['t']<by['PROOF_REQUEST']['t']<e['t']<by['HOST_FAULT_STOP']['t'],'partial closure before corruption')
        require((e['state'],e['awleft'],e['wleft'],e['partial'])==(1,0,1,1),'one retained partial obligation in FINISH')

    bus={side:rows('BUS side='+side) for side in ['S','H','D']}
    host=rows('HOST');device=rows('DEVICE');mgmt=rows('MGMT')
    boxes={box:rows('MB box='+box) for box in ['upstream','host','device']}
    stats={};transfers={};times={};fault_edges={}
    def normal(ch):return [3,(0x100000000 if ch=='aw' else 0x200000000)+3*4096,15,6,3,1,15,0,3,12]
    special=[4,0x100008000,255 if kind==2 else 3,4 if kind==2 else 6,2,1,7,0,4,11]
    for side,trace in bus.items():
        step,offset=(period,600) if side=='D' else (7000,0)
        ts=[x['t'] for x in trace];stop=by['DEVICE_FAULT_STOP' if side=='D' else 'HOST_FAULT_STOP']['t'];resume=by['CLOCKS_RESUME' if side=='D' else 'HOST_FAULT_RESUME']['t']
        require(trace and all(a<b for a,b in zip(ts,ts[1:])),'strict bus timestamps')
        first=offset+step//2
        require(ts[0]==first+((by['START']['t']-first)//step+1)*step,'first bus observation')
        require(0<=by['TERMINAL']['t']-ts[-1]<step,'last bus observation')
        require(all((t-offset)%(step//2)==0 for t in ts),'clock grid')
        require(stop<resume and not any(stop<t<resume for t in ts),'fault source clock stopped')
        for a,b in zip(ts,ts[1:]):
            if not (a<=resume and stop<=b):require(b-a==step,'complete bus observations')
        counts=Counter();stalls=Counter();previous=None
        accepted={ch:[] for ch in FIELDS};when={ch:[] for ch in FIELDS}
        for row in trace:
            require(row['rst']==1,'no transport reset')
            for ch,fields in FIELDS.items():
                require(row[ch+'valid'] in {0,1} and row[ch+'ready'] in {0,1},'known handshake')
                if row[ch+'valid']:require(all(row[ch+f] is not None for f in fields),'known offered payload')
                if previous and previous[ch+'valid'] and not previous[ch+'ready']:
                    require(row[ch+'valid'] and all(row[ch+f]==previous[ch+f] for f in fields),'blocked offer held unchanged')
                if row[ch+'valid'] and not row[ch+'ready']:stalls[ch]+=1
            for ch in ['aw','ar']:
                if not row[ch+'valid']:continue
                expected=special if ch=='aw' and counts[ch]==1 else normal(ch)
                require([row[ch+f] for f in FIELDS[ch]]==expected,'full offered request attributes')
                size=1<<row[ch+'size'];addr=row[ch+'addr'];length=row[ch+'len']+1
                require(row[ch+'burst']==1 and size<=64 and addr%size==0 and
                        (addr&4095)+length*size<=4096,'descriptor within aligned 4KiB page')
            if row['wvalid']:
                seq=counts['w'];beat=seq if seq<16 else seq-16
                expected=(pattern(3*4096^(beat<<8)),mask(beat,6),int(beat==15)) if seq<16 else \
                         (pattern(0xbad00000^beat),mask(beat,4 if kind==2 else 6),int(kind!=2 and beat==1))
                require(tuple(row['w'+f] for f in FIELDS['w'])==expected,'full W data, strobe and declared boundary')
            if row['bvalid']:
                require(counts['aw']>=1 and counts['w']>=16 and counts['b']==0,'only healthy causal B offered')
                require((row['bid'],row['bresp'])==(3,0),'healthy B identity and response')
            if row['rvalid']:
                beat=counts['r'];require(counts['ar']==1 and beat<16,'only healthy causal R offered')
                require((row['rid'],row['rdata'],row['rresp'],row['rlast'])==
                        (3,pattern(3*4096^(beat<<8)^3),[0,2,3][beat%3],int(beat==15)),'healthy R complete payload and boundary')
            for ch,fields in FIELDS.items():
                if row[ch+'valid'] and row[ch+'ready']:
                    counts[ch]+=1;accepted[ch].append(tuple(row[ch+f] for f in fields));when[ch].append(row['t'])
            previous=row
        expected_counts=dict(aw=2,w=272 if kind==2 else 18,ar=1,b=1,r=16)
        require(dict(counts)==expected_counts,'exact external accepted totals')
        require(when['aw'][1]>by['BAD_START']['t'] and when['w'][16]>by['BAD_START']['t'],'bad transaction starts after held healthy replies')
        # AXI channels may cross the converter with different latencies.
        # The source order is a stimulus guarantee, not an ordering rule on D.
        if side!='D':
            if kind==1:require(when['w'][-1]<when['aw'][1],'bad W precedes bad AW at source')
            else:require(when['aw'][1]<when['w'][16],'bad AW precedes bad W at source')
        # Independent causal fault edge: paired mismatch uses both prior histories;
        # absence of LAST at the maximum allowed length is visible on that beat.
        detected=when['w'][-1] if kind==2 else max(when['aw'][1],when['w'][-1])+step
        if side!='S':
            require(detected+1==stop,'first registered W fault follows external cause')
            fault_edges[side]=detected
        if side=='S':
            require(when['b'][0]>=by['REPLIES_RELEASE']['t'] and when['r'][-1]>=by['REPLIES_RELEASE']['t'],'healthy protected replies retained until release')
        else:
            require(max(when['b']+when['r'])<by['BAD_START']['t'],'raw healthy returns drained before new write')
        transfers[side]=accepted;times[side]=when
        stats[side]=dict(samples=len(trace),transfers=dict(counts),stalls=dict(stalls),
                         actual_failed_write_obligations=1,malformed_complete_pairs=int(kind!=2),unterminated_w_beats=256 if kind==2 else 0)
    for ch in FIELDS:
        require(transfers['S'][ch]==transfers['H'][ch]==transfers['D'][ch],'every accepted field conserved across retained section')
        if ch in {'aw','w','ar'}:
            require(times['S'][ch]==times['H'][ch] and all(a<=b for a,b in zip(times['H'][ch],times['D'][ch])),'causal forward timing')
        else:require(all(a<=b<=c for a,b,c in zip(times['D'][ch],times['H'][ch],times['S'][ch])),'causal return timing')
    require([x['t'] for x in host]==[x['t'] for x in bus['S']]==[x['t'] for x in bus['H']],'complete host diagnostics')
    for row,source in zip(host,bus['S']):
        require(row['cold']==row['hrst']==row['drst']==1,'no reset of retained section')
        require((row['sbv'],row['srv'])==(source['bvalid'],source['rvalid']),'protected offers bind to diagnostic trace')
        faulted=row['t']>fault_edges['H']
        require((row['af'],row['hf'])==((4,8) if faulted else (0,0)),'only declared registered host write alarms')
        if faulted:
            require((row['ap'],row['wp'],row['partial'])==((2,1,255) if kind==2 else (2,2,0)),'conservative guard W inventory frozen')
            require((row['hap'],row['hwp'],row['har'])==((1,0,0) if kind==2 else (1,1,0)),'raw host write inventory frozen')
    require([x['t'] for x in device]==[x['t'] for x in bus['D']],'complete device diagnostics')
    for row in device:
        require(row['cold']==row['rst']==1,'device retained reset')
        faulted=row['t']>fault_edges['D'];require(row['df']==(8 if faulted else 0),'registered device W alarm')
        if faulted:require((row['ap'],row['wp'],row['ar'],row['partial'])==((1,0,0,255) if kind==2 else (1,1,0,0)),'device inventory freezes including detecting edge')

    require(mgmt and all(b['t']-a['t']==10000 for a,b in zip(mgmt,mgmt[1:])),'complete management observations')
    require(mgmt[0]['t']==5001+((by['START']['t']-5001)//10000+1)*10000,'first management observation')
    require(0<=by['TERMINAL']['t']-mgmt[-1]['t']<=10000,'last management observation')
    seen=0;req_start=by['PROOF_REQUEST' if kind==3 else 'FAULT_FRESH_REQUEST']['t']
    for row in mgmt:
        delta=0x100000001 if row['t']>=by['FAULT_FRESH_REQUEST']['t'] else 0
        require((row['gen'],row['tag'])==(0x100000001+delta,0x200000001+delta),'fresh full-width request identity')
        require(row['req']==(row['t']>=req_start),'request lifetime')
        require((row['af'],row['hf'],row['df'])==
                (4 if row['t']>fault_edges['H'] else 0,8 if row['t']>fault_edges['H'] else 0,
                 8 if row['t']>fault_edges['D'] else 0),'management trace binds actual registered fault sources')
        require(row['fm'] is not None and row['fm']&~final_fault==0 and (row['fm']|seen)==row['fm'],'only declared sticky management sources')
        seen=row['fm'];require(not any(row[k] for k in ['ack','uc','pc','hc','dc']),'incomplete inventory never certifies')
        if delta:require(row['fm']==final_fault,'fresh identity cannot clear write alarms')
    for box,trace in boxes.items():
        require([x['t'] for x in trace]==[x['t'] for x in mgmt],'complete mailbox observations')
        previous=None
        for row,m in zip(trace,mgmt):
            require(all(v is not None for v in row.values()),'known mailbox fields')
            req=m['req'] and not m['fm'] and box=='upstream'
            require(row['req']==req and (row['gen'],row['tag'],row['kind'])==(m['gen'],m['tag'],1),'mailbox request gating and identity')
            require(not any(row[k] for k in ['ack','rcv','obsolete','peer_ack','dest_ack']),'no completion of an incomplete proof')
            if previous and previous['send']:
                require(row['send'] and tuple(row['payload_'+k] for k in ['gen','tag','kind'])==
                        tuple(previous['payload_'+k] for k in ['gen','tag','kind']),'in-flight request survives failed proof and new identity')
            if kind!=3 or box!='upstream':require(not row['send'] and not row['busy'],'no unsupported proof launched')
            previous=row
        if kind==3 and box=='upstream':require(trace[-1]['send']==trace[-1]['busy']==1,'old upstream proof remains pending without acknowledgement')
    latencies={}
    for side,bits in [('H',1028),('D',65536)]:
        stop=by['HOST_FAULT_STOP' if side=='H' else 'DEVICE_FAULT_STOP']['t']
        first=next(row for row in mgmt if row['fm']&bits==bits)
        latency=first['t']-stop;require(0<=latency<=30001 and first['t']<by['CLOCKS_RESUME' if side=='D' else 'HOST_FAULT_RESUME']['t'],'alarm crosses while its source clock remains stopped')
        latencies[side]=latency
    require(by['FAULT_REQUEST_WAIT']['t']-by['FAULT_FRESH_REQUEST']['t']>=190000,'new identity rejected for twenty management edges')
    for name in ['HEALTHY_HELD','FAULT_REPORTED','HELD_AFTER_FAULT']:
        e=by[name];require((e['sbv'],e['srv'],e['sbs'],e['srs'])==(1,1,0,15),'healthy held slots survive independently')
    end=by['TERMINAL']
    require((end['state'],end['fm'],end['af'],end['hf'],end['df'])==(1,final_fault,4,8,8),'terminal retained FINISH and independent alarms')
    require(not any(end[k] for k in ['ack','idle','sbv','srv','law','lwd','lb','lwbeats','lrleft']),'no proof, false idle or manufactured local completion')
    require((end['sbs'],end['srs'],end['arp'],end['awleft'],end['wleft'],end['arleft'],end['fenced'])==
            (1,16,0,0,int(kind==2),0,int(kind!=2)),'healthy retirement with explicit finite-prefix remainder')
    require((end['ap'],end['wp'],end['partial'])==((2,1,255) if kind==2 else (2,2,0)),'terminal frozen guard accounting')
    require((end['daw'],end['dw'],end['dar_count'],end['dbs'],end['drs'])==
            (2,1 if kind==2 else 2,1,1,16),'terminal device transaction counts')
    return dict(status='PASS_FORWARDED_AXI4_WRITE_FAULT',kind=kind,device_period_ps=period,profiles=stats,
                registered_fault_edges_ps=fault_edges,fault_reporting_latency_ps=latencies,
                conservative_guard_inventory=[end['ap'],end['wp'],end['partial']],
                actual_failed_write_obligations=1,physical_timing_qualified=False,native_endpoint_qualified=False,
                scope='Actual CDC, malformed write in RUN/FINISH, complete external beat accounting, healthy response retirement and frozen-inventory proof rejection.')
