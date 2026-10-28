"""Independent bus and mailbox accounting for external request-side faults."""
from collections import Counter, deque
import re
from check_ntrpu_axi4_drain import pattern, mask, require
from check_ntrpu_axi4_pair_r2 import FIFO_WARNING
from check_ntrpu_axi4_section_cancel import parse, FIELDS


def review(text, kind, period):
    require(kind in range(5) and period in {5000,37000}, 'case parameters')
    lines=text.splitlines(); fault=16 if kind<3 else 32
    require(lines.count(FIFO_WARNING)==5, 'five functional-only FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|WARNING:|Warning:)',
                         '\n'.join(x for x in lines if x!=FIFO_WARNING),re.M), 'runtime diagnostic')
    require(lines.count(f'AXI4_REQUEST_PROFILE kind={kind} device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000')==1,'profile')
    require(lines.count(f'AXI4_REQUEST_SUITE_PASS kind={kind} device_period_ps={period} fault={fault} native_endpoint=0')==1,'suite marker')
    events=[parse(x) for x in lines if x.startswith('AXI4_REQUEST_EVENT ')]
    names=['START','HEALTHY_HELD','PROOF_REQUEST','NORMAL_DRAINED','PROOF_COMPLETE','FAULT_READY',
           'HOST_FAULT_STOP','FAULT_REPORTED','FAULT_FRESH_REQUEST','FAULT_REQUEST_WAIT','HOST_FAULT_RESUME',
           'HELD_AFTER_FAULT','REPLIES_RELEASE','TERMINAL']
    names+=['LOCAL_HELD','POISON_OFFER','POISON_MUTATE','POISON_ACCEPT'] if kind<3 else ['LOCAL_READ_HELD','MALFORMED_WRITE_START']
    require(Counter(x['name'] for x in events)==Counter(names),'complete unique event inventory')
    require(all(a['t']<=b['t'] for a,b in zip(events,events[1:])),'monotonic events')
    by={e['name']:e for e in events}
    order=['START','HEALTHY_HELD','PROOF_REQUEST','NORMAL_DRAINED','PROOF_COMPLETE','FAULT_READY','HOST_FAULT_STOP',
           'FAULT_REPORTED','FAULT_FRESH_REQUEST','FAULT_REQUEST_WAIT','HOST_FAULT_RESUME','HELD_AFTER_FAULT','REPLIES_RELEASE','TERMINAL']
    require(all(by[a]['t']<=by[b]['t'] for a,b in zip(order,order[1:])),'lifetime order')
    for name in ['PROOF_COMPLETE','FAULT_READY']:
        require(by[name]['ack']==1 and by[name]['fm']==0 and by[name]['state']==2,'healthy current proof before fault')
        require(by[name]['ub']==by[name]['hb']==by[name]['db']==0,'proof handshakes retired before request fault')
    require(by['PROOF_COMPLETE']['idle']==1 and by['FAULT_READY']['idle']==0,'local reply occupancy is separate from proof')
    require(by['FAULT_READY']['sbs']==1 and by['FAULT_READY']['srs']==16 and by['FAULT_READY']['srv']==1,'healthy local read held')
    require(by['FAULT_READY']['sbv']==(kind<3),'held local write matches case')

    bus={side:[parse(x) for x in lines if x.startswith('AXI4_REQUEST_BUS side='+side+' ')] for side in ['S','H','D']}
    host=[parse(x) for x in lines if x.startswith('AXI4_REQUEST_HOST ')]
    mgmt=[parse(x) for x in lines if x.startswith('AXI4_REQUEST_MGMT ')]
    boxes={box:[parse(x) for x in lines if x.startswith('AXI4_REQUEST_MB box='+box+' ')] for box in ['upstream','host','device']}
    poisoned=['aw','w','ar'][kind] if kind<3 else None
    normal_addr=lambda ch:[3,(0x100000000 if ch=='aw' else 0x200000000)+3*4096,15,6,3,1,15,0,3,12]
    special_addr=lambda ch:[4,(0x100008000 if ch=='aw' else 0x200008000),
                            (0 if ch=='ar' else (255 if kind==4 else 3)),6,2,1,7,0,4,11]
    poisoned_original=(tuple(special_addr(poisoned)) if poisoned in {'aw','ar'} else (pattern(0xca115eed),mask(0,6),1)) if kind<3 else None
    stats={};transfers={};snapshots={};mutation_edges=[];source_accept=None
    for side,rows in bus.items():
        step,offset=(period,600) if side=='D' else (7000,0)
        times=[row['t'] for row in rows]
        require(rows and all(a<b for a,b in zip(times,times[1:])),'strict bus timestamps')
        first=offset+step//2
        require(times[0]==first+((by['START']['t']-first)//step+1)*step,'first bus observation')
        require(0<=by['TERMINAL']['t']-times[-1]<step,'last bus observation')
        require(all((t-offset)%(step//2)==0 for t in times),'clock grid')
        stop,resume=by['HOST_FAULT_STOP']['t'],by['HOST_FAULT_RESUME']['t']
        for a,b in zip(times,times[1:]):
            if side=='D' or not (a<=resume and stop<=b):require(b-a==step,'missing clock observation')
        if side!='D':require(stop<resume and not any(stop<t<resume for t in times),'host clock remains stopped')
        counts=Counter();observed={ch:[] for ch in FIELDS};stalls=Counter();previous=None
        aw=deque();writes=deque();complete_w=deque();reads=deque();partial=[];bad_pairs=0;state={}
        for row in rows:
            when=row['t'];require(row['rst']==1,'no reset of retained transport')
            state[when]=dict(b=counts['b'],r=counts['r'],aw=counts['aw'],w=counts['w'],ar=counts['ar'])
            for ch,fields in FIELDS.items():
                require(row[ch+'valid'] in {0,1} and row[ch+'ready'] in {0,1},'known handshake')
                if row[ch+'valid']:require(all(row[ch+f] is not None for f in fields),'known valid payload')
                if previous and previous[ch+'valid'] and not previous[ch+'ready']:
                    require(row[ch+'valid'],'blocked offer withdrawn')
                    changed=any(row[ch+f]!=previous[ch+f] for f in fields)
                    if changed:
                        require(side=='S' and kind<3 and ch==poisoned and previous['t']<by['POISON_MUTATE']['t']<=when,'only declared external corruption')
                        mutation_edges.append(when)
                if row[ch+'valid'] and not row[ch+'ready']:stalls[ch]+=1
            if side=='H' and kind<3 and by['POISON_MUTATE']['t']<=when<by['POISON_ACCEPT']['t']:
                require(tuple(row[poisoned+f] for f in FIELDS[poisoned])==poisoned_original,'original captured payload survives source mutation')
                require(not row[poisoned+'valid'],'poisoned local request never forwarded')
            for ch in ['b','r']:
                if not row[ch+'valid']:continue
                q=writes if ch=='b' else reads;require(q,'response has causal accepted request')
                desc=q[0];local=side=='S' and desc['seq']>0
                require(row[ch+'id']==desc['id'],'response ID and ordered ownership')
                if ch=='b':require(row['bresp']==(2 if local else 0),'B response semantics')
                else:
                    require(row['rlast']==(desc['beat']==desc['length']-1),'R LAST association')
                    require(row['rresp']==(2 if local else [0,2,3][desc['beat']%3]),'R response semantics')
                    require(row['rdata']==(0 if local else pattern(3*4096^(desc['beat']<<8)^3)),'complete response payload')
                if row[ch+'ready']:
                    counts[ch]+=1;observed[ch].append(tuple(row[ch+f] for f in FIELDS[ch]))
                    if ch=='b':q.popleft()
                    else:
                        desc['beat']+=1
                        if row['rlast']:q.popleft()
            for ch in ['aw','ar']:
                if not row[ch+'valid']:continue
                seq=counts[ch]
                special=side=='S' and ((ch=='aw' and ((kind==0 and seq==2) or (kind>=3 and seq==1))) or (ch=='ar' and kind==2 and seq==2))
                expected=special_addr(ch) if special else normal_addr(ch)
                if special and kind<3 and when>=by['POISON_MUTATE']['t']:expected[1]^=1<<32
                require([row[ch+f] for f in FIELDS[ch]]==expected,'full request attributes')
                if row[ch+'ready']:
                    counts[ch]+=1;observed[ch].append(tuple(row[ch+f] for f in FIELDS[ch]))
                    desc=dict(seq=seq,id=row[ch+'id'],length=row[ch+'len']+1,beat=0)
                    if ch=='aw':aw.append(desc)
                    else:reads.append(desc)
                    if side=='S' and kind<3 and ch==poisoned and special:source_accept=when
            if row['wvalid']:
                seq=counts['w'];special=side=='S' and ((kind==1 and seq>=32) or (kind>=3 and seq>=16))
                if special and kind==1:
                    expected=[pattern(0xca115eed),mask(0,6),1]
                    if when>=by['POISON_MUTATE']['t']:expected[0]^=1<<511
                elif special:
                    beat=seq-16;expected=[pattern(0xbad00000^beat),mask(beat,6),int(kind==3 and beat==1)]
                else:
                    beat=seq%16;expected=[pattern(3*4096^(beat<<8)),mask(beat,6),int(beat==15)]
                require([row['w'+f] for f in FIELDS['w']]==expected,'complete W payload and boundary')
                if row['wready']:
                    counts['w']+=1;partial.append(expected);observed['w'].append(tuple(expected))
                    if row['wlast']:complete_w.append(partial);partial=[]
                    if side=='S' and kind==1 and special:source_accept=when
            while aw and complete_w:
                desc=aw.popleft();beats=complete_w.popleft()
                if len(beats)!=desc['length']:
                    require(side=='S' and kind==3 and desc['id']==4 and len(beats)==2 and desc['length']==4,'only declared malformed local burst')
                    bad_pairs+=1
                else:writes.append(desc)
            previous=row
        expected_counts=dict(aw=1,w=16,ar=1,b=1,r=16)
        if side=='S':expected_counts=[dict(aw=3,w=32,ar=2,b=2,r=32),dict(aw=2,w=33,ar=2,b=2,r=32),
                                      dict(aw=2,w=32,ar=3,b=2,r=33),dict(aw=2,w=18,ar=2,b=1,r=32),dict(aw=2,w=272,ar=2,b=1,r=32)][kind]
        require(dict(counts)==expected_counts,'exact accepted transaction totals')
        require(not writes and not reads,'all valid response credits retired')
        expected_pending=(int(kind in {0,4}),int(kind==1),256 if kind==4 else 0,int(kind==3)) if side=='S' else (0,0,0,0)
        require((len(aw),len(complete_w),len(partial),bad_pairs)==expected_pending,'only declared failed or unmatched write obligations remain')
        snapshots[side]=state;transfers[side]=observed;stats[side]=dict(samples=len(rows),transfers=dict(counts),stalls=dict(stalls),pending_write_state=expected_pending)
    for ch in FIELDS:
        require(transfers['H'][ch]==transfers['D'][ch],'normal native CDC conservation')
        require(transfers['S'][ch][:len(transfers['H'][ch])]==transfers['H'][ch],'normal source conservation')
    if kind<3:
        require(len(mutation_edges)==1 and mutation_edges[0]+1==by['HOST_FAULT_STOP']['t'],'source mutation causes next registered fault')
        require(source_accept+1==by['POISON_ACCEPT']['t'] and source_accept>by['REPLIES_RELEASE']['t'],'captured offer accepts only after healthy reply release')
        require(by['POISON_OFFER']['t']<by['POISON_MUTATE']['t']<by['HOST_FAULT_STOP']['t'],'blocked-source corruption order')
    else:require(not mutation_edges,'malformed W case has stable offered payloads')

    require([row['t'] for row in host]==[row['t'] for row in bus['S']]==[row['t'] for row in bus['H']],'complete host diagnostics')
    for row,s,h in zip(host,bus['S'],bus['H']):
        require(row['cold']==row['hrst']==row['drst']==1 and row['hf']==row['df']==0,'retained raw transport remains fault-free')
        require((row['sbv'],row['srv'])==(s['bvalid'],s['rvalid']),'protected offers match diagnostic trace')
        require(row['af']==(fault if row['t']>by['HOST_FAULT_STOP']['t'] else 0),'registered request-side fault timeline')
        if row['t']>by['PROOF_COMPLETE']['t']:
            require(row['state']==2 and row['fenced']==1,'closed source never reopens')
            require(not any(h[ch+'valid'] for ch in ['aw','w','ar']),'all subsequent requests remain local')
            require(not any(row[k] for k in ['ap','wp','arp','partial','hap','hwp','har','dap','dwp','dar']),'native inventories remain drained')
        if row['af']:require(not row['idle'],'fault is never local idle')
    require(mgmt and all(b['t']-a['t']==10000 for a,b in zip(mgmt,mgmt[1:])),'complete management observations')
    require(mgmt[0]['t']==5001+((by['START']['t']-5001)//10000+1)*10000,'first management observation')
    require(0<=by['TERMINAL']['t']-mgmt[-1]['t']<=10000,'last management observation')
    seen=0
    for row in mgmt:
        changed=row['t']>=by['FAULT_FRESH_REQUEST']['t'];delta=0x100000001 if changed else 0
        require((row['gen'],row['tag'])==(0x100000001+delta,0x200000001+delta),'full-width fresh request identity')
        require(row['req']==(row['t']>=by['PROOF_REQUEST']['t']),'management request timeline')
        require(row['fm'] in {0,fault} and (row['fm']|seen)==row['fm'],'only declared sticky management fault')
        seen=row['fm'];require(row['ack']==(row['uc'] and row['pc']),'complete section proof binding')
        if row['fm']:require(not (row['ack'] or row['uc'] or row['pc']),'fault revokes section and component proof')
        if changed:require(row['fm']==fault and not any(row[k] for k in ['ack','ub','hb','db']),'fresh identity cannot launch another proof')
        if row['ack']:require(row['state']==2 and row['fenced']==1 and not row['fm'],'proof excludes open source or reported fault')
    for box,rows in boxes.items():
        require([row['t'] for row in rows]==[row['t'] for row in mgmt],'complete mailbox observations')
        previous=None
        for row,m in zip(rows,mgmt):
            require(all(v is not None for v in row.values()) and row['obsolete']==0,'known mailbox fields with no stale in-flight completion')
            req=m['req'] and not m['fm']
            if box!='upstream':req=req and m['uc']
            if box=='device':req=req and m['hc']
            require(row['req']==req and (row['gen'],row['tag'],row['kind'])==(m['gen'],m['tag'],1),'mailbox identity and gating')
            if row['ack']:require(row['req'] and (row['ack_gen'],row['ack_tag'],row['ack_kind'])==(m['gen'],m['tag'],1),'fresh mailbox acknowledgement')
            if previous and previous['send'] and not previous['rcv']:
                require(row['send'] and tuple(row['payload_'+k] for k in ['gen','tag','kind'])==tuple(previous['payload_'+k] for k in ['gen','tag','kind']),'issued request held until receipt')
            previous=row
    for i,m in enumerate(mgmt):
        u,h,d=[boxes[box][i] for box in ['upstream','host','device']]
        require(m['uc']==(u['ack'] and not m['fm']) and m['hc']==h['ack'] and m['dc']==d['ack'],'component certificates bound to mailboxes')
        require(m['pc']==(h['ack'] and d['ack'] and not m['fm']),'pair certificate binding')
    detected=next(row for row in mgmt if row['fm']==fault)
    latency=detected['t']-by['HOST_FAULT_STOP']['t'];require(0<=latency<=30001,'fault crosses without another host edge')
    require(by['FAULT_REQUEST_WAIT']['t']-by['FAULT_FRESH_REQUEST']['t']>=190000,'fresh identity held rejected for twenty management edges')
    held=by['HELD_AFTER_FAULT'];require(held['srv']==1 and held['sbv']==(kind<3) and held['sbs']==1 and held['srs']==16,'healthy local offers retained through fault and stopped clock')
    terminal=by['TERMINAL'];require(terminal['fm']==terminal['af']==fault and terminal['ack']==terminal['idle']==terminal['sbv']==terminal['srv']==0,'terminal quarantine and completed healthy responses')
    require((terminal['law'],terminal['lwd'],terminal['lb'],terminal['lwbeats'],terminal['lrleft'])==
            [(1,0,0,0,0),(0,1,0,1,0),(0,0,0,0,0),(1,1,0,2,0),(1,0,0,255,0)][kind],'terminal local obligations and malformed-write saturation')
    return dict(status='PASS_EXTERNAL_AXI4_REQUEST_FAULT',kind=kind,device_period_ps=period,fault=fault,profiles=stats,
                fault_reporting_latency_ps=latency,source_mutation_edge=mutation_edges[0] if mutation_edges else None,
                captured_offer_acceptance_ps=source_accept,physical_timing_qualified=False,native_endpoint_qualified=False,
                scope='Actual AMD CDC, local healthy reply retention, externally corrupted blocked request or malformed local W; no internal fault forcing.')
