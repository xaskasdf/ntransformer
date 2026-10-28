"""Review real-owner join composition, preserving full public AXI transaction audits."""
import re
from collections import Counter,deque
from check_ntrpu_drain_sections_r2 import (axi4_history,lite_history,vcd_rows,EVENT_MAP,LITE_FIELDS,FIELDS,parse,FIFO_WARNING,markers,require)
OWNER_EVENT_MAP=dict(own_enable='owner_enable',own_cold='owner_cold',own_lost='owner_lost',own_invalid='owner_invalid',own_cmd='owner_cmd',own_data='owner_data',own_result='owner_result',own_busy='owner_busy',own_done='owner_done',own_reuse='owner_reuse_tested')

def review_transports(sim, vcd, period, mode):
    require(mode in {0,1,2,3} and period in {5000,37000}, 'known owned profile')
    kind='owned';external=int(mode==3);shared=False
    owner_phase=1 if mode==1 else 2 if mode==2 else 0
    present=121 | (2 if mode&1 else 0) | (4 if mode&2 else 0)
    require(not shared or external == 0, 'shared profile uses absent external contract')
    lines = sim.splitlines()
    require(lines.count(FIFO_WARNING) == 5, 'five functional FIFO notices')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|Warning:|WARNING:|CRITICAL WARNING:|DRAIN_CAPTURE_FATAL)',
                          '\n'.join(x for x in lines if x != FIFO_WARNING), re.M), 'simulator/capture diagnostic')
    pass_name = 'DRAIN_OWNED_PASS'
    require(lines.count(f'{pass_name} mode={mode} period={period} owner_phase={owner_phase} actual_ownership=1 actual_sections=2 local_providers=2 native_endpoint=0 host_xdma_qualified=0 reset_driven=0') == 1, 'unique scoped PASS')
    finishes = re.findall(r'\$finish called at time\s*:\s*(\d+) ps', sim)
    require(len(finishes) == 1, 'unique native finish at ps precision')
    finish = int(finishes[0])
    events = markers(sim, 'DRAIN_OWNED_EVENT')
    names = ['BEGIN_CLOSE','LIVE_OWNER_BLOCKED','OWNER_STOPPED_BEFORE_LOSS','OWNER_REPLACED_WHILE_STOPPED','OWNER_LOSS_REGISTERED_STOPPED','OWNER_CERTIFIED','OWNER_REUSE_BLOCKED','FORWARD_HELD','LOCAL_HELD']
    if external:
        names += ['LOCAL_PROOFS_PENDING']
    names += ['REPLACE_IDENTITY','STOPPED_NEW_ID','OWNER_ONLY_RENEWED','SECTIONS_RENEWED','OWNER_OBLIGATION_HELD','AXIL_LOCAL_DRAINED','OWNER_DATA_TRANSFERRED','LOCAL_DRAINED']
    if external:
        names += ['READINESS_LOST', 'READINESS_REPROVEN', 'LATE_REQUEST_SOURCE_STOPPED', 'LATE_REQUEST_REVOKED', 'FAULT_RETAINED']
    if shared:
        names += ['SHARED_FAULT_MGMT_STOPPED', 'SHARED_FAULT_SOURCE_STOPPED', 'SHARED_FAULT_DELIVERED', 'SHARED_FAULT_RETAINED']
    names += ['TERMINAL']
    require([e['name'] for e in events] == names, 'ordered scenario inventory')
    require(all(int(a['ps']) < int(b['ps']) for a, b in zip(events, events[1:])), 'event times')
    require(all(int(e['period']) == period and int(e['external']) == external and int(e['mode'])==mode and int(e['own_phase'])==owner_phase for e in events), 'event profile')
    by = {e['name']: e for e in events}
    require(finish == int(by['TERMINAL']['ps'])+1, 'terminal marker immediately precedes finish')
    number = lambda name, key: int(by[name][key], 16 if key in {'gen', 'tag', 'pg', 'pt'} else 2)
    require(number('LOCAL_HELD', 'sections') == 3 and number('LOCAL_HELD', 'idle') == 0 and
            number('LOCAL_HELD', 'pending') == 3 and number('LOCAL_HELD', 'complete') == 0, 'local replies block composed proof')
    require(number('AXIL_LOCAL_DRAINED', 'idle') == 1 and number('AXIL_LOCAL_DRAINED', 'pending') & 2 and
            not number('AXIL_LOCAL_DRAINED', 'complete'), 'independent AXI4 local hold')
    require(number('STOPPED_NEW_ID', 'enable') == 1 and not number('STOPPED_NEW_ID', 'sections') & 2 and
            not number('STOPPED_NEW_ID', 'complete'), 'stopped peer cannot renew')
    require(number('LOCAL_DRAINED', 'idle') == 3 and number('LOCAL_DRAINED', 'pending') == 0 and
            number('LOCAL_DRAINED', 'complete') == external, 'external contract gate')
    if external:
        require(min(int(by['LOCAL_DRAINED'][k]) for k in ('obsolete_l', 'obsolete_a')) >= 1, 'obsolete local mailboxes retired')
    buses = {s: [parse(l) for l in lines if l.startswith('AXI4_SECTION_BUS side='+s+' ')] for s in ('S', 'H', 'D')}
    histories, stats = {}, {}
    for side, rows in buses.items():
        histories[side], stats[side] = axi4_history(rows, side, external, shared,
            int(by['SHARED_FAULT_MGMT_STOPPED']['ps']) if shared else 0)
    for ch in FIELDS:
        raw = histories['D'][ch][:-1] if shared and ch == 'r' else histories['D'][ch]
        require(histories['H'][ch] == raw, 'raw CDC payload conservation')
        require(histories['S'][ch][:len(raw)] == raw, 'protected normal payload conservation')
    lite = markers(sim, 'FENCE_BUS')
    lite_stats = lite_history(lite)
    require(vcd.count('$var ') > 100, 'nonempty capture')
    root_match = re.search(r'\$scope module (\S+) \$end', vcd)
    require(root_match, 'VCD root')
    root = '/'+root_match[1]+'/'
    require(root_match[1].lstrip('\\').split('(')[0] ==
            'tb_ntrpu_drain_owned_sections', 'correct VCD top')
    bus_lookup = {(s, r['t']): r for s, rs in buses.items() for r in rs}
    require(len(bus_lookup) == sum(map(len, buses.values())), 'unique AXI4 cycle samples')
    lite_lookup = {(r['side'], int(r['ps']), r['channel']): r for r in lite}
    require(len(lite_lookup) == len(lite), 'unique AXI-Lite handshakes')
    seen_bus, seen_lite = set(), set()
    event_index = 0
    mgmt_edges = positive_edges = complete_rises = 0
    fresh_low = 0
    identity = None
    last_fault = 0
    source_revoked = global_revoked = shared_source = shared_delivered = None
    host_edges, management_edges = [], []
    clocks = {n: [] for n in ('lite/host_clk', 'lite/device_clk', 'axi4/device_clk')}
    final = 0
    def covered_rows():
        last = None
        for row in vcd_rows(vcd):
            last = row
            yield row
        # VCD omits timestamps with no changes. The final two mark() calls
        # add four 1-ps delays without signal changes; bound this tail strictly.
        require(last is not None and 0 <= finish-last[0] <= 4, 'VCD reaches native finish')
        if last[0] < finish:
            yield finish, last[2], last[2]

    for t, before, after in covered_rows():
        final = t
        get = lambda name, state=after: state[root+name]
        rising = lambda name: get(name, before) == 0 and get(name) == 1
        for clock in clocks:
            if rising(clock):
                clocks[clock].append(t)
        if t >= int(by['BEGIN_CLOSE']['ps']):
            require(get('cold') == 1 and get('peer_cold') == 3 and
                    all(get(n) == 1 for n in ('lite/manager_reset_n', 'lite/hn', 'lite/dn',
                                             'axi4/host_rst_n', 'axi4/device_rst_n')), 'no operational reset during composed proof')
        if int(by['LOCAL_HELD']['ps']) <= t < int(by['SECTIONS_RENEWED']['ps']):
            require(get('replies_pending') == 3 and get('local_idle') == 0 and get('complete') == 0,
                    'local replies continuously retained across identity replacement')
        while event_index < len(events) and int(events[event_index]['ps']) <= t:
            e = events[event_index]
            qt = int(e['ps'])
            expected = {root+signal: int(e[k], 16 if k in {'gen', 'tag', 'pg', 'pt'} else 2) for k, signal in EVENT_MAP.items()}
            expected[root+'obsolete'] = int(e['obsolete_l']) | (int(e['obsolete_a']) << 32)
            expected.update({root+signal:int(e[field],2) for field,signal in OWNER_EVENT_MAP.items()})
            expected[root+'owner_obsolete']=int(e['own_obsolete'])
            expected[root+'owner/proof_ack']=int(e['own_ack'],2)
            if shared:
                expected.update({root+'mgmt_enable': int(e['mgmt_enable'], 2), root+'axi4_sources': int(e['asrc'], 2),
                                 root+'axi4_fault': int(e['local_axi4_fault'], 2)})
            candidates = [before, after] if qt == t else [before]
            require(any(all(s[k] == v for k, v in expected.items()) and
                        (not shared or (s[root+'axi4/fault_mgmt'] & 127) == int(e['asrc_mgmt'], 2)) for s in candidates),
                    'whole event/VCD snapshot: '+e['name'])
            event_index += 1
        for side, clock in [('S', 'host_clk'), ('H', 'host_clk'), ('D', 'device_clk')]:
            if rising('axi4/'+clock) and get('axi4/active', before):
                key = (side, t)
                require(key in bus_lookup, 'missing AXI4 cycle sample')
                row = bus_lookup[key]
                for ch, fields in FIELDS.items():
                    keys = [ch+'valid', ch+'ready']+[ch+f for f in fields] if row[ch+'valid'] else [ch+'valid', ch+'ready']
                    for key_field in keys:
                        require(get('axi4/'+side.lower()+'_'+key_field, before) == row[key_field],
                                'AXI4 VCD/log '+side+' '+key_field+' at '+str(t))
                require(get('axi4/phase', before) == row['phase'] and
                        get('axi4/'+('device_rst_n' if side == 'D' else 'host_rst_n'), before) == row['rst'], 'AXI4 phase/reset VCD')
                seen_bus.add(key)
        for side, clock, reset in [('U', 'host_clk', 'manager_reset_n'), ('H', 'host_clk', 'hn'), ('D', 'device_clk', 'dn')]:
            if rising('lite/'+clock) and get('lite/'+reset, before):
                for ch, fields in LITE_FIELDS.items():
                    prefix = 'lite/'+side.lower()+'_'+ch.lower()
                    if get(prefix+'valid', before) and get(prefix+'ready', before):
                        key = (side, t, ch)
                        require(key in lite_lookup, 'missing AXI-Lite handshake')
                        row = lite_lookup[key]
                        require(all(get(prefix+f, before) == int(row[f], 16) for f in fields), 'AXI-Lite VCD payload')
                        require(get('lite/shutdown_request', before) == int(row['close'], 2), 'AXI-Lite close marker')
                        seen_lite.add(key)
        if rising('axi4/host_clk'):
            host_edges.append(t)
        current_identity = (get('pg'), get('pt'))
        if current_identity != identity or not get('cold'):
            fresh_low = 0
            identity = current_identity
        if rising('mgmt_clk'):
            management_edges.append(t)
            mgmt_edges += 1
            if get('cold', before) and get('req', before) and (get('pg', before), get('pt', before)) == identity:
                fresh_low |= get('proof_req', before) & (~get('ack', before) & 127)
            if get('complete'):
                positive_edges += 1
        if get('cold'):
            faults = get('retained_faults')
            require(faults | last_fault == faults, 'retained management fault monotonicity')
            last_fault = faults
            if not external:
                require(get('complete') == 0 and not ((get('proof_req') | get('ack') | get('accepted')) & (~present & 127)), 'each absent host/XDMA contract blocks')
        if get('cold'):
            require(get('present')==present,'exact independently missing external contracts')
            require(get('owner_cold')==1 and get('owner/fault_mgmt')==0,'retained real owner configuration')
            require(get('owner/proof_request')==(get('proof_req')&1) and get('owner/proof_ack')==(get('ack')&1),'actual slot-zero wiring')
            require(get('owner/generation')==get('pg') and get('owner/proof_tag')==get('pt'),'owner challenge is join identity')
            if get('accepted')&1:
                require(get('ack')&1 and get('owner_lost')==get('owner_invalid')==1 and get('owner/epoch')==get('pg')==get('owner/bridge_epoch'), 'uncertified ownership accepted')
                require((get('ack_gen')&((1<<64)-1))==get('owner/proof_generation')==get('pg') and (get('ack_tag')&((1<<64)-1))==get('owner/proof_ack_tag')==get('pt'),'slot-zero coherent identity')
        if get('complete'):
            require(external and get('cold') == get('req') == 1 and all(get(n) == 127 for n in ('present', 'ready', 'ack', 'accepted')) and
                    all(get(n) == 0 for n in ('retained_faults', 'fault', 'revoked', 'replay')), 'every complete state is authorized')
            require(get('generation') == get('pg') == get('cg') and get('tag') == get('pt') == get('ct') and fresh_low == 127,
                    'current identity and seven independent low challenges')
            for i in range(7):
                require((get('ack_gen') >> (64*i)) & ((1 << 64)-1) == get('pg') and
                        (get('ack_tag') >> (64*i)) & ((1 << 64)-1) == get('pt'), 'per-slot ACK identity')
            if not get('complete', before):
                complete_rises += 1
                require(get('local_idle') == 3 and get('requests_present') == get('replies_pending') == 0, 'local drain at complete assertion')
        for scope in ('local_lite/', 'local_axi4/'):
            if get(scope+'peer_answer'):
                require(get(scope+'observed_idle') == get(scope+'peer_current') == 1 and
                        get(scope+'source_faults') == get(scope+'revoked_source') == 0 and get(scope+'quiet') == 3 and
                        get(scope+'saved_generation') == get(scope+'peer_generation') and get(scope+'saved_tag') == get(scope+'peer_tag'), 'valid local peer answer')
        if get('local_revoked') is not None and get('local_revoked') & 2 and source_revoked is None:
            source_revoked = t
        if get('retained_faults') is not None and get('retained_faults') & 64 and global_revoked is None:
            global_revoked = t
        # Ignore configuration-time X values only while the composed bank
        # remains in cold initialization. Active fault observations stay strict.
        if shared and get('cold') == 1:
            if get('axi4_sources') & 2 and shared_source is None:
                shared_source = t
            if get('axi4_fault') & 1 and shared_delivered is None:
                shared_delivered = t
            require(not (get('local_revoked') & 2 or get('axi4_fault') & 2), 'shared fault not hidden by own revocation')
    require(seen_bus == set(bus_lookup) and seen_lite == set(lite_lookup), 'all logged bus observations witnessed in VCD')
    require(event_index == len(events) and 0 <= final-int(by['TERMINAL']['ps']) <= 2, 'complete capture through terminal')
    lo, hi = int(by['REPLACE_IDENTITY']['ps']), int(by['STOPPED_NEW_ID']['ps'])
    require(hi-lo >= 480000 and not any(lo < t < hi for t in host_edges), 'stopped host clock across identity replacement')
    for clock, times in clocks.items():
        step = 7000 if clock == 'lite/host_clk' else period
        offset = 700 if clock == 'lite/host_clk' else 1100 if clock == 'lite/device_clk' else 600
        require(times and times[0] == offset+step//2 and all(b-a == step for a, b in zip(times, times[1:])),
                'continuous clock grid '+clock)
        require(0 <= finish-times[-1] < step, 'clock captured through finish '+clock)
    for a, b in zip(host_edges, host_edges[1:]):
        if not (a < lo and b > hi):
            require(b-a == 7000, 'AXI4 host edges outside declared stop')
    for a, b in zip(management_edges, management_edges[1:]):
        if shared and a < int(by['SHARED_FAULT_MGMT_STOPPED']['ps']) < int(by['SHARED_FAULT_SOURCE_STOPPED']['ps']) < b:
            continue
        require(b-a == 10000, 'management clock grid')
    latencies = {}
    if external:
        require(complete_rises == 2 and positive_edges > 0, 'initial and fresh readiness proof')
        require(source_revoked is not None and global_revoked is not None and 0 < global_revoked-source_revoked <= 30000,
                'bounded three-stage local fault crossing')
        require(not any(source_revoked < t for t in host_edges), 'fault delivered without another host edge')
        latencies['own_revocation_ps'] = global_revoked-source_revoked
    else:
        require(complete_rises == positive_edges == 0, 'no complete proof with missing contract')
    if shared:
        require(shared_source is not None and shared_delivered is not None and global_revoked == shared_delivered, 'shared fault observed by joiner')
        stop = int(by['SHARED_FAULT_MGMT_STOPPED']['ps'])
        resume = int(by['SHARED_FAULT_SOURCE_STOPPED']['ps'])
        require(stop < shared_source < resume and not any(stop < t < resume for t in management_edges), 'registered source while management stopped')
        require(not any(shared_source < t for t in host_edges) and 0 < shared_delivered-resume <= 30000, 'shared fault crosses without another host edge')
        require(number('SHARED_FAULT_RETAINED', 'fault') & 80 == 80, 'both section/local fault retained after new identity')
        latencies['shared_delivery_after_resume_ps'] = shared_delivered-resume
    return dict(status='DRAIN_SECTION_TRANSACTION_VCD_PASS', kind=kind, external=external, mode=mode, owner_phase=owner_phase, period_ps=period,
                axi4=stats, axil=lite_stats, events=len(events), mgmt_edges=mgmt_edges, positive_edges=positive_edges,
                complete_rises=complete_rises, latency_ps=latencies, final_ps=final, native_endpoint=False,
                external_providers_qualified=False, reset_driven=False, physical_qualification=False)


def review_owner(sim, vcd, period, mode):
    events={r['name']:r for r in markers(sim,'DRAIN_OWNED_EVENT')}
    root='/'+re.search(r'\$scope module (\S+) \$end',vcd)[1]+'/'
    p=root+'owner/'
    phase=1 if mode==1 else 2 if mode==2 else 0
    epoch=0x100000001;tag=0x800000010000abcd;address=0x1fffffffc0
    payload=sum(((44+19*k)&255)<<(8*k) for k in range(64))
    counts=Counter();aw=deque();w=deque();previous=None
    lost_at=None;first_ack=None;enabled=False;edges=[];ack_rises=[]
    active=False;disabled=False;last_clock=None;last_complete=0
    signals='''clk rst_n source_enable epoch bridge_epoch lease_held bridge_lease_held
        session_lost session_invalid pool_admit pool_restart_required arm_valid arm_ready arm_epoch arm_tag
        arm_destination retire_ready line_valid line_ready line_epoch line_tag line_address line_data
        cmd_valid cmd_ready cmd_addr cmd_tag cmd_channel cmd_len data_valid data_ready data
        done_valid done_ready done_epoch done_tag done_status result_valid result_ready
        result_epoch result_tag result_status ack_valid ack_ready cpl_valid cpl_ready
        next_cookie proof_request generation proof_tag proof_ack proof_generation proof_ack_tag
        fault_mgmt revoked_source obsolete owner_done s_axi_awvalid s_axi_awready s_axi_awaddr
        s_axi_awid s_axi_awlen s_axi_awsize s_axi_awburst s_axi_awlock s_axi_wvalid s_axi_wready
        s_axi_wdata s_axi_wstrb s_axi_wlast s_axi_bvalid s_axi_bready s_axi_bid s_axi_bresp'''.split()
    for time,before,after in vcd_rows(vcd):
        require(all(p+k in after for k in signals),'all actual owner ports captured')
        s={k:after[p+k] for k in signals};old={k:before[p+k] for k in signals}
        if s['rst_n']!=1:
            require(not active,'owner retention was reset');continue
        active=True
        require(all(value is not None for value in s.values()),'known owner ports after configuration')
        if not s['source_enable']:disabled=True
        if s['session_lost']:
            if lost_at is None:lost_at=time
            require(s['session_invalid'] and s['arm_ready']==s['retire_ready']==s['line_ready']==s['cpl_ready']==0,'lost owner cannot reopen')
            require(s['epoch']==s['bridge_epoch']==epoch and s['lease_held']==s['bridge_lease_held']==1,'actual ownership ledger retained')
        else:require(lost_at is None,'registered ownership loss remains sticky')
        require(s['fault_mgmt']==s['revoked_source']==0,'no spurious ownership contradiction')
        if s['proof_ack']:
            require(s['session_lost']==s['session_invalid']==1 and s['proof_request']==1,'owner proof needs registered loss and request')
            require(s['proof_generation']==s['generation']==epoch and s['proof_ack_tag']==s['proof_tag'],'owner proof identity bound to retained epoch')
        if old['proof_ack']==0 and s['proof_ack']==1:
            ack_rises.append((time,s['proof_ack_tag']))
            if first_ack is None:first_ack=time
        complete=after[root+'complete']
        if complete and not last_complete:
            require(s['owner_done']==1 and s['cmd_valid']==s['data_valid']==0,'control profile waits for exposed owner data transfers')
        last_complete=complete
        if old['clk']!=0 or s['clk']!=1 or old['rst_n']!=1:continue
        require(old['source_enable']==1,'no owner edge while source disabled')
        if last_clock is not None:
            require(time-last_clock>=period and (disabled or time-last_clock==period),'actual owner clock grid')
        last_clock=time;disabled=False;edges.append(time)
        for valid,ready,fields in [
            ('cmd_valid','cmd_ready',['cmd_addr','cmd_tag','cmd_channel','cmd_len']),
            ('data_valid','data_ready',['data']),
            ('line_valid','line_ready',['line_epoch','line_tag','line_address','line_data']),
            ('result_valid','result_ready',['result_epoch','result_tag','result_status']),
            ('s_axi_bvalid','s_axi_bready',['s_axi_bid','s_axi_bresp'])]:
            if previous and previous[valid] and not previous[ready]:
                require(old[valid] and all(old[k]==previous[k] for k in fields),'owner held offer changed: '+valid)
        for name,a,b in [('arm','arm_valid','arm_ready'),('line','line_valid','line_ready'),
            ('cmd','cmd_valid','cmd_ready'),('data','data_valid','data_ready'),
            ('done','done_valid','done_ready'),('retire','ack_valid','ack_ready')]:
            if old[a] and old[b]:counts[name]+=1
        counts['completion']+=(old['cpl_valid']&old['cpl_ready']).bit_count()
        if old['arm_valid'] and old['arm_ready']:
            require((old['arm_epoch'],old['arm_tag'],old['arm_destination'])==(epoch,tag,address),'real owner descriptor')
        if old['s_axi_awvalid'] and old['s_axi_awready']:
            require([old['s_axi_'+k] for k in ['awaddr','awid','awlen','awsize','awburst','awlock']]==[0x1000fffc0,1,0,6,1,0],'owner ingress AW attributes')
            aw.append(time);counts['aw']+=1
        if old['s_axi_wvalid'] and old['s_axi_wready']:
            half=counts['w'];require(half<2,'two owner frame halves')
            expected=0x313046555052544e|(epoch<<64)|(tag<<128)|((half*32)<<192)|(((payload>>(half*256))&((1<<256)-1))<<256)
            require(old['s_axi_wdata']==expected and old['s_axi_wstrb']==(1<<64)-1 and old['s_axi_wlast']==1,'full owner frame payload')
            w.append(time);counts['w']+=1
        if old['s_axi_bvalid']:
            require(aw and w and old['s_axi_bid']==1 and old['s_axi_bresp']==0,'owner B causality and identity')
            if old['s_axi_bready']:aw.popleft();w.popleft();counts['b']+=1
        if old['line_valid']:
            require((old['line_epoch'],old['line_tag'],old['line_address'],old['line_data'])==(epoch,tag,address,payload),'owner LINE payload')
        if old['cmd_valid']:
            require((old['cmd_addr'],old['cmd_tag'],old['cmd_channel'],old['cmd_len'])==(address,0xffffffff,1,0),'owner CMD context')
        if old['data_valid']:require(counts['cmd']==1 and old['data']==payload,'owner full DATA and causal CMD')
        if old['done_valid']:
            require(phase==2 and (old['done_epoch'],old['done_tag'],old['done_status'])==(epoch,tag,0),'only pre-loss owner DONE')
        if old['result_valid']:
            require(phase==2 and (old['result_epoch'],old['result_tag'],old['result_status'])==(epoch,tag,0),'owner held result remains immutable')
        previous=old
    require(counts==Counter(arm=1,line=1,cmd=1,data=1,done=int(phase==2),retire=0,completion=int(phase==2),aw=2,w=2,b=2),'exact real owner transfer scoreboard')
    require(not aw and not w,'owner ingress fully accounted')
    expected_tags=[2,3,4] if mode==3 else [2,3]
    require([tag for t,tag in ack_rises]==expected_tags,'real-owner fresh certificates and no obsolete publication')
    number=lambda name,key:int(events[name][key],2)
    require(number('LIVE_OWNER_BLOCKED','own_ack')==number('LIVE_OWNER_BLOCKED','own_lost')==0,'live ownership stays unqualified')
    require(number('OWNER_STOPPED_BEFORE_LOSS','own_lost')==0 and number('OWNER_STOPPED_BEFORE_LOSS','own_invalid')==1,'combinational loss is not registered proof')
    require(number('OWNER_LOSS_REGISTERED_STOPPED','own_lost')==1 and number('OWNER_LOSS_REGISTERED_STOPPED','own_ack')==0,'one source edge does not complete proof')
    for name in ['OWNER_STOPPED_BEFORE_LOSS','OWNER_REPLACED_WHILE_STOPPED','OWNER_LOSS_REGISTERED_STOPPED','STOPPED_NEW_ID']:
        require(number(name,'own_enable')==number(name,'own_ack')==0,'stopped owner cannot certify renewed identity')
    for name in ['OWNER_CERTIFIED','OWNER_REUSE_BLOCKED','OWNER_ONLY_RENEWED','OWNER_OBLIGATION_HELD','OWNER_DATA_TRANSFERRED','LOCAL_DRAINED']:
        require(number(name,'own_ack')==1 and int(events[name]['own_obsolete'])==1,'actual owner valid certificate and obsolete inventory')
    held={0:'own_cmd',1:'own_data',2:'own_result'}[phase]
    require(all(number(name,held)==1 for name in ['LIVE_OWNER_BLOCKED','OWNER_CERTIFIED','OWNER_REUSE_BLOCKED']),'ownership invalidation preserves exposed obligation')
    require(number('OWNER_REUSE_BLOCKED','own_reuse')==1 and number('OWNER_DATA_TRANSFERRED','own_done')==1,'reuse attempt and destination transfer exercised')
    lo=int(events['REPLACE_IDENTITY']['ps']);hi=int(events['STOPPED_NEW_ID']['ps'])
    require(hi-lo>=480000 and not any(lo<t<hi for t in edges),'owner clock truly stopped across renewal')
    lo=int(events['OWNER_STOPPED_BEFORE_LOSS']['ps']);hi=int(events['OWNER_REPLACED_WHILE_STOPPED']['ps'])
    require(not any(lo<t<hi for t in edges),'owner clock truly stopped before loss edge')
    return dict(owner_phase=phase,source_clock_edges=len(edges),transfers=dict(counts),
        owner_ack_tags=expected_tags,loss_to_first_certificate_ps=first_ack-lost_at,
        retained_cookie=0xffffffff,next_cookie=0x100000000,
        integrated_pcie_ddr_datapath=False,owner_and_section_stimuli_independent=True)


def review(sim,vcd,period,mode):
    result=review_transports(sim,vcd,period,mode)
    result['owner']=review_owner(sim,vcd,period,mode)
    result['status']='PASS_REAL_OWNERSHIP_JOIN_AND_SECTION_TRANSACTIONS'
    result['qualified_complete_native_drain']=False
    return result
