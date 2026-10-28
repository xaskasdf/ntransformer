"""Accepted command, pending DATA, real source watchdog and subsequent DDR write."""
import re
from collections import Counter
from check_ntrpu_retained_paths_mig import rows, fields, require, number, events, CTL, CTL_CHANNELS, DATA_CHANNELS, pattern, address
from check_ntrpu_retained_paths_mig_r2 import review_transport, review_core, canonical_core_trace, periodic, rank_witness

def review_closure(text, period):
    markers = rows(text, 'RETAINED_CMD_MIG_EVENT ')
    names = ['CLOSE_WITH_PENDING_RESPONSES', 'SOURCE_STOPPED_BEFORE_LOSS',
             'REPLACED_WHILE_SOURCE_STOPPED', 'LOSS_LATCHED_SOURCE_STOPPED',
             'OWNER_CERTIFIED_RESPONSES_HELD', 'OWNER_RENEWED_HOST_STOPPED',
             'SHARED_SECTIONS_DRAINED', 'MEMORY_READBACK_WITH_CMD_HELD', 'TERMINAL']
    require([r['name'] for r in markers] == names, 'all closure phases in order')
    mark = {r['name']: r for r in markers}
    require([int(r['ps']) for r in markers] == sorted({int(r['ps']) for r in markers}), 'unique ordered phase timestamps')
    epoch, cookie = 0x100000013, 0x100000010
    for r in markers:
        require(int(r['period']) == period and number(r['epoch']) == epoch and number(r['cookie']) == cookie, 'retained identity at every phase')
        require(r['held'] == r['bridge'] == '1' and r['reset'] == 'f', 'leases and cold reset rails survive')
        require(all(number(r[k]) == 0 for k in ('complete','local','fault')), 'missing host/XDMA contracts never certify global drain')
        require(number(r['gen']) == epoch, 'management generation matches actual owner')
    for name in names[:6]:
        r = mark[name]
        require(r['sections'] == '0' and r['bv'] == r['rv'] == r['fb'] == '1', 'pending real responses prevent section drain')
    for name in names[1:4]:
        require(mark[name]['source'] == mark[name]['proof'] == '0', 'no stopped-source ownership certificate')
    require(mark[names[1]]['lost'] == '0' and mark[names[3]]['lost'] == '1', 'one actual edge latches loss')
    require([number(r['tag']) for r in markers] == [1,1,2,2,2,3,3,3,3], 'fresh request replacement and renewal')
    require(mark[names[4]]['proof'] == mark[names[5]]['proof'] == '1', 'owner certified before response drain')
    require(mark[names[4]]['accepted'] == '01' and mark[names[5]]['host'] == '0', 'owner-only acceptance and stopped host renewal')
    for name in names[6:]:
        r = mark[name]
        require(number(r['accepted']) == 0x19 and r['sections'] == '3' and r['proof'] == '1', 'only owner and actual sections accepted')
        require(r['bv'] == r['rv'] == r['fb'] == '0', 'old host responses drained')
    require(mark[names[-1]]['cv'] == '0' and mark[names[-1]]['cr'] == '0', 'no completion for unaccepted command')

    mgmt = rows(text, 'RETAINED_CMD_MIG_MGMT ')
    require(mgmt and [int(r['ps']) for r in mgmt] == list(range(int(mgmt[0]['ps']), int(mgmt[-1]['ps'])+1, 10000)), 'continuous management observation')
    ack_tags = set()
    for r in mgmt:
        require(number(r['req']) & 0x66 == 0 and all(number(r[k]) == 0 for k in ('complete','local','fault')), 'absent upstream and staged local proof gates')
        require(number(r['accepted']) & ~0x19 == 0, 'no invented provider acceptance')
        if r['ack'] == '1':
            require(r['pg'] == r['gen'] and r['pt'] == r['tag'], 'ACK bound to current identity')
            ack_tags.add(number(r['pt']))
    require(ack_tags == {2,3} and int(mgmt[-1]['obsolete']) == 1, 'old reply rejected; both new identities certified')

    hd = rows(text, 'NTRD_HOST_DATA '); hc = rows(text, 'NTRD_HOST_CTL ')
    life = rows(text, 'NTRD case=0 LIFE_TRACE ')
    own = rows(text, 'NTRD case=0 OWN_TRACE '); ctl = rows(text, 'NTRD case=0 CTL_TRACE ')
    reset = rows(text, 'NTRD_HOST_RESET ')
    require(len(hd) == len(hc) and len(life) == len(own) == len(ctl) == len(reset), 'complete paired full-run traces')
    for rr in (hd,hc,life,own,ctl,reset):
        require([int(r['cycle']) for r in rr] == list(range(1,len(rr)+1)), 'no missing active clock samples')
    mapping = dict(av='awv',ar='awr',addr='awa',id='awi',len='awl',size='aws',burst='awb',lock='awk',wv='wv',wr='wr',data='wd',strb='ws',last='wl',bv='bv',br='br',bid='bi',resp='bs')
    sd = [dict(ps=l['ps'],**{k:o[v] for k,v in mapping.items()}) for o,l in zip(own,life)]
    sc = [dict(ps=l['ps'],**{k:c[k] for k in CTL}) for c,l in zip(ctl,life)]
    de, _ = events(sd, DATA_CHANNELS, [r['transport']=='1' for r in reset])
    ce, _ = events(sc, CTL_CHANNELS, [r['transport']=='1' for r in reset])
    he, hs = events(hd, DATA_CHANNELS, [r['reset']=='1' for r in hd])
    le, ls = events(hc, CTL_CHANNELS, [r['reset']=='1' for r in hc])
    for host, dest, inventory in [(he,de,dict(AW=(38,37),W=(38,37),B=(38,37))),
                                 (le,ce,dict(AW=(288,287),W=(288,287),B=(288,287),AR=(12,11),R=(12,11)))]:
        for channel, (hn,dn) in inventory.items():
            require((len(host[channel]),len(dest[channel])) == (hn,dn), 'full-run endpoint inventory '+channel)
            require([e['payload'] for e in host[channel][:dn]] == [e['payload'] for e in dest[channel]], 'all forwarded payloads and responses preserved')
            require(host[channel][-1]['ps'] > int(mark[names[6]]['ps']), 'extra request served only after closure')
    tag = 0x800000010000abcd
    frame = lambda half: 0x313046555052544e | epoch<<64 | tag<<128 | (half*32)<<192 | ((pattern(18)>>(half*256)) & ((1<<256)-1))<<256
    require([e['payload'] for e in he['W'][-3:]] == [(frame(i), (1<<64)-1, 1) for i in [0,1,0]], 'two actual data halves and local post-close rejection')
    require([e['payload'] for e in he['B'][-3:]] == [(2,0),(2,0),(2,2)], 'frame responses including local SLVERR')
    desc = [(0x20,epoch & 0xffffffff),(0x24,epoch>>32),(0x28,tag & 0xffffffff),(0x2c,tag>>32),
            (0x30,0xfffc0),(0x34,1),(0x38,address(18)&0xffffffff),(0x3c,address(18)>>32),(0xb0,0),(0x10,1),(0x20,0xdeadbeef),(0x10,1)]
    require([e['payload'] for e in le['AW'][-12:]] == [(a,) for a,_ in desc], 'real MMIO descriptor and ARM inventory')
    require([e['payload'] for e in le['W'][-12:]] == [(v,15) for _,v in desc], 'all MMIO data and strobes')
    require([e['payload'] for e in le['B'][-12:]] == [(0,)]*10+[(2,),(2,)], 'busy NTRD and closed admission reject separately')
    require([e['payload'] for e in le['R'][-2:]] == [(0x4452544e,0),(0,2)], 'held real ID response survives; closed read rejects locally')
    for stream,channel in [(he,'B'),(le,'B'),(le,'R')]:
        pending = stream[channel][-2]
        require(pending['offered_ps'] < int(mark[names[0]]['ps']) < int(mark[names[5]]['ps']) < pending['ps'] < int(mark[names[6]]['ps']), 'actual held response spans loss and host stop')

    for r in markers:
        require(r['receiver_reset']=='1' and r['memreset']=='3' and r['peer_reset']==r['forced_invalid']=='0', 'runtime watchdog never resets receiver or forces owner input')
    for r in markers[1:]:
        require(r['restart']=='1' and r['admit']==r['running']==r['rearm_ready']=='0', 'actual clock fault never reopens admission')
    require(all(mark[n]['source_alive']=='0' for n in names[1:4]), 'watchdog detects both long stopped-source intervals')
    require(mark[names[-1]]['source_alive']=='1', 'clock probe recovers without reopening session')
    first_stopped=next(int(r['ps']) for r in mgmt if r['source']=='0')
    first_dead=next(int(r['ps']) for r in mgmt if r['alive']=='0')
    first_fault=next(int(r['ps']) for r in mgmt if r['restart']=='1')
    require(250000<first_dead-first_stopped<750000 and 0<first_fault-first_dead<=10000, 'actual unchanged watchdog detection then management quarantine')
    for r in mgmt:
        require(r['forced_invalid']=='0' and r['receiver_reset']=='1' and r['memreset']=='3' and r['peer_reset']=='0', 'no testbench invalidation or runtime reset')
        if int(r['ps'])>=first_fault:
            require(r['restart']=='1' and r['admit']=='0', 'sticky quarantine sampled continuously')
    require(int(mark[names[2]]['ps'])-int(mark[names[0]]['ps'])>960000 and int(mark[names[3]]['ps'])-int(mark[names[2]]['ps'])>=960000, 'both source pauses exceed 64 management cycles')
    counts = Counter()
    times = {}
    loss_time = int(mark[names[3]]['ps'])
    source_times = [int(l['ps']) for l in life]
    require(not any(int(mark[names[1]]['ps']) < t < int(mark[names[2]]['ps']) for t in source_times), 'no source edge while replacement is issued')
    require(sum(int(mark[names[2]]['ps']) < t < loss_time for t in source_times)==1, 'exactly one source edge before second stopped window')
    for o,l in zip(own,life):
        now = int(l['ps'])
        for name,v,r in [('ARM','av','ar'),('LINE','lv','lr'),('CMD','cv','cr'),('DATA','dv','dr'),('DONE','ov','ore'),('ACK','kv','kr')]:
            if o[v] == o[r] == '1':
                counts[name] += 1
                if o['case'] == '18': times[name] = now
        if now >= loss_time:
            require(l['cold']==l['poolreset']==l['restart']=='1' and l['admit']=='0' and l['memreset']=='3', 'receiver retained after actual watchdog admission loss')
            require(number(o['epoch']) == epoch and number(o['next']) == cookie and o['held']==o['bh']=='1', 'actual owner state retained every source edge')
            require(o['lost']==o['invalid']=='1' and o['ar']==o['kr']==o['ov']==o['rv']=='0', 'no owner reuse or publication after loss')
            require(o['pr']=='0', 'late actual CPL is not accepted')
    require(dict(counts)==dict(ARM=19,LINE=17,CMD=16,DATA=16,DONE=16,ACK=18), 'source obligations and no acceptance of lost CMD')
    require(set(times)=={'ARM','LINE'} and times['ARM']<times['LINE'], 'job18 line accepted without CMD or DATA handshake')
    line_edge = [(o,l) for o,l in zip(own,life) if int(l['ps'])==times['LINE']]
    require(len(line_edge)==1, 'one real LINE handshake')
    o,l=line_edge[0]
    require(number(o['le'])==epoch and number(o['lt'])==tag and number(o['la'])==address(18) and number(o['ld'])==pattern(18), 'real accepted line identity and every payload byte')
    stops=rows(text,'RETAINED_CMD_MIG_STOP ')
    require(len(stops)==1,'one source stop with pending CMD')
    pending=stops[0];stop_ps=int(pending['ps'])
    require(int(pending['line_ps'])==times['LINE']<stop_ps<int(mark[names[0]]['ps']), 'real LINE precedes stopped unaccepted CMD')
    require(all(int(pending[k])==16 for k in ('cmd','data','aw','w','b')), 'no job18 CMD DATA or memory write at stop')
    require(number(pending['banks'])==number(pending['pending'])==0, 'no receiver reservation before CMD acceptance')
    require(number(pending['cookie'])==cookie and number(pending['epoch'])==epoch and number(pending['payload'])==pattern(18), 'retained line bytes and ownership')
    require(pending['cv']==pending['cr']=='1' and number(pending['channel'])==0 and number(pending['address'])==address(18) and number(pending['ctag'])==cookie-1 and number(pending['len'])==0, 'public CMD offered before the stopped edge')
    require(0<first_stopped-stop_ps<=10000, 'source stop observed on next management edge')
    require(not any(stop_ps<t<first_fault for t in source_times), 'no source handshake before watchdog closes admission')
    require(first_fault<int(mark[names[2]]['ps'])<loss_time,'watchdog precedes single source recovery edge')
    pending_rows=rows(text,'RETAINED_CMD_MIG_PENDING ')
    require([r['ps'] for r in pending_rows]==[r['ps'] for r in mgmt], 'continuous pending CMD observation on management clock')
    for r in pending_rows:
        require(r['cv']=='1' and r['dv']=='0' and int(r['phase'])==2, 'offered CMD retained without producing DATA')
        require(number(r['channel'])==0 and number(r['address'])==address(18) and number(r['tag'])==cookie-1 and number(r['len'])==0 and number(r['data'])==pattern(18), 'CMD descriptor and pending payload stable throughout clock stop')
        require(number(r['banks'])==number(r['pending'])==0 and all(int(r[k])==16 for k in ('cmd','writes','aw','w','b')), 'receiver stays empty without losing producer obligation')
        if int(r['ps'])>=first_fault:require(r['cr']=='0','closed receiver never accepts CMD')
    for o,l in zip(own,life):
        if int(l['ps'])<=stop_ps:continue
        require(o['cv']=='1' and o['cr']==o['dv']==o['pv']=='0' and int(o['phase'])==2, 'CMD remains offered and unaccepted on every resumed source edge')
        require(number(o['cc'])==0 and number(o['ca'])==address(18) and number(o['ct'])==cookie-1 and number(o['cl'])==0 and number(o['data'])==pattern(18), 'same actual pending command and bytes on source clock')
    for kind in ('AW','W','B'):
        memory=rows(text,'MANAGED_MIG_'+kind+' ')
        require(len(memory)==16 and all(int(r['job'])<18 and int(r['ps'])<stop_ps for r in memory), 'no DDR '+kind+' for unaccepted command')
    read=[r for r in rows(text,'MANAGED_MIG_R ') if int(r['ps'])>int(mark[names[6]]['ps'])]
    ar=[r for r in rows(text,'MANAGED_MIG_AR ') if int(r['ps'])>int(mark[names[6]]['ps'])]
    require(len(read)==len(ar)==1 and read[0]['identity']==ar[0]['identity']=='16' and read[0]['channel']==ar[0]['channel']=='0', 'one readback of prior data at lost-command destination')
    require(address(16)==address(18) and pattern(16)!=pattern(18),'prior and proposed bytes differ at the same address')
    require(number(ar[0]['address'])==address(18) and number(read[0]['data'])==pattern(16) and number(read[0]['status'])==number(read[0]['id'])==0 and number(read[0]['last'])==1, 'independent read proves old destination bytes unchanged')
    require(int(mark[names[6]]['ps'])<int(ar[0]['ps'])<int(read[0]['ps'])<int(mark[names[7]]['ps']), 'prior-data read follows section proof')
    late=[(o,l) for o,l in zip(own,life) if int(l['ps'])>int(mark[names[7]]['ps'])]
    require(late and late[-1][1]['synced']=='1','pending CMD observed after readback and restart synchronization')
    end=rows(text,'RETAINED_CMD_MIG_PASS ')
    expected=dict(period=period,arms=19,lines=17,commands=16,data=16,cpl=16,done=16,acks=18,aw=16,w=16,b=16,reads=29,frames=38,rejects=2,mmio_writes=288,mmio_reads=12)
    require(len(end)==1 and all(int(end[0][k])==v for k,v in expected.items()) and number(end[0]['accepted'])==0x19 and end[0]['complete']=='0','terminal inventory agrees with independent ledgers')
    return dict(status='PASS_SHARED_PATH_UNACCEPTED_CMD_RETAINED_AFTER_WATCHDOG',counts=dict(counts),ack_tags=sorted(ack_tags),markers=len(markers),
                host_stalls=dict(data=hs,control=ls),memory_write_bytes_after_stop=0,memory_read_bytes_after_stop=64,
                pending_producer_command=hex(cookie-1),accepted_mask='0x19',complete_ack=False,
                read_latency_ps=int(read[0]['ps'])-int(ar[0]['ps']),
                first_management_sample_to_fault_ps=first_fault-first_stopped,source_stop_to_fault_sample_ps=first_fault-stop_ps,
                line_ps=times['LINE'],source_stop_ps=stop_ps,management_pending_samples=len(pending_rows),
                scope='Actual pending CMD retained across watchdog; no receiver CMD/DATA acceptance or DDR write. Existing destination bytes verified independently. Generic dual MIG/shared AXI; no native host/XDMA, global drain, operational reset, exact DIMM, physical board or full-model qualification.')



def review_base(text, source_period_ps):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text), 'no native/model failure')
    require(text.count('RETAINED_CMD_MIG_HEALTHY_END ')==1, 'one healthy boundary')
    healthy=text.split('RETAINED_CMD_MIG_HEALTHY_END ')[0]
    canonical,normalized=canonical_core_trace(healthy)
    return dict(status='PASS_RETAINED_CMD_WATCHDOG_REAL_DUAL_MIG',source_period_ps=source_period_ps,
                core=review_core(canonical,source_period_ps),transport=review_transport(healthy,source_period_ps),
                closure=review_closure(text,source_period_ps),periodic=periodic(text),ranks=rank_witness(text),
                inactive_unknown_request_payloads=normalized)


def review(text, source_period_ps):
    result = review_base(text, source_period_ps)
    requests = rows(text, 'RETAINED_CMD_MIG_MGMT_REQUEST ')
    require(len(requests)==1, 'one management request before source-domain delivery')
    request=requests[0]
    require(number(request['req'])==0x19 and number(request['gen'])==0x100000013 and number(request['tag'])==1 and request['peer']==request['source']=='0', 'management request identity with source stopped')
    start=rows(text, 'RETAINED_CMD_MIG_STOP ')[0]
    close=next(r for r in rows(text, 'RETAINED_CMD_MIG_EVENT ') if r['name']=='CLOSE_WITH_PENDING_RESPONSES')
    require(int(start['ps'])<int(request['ps'])<int(close['ps']), 'management request precedes close marker after CMD stop')
    result['management_request_witness']=request
    rr = rows(text, 'RETAINED_CMD_MIG_FORENSIC_AR ')
    require(len(rr) == 1, 'one independent post-quarantine memory read')
    r = rr[0]
    expected = dict(channel=0, identity=16, address=address(18), calibrated=3,
                    ui_rst=0, admit=0, restart=1, source_reset=1, memory_reset=3,
                    cpl=0, ready=0, cookie=0x10000000f, next=0x100000010,
                    epoch=0x100000013, forced=0, global_ack=0)
    for key, value in expected.items():
        observed = int(r[key]) if key in {'channel', 'identity'} else number(r[key])
        require(observed == value, 'independent read context ' + key)
    ar = [row for row in rows(text, 'MANAGED_MIG_AR ') if row['ps'] == r['ps']]
    require(len(ar) == 1 and ar[0]['ps'] == r['ps'], 'fixture read matches actual MIG handshake')
    result['independent_post_quarantine_read'] = dict(expected, ps=int(r['ps']),
        host_read_qualified=False, memory_fixture_only=True)
    return result
