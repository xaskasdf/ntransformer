"""Independent endpoint conservation for shared retained host paths."""
import re
from collections import Counter
from check_ntrpu_pcie_fragment_pool import fields, require
from check_ntrpu_guarded_mig_r2 import CTL, DATA, CTL_CHANNELS, DATA_CHANNELS, number, events, canonical_core_trace
from check_ntrpu_retained_mig_core import review as review_core, periodic, rank_witness, pattern, address

def review_transport(text, source_period_ps):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text),'simulator failure')
    def rows(prefix):return [fields(x[len(prefix):]) for x in text.splitlines() if x.startswith(prefix)]
    hd=rows('NTRD_HOST_DATA ');hc=rows('NTRD_HOST_CTL ');reset=rows('NTRD_HOST_RESET ')
    own=rows('NTRD case=0 OWN_TRACE ');ctl=rows('NTRD case=0 CTL_TRACE ');life=rows('NTRD case=0 LIFE_TRACE ')
    end=rows('NTRD_HOST_PASS ')
    require(len(end)==1 and set(end[0])==set('cycles ps writes reads frames'.split()),'one host summary')
    require(hd and len(hd)==len(hc) and len(own)==len(ctl)==len(life)==len(reset),'paired trace domains')
    for rr,keys in [(hd,DATA),(hc,CTL)]:
        require(all(set(x)=={'cycle','ps','reset',*keys} for x in rr),'exact host schema')
        require([int(x['cycle']) for x in rr]==list(range(1,len(rr)+1)),'host trace from cycle1')
        require([int(x['ps']) for x in rr]==[4200+7000*i for i in range(len(rr))],'independent host phase/period')
    require(all(set(x)==set('cycle ps host transport lifetime'.split()) for x in reset),'reset schema')
    require([int(x['cycle']) for x in reset]==list(range(1,len(reset)+1)),'source reset trace from cycle1')
    require([int(x['ps']) for x in reset]==[source_period_ps//2+source_period_ps*i for i in range(len(reset))],'source reset timestamps')
    for rr,key in [(hd,'reset'),(hc,'reset'),(reset,'host'),(reset,'transport'),(reset,'lifetime')]:
        bits=[number(x[key]) for x in rr];require(set(bits)=={0,1} and bits==sorted(bits),'one cold release '+key)
    require(all(a['reset']==c['reset'] for a,c in zip(hd,hc)),'host reset pairing')
    require(all(a['ps']==b['ps'] and a['lifetime']==b['cold'] for a,b in zip(reset,life)),'retained core/source reset pair')
    first=lambda rr,key:int(next(x['ps'] for x in rr if x[key]=='1'))
    require(first(reset,'lifetime')<min(first(reset,'transport'),first(hd,'reset')),'retained lifetime before transport release')
    lifetime=first(reset,'lifetime')
    # LIFE_TRACE samples before the source NBA update: lifetime_started rose
    # just after the preceding source edge, potentially before a host edge
    # that precedes the first source observation of the new lifetime.
    lifetime_rise=lifetime-source_period_ps
    require(sum(int(x['ps'])>lifetime_rise and x['reset']=='0' for x in hd)>=3,'cold host release synchronizer depth')
    require(sum(int(x['ps'])>=lifetime and x['transport']=='0' for x in reset)>=3,'cold source release synchronizer depth')
    for row in hd:
        if row['reset']=='0':require(row['av']==row['wv']=='0','no host data while reset')
    for row in hc:
        if row['reset']=='0':require(row['av']==row['wv']==row['rv']=='0','no MMIO requests while reset')
    source_data=[];source_ctl=[]
    mapping=dict(av='awv',ar='awr',addr='awa',id='awi',len='awl',size='aws',burst='awb',lock='awk',wv='wv',wr='wr',data='wd',strb='ws',last='wl',bv='bv',br='br',bid='bi',resp='bs')
    for a,c,l,r in zip(own,ctl,life,reset):
        require(a['cycle']==c['cycle']==l['cycle']==r['cycle'],'paired source cycles')
        source_data.append(dict(ps=l['ps'],**{k:a[v] for k,v in mapping.items()}))
        source_ctl.append(dict(ps=l['ps'],**{k:c[k] for k in CTL}))
        if r['transport']=='0':require(a['awv']==a['wv']==c['av']==c['wv']==c['rv']=='0','CDC cannot issue before destination reset release')
    ha=[x['reset']=='1' for x in hd];sa=[x['transport']=='1' for x in reset]
    hd_events,hd_stalls=events(hd,DATA_CHANNELS,ha);hc_events,hc_stalls=events(hc,CTL_CHANNELS,ha)
    sd_events,sd_stalls=events(source_data,DATA_CHANNELS,sa);sc_events,sc_stalls=events(source_ctl,CTL_CHANNELS,sa)
    stats={};counts={}
    for path,h,d in [('data',hd_events,sd_events),('control',hc_events,sc_events)]:
        counts[path]={}
        for channel in h:
            lhs,rhs=(d[channel],h[channel]) if channel in ('B','R') else (h[channel],d[channel])
            require(lhs and len(lhs)==len(rhs),'all CDC '+path+' '+channel+' obligations')
            require([x['payload'] for x in lhs]==[x['payload'] for x in rhs],'lossless ordered CDC '+path+' '+channel)
            require(all(a['offered_ps']<b['offered_ps']<=b['ps'] for a,b in zip(lhs,rhs)),'producer presents data before consumer '+path+' '+channel)
            delivery=[b['ps']-a['offered_ps'] for a,b in zip(lhs,rhs)]
            acceptance=[b['ps']-a['ps'] for a,b in zip(lhs,rhs)]
            # AXI-Lite uses a returning acknowledgement: consumer acceptance
            # can precede the producer handshake. Full AXI uses FIFO transport.
            if path=='data':require(min(acceptance)>0,'FIFO producer acceptance before consumer')
            stats[path+'_'+channel]=dict(count=len(lhs),offer_to_consumer_accept_ps=[min(delivery),max(delivery)],consumer_minus_producer_accept_ps=[min(acceptance),max(acceptance)])
            counts[path][channel]=len(lhs)
    require(counts==dict(data=dict(AW=35,W=35,B=35),control=dict(AW=276,W=276,B=276,AR=10,R=10)),'full fixture inventory at both ends')
    require(Counter(x['payload'][0] for x in hc_events['B'])=={0:274,2:1,3:1},'host write response codes')
    require(Counter(x['payload'][1] for x in hc_events['R'])=={0:9,3:1},'decoded RRESP survives recode transport')
    require(Counter(x['payload'][1] for x in hd_events['B'])=={0:34,2:1},'fragment rejection reaches host')
    # An idle collector alone does not establish the external barrier. At
    # every ARM, first publication and ACK, all accepted host data transactions
    # must have returned their B through CDC and no host request is asserted.
    barriers=Counter();host_index=0;previous_result=False
    for s,l in zip(own,life):
        now=int(l['ps']);kinds=[]
        if s['av']==s['ar']=='1':kinds.append('arm')
        if s['rv']=='1' and not previous_result:kinds.append('publication')
        if s['rc']=='1':kinds.append('ack')
        previous_result=s['rv']=='1'
        if not kinds:continue
        while host_index+1<len(hd) and int(hd[host_index+1]['ps'])<now:host_index+=1
        require(int(hd[host_index]['ps'])<now and hd[host_index]['av']==hd[host_index]['wv']=='0','external barrier has no presented host data')
        for stream in (hd_events,sd_events):
            totals=[sum(x['ps']<now for x in stream[c]) for c in ('AW','W','B')]
            require(totals[0]==totals[1]==totals[2],'barrier drains accepted data at both CDC endpoints')
        require(sum(x['ps']<now for x in hd_events['B'])==sum(x['ps']<now for x in sd_events['B']),'barrier includes queued CDC responses')
        barriers.update(kinds)
    require(dict(barriers)==dict(arm=18,publication=18,ack=18),'all54 external data barriers witnessed')
    require(int(end[0]['cycles'])==len(hd) and 0<int(end[0]['ps'])-int(hd[-1]['ps'])<7000,'host finish edge')
    require((int(end[0]['writes']),int(end[0]['reads']),int(end[0]['frames']))==(276,10,35),'summary independently witnessed')
    require(hc_stalls.get('B',0)>0 and hc_stalls.get('R',0)>0 and hd_stalls.get('B',0)>0,'held host responses exercised')
    drain=rows('NTRD_HOST_DRAIN ');quiet=rows('NTRD_HOST_DRAIN_QUIET ')
    require(len(drain)==len(quiet)==1 and set(drain[0])==set('ps aw w b ar r host_writes host_reads'.split()) and set(quiet[0])=={'ps','cycles'},'explicit complete control drain')
    d=drain[0];require([int(d[k]) for k in ('aw','w','b','ar','r','host_writes','host_reads')]==[276,276,276,10,10,276,10],'drain counters equal independent endpoint inventory')
    begin=int(d['ps']);until=int(quiet[0]['ps'])
    require(quiet[0]['cycles']=='8' and until-begin==8*source_period_ps and until<int(end[0]['ps']),'eight quiet source cycles before finish')
    for rr in [hc,source_ctl]:
        require(all(all(x[k]=='0' for k in ('av','wv','bv','rv','ov')) for x in rr if int(x['ps'])>=begin),'no late valid after declared drain')
    require(max(x['ps'] for stream in [hc_events,sc_events] for channel in stream.values() for x in channel)<begin,'drain follows all accepted control transfers')
    return dict(status='PASS_SHARED_RETAINED_HOST_CDC_CONSERVATION',counts=counts,latencies=stats,host_samples=len(hd),source_samples=len(own),barriers=dict(barriers),
                stalls=dict(host_data=hd_stalls,host_control=hc_stalls,source_data=sd_stalls,source_control=sc_stalls))


def rows(text, prefix):
    return [fields(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]


def review_closure(text, period):
    markers = rows(text, 'RETAINED_MIG_EVENT ')
    names = ['CLOSE_WITH_PENDING_RESPONSES', 'SOURCE_STOPPED_BEFORE_LOSS',
             'REPLACED_WHILE_SOURCE_STOPPED', 'LOSS_LATCHED_SOURCE_STOPPED',
             'OWNER_CERTIFIED_RESPONSES_HELD', 'OWNER_RENEWED_HOST_STOPPED',
             'SHARED_SECTIONS_DRAINED', 'MEMORY_READBACK_WITH_CPL_HELD', 'TERMINAL']
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
    require(mark[names[-1]]['cv'] == '1' and mark[names[-1]]['cr'] == '0', 'real receiver completion remains outstanding')

    mgmt = rows(text, 'RETAINED_MIG_MGMT ')
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
            require(l['cold']==l['poolreset']==l['admit']=='1' and l['memreset']=='3' and l['restart']==l['synced']=='0', 'receiver lifetime not reset during bounded source stall')
            require(number(o['epoch']) == epoch and number(o['next']) == cookie and o['held']==o['bh']=='1', 'actual owner state retained every source edge')
            require(o['lost']==o['invalid']=='1' and o['ar']==o['kr']==o['ov']==o['rv']=='0', 'no owner reuse or publication after loss')
            require(o['pr']=='0', 'late actual CPL is not accepted')
    require(dict(counts)==dict(ARM=19,LINE=17,CMD=17,DATA=17,DONE=16,ACK=18), 'source obligations and no retirement of lost job')
    require(set(times)=={'ARM','LINE','CMD','DATA'} and times['ARM']<times['LINE']<times['CMD']<times['DATA'], 'job18 uses same owner and real receiver')
    mem = {}
    for kind in ('AW','W','B'):
        rr = [r for r in rows(text,'MANAGED_MIG_'+kind+' ') if r['job']=='18']
        require(len(rr)==1 and rr[0]['channel']=='0','one real job18 memory '+kind)
        mem[kind] = rr[0]
    require(number(mem['AW']['address'])==address(18) and number(mem['W']['data'])==pattern(18) and number(mem['W']['strb'])==(1<<64)-1,'actual memory address and all bytes')
    require(mem['B']['status']==mem['B']['id']=='0', 'actual MIG successful B')
    require(times['CMD']<int(mem['AW']['ps'])<int(mark[names[0]]['ps']) and times['DATA']<int(mem['W']['ps'])<int(mark[names[0]]['ps']), 'accepted MIG request precedes loss')
    require(int(mark[names[6]]['ps'])<int(mem['B']['ps'])<int(mark[names[7]]['ps']), 'MIG B retained beyond section proof')
    read = [r for r in rows(text,'MANAGED_MIG_R ') if r['identity']=='18']
    ar = [r for r in rows(text,'MANAGED_MIG_AR ') if r['identity']=='18']
    require(len(read)==len(ar)==1 and read[0]['channel']==ar[0]['channel']=='0', 'actual job18 readback')
    require(number(ar[0]['address'])==address(18) and number(read[0]['data'])==pattern(18) and number(read[0]['status'])==0 and number(read[0]['last'])==1,'independent full payload readback')
    require(int(mem['B']['ps'])<int(ar[0]['ps'])<int(read[0]['ps'])<int(mark[names[7]]['ps']), 'DDR readback follows real write response')
    late = [(o,l) for o,l in zip(own,life) if int(l['ps'])>int(mark[names[7]]['ps'])]
    require(late, 'late completion observed through finish')
    for o,l in late:
        require(o['pv']=='1' and o['pr']=='0' and number(o['pt'])&((1<<64)-1)==cookie-1 and number(o['ps'])&3==0,'same real pending completion retained through finish')
    end = rows(text,'RETAINED_MIG_PASS ')
    expected=dict(period=period,arms=19,lines=17,commands=17,data=17,cpl=16,done=16,acks=18,aw=17,w=17,b=17,reads=29,frames=38,rejects=2,mmio_writes=288,mmio_reads=12)
    require(len(end)==1 and all(int(end[0][k])==v for k,v in expected.items()) and number(end[0]['accepted'])==0x19 and end[0]['complete']=='0','terminal inventory agrees with independent ledgers')
    return dict(status='PASS_SHARED_PATH_RETAINED_OBLIGATIONS',counts=dict(counts),ack_tags=sorted(ack_tags),markers=len(markers),
                host_stalls=dict(data=hs,control=ls),memory_write_bytes=64,memory_read_bytes=64,
                pending_receiver_completion=hex(cookie-1),accepted_mask='0x19',complete_ack=False,
                b_to_readback_ps=int(read[0]['ps'])-int(mem['B']['ps']),
                scope='Actual shared AXI paths and generic dual MIG; short source stalls only. No host/XDMA closure, operational reset or exact DIMM qualification.')


def review(text, source_period_ps):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text), 'no native/model failure')
    require(text.count('RETAINED_MIG_HEALTHY_END ')==1, 'one healthy boundary')
    healthy=text.split('RETAINED_MIG_HEALTHY_END ')[0]
    canonical, normalized=canonical_core_trace(healthy)
    return dict(status='PASS_RETAINED_PATHS_REAL_DUAL_MIG',source_period_ps=source_period_ps,
                core=review_core(canonical,source_period_ps),transport=review_transport(healthy,source_period_ps),
                closure=review_closure(text,source_period_ps),periodic=periodic(text),ranks=rank_witness(text),
                inactive_unknown_request_payloads=normalized)

