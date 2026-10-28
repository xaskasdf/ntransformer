"""R2: compare cold reset releases with the actual source NBA lifetime edge.

At a 37 ns source period, three 7 ns host edges can occur before the next
source trace samples lifetime=1. Preserve R1; parameterize only that
observation relationship, never the reset depth or real handshake checks.
"""
from check_ntrpu_retained_paths_mig import (
    re, Counter, fields, require, CTL, DATA, CTL_CHANNELS, DATA_CHANNELS,
    number, events, canonical_core_trace, review_core, periodic, rank_witness,
    review_closure,
)

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
    lifetime=first(reset,'lifetime')
    require(lifetime-source_period_ps<min(first(reset,'transport'),first(hd,'reset')), 'actual retained lifetime precedes both reset releases')
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


def review(text, source_period_ps):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text), 'no native/model failure')
    require(text.count('RETAINED_MIG_HEALTHY_END ')==1, 'one healthy boundary')
    healthy=text.split('RETAINED_MIG_HEALTHY_END ')[0]
    canonical, normalized=canonical_core_trace(healthy)
    return dict(status='PASS_RETAINED_PATHS_REAL_DUAL_MIG',source_period_ps=source_period_ps,
                core=review_core(canonical,source_period_ps),transport=review_transport(healthy,source_period_ps),
                closure=review_closure(text,source_period_ps),periodic=periodic(text),ranks=rank_witness(text),
                inactive_unknown_request_payloads=normalized)

