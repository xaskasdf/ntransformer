"""Cold host CDC conservation, plus the frozen NTRD/two-MIG reference model."""
import re
from collections import Counter
from check_ntrpu_pcie_managed_dual_mig import review as review_core,periodic,rank_witness
from check_ntrpu_pcie_fragment_pool import fields,require

CTL='aw av ar wd ws wv wr bv br bs ra rv rr ov ore od os'.split()
DATA='av ar addr id len size burst lock wv wr data strb last bv br bid resp'.split()
CTL_CHANNELS=[('AW','av','ar','aw'),('W','wv','wr','wd ws'),('B','bv','br','bs'),('AR','rv','rr','ra'),('R','ov','ore','od os')]
DATA_CHANNELS=[('AW','av','ar','addr id len size burst lock'),('W','wv','wr','data strb last'),('B','bv','br','bid resp')]

def number(v):
    require(re.fullmatch('[0-9a-f]+',v) is not None,'known active transport value')
    return int(v,16)

def events(rows,channels,active):
    streams={k:[] for k,_,_,_ in channels};held={};offered={};stalls=Counter()
    for i,row in enumerate(rows):
        if not active[i]:
            require(not held,'no reset may erase a held transaction')
            continue
        ps=int(row['ps'])
        for kind,v,r,keys in channels:
            valid=number(row[v]);ready=number(row[r]);require(valid in (0,1) and ready in (0,1),'boolean handshake')
            payload=tuple(number(row[k]) for k in keys.split()) if valid else None
            if valid and kind not in offered:offered[kind]=ps
            if kind in held:require(valid and held.pop(kind)==payload,'stable held '+kind)
            if valid and ready:streams[kind].append(dict(ps=ps,offered_ps=offered.pop(kind),payload=payload))
            if valid and not ready:held[kind]=payload;stalls[kind]+=1
    require(not held,'all observed transport transactions drained')
    return streams,dict(stalls)

def review_transport(text):
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
        require([int(x['ps']) for x in rr]==[4381+7300*i for i in range(len(rr))],'independent host phase/period')
    require(all(set(x)==set('cycle ps host transport lifetime'.split()) for x in reset),'reset schema')
    require([int(x['cycle']) for x in reset]==list(range(1,len(reset)+1)),'source reset trace from cycle1')
    require([int(x['ps']) for x in reset]==[2500+5000*i for i in range(len(reset))],'source reset timestamps')
    for rr,key in [(hd,'reset'),(hc,'reset'),(reset,'host'),(reset,'transport'),(reset,'lifetime')]:
        bits=[number(x[key]) for x in rr];require(set(bits)=={0,1} and bits==sorted(bits),'one cold release '+key)
    require(all(a['reset']==c['reset'] for a,c in zip(hd,hc)),'host reset pairing')
    require(all(a['ps']==b['ps'] and a['lifetime']==b['cold'] for a,b in zip(reset,life)),'retained core/source reset pair')
    first=lambda rr,key:int(next(x['ps'] for x in rr if x[key]=='1'))
    require(first(reset,'lifetime')<first(reset,'transport')<first(hd,'reset'),'ordered cold release, memory then source CDC then host')
    # This bench releases destination transport on the fifth source falling
    # edge after the retained lifetime rises, then host reset on the fifth
    # host falling edge. Check both sampled views of the same reset signal.
    # Monotonic release alone could hide a one-sample mutation at the boundary.
    require(first(reset,'transport')==first(reset,'lifetime')+20000,'five source falling edges before transport release')
    transport_release=first(reset,'transport')-2500
    host_release=8031+7300*((transport_release-8031)//7300+5)
    require(first(hd,'reset')==host_release+3650,'five host falling edges before host release')
    require(all(number(x['host'])==int(int(x['ps'])>=host_release) for x in reset),'source samples of host reset release')
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
    require(int(end[0]['cycles'])==len(hd) and 0<int(end[0]['ps'])-int(hd[-1]['ps'])<7300,'host finish edge')
    require((int(end[0]['writes']),int(end[0]['reads']),int(end[0]['frames']))==(276,10,35),'summary independently witnessed')
    require(hc_stalls.get('B',0)>0 and hc_stalls.get('R',0)>0 and hd_stalls.get('B',0)>0,'held host responses exercised')
    drain=rows('NTRD_HOST_DRAIN ');quiet=rows('NTRD_HOST_DRAIN_QUIET ')
    require(len(drain)==len(quiet)==1 and set(drain[0])==set('ps aw w b ar r host_writes host_reads'.split()) and set(quiet[0])=={'ps','cycles'},'explicit complete control drain')
    d=drain[0];require([int(d[k]) for k in ('aw','w','b','ar','r','host_writes','host_reads')]==[276,276,276,10,10,276,10],'drain counters equal independent endpoint inventory')
    begin=int(d['ps']);until=int(quiet[0]['ps'])
    require(quiet[0]['cycles']=='8' and until-begin==40000 and until<int(end[0]['ps']),'eight quiet source cycles before finish')
    for rr in [hc,source_ctl]:
        require(all(all(x[k]=='0' for k in ('av','wv','bv','rv','ov')) for x in rr if int(x['ps'])>=begin),'no late valid after declared drain')
    require(max(x['ps'] for stream in [hc_events,sc_events] for channel in stream.values() for x in channel)<begin,'drain follows all accepted control transfers')
    return dict(status='PASS_COLD_HOST_CDC_CONSERVATION',counts=counts,latencies=stats,host_samples=len(hd),source_samples=len(own),barriers=dict(barriers),
                stalls=dict(host_data=hd_stalls,host_control=hc_stalls,source_data=sd_stalls,source_control=sc_stalls))

def canonical_core_trace(text):
    # AMD CDC payload registers may be unspecified while VALID=0. The frozen
    # core oracle needs numeric unused inputs. Normalize only unknown INACTIVE
    # request payloads, never VALID/READY, responses, state or accepted data.
    maps={'OWN_TRACE':[('awv','awa awi awl aws awb awk'),('wv','wd ws wl')],
          'CTL_TRACE':[('av','aw'),('wv','wd ws'),('rv','ra')]}
    out=[];changed=Counter()
    for line in text.splitlines():
        for kind,groups in maps.items():
            prefix='NTRD case=0 '+kind+' '
            if not line.startswith(prefix):continue
            f=fields(line[len(prefix):])
            for valid,keys in groups:
                if f[valid]!='0':continue
                for key in keys.split():
                    if re.search('[xz]',f[key],re.I):
                        line=re.sub(r'\b'+key+r'=\S+',key+'=0',line,count=1);changed[kind+'.'+key]+=1
        out.append(line)
    return '\n'.join(out),dict(changed)

def review(text):
    transport=review_transport(text)
    canonical,normalizations=canonical_core_trace(text)
    core=review_core(canonical)
    return dict(status='PASS_NTRD_COLD_HOST_CDC_TWO_MIG',transport=transport,core=core,inactive_unknown_request_payloads=normalizations,
                scope='Two actual AMD host CDCs around retained NTRD collector/bridge/supervisor/pool and two generated MIG/generic DDR4 models. Cold coordinated resets; host is AXI/AXI-Lite BFM. No native PCIe/PERST recovery, IRQ-to-MSI CDC, commercial DIMM, routed timing, sustained throughput or inference qualification.')
