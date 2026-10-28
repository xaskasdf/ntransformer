"""Cold host CDC conservation, plus the frozen NTRD/two-MIG reference model."""
import re
from collections import Counter
from check_ntrpu_guarded_mig_core import review as review_core,periodic,rank_witness
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
    require(first(reset,'lifetime')<min(first(reset,'transport'),first(hd,'reset')),'retained lifetime before transport release')
    lifetime=first(reset,'lifetime')
    require(sum(int(x['ps'])>=lifetime and x['reset']=='0' for x in hd)>=64,'host reset at least64 actual edges')
    require(sum(int(x['ps'])>=lifetime and x['transport']=='0' for x in reset)>=64,'source reset at least64 actual edges')
    mgmt=rows('GUARDED_MIG_MGMT ')
    require(mgmt and all(x['loss']==x['invalid']==x['ack']=='0' for x in mgmt),'healthy guard never invalidates owner')
    require(any(x['state']=='3' and x['ha']==x['sa']=='1' for x in mgmt),'automatic guard admits after boot')
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

def review_healthy(text):
    transport=review_transport(text)
    canonical,normalizations=canonical_core_trace(text)
    core=review_core(canonical)
    return dict(status='PASS_GUARDED_HOST_CDC_TWO_MIG_HEALTHY',transport=transport,core=core,inactive_unknown_request_payloads=normalizations,
                scope='Two actual AMD host CDCs around retained NTRD collector/bridge/supervisor/pool and two generated MIG/generic DDR4 models. Automatic guarded cold release; host is AXI/AXI-Lite BFM. No native PCIe/PERST recovery, IRQ-to-MSI CDC, commercial DIMM, routed timing, sustained throughput or inference qualification.')


def review_loss(text):
    """Check actual ownership, DDR bytes and reset order across the last loss."""
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text),'simulator failure')
    def rows(prefix):return [fields(x[len(prefix):]) for x in text.splitlines() if x.startswith(prefix)]
    endings=rows('GUARDED_MIG_PASS '); markers=rows('GUARDED_MIG_HEALTHY_END ')
    require(len(endings)==len(markers)==1,'one complete guarded fixture')
    end=endings[0]; begin=int(markers[0]['ps'])
    events_=rows('GUARDED_MIG_EVENT ')
    require([x['name'] for x in events_]==['ready','link_loss','release_memory_b'],'loss event order')
    ready,loss,release=map(lambda x:int(x['ps']),events_)
    require(begin<ready<loss<release<int(end['ps']),'ordered loss phases')
    cookie=0x100000010; epoch=0x100000013
    require(int(events_[0]['cookie'],16)==cookie and int(events_[0]['epoch'],16)==epoch,'independent last identity')
    loss_rows=rows('GUARDED_MIG_LOSS ')
    require(len(loss_rows)==int(end['loss_samples'])>100,'complete loss sample inventory')
    times=[int(x['ps']) for x in loss_rows]
    require(0<times[0]-ready<=5000 and times==list(range(times[0],times[-1]+1,5000)),'consecutive source edges after loss arm')
    require(0<int(end['ps'])-times[-1]<5000,'loss trace reaches finish')
    for row in loss_rows:
        require(int(row['cookie'],16)==cookie and int(row['epoch'],16)==epoch and row['bh']=='1','ownership preserved every sampled edge')
        require(row['mrst']=='3' and row['lifetime']=='1' and int(row['fault'],16)==0,'memory lifetime and boundary integrity')
        require(row['occupied0']=='1' and row['occupied1']=='0','completion descriptor retained independently of reusable staging bank')
        if row['invalid']=='1':require(row['boundary']=='1','invalidation follows boundary loss')
        if row['ack']=='1':require(row['invalid']==row['boundary']=='1','owner ACK requires latched invalidation')
        if int(row['cv'],16):
            require(int(row['cv'],16)==1 and int(row['cr'],16)==0 and int(row['ct'],16)==cookie-1 and int(row['cs'],16)==0,'late completion held intact')
    invalid_at=int(next(x['ps'] for x in loss_rows if x['invalid']=='1'))
    source_ack_at=int(next(x['ps'] for x in loss_rows if x['ack']=='1'))
    require(source_ack_at==invalid_at+5000,'owner latches invalidation on next actual source edge')
    mgmt=[x for x in rows('GUARDED_MIG_MGMT ') if int(x['ps'])>=loss]
    first=lambda key,value:int(next(x['ps'] for x in mgmt if x[key]==value))
    ack_at=first('ack','1'); reset_at=first('req','1')
    require(loss<first('loss','1')<=ack_at<reset_at<release,'management loss then real owner ACK then transport reset')
    faulted=[x for x in mgmt if x['state']=='8']
    require(faulted and all(x['ha']==x['sa']=='0' and x['hrst']==x['srst']=='1' and x['loss']==x['ack']=='1' for x in faulted),'faulted transport restored with reuse closed')
    for key in ['invalid','ack','boundary']:
        bits=[int(x[key],16) for x in loss_rows]
        require(set(bits)=={0,1} and bits==sorted(bits),'sticky '+key)
    for key in ['hrst','srst']:
        bits=[int(x[key],16) for x in loss_rows]
        groups=[v for i,v in enumerate(bits) if i==0 or v!=bits[i-1]]
        require(groups==[1,0,1],'one guarded runtime reset '+key)
    host=[x for x in rows('NTRD_HOST_DATA ') if int(x['ps'])>=loss]
    require(sum(x['reset']=='0' for x in host)>=64,'runtime reset spans64 real host edges')
    require(sum(x['srst']=='0' for x in loss_rows)>=64,'runtime reset spans64 real source edges')
    life=[x for x in rows('NTRD case=0 LIFE_TRACE ') if int(x['ps'])>begin]
    require(life and all(x['cold']==x['poolreset']==x['admit']=='1' and x['memreset']=='3' and x['restart']==x['synced']=='0' for x in life),'no memory pool reset or admission loss')
    cycles={x['cycle']:int(x['ps']) for x in life}
    own=[x for x in rows('NTRD case=0 OWN_TRACE ') if x['cycle'] in cycles]
    counts=Counter()
    pairs={'arms':('av','ar'),'lines':('lv','lr'),'commands':('cv','cr'),'data':('dv','dr'),'done':('ov','ore'),'ack':('kv','kr')}
    witnessed={}
    for row in own:
        for name,(v,r) in pairs.items():
            if row[v]==row[r]=='1':counts[name]+=1;witnessed[name]=cycles[row['cycle']]
        counts['returns']+=bool(int(row['pv'],16)&int(row['pr'],16))
        require(row['rv']=='0','no last-job publication')
        if cycles[row['cycle']]>=ready:require(row['kr']=='0' and row['ov']=='0','late completion cannot become DONE or permit retirement')
    require({k:counts[k] for k in [*pairs,'returns']}==dict(arms=1,lines=1,commands=1,data=1,done=0,ack=0,returns=0),'last job transfer obligations')
    pattern=sum(((18*37+k*19+7*7)&255)<<(k*8) for k in range(64))
    memory={kind:[x for x in rows('MANAGED_MIG_'+kind+' ') if int(x['ps'])>begin] for kind in ['AW','W','B','AR','R']}
    require(all(len(v)==1 for v in memory.values()),'one last DDR write/read transaction')
    aw,w,b,ar,r=[memory[k][0] for k in ['AW','W','B','AR','R']]
    require(all(x['channel']=='0' for x in [aw,w,b,ar,r]),'last transfer to channel0')
    require(all(x['job']=='18' for x in [aw,w,b]) and ar['identity']==r['identity']=='18','last DDR identity')
    require(int(aw['address'],16)==int(ar['address'],16)==0x1fffffffc0,'last line of128GiB')
    require([int(aw[k],16) for k in ['len','id','size','burst','cache']]==[0,0,6,1,0],'last AW attributes')
    require(int(w['data'],16)==int(r['data'],16)==pattern and int(w['strb'],16)==2**64-1 and w['last']==r['last']=='1','all64 bytes survive host reset and DDR readback')
    require(b['status']==r['status']=='0' and b['id']==r['id']=='0','successful DDR responses')
    cpl_at=int(next(x['ps'] for x in loss_rows if x['cv']=='1'))
    require(witnessed['commands']<int(aw['ps'])<ready and witnessed['data']<int(w['ps'])<ready,'pool acceptance before DDR acceptance')
    require(release<int(b['ps'])<cpl_at<int(ar['ps'])<int(r['ps'])<int(end['ps']),'late B then quarantined CPL then readback')
    expected=dict(arms=19,commands=17,data=17,returns=16,done=16,ack=18,aw=17,w=17,b=17,reads=29,frames=37)
    require(all(int(end[k])==v for k,v in expected.items()),'whole bank conservation summary')
    require(int(end['cookie'],16)==cookie and int(end['epoch'],16)==epoch and int(end['banks'],16)==0 and int(end['pending'],16)==0 and end['occupied0']=='1' and end['occupied1']=='0','last reservation remains owned')
    return dict(status='PASS_GUARDED_MIG_LOSS_RETENTION',counts=expected,loss_ps=loss,owner_ack_ps=ack_at,transport_reset_ps=reset_at,
                memory_b_ps=int(b['ps']),quarantined_cpl_ps=cpl_at,readback_ps=int(r['ps']),loss_source_samples=len(loss_rows),
                physical_throughput_claimed=False)


def review(text):
    marker='GUARDED_MIG_HEALTHY_END '
    require(text.count(marker)==1,'one healthy/loss boundary')
    healthy=text.split(marker,1)[0]
    return dict(healthy=review_healthy(healthy),loss=review_loss(text))
