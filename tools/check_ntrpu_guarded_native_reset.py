"""Native PERST, retained DDR operation and explicitly blocked peer coordination.

The healthy prefix is replayed through the frozen native checker. Prefix views
only truncate at an actual recorded boundary; they never rewrite transactions.
"""
import re
from collections import Counter
from check_ntrpu_pcie_bypass_native import bits,require
from check_ntrpu_pcie_bypass_smoke import packet_blocks
from check_ntrpu_guarded_native_r2 import review as cold_review,sample,pattern
from check_ntrpu_guarded_native_startup_r2 import sample_management,PINS
from check_ntrpu_guarded_native_transport import fields,events,CTL_CHANNELS,DATA_CHANNELS
from check_ntrpu_guarded_native_reset_loss import review as loss_review
from check_ntrpu_guarded_mig_periodic import review as periodic_review
from check_ntrpu_guarded_native_prefix_periodic import review as prefix_periodic_review

REFS={'/board/rp_sys_clk_p','/board/ep_sys_clk_p'}


def reference_view(vcd):
    """Verify both added reference clocks before removing them from old views."""
    header,body=vcd.split('$enddefinitions $end',1)
    scope=[];names={};widths={};kept=[]
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+'/'.join(scope+[f[4]])
            require(name not in names,'unique runtime pin');names[name]=f[3];widths[f[3]]=int(f[2])
            if name in REFS:require(int(f[2])==1,'reference clock width');keep=False
        if keep:kept.append(line)
    require(len(names)==86 and REFS<=set(names),'86 runtime pins')
    codes={}
    for name in REFS:codes.setdefault(names[name],[]).append(name)
    values={};edges={n:[] for n in REFS};out=[];now=0
    for line in body.splitlines():
        clean=line.strip()
        if not clean or clean[0]=='$':out.append(line);continue
        if clean[0]=='#':now=int(clean[1:]);out.append(line);continue
        value,code=clean[1:].split() if clean[0]=='b' else (clean[0],clean[1:])
        if code not in codes:out.append(line);continue
        require(value in {'0','1'},'known reference clock')
        if values.get(code)=='0' and value=='1':
            for name in codes[code]:edges[name].append(now)
        values[code]=value
    for name,times in edges.items():
        require(times==list(range(5000,now+1,10000)),'continuous actual100MHz reference clock '+name)
    return '\n'.join(kept)+'\n$enddefinitions $end\n'+'\n'.join(out),edges


def cut_vcd(vcd,finish):
    header,body=vcd.split('$enddefinitions $end',1)
    out=[];seen=False
    for line in body.splitlines():
        if line.startswith('#'):
            time=int(line[1:])
            if time>finish:break
            if time==finish:seen=True
        out.append(line)
    require(seen,'healthy boundary is an actual VCD timestamp')
    return header+'$enddefinitions $end\n'+'\n'.join(out)+'\n'


def packet_prefix(text,finish):
    blocks=list(re.finditer(r'\[\s*(\d+)\]\s*:\s*([^\n]+)\n(.*?)(?=\[\s*\d+\]\s*:|\Z)',text,re.S))
    return ''.join(m[0] for m in blocks if int(m[1])<=finish)


def review(sim,tx,rx,vcd):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',sim),'runtime failure')
    def rows(prefix):return [fields(x[len(prefix):]) for x in sim.splitlines() if x.startswith(prefix)]
    markers=rows('NTRD_NATIVE_RESET_HEALTHY_END ');endings=rows('NTRD_NATIVE_RESET_ROOT_PASS ')
    require(len(markers)==len(endings)==1,'healthy boundary and final native root witness')
    begin=int(markers[0]['ps']);finish=int(endings[0]['ps'])
    require({k:int(v) for k,v in endings[0].items()}==dict(writes=304,reads=8,frames=37,ps=finish),'final root inventory')
    loss=loss_review(sim)
    classic,refs=reference_view(vcd)
    prefix=sim.split('NTRD_NATIVE_RESET_HEALTHY_END ',1)[0]
    healthy=cold_review(prefix,packet_prefix(tx,begin),packet_prefix(rx,begin),cut_vcd(classic,begin))
    require(healthy['final_ps']==begin,'healthy native boundary identity')
    healthy_periodic=prefix_periodic_review(prefix)
    require(healthy_periodic['finish_ps']==begin,'healthy periodic boundary identity')
    native,host,source=sample(classic)
    require(native['end_ps']==finish,'runtime VCD reaches terminal')
    guard=sample_management(classic);mgmt=rows('NTRD_GUARD_NATIVE ')
    require([int(x['ps']) for x in mgmt]==[x['ps'] for x in guard if x['ps']<=int(mgmt[-1]['ps'])],'all management samples')
    require([x['ps'] for x in guard]==list(range(5000,finish+1,10000)),'continuous independent management clock')
    require(all(x['ps']==finish for x in guard[len(mgmt):]),'no missing preterminal guard sample')
    for a,z in zip(mgmt,guard):
        require(set(a)=={'ps',*(set(PINS)-{'arm'})},'runtime guard schema')
        for k in PINS:
            if k=='arm':continue
            if k=='state':require(int(a[k],16)==bits(z[k]),'actual runtime guard state')
            else:require(a[k]==z[k],'actual runtime guard pin '+k)
    require(int(rows('NTRD_NATIVE_RESET_RETENTION_PASS ')[0]['ps'])==finish,'root/retention terminal agreement')
    ev={x['name']:x for x in rows('NTRD_NATIVE_RESET_EVENT ')}
    ready=int(ev['ready']['ps']);asserted=int(ev['perst_assert']['ps']);released=int(ev['perst_release']['ps']);returned=int(ev['link_return']['ps'])
    require(ready>begin and sum(asserted<t<released for t in refs['/board/rp_sys_clk_p'])>=500,'PERST held for500 real reference edges')
    after=[x for x in guard if x['ps']>asserted]
    require(all(x['perst']==('0' if x['ps']<=released else '1') for x in after),'actual board PERST interval')
    down=[x for x in after if x['epreset']==x['link']=='0' and x['ps']<released]
    require(down and down[0]['ps']<=int(ev['endpoint_down']['ps'])+10000,'actual native endpoint reset and link loss')
    require(any(x['epreset']==x['link']=='1' and released<x['ps']<=returned+10000 for x in after),'actual native link returns')
    require(all(x['held']==x['ready']=='0' for x in after),'no fabricated runtime peer acknowledgement')
    require(all(x['req']=='1' and x['hrst']==x['srst']==x['ha']==x['sa']=='0' and int(x['state'],2)==6 for x in after if x['ps']>returned),'link return cannot authorize transport release')
    # Match native host/source clocks and active payloads without assuming that
    # endpoint reset waits for the delayed guard transport reset.
    hd=rows('NTRD_HOST_DATA ');hc=rows('NTRD_HOST_CTL ');resets=rows('NTRD_HOST_RESET ')
    require(len(hd)==len(hc) and [int(x['ps']) for x in hd]==[x['ps'] for x in host if x['ps']<=int(hd[-1]['ps'])],'all actual host samples')
    require(all(x['ps']==finish for x in host[len(hd):]),'no missing preterminal host sample')
    require([int(x['ps']) for x in resets]==[x['ps'] for x in source],'all actual source reset samples')
    for a,z in zip(resets,source):require(all(a[k]==z[k] for k in ('host','transport','lifetime')),'actual reset pins across domains')
    for rr,name,channels in [(hd,'data',DATA_CHANNELS),(hc,'control',CTL_CHANNELS)]:
        for a,z in zip(rr,host):
            require(a['reset']==z['reset'],'host reset trace matches pin')
            for _,v,r,keys in channels:
                require(a[v]==z[name][v] and a[r]==z[name][r],'runtime actual valid/ready')
                if a[v]=='1':require(all(int(a[k],16)==bits(z[name][k]) for k in keys.split()),'runtime active payload')
            if z['ps']>=ready:
                require(all(z[name][v]=='0' for _,v,_,_ in channels),'no presented native transaction across PERST')
    # All additional traffic must be completed before the reset-ready event.
    active=[z['reset']==z['ep_reset']=='1' for z in host[:len(hd)]]
    he,_=events(hc,CTL_CHANNELS,active);de,_=events(hd,DATA_CHANNELS,active)
    require({k:len(v) for k,v in he.items()}==dict(AW=304,W=304,B=304,AR=9,R=9),'complete native control inventory')
    require({k:len(v) for k,v in de.items()}==dict(AW=37,W=37,B=37),'complete native frame inventory')
    life=rows('NTRD case=0 LIFE_TRACE ');own=rows('NTRD case=0 OWN_TRACE ');ctl=rows('NTRD case=0 CTL_TRACE ')
    require(len(own)==len(ctl)==len(life)==len(resets),'complete source ledgers')
    mapping=dict(av='awv',ar='awr',addr='awa',id='awi',len='awl',size='aws',burst='awb',lock='awk',wv='wv',wr='wr',data='wd',strb='ws',last='wl',bv='bv',br='br',bid='bi',resp='bs')
    sd=[];sc=[]
    for i,(o,c,l,r) in enumerate(zip(own,ctl,life,resets),1):
        require(o['cycle']==c['cycle']==l['cycle']==r['cycle']==str(i) and l['ps']==r['ps'],'source cycle/time pairing')
        sd.append(dict(ps=l['ps'],**{k:o[v] for k,v in mapping.items()}))
        sc.append(dict(ps=l['ps'],**{k:c[k] for k in 'aw av ar wd ws wv wr bv br bs ra rv rr ov ore od os'.split()}))
        if r['transport']=='0':require(o['awv']==o['wv']==c['av']==c['wv']==c['rv']=='0','no request escapes reset transport')
        if o['av']==o['ar']=='1' and int(l['ps'])>begin:
            require([int(o[k],16) for k in ['ae','at','aw','aa','ac']]==[0x100000013,0x800000010000abcd,0x1000fffc0,0x1fffffffc0,0],'last captured descriptor')
    se,_=events(sd,DATA_CHANNELS,[x['transport']=='1' for x in resets])
    ce,_=events(sc,CTL_CHANNELS,[x['transport']=='1' for x in resets])
    for name,h,c in [('data',de,se),('control',he,ce)]:
        for channel in h:
            require(len(h[channel])==len(c[channel]) and [x['payload'] for x in h[channel]]==[x['payload'] for x in c[channel]],'full runtime CDC conservation '+name+' '+channel)
            producer,consumer=(c[channel],h[channel]) if channel in ('B','R') else (h[channel],c[channel])
            require(all(a['offered_ps']<z['offered_ps']<=z['ps']<ready for a,z in zip(producer,consumer)),'all CDC offers delivered before reset '+name+' '+channel)
            if name=='data':require(all(a['ps']<z['ps'] for a,z in zip(producer,consumer)),'FIFO acceptance order '+channel)
    tail={k:[x for x in v if x['ps']>begin] for k,v in he.items()}
    require(all(len(tail[k])==10 for k in ['AW','W','B']) and not tail['AR'] and not tail['R'],'exact ten loss-job control writes')
    epoch=0x100000013;tag=0x800000010000abcd;window=0x1000fffc0;dest=0x1fffffffc0
    expected=[]
    for i,value in enumerate([epoch,tag,window,dest]):expected.extend([(0x20+8*i,value&0xffffffff),(0x24+8*i,value>>32)])
    expected.extend([(0xb0,0),(0x10,1)])
    for i,(address,value) in enumerate(expected):
        a,w,b=[tail[k][i] for k in ['AW','W','B']]
        require(a['payload']==(address,) and w['payload']==(value,15) and b['payload']==(0,),'additional MMIO bytes/verdict')
        require(max(a['ps'],w['ps'])<b['ps']<ready,'loss setup control drained before reset')
    observed=[x for x in native['events'] if x['ps']>begin]
    groups={k:[x for x in observed if x['channel']==k] for k in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require([len(groups[k]) for k in groups]==[14,0,0,2,2,2,0,0],'complete extra native packet inventory')
    packets=[p for p in packet_blocks(tx) if p['ps']>begin]
    require(len(packets)==12 and not [p for p in packet_blocks(rx) if p['ps']>begin],'extra decoded packets and no unexpected completions')
    for i,((address,value),req,packet) in enumerate(zip(expected,groups['RQ'][:10],packets[:10])):
        d=req['tdata'];v=packet['values']
        require(bits(d,63,0)==address and bits(d,74,64)==1 and bits(d,78,75)==1 and bits(d,127,120)==0,'loss setup native request descriptor')
        require(bits(d,95,80)==0xaf and bits(d,119,104)==0x1a0 and bits(d,103,96)==0x20 and bits(d,159,128)==value,'loss setup native request identity/data')
        require(bits(req['tkeep'])==31 and bits(req['tlast'])==1 and bits(req['tuser'],15,0)==15,'loss setup native framing')
        require(packet['ps']==req['ps'] and packet['kind']=='Memory Write-64 Frame' and packet['payload']==value.to_bytes(4,'little'),'loss setup decoded bytes')
        require((v['Address High']<<32)|v['Address Low']==address and v['Length']==1 and v['Tag']==0x20 and v['Last and First Byte Enables']==15,'loss setup decoded header')
        require(begin<req['ps']<=min(tail['AW'][i]['ps'],tail['W'][i]['ps'])<ready,'native setup reaches actual AXI')
    frames=[]
    for i in range(2):
        payload=b''.join(x.to_bytes(8,'little') for x in [0x313046555052544e,epoch,tag,i*32])+((pattern(18)>>(i*256))&((1<<256)-1)).to_bytes(32,'little')
        a,z=groups['RQ'][10+2*i:12+2*i];d=a['tdata'];aw,w,b=[groups[k][i] for k in ['AW','W','B']];p=packets[10+i];v=p['values']
        require(bits(d,63,0)==0x9000fffc0 and bits(d,74,64)==16 and bits(d,78,75)==1 and bits(d,127,120)==0,'last frames native address/type')
        require(bits(d,95,80)==0xaf and bits(d,119,104)==0x1a0 and bits(d,103,96)==0x63+i,'last frames native identity')
        require(bits(a['tkeep'])==65535 and bits(z['tkeep'])==15 and bits(a['tlast'])==0 and bits(z['tlast'])==1,'last frames native beat shape')
        require(bits(a['tuser'],35,0)==0x100f0f and bits(z['tuser'],35,0)==(13<<26)|0xf0f,'last frames SOP/EOP and byte enables')
        require(bits(d,511,128).to_bytes(48,'little')+bits(z['tdata'],127,0).to_bytes(16,'little')==payload and bits(z['tdata'],511,128)==0,'last frames all64 native bytes')
        require(p['ps']==z['ps'] and p['kind']=='Memory Write-64 Frame' and p['payload']==payload,'last frames all64 decoded bytes')
        require((v['Address High']<<32)|v['Address Low']==0x9000fffc0 and v['Length']==16 and v['Tag']==0x63+i and v['Last and First Byte Enables']==255,'last frames decoded headers')
        require([bits(aw[k]) for k in ['addr','id','len','size','burst']]==[window,0,0,6,1],'last frames native AXI attributes')
        require(bits(w['data'])==int.from_bytes(payload,'little') and bits(w['strb'])==(1<<64)-1 and bits(w['last'])==1 and bits(b['id'])==bits(b['resp'])==0,'last frames AXI bytes and B')
        require(a['ps']<z['ps']<=min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<ready,'last native frame delivered before reset')
        frames.append(dict(rq_ps=a['ps'],b_ps=b['ps']))
    # Keep all physical events and warnings; remove only the previously
    # witnessed healthy periodic summaries before checking the final totals.
    keep=[];removed=Counter();before=True
    for line in sim.splitlines():
        if line.startswith('NTRD_NATIVE_RESET_HEALTHY_END '):before=False
        m=re.match(r'DUAL_PER ([01]) (NTRPU_DDR_PERIODIC_(?:PASS|DRAIN)) ',line)
        if before and m:removed[(m[1],m[2])]+=1;continue
        keep.append(line)
    require(len(removed)==4 and set(removed.values())=={1},'one healthy periodic summary/drain per controller')
    periodic=periodic_review('\n'.join(keep))
    return dict(status=loss['status'],healthy=healthy,healthy_periodic=healthy_periodic,loss=loss,periodic=periodic,signals=86,
        native_loss_frames=frames,reference_edges_while_perst=sum(asserted<t<released for t in refs['/board/rp_sys_clk_p']),
        source_samples=len(source),host_samples=len(host),final_ps=finish,
        endpoint_reset_coordination_qualified=False,runtime_reuse_qualified=False,physical_design_qualified=False)
