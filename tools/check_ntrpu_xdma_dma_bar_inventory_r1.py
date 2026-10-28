"""Reconstruct local XDMA register transactions and retained public responses."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review_traffic(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    names = ['BEGIN', 'BAR_PENDING', 'BAR_RELEASE', 'BAR_RECEIPT', 'BAR2_HOLD', 'BAR2_PENDING', 'H2C_PENDING',
             'H2C_RELEASE', 'REHOLD', 'SECOND_PENDING', 'SECOND_RELEASE', 'BAR2_RELEASE', 'BAR2_EMPTY', 'PASS']
    events = {}
    for name in names:
        prefix='XDMA_DMA_BAR_'+dict(BAR2_HOLD='REHOLD',BAR2_PENDING='PENDING',BAR2_RELEASE='RELEASE',BAR2_EMPTY='EMPTY')[name] if name.startswith('BAR2') else 'XDMA_DMA_BAR_NATIVE_PASS' if name=='PASS' else ('XDMA_STOP_CHAIN_' if name in ('REHOLD','SECOND_PENDING','SECOND_RELEASE') else 'XDMA_LOCAL_')+name
        rows = markers(sim,prefix)
        require(len(rows) == 1, 'unique ' + name)
        events[name] = rows[0]
    times = {n:int(r['t']) for n,r in events.items()}
    start, finish = times['BEGIN'], times['PASS']
    require(90000000 < start < finish < 240000000 and list(times.values()) == sorted(set(times.values())), 'native causal marker order')
    require('$finish called at time' in sim, 'native finish')
    require({k:events['PASS'][k] for k in ['local_writes','host_bytes','faults','dma_bytes','done_events','completed','native_xdma','bar_writes','observer_instances','source_fence_tested','c2h_tested','driver_closure','global_drain','reset_tested']} ==
            dict(local_writes='9',host_bytes='0',faults='200c',dma_bytes='128',done_events='2',completed='2',native_xdma='1',bar_writes='2',observer_instances='2',source_fence_tested='0',c2h_tested='0',driver_closure='0',global_drain='0',reset_tested='0'), 'fixture scope')
    ep,tb,rp = '/board/EP/','/board/ntrpu_stop_probe/','/board/RP/'
    lf = {'aw':['awaddr','awprot'], 'w':['wdata','wstrb'], 'b':['bresp'], 'ar':['araddr','arprot'], 'r':['rdata','rresp']}
    af = {'aw':['awid','awaddr','awlen','awsize'], 'w':['wdata','wstrb','wlast'], 'b':['bid','bresp'],
          'ar':['arid','araddr','arlen','arsize'], 'r':['rid','rdata','rresp','rlast']}
    local={c:[] for c in lf}
    accepted={p:{c:[] for c in af} for p in ['m_axi','m_axib']}
    packets={p:[] for p in ['s_axis_rq','m_axis_rc','m_axis_cq','s_axis_cc']}
    retained, stalled, phases = {}, {}, []
    hold_cycles={k:0 for k in ['BAR','BAR2','H2C','SECOND']}
    response_stalls={'b':0,'r':0}
    done={'h2c':[],'c2h':[]}
    states=[]
    memory=bytearray(64);written=0;host_packets=0;host_active=False
    regs=markers(sim,'XDMA_LOCAL_REG')
    query={int(r['t']):r for r in regs}
    qt=sorted(query); qi=0
    require(len(qt)==len(regs)==4,'native register logs')
    last=0
    for t,before,after in vcd_rows(vcd):
        last=t
        while qi<len(qt) and qt[qi]<=t:
            row=query[qt[qi]]
            samples=[before,after] if qt[qi]==t else [before]
            require(any(s[rp+'tx_usrapp/P_READ_DATA']==int(row['value'],16) and s[tb+'phase']==int(row['phase']) for s in samples),'native register log/VCD')
            qi+=1
        if t<start:continue
        require(all(after[n]==1 for n in ['/board/sys_rst_n',ep+'user_resetn',ep+'user_lnk_up',rp+'user_lnk_up',ep+'ntrpu_xdma_i/dma_bridge_resetn']), 'continuous native readiness without optional reset')
        phase=after[tb+'phase'];require(phase in range(1,8),'phase range')
        if not phases or phases[-1]!=phase:phases.append(phase)
        for label,app,port,channel,hold in [('BAR','bypass','m_axib','b','hold_bypass_b'),('BAR2','bypass','m_axib','b','hold_bypass_b'),('H2C','dma','m_axi','b','hold_dma_b'),('SECOND','dma','m_axi','b','hold_dma_b')]:
            release=times[label+'_RELEASE']
            expected_hold=int(t<times['H2C_RELEASE'] or times['REHOLD']<=t<times['SECOND_RELEASE']) if hold=='hold_dma_b' else int(t<times['BAR_RELEASE'] or times['BAR2_HOLD']<=t<times['BAR2_RELEASE'])
            require(after[tb+hold]==expected_hold and after[tb+'hold_dma_r']==1,'exact hold/release/rehold controls')
            if times[label+'_PENDING']<=t<release:
                require(after[ep+'ntrpu_app_'+app+'_'+channel+'valid']==1 and after[ep+'ntrpu_app_'+app+'_'+channel+'ready']==0 and after[ep+port+'_'+channel+'valid']==0,'actual retained '+label)
                payload=[after[ep+port+'_'+f] for f in af[channel]]
                require(None not in payload,'known held response')
                if label not in retained:retained[label]=payload
                require(payload==retained[label],'stable held response '+label)
        if before[ep+'user_clk']==0 and after[ep+'user_clk']==1:
            require(before[ep+'m_axil_awvalid']==before[ep+'m_axil_arvalid']==0,'unused BAR0 has no request')
            for label in hold_cycles:
                if times[label+'_PENDING']<=t<times[label+'_RELEASE']:hold_cycles[label]+=1
            for c,fields in lf.items():
                p=tb+'local_'; valid,ready=before[p+c+'valid'],before[p+c+'ready']
                require(valid in (0,1) and ready in (0,1),'known local handshake')
                payload=[before[p+f] for f in fields]
                if c in stalled:
                    require(valid==1 and payload==stalled[c],'local valid/payload held until acceptance')
                if valid:
                    require(None not in payload,'known local payload')
                    if ready:
                        local[c].append([t,before[tb+'phase']]+payload)
                        stalled.pop(c,None)
                    else:
                        stalled[c]=payload
                        if c in response_stalls:response_stalls[c]+=1
            for port in accepted:
                for c,fields in af.items():
                    p=ep+port+'_'
                    valid,ready=before[p+c+'valid'],before[p+c+'ready']
                    require(valid in (0,1) and ready in (0,1),'known public AXI handshake')
                    if valid==ready==1:
                        row=[t]+[before[p+f] for f in fields]
                        require(None not in row,'known public AXI payload')
                        accepted[port][c].append(row)
            pair=[]
            for channel in done:
                value=before[tb+channel+'_status'];require(value is not None,'known public channel status')
                require(value&0x80==0,'reserved channel status bit')
                if value&8:done[channel].append(t)
                pair.append(value)
                require(after[tb+channel+'_done_events']==len(done[channel]),'descriptor pulse accounting')
            if not states or states[-1][1:]!=pair:states.append([t]+pair)
        require(all(after[tb+'dma_'+c]==len(accepted['m_axi'][c]) for c in af),'native DMA counters match actual interface events')
        if before[rp+'user_clk']==0 and after[rp+'user_clk']==1:
            for prefix in packets:
                p=rp+prefix+'_';valid,ready=before[p+'tvalid'],before[p+'tready']
                require(valid in (0,1),'known native PCIe valid')
                if valid:
                    require(ready is not None,'known native PCIe ready')
                    if ready&1:
                        row=[t]+[before[p+f] for f in ['tdata','tkeep','tlast','tuser']]
                        require(None not in row,'known native PCIe payload')
                        packets[prefix].append(row)
                        if prefix=='m_axis_cq':require((row[1]>>75)&15==0,'no C2H posted writes in chain fixture')
        require(after[tb+'host_memory']==int.from_bytes(memory,'little') and after[tb+'host_written']==2**written-1 and
                after[tb+'host_bytes']==written and after[tb+'host_packets']==host_packets and after[tb+'host_active']==int(host_active),
                'host memory changes only at accepted native payload bytes')
    require(phases==list(range(1,8)) and finish-1<=last<=finish and qi==len(qt),'complete native VCD')
    require(not stalled and min(hold_cycles.values())>0 and response_stalls=={'b':0,'r':0},'direct master consumes native B/R without response gating')
    nreads=int(events['PASS']['local_reads'])
    require(nreads>=35 and (nreads-11)%6==0,'at least four complete autonomous CSR rounds')
    require({c:len(v) for c,v in local.items()}==dict(aw=9,w=9,b=9,ar=nreads,r=nreads),'local transfer inventory')
    reads=[];writes=[]
    previous=0
    for ar,r in zip(local['ar'],local['r']):
        require(previous<ar[0]<r[0] and ar[1]<=r[1] and ar[3]==0 and r[3]==0,'local read lifecycle/protection/response')
        reads.append(dict(request_ps=ar[0],response_ps=r[0],phase=ar[1],response_phase=r[1],address=ar[2],value=r[2]))
        previous=r[0]
    previous=0
    for aw,w,b in zip(local['aw'],local['w'],local['b']):
        require(previous<min(aw[0],w[0]) and max(aw[0],w[0])<b[0] and aw[1]==w[1]==b[1] and aw[3]==0 and w[3]==15 and b[2]==0,'local write lifecycle/protection/strobes/response')
        writes.append(dict(aw_ps=aw[0],w_ps=w[0],response_ps=b[0],phase=b[1],address=aw[2],value=w[2]))
        previous=b[0]
    expected_reads=[(1,a,0) for a in [4,0x40,0x48,0x1004,0x1040,0x1048]]+[(1,4,0xfffe7e),
                    (2,0x40,0),(2,0x1040,0),(2,4,0xfffe7c),(2,0x1004,0xfffe7c)]
    for actual,wanted in zip(reads[:11],expected_reads):
        require((actual['phase'],actual['address'],actual['value'])==wanted,'fixture setup reads')
    sequence=[4,0x40,0x48,0x1004,0x1040,0x1048]
    for i,row in enumerate(reads[11:]):
        require(row['phase'] in (4,5,6) and row['address']==sequence[i%6],'autonomous six-register polling order')
        value=row['value'];addr=row['address']
        require(value==(0xfffe7e if addr==4 else 0xfffe7c) if addr in (4,0x1004) else value&~0x47==0 if addr in (0x40,0x1040) else value in (0,1,2) if addr==0x48 else value==0,'native CSR values')
    expected_writes=[(1,0x1004,0xfffe7e),(2,0xc,2),(2,0x100c,2),(3,0x4080,0x100),(3,0x4084,0),(3,0x4088,1),
                     (3,4,0xfffe7f),(4,0xc,1),(4,0x100c,1)]
    require([(r['phase'],r['address'],r['value']) for r in writes]==expected_writes,'fixture configuration then exactly two RTL W1C operations')
    for kind,rows in [('READ',reads),('WRITE',writes)]:
        starts,ends=markers(sim,'XDMA_LOCAL_'+kind+'_BEGIN'),markers(sim,'XDMA_LOCAL_'+kind+'_END')
        setup_count=11 if kind=='READ' else 7
        require(len(starts)==setup_count and len(ends)==len(rows),'setup and complete response marker inventory')
        for i,(end,row) in enumerate(zip(ends,rows)):
            require(row['response_ps']==int(end['t']) and int(end['address'],16)==row['address'] and int(end['value'],16)==row['value'] and int(end['phase'])==row.get('response_phase',row['phase']),'actual response marker/VCD')
            if i<setup_count:
                begin=starts[i]
                require(int(begin['t'])<row.get('request_ps',min(row.get('aw_ps',0),row.get('w_ps',0))) and int(begin['address'],16)==row['address'],'setup marker causality')
    a,z=times['BAR_PENDING'],times['BAR_RELEASE']
    rr=[r for r in reads if a<r['request_ps']<r['response_ps']<z]
    ww=[w for w in writes if a<min(w['aw_ps'],w['w_ps'])<=max(w['aw_ps'],w['w_ps'])<w['response_ps']<z]
    require(len(rr)==4 and len(ww)==2 and all(r['phase']==2 for r in rr+ww),'six local operations during actual held BAR response')
    stop_writes=writes[-2:];first_poll=reads[11:17]
    require(max(w['response_ps'] for w in stop_writes)<first_poll[0]['request_ps'],'both native stop B responses precede any status round')
    require(first_poll[-1]['response_ps']<times['H2C_RELEASE']<times['SECOND_RELEASE'],'full initial round before either DMA release')
    require([r['value']&1 for r in first_poll]==[0,1,0,0,0,0],'H2C stopped but busy and C2H idle with zero completed descriptors')
    require(len(done['h2c'])==2 and not done['c2h'],'exact two native H2C Done events and no C2H event')
    require(times['H2C_RELEASE']<done['h2c'][0]<times['SECOND_RELEASE']<done['h2c'][1],'Done follows each separately released response')
    require(states[-1][1]&0x41==states[-1][2]&0x41==0,'final live Run/Busy cleared')
    pattern=bytes(((i*13)^0xa5)&255 for i in range(64))
    second=bytes(((i*7)^0x3c)&255 for i in range(64))
    require(not any(memory) and host_packets==0 and not host_active,'no C2H host write')
    dma=accepted['m_axi'];bar=accepted['m_axib']
    require({k:len(v) for k,v in dma.items()}==dict(aw=2,w=2,b=2,ar=0,r=0),'two complete native H2C writes only')
    for i,(aw,w,b) in enumerate(zip(dma['aw'],dma['w'],dma['b'])):
        require(max(aw[0],w[0])<b[0] and b[1:]==[aw[1],0] and w[3]==1,'DMA response causality and identity')
        require(aw[2:]==[0x1000+i*0x100,0,6] and w[1].to_bytes(64,'little')==[pattern,second][i] and w[2]==2**64-1,'two distinct destinations and all128 payload bytes')
    require(times['H2C_RELEASE']<dma['b'][0][0]<times['REHOLD']<times['SECOND_RELEASE']<dma['b'][1][0],'one and only one B released before rehold')
    require(all(b[0]<t for b,t in zip(dma['b'],done['h2c'])),'every native Done follows its actual B response')
    require(max(x[0] for x in dma['aw']+dma['w'])<stop_writes[0]['response_ps'],'both prefetched writes remain outstanding when Run clears')
    require({k:len(v) for k,v in bar.items()}==dict(aw=2,w=2,b=2,ar=1,r=1),'two BAR writes and one read inventory')
    aw,w,b,ar,r=[bar[c][0] for c in ['aw','w','b','ar','r']]
    require(max(aw[0],w[0])<b[0] and ar[0]<r[0] and b[1:]==[aw[1],0] and r[1]==ar[1] and r[3:]==[0,1] and w[3]==1,'BAR causality and identities')
    require(aw[2:]==ar[2:]==[0x100002000,0,2] and w[1]&0xffffffff==r[2]&0xffffffff==0xd3c2b1a0 and w[2]==15,'BAR AXI bytes/address')
    require(times['BAR_RELEASE']<b[0]<ar[0]<r[0]<=times['BAR_RECEIPT'],'BAR receipt after release')
    aw2,w2,b2=[bar[c][1] for c in ['aw','w','b']]
    require(max(aw2[0],w2[0])<b2[0] and b2[1:]==[aw2[1],0] and w2[3]==1,'second BAR write actual response identity')
    require(aw2[2:]==[0x100002040,0,2] and w2[1]&0xffffffff==0x6f5e4d3c and w2[2]==15,'second BAR write enabled bytes and destination')
    require(times['BAR2_HOLD']<min(aw2[0],w2[0])<times['BAR2_PENDING']<times['H2C_PENDING']<times['SECOND_RELEASE']<times['BAR2_RELEASE']<b2[0]<times['BAR2_EMPTY'],'second BAR obligation spans native descriptor retirement')
    native=review_native_packets(sim,packets,regs,events,accepted,pattern)
    return dict(status='PASS_NATIVE_DIRECT_LOCAL_MANAGEMENT_TRAFFIC',start_ps=start,finish_ps=finish,
                local_reads=reads,local_writes=writes,accepted=accepted,channel_status=states,descriptor_done_ps=done,
                held_cycles=hold_cycles,held_ps={k:times[k+'_RELEASE']-times[k+'_PENDING'] for k in hold_cycles},
                response_stall_cycles=response_stalls,native=native,qualified_global_drain=False,
                qualified_endpoint_reset=False,qualified_physical_design=False,
                scope='Local public XDMA management during three actual retained responses; no OS host memory/coherence, protected global inventory, operational reset or physical throughput approval.')


def review_native_packets(sim, packets, regs, unique, accepted, pattern):
    expected=[(1,4,0xfffe7e),(1,0x1004,0xfffe7e),(2,4,0xfffe7c),(2,0x1004,0xfffe7c)]
    require(len(regs)==len(expected),'native read contract')
    for row,(phase,address,value) in zip(regs,expected):
        actual=int(row['value'],16)
        require((int(row['phase']),int(row['address'],16))==(phase,address),'native read order')
        require((actual&1==value and actual&~0x47==0) if address in (0x40,0x1040) else actual==value,'native register comparison')
    read_requests=[];write_requests=[]
    for t,data,keep,final,user in packets['s_axis_rq']:
        kind=(data>>75)&15
        require(kind in (0,1) and (data>>64)&2047==1 and data&3==0 and final==1 and keep==(15 if kind==0 else 31),'native request framing')
        require((data>>121)&7==0 and (data>>124)&7==0,'root-request TC/attributes')
        (read_requests if kind==0 else write_requests).append([t,data&(2**64-1),(data>>128)&0xffffffff])
    logs=[dict(t=int(r['t']),address=0x10000+int(r['address'],16),value=int(r['value'],16)) for r in regs]
    logs.append(dict(t=int(unique['BAR_RECEIPT']['t']),address=int(unique['BEGIN']['bypass_base'],16)+0x2000,value=0xd3c2b1a0))
    logs.sort(key=lambda r:r['t'])
    require(len(read_requests)==len(packets['m_axis_rc'])==len(logs)==5,'exact native read/completion inventory')
    previous=0
    for request,completion,log in zip(read_requests,packets['m_axis_rc'],logs):
        t,data,keep,final,user=completion
        require(previous<request[0]<t<=log['t'] and request[1]==log['address'],'serial native request correlation')
        require(keep==15 and final==1 and (data>>32)&2047==1 and (data>>43)&15==0 and (data>>96)&0xffffffff==log['value'],'native response bytes/status')
        previous=t
    require([(r[1],r[2]) for r in write_requests]==[(0x10004,0xfffe7e),(int(unique['BEGIN']['bypass_base'],16)+0x2000,0xd3c2b1a0),(int(unique['BEGIN']['bypass_base'],16)+0x2040,0x6f5e4d3c)],'native posted control and BAR write')
    controls=markers(sim,'XDMA_LOCAL_WRITE')
    require(len(controls)==1 and int(controls[0]['address'],16)==4 and int(controls[0]['value'],16)==0xfffe7e,'native control log')
    cq=packets['m_axis_cq'];cc=packets['s_axis_cc']
    require(len(cq)==3 and len(cc)==6,'exact native descriptor/payload fetch and completion inventory')
    desc=bytearray(64)
    for index,source,destination,next_ptr,control in [(0,0x400,0x1000,0x120,0xad4b0012),(1,0x500,0x1100,0,0xad4b0013)]:
        for offset,value in [(0,control),(4,64),(8,source),(16,destination),(24,next_ptr)]:
            desc[index*32+offset:index*32+offset+4]=value.to_bytes(4,'little')
    payloads={0x100:bytes(desc),0x400:pattern,0x500:bytes(((i*7)^0x3c)&255 for i in range(64))}
    host_reads=[]
    for index,(request,address) in enumerate(zip(cq,[0x100,0x400,0x500])):
        t,data,keep,final,user=request;group=cc[2*index:2*index+2];header=group[0][1]
        require(data&(2**64-1)==address and (data>>64)&2047==16 and (data>>75)&15==0 and keep==15 and final==1 and user&65535==0xf0f,'host descriptor/payload fetch')
        require((data>>121)&7==0 and (data>>124)&7==2,'host fetch attributes')
        tag,requester=(data>>96)&255,(data>>80)&65535
        require(t<group[0][0]<group[1][0] and (header>>64)&255==tag and (header>>48)&65535==requester,'CC identity and causal response')
        require(header&127==0 and (header>>16)&4095==64 and (header>>32)&2047==16 and (header>>43)&31==0 and (header>>31)&1==0 and (header>>88)&1==1 and (header>>89)&7==0 and (header>>92)&7==2,'CC header')
        require([g[2:] for g in group]==[[65535,0,1],[7,1,0x240]],'CC segmentation')
        actual=group[0][1].to_bytes(64,'little')[12:]+group[1][1].to_bytes(64,'little')[:12]
        require(actual==payloads[address],'all64 descriptor bytes and128 seed bytes')
        host_reads.append(dict(address=address,bytes=64,request_ps=t,completion_ps=group[-1][0],tag=tag))
    dma=accepted['m_axi']
    require(host_reads[0]['completion_ps']<min(r[0] for r in dma['aw']),'both descriptors fetched before AXI addresses')
    for i in range(2):
        require(host_reads[i+1]['completion_ps']<dma['w'][i][0]<dma['b'][i][0],'payload fetch to actual AXI write/response')
    require(max(r['completion_ps'] for r in host_reads)<int(unique['H2C_RELEASE']['t']),'both payloads prefetched before first release')
    starts,beats,ends=[markers(sim,'NTRPU_HOST_CPL_'+n) for n in ['BEGIN','BEAT','END']]
    require(len(starts)==len(ends)==3 and len(beats)==6,'completion driver log inventory')
    require([int(x['offset']) for x in beats]==[0,52]*3,'completion log offsets')
    for log,actual in zip(beats,cc):
        require([int(log['t']),int(log['data'],16),int(log['keep'],16),int(log['last']),int(log['user'],16)]==actual,'CC log/VCD identity')
    for begin,end,item in zip(starts,ends,host_reads):
        require(int(begin['address'],16)==item['address'] and int(begin['tag'],16)==int(end['tag'],16)==item['tag'] and int(begin['bytes'])==int(end['bytes'])==64 and int(begin['attr'],16)==2 and int(begin['tc'],16)==0,'completion driver identity')
        require(item['request_ps']<=int(begin['t'])<item['completion_ps']<int(end['t']),'completion driver lifetime')
    require(not markers(sim,'XDMA_LOCAL_HOST_BEAT'),'no host posted write log')
    return dict(read_requests=5,posted_writes=3,host_reads=host_reads,host_write_beats=0,host_received_bytes=0,
                register_correlation='serial one-outstanding public RQ/RC; core allocated tags not traced')


def review_discovery(sim, vcd, result):
    start = result['start_ps']
    ep, tb, rp = '/board/EP/', '/board/ntrpu_stop_probe/', '/board/RP/'
    ar, r, requests, completions = [], [], [], []
    held = False
    for t, before, after in vcd_rows(vcd):
        if t >= start:
            require(after[tb+'init_ar_count'] == after[tb+'init_r_count'] == 1 and after[tb+'init_rvalid'] == 0,
                    'completed discovery remains quiescent')
            continue
        if before[ep+'user_resetn'] != 1:
            continue
        if before[ep+'user_clk'] == 0 and after[ep+'user_clk'] == 1:
            require(before[ep+'m_axil_awvalid'] == 0, 'no discovery write')
            if held:
                require(before[ep+'m_axil_rvalid'] == 1 and before[ep+'m_axil_rdata'] == 0 and before[ep+'m_axil_rresp'] == 0,
                        'stable discovery read response')
            if before[ep+'m_axil_arvalid'] == before[ep+'m_axil_arready'] == 1:
                require(before[ep+'m_axil_araddr'] == 0 and before[tb+'phase'] == 0, 'bounded discovery AR')
                ar.append(t)
            if before[ep+'m_axil_rvalid'] == 1:
                require(before[ep+'m_axil_rdata'] == before[ep+'m_axil_rresp'] == 0, 'discovery zero read value')
                held = before[ep+'m_axil_rready'] != 1
                if not held:
                    r.append(t)
            require(after[tb+'init_ar_count'] == len(ar) and after[tb+'init_r_count'] == len(r), 'discovery counters')
        if before[rp+'user_clk'] == 0 and after[rp+'user_clk'] == 1:
            for prefix, rows in [('s_axis_rq', requests), ('m_axis_rc', completions)]:
                p = rp+prefix+'_'
                if before[p+'tvalid'] == 1 and before[p+'tready'] is not None and before[p+'tready'] & 1:
                    rows.append([t]+[before[p+f] for f in ['tdata','tkeep','tlast']])
    require(len(ar) == len(r) == 1 and ar[0] < r[0] < start and not held, 'one complete discovery transaction')
    for name, actual in [('AR', ar), ('R', r)]:
        logs = markers(sim, 'XDMA_LOCAL_DISCOVERY_'+name)
        require(len(logs) == 1 and int(logs[0]['t']) == actual[0], 'discovery marker/VCD identity')
    reads = [row for row in requests if (row[1] >> 75) & 15 == 0]
    require(len(reads) == 2 and [row[1] & (2**64-1) for row in reads] == [0,0x10000], 'BAR0 then native BAR1 discovery')
    require(reads[0][0] < ar[0] < r[0] < reads[1][0], 'native discovery read causality')
    replies = [row for row in completions if r[0] < row[0] < reads[1][0]]
    require(len(replies) == 1, 'one native discovery completion')
    _, data, keep, last = replies[0]
    require(keep == 15 and last == 1 and (data >> 32) & 2047 == 1 and (data >> 43) & 15 == 0 and (data >> 96) & 0xffffffff == 0,
            'native discovery completion bytes/status')
    result['discovery'] = dict(ar_ps=ar[0], r_ps=r[0], completion_ps=replies[0][0], returned_word=0,
                               scope='One public BAR0 zero-word response used only for AMD enumeration; not a production BAR0 memory implementation.')
    return result


def review_ledger(sim, vcd, native):
    p = '/board/ntrpu_stop_probe/'
    images = []
    for index,(source, dest) in enumerate([(0x400, 0x1000), (0x500, 0x1100)]):
        raw = bytearray(32)
        for offset, value in [(0, 0xad4b0012 if index==0 else 0xad4b0013), (4, 64), (8, source), (16, dest), (24,0x120 if index==0 else 0)]:
            raw[offset:offset+4] = value.to_bytes(4, 'little')
        images.append(int.from_bytes(raw, 'little'))
    active = closed = faults = started = 0
    published, completed, previous_run = [0, 0], [0, 0], [0, 0]
    publications, done, arm_times, revoked, accounted_times = [], [[], []], [], [], []
    held_samples = {'zero': 0, 'one': 0}
    checkpoints = {int(row['t']): row for row in markers(sim, 'XDMA_STOP_CHAIN_ACCOUNTED')}
    require(len(checkpoints) == 1, 'one logged descriptor accounting point')
    csr = {row['response_ps']: row for row in native['local_reads'] if row['address'] in (0x48, 0x1048)}
    csr_compared = []
    seen_checkpoints = 0
    for t, before, after in vcd_rows(vcd):
        if t < native['start_ps']:
            continue
        a = {name[len(p):]: value for name, value in after.items() if name.startswith(p)}
        b = {name[len(p):]: value for name, value in before.items() if name.startswith(p)}
        observed = [n for n in a if n.startswith('ledger_') and n not in ('ledger_first_image', 'ledger_second_image')]
        require(all(a[n] is not None for n in observed), 'known ledger interface')
        require(a['ledger_ready'] == 1, 'native readiness remains asserted')
        edge = before['/board/EP/user_clk'] == 0 and after['/board/EP/user_clk'] == 1
        if edge:
            if b['ledger_arm_valid'] and b['ledger_arm_ready']:
                require(not active and not arm_times, 'one native ledger session')
                active = 1
                arm_times.append(t)
            if b['ledger_publish_valid']:
                c = b['ledger_publish_channel']
                index=published[0]
                require(active and c==0 and index<2 and b['ledger_publish_index']==index,'two ordered owned H2C publications')
                require(b['ledger_publish_address'] == (0x100+32*index), 'actual descriptor address')
                require(b['ledger_publish_image'] == images[index] == b['ledger_' + ('first' if index==0 else 'second') + '_image'],
                        'notification contains actual complete host descriptor bytes')
                require(not (b['h2c_status'] & 0x41 or b['c2h_status'] & 0x41), 'publication precedes engine enable')
                published[c] += 1
                publications.append(dict(t=t, channel=c, index=index, address=b['ledger_publish_address'],
                                         image=images[index], generation=31, tag=301))
            for c, status in enumerate(('h2c_status', 'c2h_status')):
                run = (b[status] >> 6) & 1
                if run and not previous_run[c]:
                    require(c==0 and published[c] == 2 and not (started & (1 << c)) and not closed, 'one published native Run lifetime')
                    started |= 1 << c
                if b[status] & 8:
                    require(active and started & (1 << c) and completed[c] < published[c], 'Done requires owned native descriptor')
                    completed[c] += 1
                    done[c].append(t)
                previous_run[c] = run
            if closed and (not b['ledger_close_request'] or not b['ledger_host_closed'] or
                           (b['ledger_host_generation'], b['ledger_host_tag']) != (31, 301)):
                if not faults:
                    revoked.append(t)
                faults |= 256
            if active and b['ledger_close_request'] and b['ledger_host_closed']:
                require((b['ledger_host_generation'], b['ledger_host_tag']) == (31, 301), 'current fixture closure identity')
                closed = 1
        require(a['ledger_active'] == active and a['ledger_closed'] == closed, 'retained native ledger lifecycle')
        require(a['ledger_faults'] == faults and a['ledger_started'] == started, 'native ledger faults and Run association')
        require((a['ledger_generation'], a['ledger_tag']) == ((31, 301) if active else (0, 0)), 'retained ledger identity')
        for c in (0, 1):
            for field, wanted in [('published', published[c]), ('completed', completed[c]),
                                  ('outstanding', published[c]-completed[c])]:
                require((a['ledger_' + field] >> (32*c)) & 0xffffffff == wanted, 'event-derived native ' + field)
        for index in range(published[0]):
            require(a['ledger_'+('first' if index==0 else 'second')+'_image']==images[index],'published descriptor bytes never change')
        idle = not (a['h2c_status'] & 0x41 or a['c2h_status'] & 0x41)
        accounted = int(active and closed and not faults and a['ledger_close_request'] and a['ledger_host_closed'] and
                        (a['ledger_host_generation'], a['ledger_host_tag']) == (31, 301) and idle and
                        not a['ledger_publish_valid'] and not (a['h2c_status'] & 8 or a['c2h_status'] & 8) and published == completed)
        require(a['ledger_accounted'] == accounted, 'continuous descriptor-only accounting predicate')
        require(a['ledger_arm_ready'] == int(not active and not faults and idle), 'native ledger arming predicate')
        if accounted:
            accounted_times.append(t)
        if a['hold_dma_b'] and published[0]==2 and completed[0]<2:
            require(not accounted,'held native descriptor cannot be retired')
            held_samples['zero' if completed[0]==0 else 'one']+=1
        if t in csr:
            row = csr[t]
            c = int(row['address'] == 0x1048)
            require(row['value']<=completed[c], 'native CSR count never precedes the corresponding Done event')
            prior=[r['count'] for r in csr_compared if r['channel']==c]
            require(not prior or row['value']>=prior[-1], 'native completed CSR count is monotonic')
            csr_compared.append(dict(channel=c, t=t, count=row['value'],observed_done=completed[c]))
        if t in checkpoints:
            row = checkpoints[t]
            # The receipt is logged before same-timestamp phase7/host withdrawal.
            checkpoint_sample=b if b['phase']==6 and a['phase']==7 and not a['ledger_host_closed'] else a
            require(checkpoint_sample['ledger_accounted'] and checkpoint_sample['ledger_published']==checkpoint_sample['ledger_completed']==2,
                    'final H2C accounting checkpoint retains actual descriptor accounting')
            require(int(row['host_bytes'])==checkpoint_sample['host_bytes']==0,'no C2H receipt asserted by this H2C test')
            seen_checkpoints += 1
    parsed = [{k: int(v, 10 if k == 't' else 2 if k == 'channel' else 16) for k, v in row.items()}
              for row in markers(sim, 'XDMA_LEDGER_PUBLISH')]
    require(parsed == publications and [r['channel'] for r in publications] == [0, 0], 'publication log/VCD correspondence')
    require(done == [native['descriptor_done_ps']['h2c'], native['descriptor_done_ps']['c2h']], 'native Done event correspondence')
    require([len(rows) for rows in done] == [2, 0] and len(arm_times) == len(revoked) == seen_checkpoints == 1,
            'complete native ledger event coverage')
    require(len(csr_compared) == len(csr) and all(n > 20 for n in held_samples.values()), 'CSR and retained descriptor observations')
    require(accounted_times and revoked[0] > accounted_times[-1] and faults == 256 and not accounted and active,
            'approval revoked permanently while identity and counts remain retained')
    for publication in publications:
        fetch = next(row for row in native['native']['host_reads'] if row['address'] == 0x100)
        require(publication['t'] < fetch['request_ps'] < fetch['completion_ps'] < done[0][publication['index']],
                'published bytes fetched through native PCIe before descriptor completion')
    return dict(publications=publications, done_ps=done, csr_comparisons=csr_compared,
                armed_ps=arm_times[0], held_observation_samples=held_samples,
                first_accounted_ps=accounted_times[0], revocation_fault_ps=revoked[0],
                final_faults=faults, descriptor_bytes_verified=64, fixture_closure_only=True,
                publication_to_done_ps=[done[0][i]-publications[i]['t'] for i in (0, 1)])


def review_master(sim,vcd,result):
    p='/board/ntrpu_stop_probe/'
    require(all(re.search(r'\$var \w+ \d+ \S+ '+n+r'(?: |\[)',vcd.split('$enddefinitions $end')[0])
                for n in ['master_req_valid','master_req_identity','master_rsp_valid','master_rsp_identity','master_busy']),
            'synthesizable master public trace required')
    transactions=sorted([dict(r,write=0) for r in result['local_reads']]+[dict(w,write=1) for w in result['local_writes']],
                        key=lambda r:r.get('request_ps',r.get('aw_ps')))
    total=len(transactions)
    require(total>=42,'native setup and autonomous transaction inventory')
    pending=None;requests=[];reports=[];measurements=[]
    for t,before,after in vcd_rows(vcd):
        if t<result['start_ps']:continue
        require(after[p+'master_timeout_latched']==after[p+'master_rsp_timeout']==0,'no unreported master timeout')
        edge=before['/board/EP/user_clk']==0 and after['/board/EP/user_clk']==1
        if edge:
            if before[p+'master_req_valid'] and before[p+'master_req_ready']:
                require(pending is None and len(requests)<total,'one master owner at a time')
                row=transactions[len(requests)];serial=len(requests)+1
                owner=before[p+'sequencer_owner']
                identity=(31<<64)|301 if owner else (0x4e545250554d4754<<64)|(row['phase']<<32)|serial
                request=dict(t=t,phase=before[p+'phase'],write=before[p+'master_req_write'],
                             address=before[p+'master_req_address'],data=before[p+'master_req_data'],
                             strobe=before[p+'master_req_strobe'],identity=before[p+'master_req_identity'])
                require([request[k] for k in ['write','address','data','strobe','identity']]==
                        [row['write'],row['address'],row['value'] if row['write'] else 0,15 if row['write'] else 0,identity],
                        'request/native register provenance')
                require(t<row.get('request_ps',min(row.get('aw_ps',0),row.get('w_ps',0))),'RTL request precedes native AXI request')
                requests.append(request)
                pending=dict(row=row,request=request,bus_seen=False,stalls=0,closed=False,owner=owner)
            if before[p+'master_rsp_valid'] and not before[p+'master_rsp_ready']:
                require(pending is not None and pending['bus_seen'],'no invented report under backpressure')
                pending['stalls']+=1
            if before[p+'master_rsp_valid'] and before[p+'master_rsp_ready']:
                require(pending is not None and pending['bus_seen'],'report after real response and admission close')
                require(pending['stalls']==0 if pending['owner'] else pending['stalls']>=3 and pending['closed'],'autonomous reports consumed; fixture reports actually retained after admission closes')
                row=pending['row'];req=pending['request']
                report=dict(t=t,phase=before[p+'phase'],data=before[p+'master_rsp_data'],code=before[p+'master_rsp_code'],
                            identity=before[p+'master_rsp_identity'],timeout=before[p+'master_rsp_timeout'])
                require([report[k] for k in ['data','code','identity','timeout']]==
                        [0 if row['write'] else row['value'],0,req['identity'],0],'native report bytes/error/identity')
                reports.append(report)
                measurements.append(dict(serial=len(reports),phase=req['phase'],write=req['write'],address=req['address'],
                    request_ps=req['t'],native_response_ps=row['response_ps'],report_ps=t,
                    request_to_native_ps=row['response_ps']-req['t'],report_stall_cycles=pending['stalls']))
                pending=None
            for channel in ('b','r'):
                if before[p+'local_'+channel+'valid'] and before[p+'local_'+channel+'ready']:
                    require(pending is not None and not pending['bus_seen'],'one real bus response per RTL request')
                    require(pending['row']['response_ps']==t and pending['row']['write']==int(channel=='b'),'native/RTL response binding')
                    pending['bus_seen']=True
        require(after[p+'master_requests']==len(requests) and after[p+'master_reports']==len(reports),'master counters/VCD')
        require(after[p+'master_busy']==int(pending is not None),'master ownership until report consumed')
        require(after[p+'master_req_ready']==int(after[p+'master_admit'] and pending is None),'admission closes without dropping owner')
        require(after[p+'master_rsp_valid']==int(pending is not None and pending['bus_seen']),'report follows actual B/R only')
        if pending:
            pending['closed'] |= after[p+'master_admit']==0
            if pending['bus_seen']:
                row=pending['row'];req=pending['request']
                require([after[p+n] for n in ['master_rsp_data','master_rsp_code','master_rsp_identity','master_rsp_timeout']]==
                        [0 if row['write'] else row['value'],0,req['identity'],0],'continuous report retention')
    require(pending is None and len(requests)==len(reports)==total,'all native-master obligations retired')
    for name,expected in [('REQ',requests),('REPORT',reports)]:
        actual=[{k:int(v,10 if k in ('t','phase','write','timeout') else 16) for k,v in r.items()}
                for r in markers(sim,'XDMA_MASTER_'+name)]
        require(actual==expected,'master log/VCD correspondence '+name)
    result.update(status='PASS_SYNTHESIZABLE_LOCAL_MASTER_NATIVE_XDMA',master_transactions=measurements,
        qualified_native_master_bounded=True,
        scope='Direct synthesizable master drives all actual native register transactions while BAR and both H2C responses are held; retained tagged reports checked. No driver certificate, protected global inventory, operational reset, physical timing or inference qualification.')
    return result


def review_sequencer(sim,vcd,native):
    """Derive the local observation from actual tagged responses, not DUT state."""
    p='/board/ntrpu_stop_probe/';ep='/board/EP/'
    order=[0xc,0x100c,4,0x40,0x48,0x1004,0x1040,0x1048]
    identity=(31<<64)|301
    mode='idle';active=0;faults=0;cursor=0;stops=mask=0;values=[0]*6
    requests=reports=0;starts=[];captures=[];rounds=[];handoffs=[];revocations=[]
    old_valid=0;start_ps=native['start_ps'];report_values=[];last_report_ps=None
    for t,raw_b,raw_a in vcd_rows(vcd):
        if t<start_ps:continue
        b={k[len(p):]:v for k,v in raw_b.items() if k.startswith(p)}
        a={k[len(p):]:v for k,v in raw_a.items() if k.startswith(p)}
        require(all(v is not None for k,v in a.items() if k.startswith('seq_') or k=='sequencer_owner'),'known sequencer public interface')
        if not b['sequencer_owner'] and a['sequencer_owner']:
            require(not handoffs and not b['master_busy'] and not b['master_req_valid'] and not b['master_rsp_valid'],'exclusive handoff only at actual idle')
            require(a['ledger_closed'] and a['ledger_published']==a['ledger_outstanding']==2,'handoff retains both closed descriptor obligations')
            handoffs.append(t)
        if handoffs:require(a['sequencer_owner']==1,'fixture cannot reclaim register ownership')
        edge=raw_b[ep+'user_clk']==0 and raw_a[ep+'user_clk']==1
        if edge:
            if active:
                if not b['ledger_host_closed'] or not b['ledger_close_request']:
                    faults|=4;revocations.append(t)
                if b['ledger_faults'] or b['ledger_published']!=2:faults|=8
                if mode=='captured' and (not b['ledger_accounted'] or b['ledger_completed']!=2):faults|=8192
            if b['seq_start_valid'] and b['seq_start_ready']:
                require(not active and len(handoffs)==1 and not starts,'one coherent sequencer session')
                active=1;mode='send';starts.append(t)
            elif mode=='assess':
                quiet=(stops==3 and mask==63 and not (values[0]&1 or values[3]&1 or values[1]&1 or values[4]&1)
                       and values[2]==2 and values[5]==0 and b['ledger_accounted'] and b['ledger_completed']==2)
                require(not faults,'no fault before deliberate closure withdrawal')
                rounds.append(dict(assessment_ps=t,values=values.copy(),qualified=bool(quiet)))
                if quiet:mode='captured'
                else:mode='send';cursor=2;mask=0
            if b['sequencer_owner'] and b['master_req_valid'] and b['master_req_ready']:
                require(active and mode=='send','single owned autonomous request')
                require([b['master_req_address'],b['master_req_write'],b['master_req_data'],b['master_req_strobe'],b['master_req_identity']]==
                        [order[cursor],int(cursor<2),int(cursor<2),15 if cursor<2 else 0,identity],'exact autonomous W1C or nonclearing CSR request')
                if cursor==2:mask=0
                requests+=1;mode='receive'
            if b['sequencer_owner'] and b['master_rsp_valid'] and b['master_rsp_ready']:
                require(mode=='receive' and reports<requests,'one actual report per autonomous request')
                require(b['master_rsp_identity']==identity and b['master_rsp_code']==b['master_rsp_timeout']==0,'native tagged successful report')
                value=b['master_rsp_data'];report_values.append(dict(t=t,index=cursor,value=value));last_report_ps=t
                if cursor<2:stops|=1<<cursor
                else:values[cursor-2]=value;mask|=1<<(cursor-2)
                reports+=1
                if cursor==7:mode='assess'
                else:cursor+=1;mode='send'
        require(a['seq_active']==active and a['seq_faults']==faults,'retained autonomous lifecycle and exact sticky revocation faults')
        require((a['seq_generation'],a['seq_tag'],a['seq_expected'])==((31,301,2) if active else (0,0,0)),'retained coherent identity and publication count')
        require(a['seq_stops']==stops and a['seq_reads']==mask,'only actual reports populate response masks')
        require([a['seq_control'],a['seq_status'],a['seq_count']]==[values[0]|values[3]<<32,values[1]|values[4]<<32,values[2]|values[5]<<32],'all six snapshot words equal native report data')
        require(a['seq_req_valid']==int(mode=='send') and a['seq_rsp_ready']==int(mode=='receive') and a['seq_pending']==int(mode in ('send','receive')),'request/report obligations retained')
        require(a['seq_admit']==active and a['seq_requests']==requests and a['seq_reports']==reports,'autonomous admission and event counters')
        valid=int(active and mode=='captured' and not faults and a['ledger_host_closed'] and a['ledger_close_request'] and
                  not a['ledger_faults'] and a['ledger_accounted'] and a['ledger_completed']==2 and not a['seq_start_valid'] and not a['master_rsp_valid'])
        require(a['seq_snapshot_valid']==valid,'continuous local snapshot gate, distinct from host receipt')
        if valid and not old_valid:captures.append(dict(t=t,host_bytes=a['host_bytes'],last_report_ps=last_report_ps))
        old_valid=valid
        if a['sequencer_owner']:
            for suffix in ('admit','req_valid','req_write','req_address','req_data','req_strobe','req_identity','rsp_ready'):
                require(a['master_'+suffix]==a['seq_'+suffix],'exclusive public register path '+suffix)
    final=markers(sim,'XDMA_DMA_BAR_NATIVE_PASS')[0]
    require(len(starts)==len(captures)==len(handoffs)==1 and revocations and requests==reports==int(final['seq_requests'])==int(final['seq_reports']),'complete autonomous lifetime')
    require(len(rounds)>=4 and all(not r['qualified'] for r in rounds[:-1]) and rounds[-1]['qualified'],'poll until all local obligations observed quiet')
    require(any(r['values'][2]==0 and not r['qualified'] for r in rounds) and any(r['values'][2]==1 and not r['qualified'] for r in rounds),'zero and partial descriptor counts never qualify')
    require(faults==0x200c and active and not old_valid and mode=='captured','restoring host level never restores approval')
    snapshot=markers(sim,'XDMA_STOP_REG_SNAPSHOT');require(len(snapshot)==1,'single logged snapshot')
    require(int(snapshot[0]['t'])==captures[0]['t']+4000 and int(snapshot[0]['count'],16)==2 and int(snapshot[0]['control'],16)==0xfffe7c00fffe7e and snapshot[0]['global_drain']=='0','snapshot marker follows independently reconstructed capture')
    receipt=markers(sim,'XDMA_STOP_CHAIN_ACCOUNTED');require(len(receipt)==1 and int(receipt[0]['host_bytes'])==0 and int(receipt[0]['reports'])==reports,'separate final accounting marker with no C2H receipt claim')
    require(starts[0]<captures[0]['t']<int(receipt[0]['t'])<revocations[0],'capture, host receipt check and deliberate revocation order')
    return dict(handoff_ps=handoffs[0],start_ps=starts[0],capture=captures[0],requests=requests,reports=reports,rounds=rounds,
                revocation_ps=revocations[0],final_faults=faults,report_values=report_values,
                start_to_snapshot_ps=captures[0]['t']-starts[0],host_receipt_separate=True,c2h_tested=False)


def review(sim,vcd):
    native=review_traffic(sim,vcd)
    review_discovery(sim,vcd,native)
    review_master(sim,vcd,native)
    native['ledger']=review_ledger(sim,vcd,native)
    native['sequencer']=review_sequencer(sim,vcd,native)
    native['axi_inventory']=review_inventory(sim,vcd,native)
    native.update(status='PASS_NATIVE_DMA_BAR_INVENTORY_COMPOSITION',
                  scope='Native DMA/BAR ports feed two production AXI4 inventory observers. Second BAR B remains outstanding after the H2C descriptor snapshot, and only its real response empties the inventory. No source fence is supplied, so ACK remains absent even when empty. Two descriptors/128 bytes, real tagged W1C/CSR and sticky closure revocation preserved; no C2H/driver/global/reset/physical/inference qualification.')
    return native


def review_inventory(sim,vcd,native):
    p='/board/ntrpu_stop_probe/';ep='/board/EP/';rp='/board/RP/'
    fields={'aw':'awaddr awid awlen awsize awprot awburst awcache awlock'.split(),
            'w':'wdata wstrb wlast'.split(),'b':'bid bresp'.split(),
            'ar':'araddr arid arlen arsize arprot arburst arcache arlock'.split(),
            'r':'rdata rid rresp rlast'.split()}
    observed={k:{c:[] for c in fields} for k in ['dma','bar']}
    stalls={k:{} for k in observed};aw_queues={k:[] for k in observed};w_queues={k:[] for k in observed}
    reads={k:[] for k in observed};writes={k:[] for k in observed}
    captured=False;saved_tag=0;closures=[];host_requests=[]
    marks={};logged={}
    for name in ['PENDING','SNAPSHOT_BLOCKED','RELEASE','EMPTY']:
        rows=markers(sim,'XDMA_DMA_BAR_'+name);require(len(rows)==1,'unique inventory '+name)
        marks[name]=int(rows[0]['t']);logged[name]=rows[0]
    samples={};overlap=[];last=None;peaks={k:[0,0,0,0] for k in observed}
    for t,b,a in vcd_rows(vcd):
        if t<native['start_ps']:continue
        require(a[p+'inventory_source_fenced']==0,'no fabricated upstream source fence')
        require(a[p+'inventory_request']==a[p+'ledger_close_request'],'inventory request follows the local closure request')
        if not b[p+'ledger_host_closed'] and a[p+'ledger_host_closed']:closures.append(t)
        edge=b[ep+'user_clk']==0 and a[ep+'user_clk']==1
        if edge:
            if not b[p+'inventory_request']:captured=False
            elif not captured or saved_tag!=b[p+'ledger_tag']:captured=True;saved_tag=b[p+'ledger_tag']
        for label,port in [('dma','m_axi'),('bar','m_axib')]:
            o=p+label+'_obs_';bus=ep+port+'_'
            require(a[bus+'rlast']==a[ep+'ntrpu_xdma_i/'+port+'_rlast'],'observer RLAST is the same public XDMA input as the implicit example net')
            if edge:
                for c,ns in fields.items():
                    valid,ready=b[bus+c+'valid'],b[bus+c+'ready']
                    require(valid in (0,1) and ready in (0,1),'known inventory handshake')
                    payload=[b[bus+n] for n in ns]
                    if c in stalls[label]:require(valid==1 and payload==stalls[label][c],'native stalled full payload remains stable')
                    if valid:
                        require(None not in payload,'known native inventory payload')
                        if ready:observed[label][c].append([t,*payload]);stalls[label].pop(c,None)
                        else:stalls[label][c]=payload
                if b[bus+'awvalid'] and b[bus+'awready']:
                    require(b[bus+'awlen']==0,'bounded single-beat native write')
                    aw_queues[label].append((b[bus+'awid'],t))
                if b[bus+'wvalid'] and b[bus+'wready']:
                    require(b[bus+'wlast']==1,'bounded completed W beat')
                    w_queues[label].append(t)
                if b[bus+'bvalid'] and b[bus+'bready']:
                    require(aw_queues[label] and w_queues[label],'B owns both accepted AW and complete W')
                    ident,aw_time=aw_queues[label].pop(0);w_time=w_queues[label].pop(0)
                    require(ident==b[bus+'bid'] and max(aw_time,w_time)<t and b[bus+'bresp']==0,'native B identity and successful retirement')
                    writes[label].append(dict(aw_ps=aw_time,w_ps=w_time,b_ps=t,identity=ident))
                if b[bus+'arvalid'] and b[bus+'arready']:
                    require(b[bus+'arlen']==0,'bounded single-beat native read')
                    reads[label].append((b[bus+'arid'],t))
                if b[bus+'rvalid'] and b[bus+'rready']:
                    require(reads[label],'R has a real AR owner')
                    ident,ar_time=reads[label].pop(0)
                    require(ident==b[bus+'rid'] and ar_time<t and b[bus+'rresp']==0 and b[bus+'rlast']==1,'native R identity and successful retirement')
            expected=[len(aw_queues[label]),len(w_queues[label]),len(reads[label]),0]
            actual=[a[o+n] for n in ['aw_pending','w_pending','ar_pending','w_partial']]
            require(actual==expected,'continuous native '+label+' outstanding inventory')
            peaks[label]=[max(old,new) for old,new in zip(peaks[label],expected)]
            require(a[o+'fault']==a[o+'fault_sources']==0,'native observer protocol evidence remains valid')
            require(a[o+'ack']==0 and a[o+'tag']==saved_tag,'tagged monitor never certifies an absent source fence')
        if b[rp+'user_clk']==0 and a[rp+'user_clk']==1:
            if b[rp+'s_axis_rq_tvalid'] and b[rp+'s_axis_rq_tready']&1:host_requests.append(t)
        if a[p+'seq_snapshot_valid'] and len(aw_queues['bar'])==1:
            require(a[p+'ledger_accounted'] and a[p+'ledger_completed']==2 and not aw_queues['dma'] and not w_queues['dma'],'descriptor snapshot coexists with outstanding BAR response')
            overlap.append(t)
        if t in marks.values():
            samples[t]=[{n[len(p):]:v for n,v in state.items() if n.startswith(p)} for state in (b,a)]
        last=a
    require(closures and host_requests and max(host_requests)<closures[0],'all actual host RQ requests precede producer closure')
    require(all(not q for q in [*aw_queues.values(),*w_queues.values(),*reads.values()]) and all(not x for x in stalls.values()),'all actual public AXI transactions finally retired')
    require([len(writes[k]) for k in ['dma','bar']]==[2,2],'two retired writes per native interface')
    require(peaks==dict(dma=[2,2,0,0],bar=[1,1,1,0]),'observed native occupancy profile')
    at=samples[marks['SNAPSHOT_BLOCKED']][1]
    require(at['seq_snapshot_valid']==at['ledger_accounted']==1 and at['bar_obs_aw_pending']==at['bar_obs_w_pending']==1 and at['dma_obs_aw_pending']==at['dma_obs_w_pending']==0,'direct contradictory local-snapshot/BAR-outstanding witness')
    # EMPTY is printed before the fixture withdraws host closure at this same
    # negedge. Check its pre-edge observation; the sequencer review checks the
    # immediate withdrawal and subsequent sticky fault in the resulting state.
    empty=samples[marks['EMPTY']][0]
    require(empty['bar_obs_aw_pending']==empty['bar_obs_w_pending']==empty['bar_obs_ack']==empty['dma_obs_ack']==0 and empty['seq_snapshot_valid']==1,'empty AXI still cannot certify without source fencing')
    require(logged['PENDING']['aw_pending']==logged['PENDING']['w_pending']=='1','pending marker agrees with native inventory')
    require({k:logged['SNAPSHOT_BLOCKED'][k] for k in ['dma_aw_pending','dma_w_pending','bar_aw_pending','bar_w_pending','snapshot','source_fenced']}==dict(dma_aw_pending='0',dma_w_pending='0',bar_aw_pending='1',bar_w_pending='1',snapshot='1',source_fenced='0'),'snapshot marker inventory agrees with VCD')
    require({k:logged['EMPTY'][k] for k in ['bar_aw_pending','bar_w_pending','source_fenced','monitor_ack']}==dict(bar_aw_pending='0',bar_w_pending='0',source_fenced='0',monitor_ack='00'),'empty marker agrees with missing-fence observation')
    capture=native['sequencer']['capture']['t'];bar_b=writes['bar'][1]['b_ps']
    require(marks['PENDING']<capture<=marks['SNAPSHOT_BLOCKED']<marks['RELEASE']<bar_b<marks['EMPTY'],'BAR response independently gates inventory after local snapshot')
    require(overlap and overlap[0]==capture,'observed gap begins at actual descriptor snapshot')
    require(last[p+'seq_faults']==0x200c and last[p+'ledger_faults']==0x100,'descriptor closure revocation retained after AXI inventory empties')
    return dict(observed=observed,write_lifetimes=writes,markers_ps=marks,
                snapshot_ps=capture,bar_b_ps=bar_b,snapshot_to_bar_b_ps=bar_b-capture,
                peak_inventory=dict(zip(peaks,[dict(zip(['aw','w','ar','partial'],v)) for v in peaks.values()])),
                enabled_write_bytes={label:sum(row[2].bit_count() for row in observed[label]['w']) for label in observed},
                observer_profile=dict(id_width=4,write_depth=32,read_depth_per_id=8,data_bits=512,native_max_burst_beats=1),
                overlap_observation_samples=len(overlap),source_fence_supplied=False,
                monitor_ack_assertions=0,host_last_request_ps=max(host_requests),host_first_closure_ps=closures[0],
                qualified_global_drain=False,scope='Exact native AXI inventory with missing source fences; no global certificate.')
