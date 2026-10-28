"""Independent native CQ/CC, register and AXI reconstruction for a two-descriptor stop."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import vcd_rows, markers, require


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)',sim,re.M),'simulator failure')
    begin,end=markers(sim,'XDMA_CHAIN_BEGIN'),markers(sim,'XDMA_CHAIN_PASS')
    require(len(begin)==len(end)==1,'unique begin/pass')
    begin,end=begin[0],end[0];start,finish=int(begin['t']),int(end['t'])
    count=int(end['completed'])
    require(90000000<start<finish<180000000 and count in (1,2),'bounded experiment')
    require({k:begin[k] for k in ['dma_bar','descriptors','adjacent']}==dict(dma_bar='1',descriptors='2',adjacent='1'),'initial scope')
    require({k:end[k] for k in ['descriptors','payload_bytes','reset_tested','c2h_tested']}==
            dict(descriptors='2',payload_bytes=str(count*64),reset_tested='0',c2h_tested='0'),'final scope')
    require('$finish called at time' in sim,'simulator finish')
    pending,release,settled=[markers(sim,'XDMA_CHAIN_'+x) for x in ['PENDING','RELEASE','SETTLED']]
    require(len(pending)==len(release)==len(settled)==1,'unique lifetime markers')
    pending,release,settled=pending[0],release[0],settled[0]
    pt,rt,st=[int(x['t']) for x in [pending,release,settled]]
    require(start<pt<rt<st<finish and int(settled['completed'])==count,'response lifetime')
    regs=markers(sim,'XDMA_CHAIN_REG')
    expected=[(1,0x40,0),(1,0x48,0),(1,0x40,1),(1,0x48,0)]
    for phase,busy,completed in [(2,1,0),(2,1,0),(3,0,count),(3,0,count)]:
        expected += [(phase,4,0xfffe7e),(phase,0x40,busy),(phase,0x48,completed),(phase,0x1004,0),(phase,0x1040,0)]
    require(len(regs)==len(expected),'all register observations')
    for row,(phase,address,value) in zip(regs,expected):
        actual=int(row['value'],16)
        require(int(row['phase'])==phase and int(row['address'],16)==address,'register order')
        require((actual&1==value and actual&~0x47==0) if address==0x40 else actual==value,'register value/errors')
    writes=[(int(r['phase']),int(r['address'],16),int(r['value'],16)) for r in markers(sim,'XDMA_CHAIN_WRITE')]
    require(writes==[(1,0x4080,0x100),(1,0x4084,0),(1,0x4088,1),(1,4,0xfffe7f),(2,0xc,1)],'native control writes')
    ep,tb='/board/EP/','/board/ntrpu_stop_probe/'
    fields={'AW':['awid','awaddr','awlen','awsize','awburst'],'W':['wdata','wstrb','wlast'],'B':['bid','bresp']}
    accepted={k:[] for k in fields};packets={k:[] for k in ['s_axis_rq','m_axis_rc','m_axis_cq','s_axis_cc']}
    queries={int(r['t']):r for r in regs};qtimes=sorted(queries);qindex=0
    require(len(queries)==len(regs),'unique register times')
    phases=[];held=0;last=0;marked={}
    for t,before,after in vcd_rows(vcd):
        last=t
        while qindex<len(qtimes) and qtimes[qindex]<=t:
            qtime=qtimes[qindex];row=queries[qtime];samples=[before,after] if qtime==t else [before]
            require(any(s['/board/RP/tx_usrapp/P_READ_DATA']==int(row['value'],16) and s[tb+'phase']==int(row['phase']) for s in samples),'register log/VCD agreement')
            qindex+=1
        if t<start:continue
        require(all(after[n]==1 for n in ['/board/sys_rst_n',ep+'user_resetn',ep+'user_lnk_up','/board/RP/user_lnk_up']),'continuous native readiness')
        phase=after[tb+'phase'];require(phase in (1,2,3,4),'phase')
        if not phases or phases[-1]!=phase:phases.append(phase)
        require(after[tb+'hold_dma_b']==int(phase<3) and after[tb+'hold_bypass_b']==1,'public response delay controls')
        if pt<=t<rt:
            require(after[ep+'ntrpu_app_dma_bvalid']==1 and after[ep+'ntrpu_app_dma_bready']==0 and after[ep+'m_axi_bvalid']==0,'continuously retained application B')
            require(accepted['AW'] and after[ep+'m_axi_bid']==accepted['AW'][0][1] and after[ep+'m_axi_bresp']==0,'stable retained response identity')
        if before[ep+'user_clk']==0 and after[ep+'user_clk']==1:
            if pt<=t<rt:held+=1
            for p in ['m_axi_ar','m_axi_r','m_axib_aw','m_axib_w','m_axib_b','m_axib_ar','m_axib_r']:
                require(not(before[ep+p+'valid']==1 and before[ep+p+'ready']==1),'no C2H or BAR transfers claimed')
            for channel,names in fields.items():
                p=ep+'m_axi_';low=channel.lower()
                if before[p+low+'valid']==1 and before[p+low+'ready']==1:
                    row=[t]+[before[p+n] for n in names];require(None not in row,'known accepted AXI payload')
                    accepted[channel].append(row)
        if t in (pt,rt):marked[t]={k:len(v) for k,v in accepted.items()}
        if before['/board/RP/user_clk']==0 and after['/board/RP/user_clk']==1:
            for prefix,rows in packets.items():
                p='/board/RP/'+prefix+'_'
                if before[p+'tvalid']==1 and before[p+'tready'] is not None and before[p+'tready']&1:
                    row=[t]+[before[p+n] for n in ['tdata','tkeep','tlast','tuser']]
                    require(None not in row,'known public PCIe packet');rows.append(row)
    require(phases==[1,2,3,4] and finish-1<=last<=finish and qindex==len(regs),'complete VCD scope')
    require(held>=64,'observable retained first B')
    for mark,time in [(pending,pt),(release,rt)]:
        require(marked[time]=={k:int(mark[k.lower()]) for k in fields},'pending/release marker inventory')
        require(marked[time]['B']==0,'no B before release')
    for channel in fields:
        rows=markers(sim,'XDMA_CHAIN_'+channel)
        require(len(rows)==len(accepted[channel])==count,'exact accepted descriptor inventory')
        keys={'AW':['id','addr','len','size'],'W':['data','strb','last'],'B':['id','resp']}[channel]
        for log,actual in zip(rows,accepted[channel]):
            logged=[int(log['t'])]+[int(log[k],2 if k=='last' else 16) for k in keys]
            require(logged==(actual[:-1] if channel=='AW' else actual),'AXI log/VCD identity')
    patterns=[bytes(((i*13)^0xa5)&255 for i in range(64)),bytes(((i*7)^0x3c)&255 for i in range(64))]
    for i,(aw,w,b) in enumerate(zip(accepted['AW'],accepted['W'],accepted['B'])):
        require(aw[2:]==[0x1000+i*0x100,0,6,1],'descriptor AXI address/burst')
        require(w[1].to_bytes(64,'little')==patterns[i] and w[2:]==[(1<<64)-1,1],'all descriptor payload bytes')
        require(aw[1]==b[1] and b[2]==0 and max(aw[0],w[0],rt)<b[0]<st,'descriptor completion causality')
    reads=[];posted=[]
    for t,data,keep,last,user in packets['s_axis_rq']:
        kind=(data>>75)&15
        require(kind in (0,1) and (data>>64)&2047==1 and data&3==0 and last==1,'native register request geometry')
        require(keep==(15 if kind==0 else 31),'native register request mask')
        (reads if kind==0 else posted).append([t,data&((1<<64)-1),(data>>128)&0xffffffff])
    require(len(reads)==len(packets['m_axis_rc'])==len(regs),'native register completion inventory')
    base=reads[0][1]-0x40;require(base==0x10000,'enumerated DMA BAR')
    previous=-1
    for request,completion,log in zip(reads,packets['m_axis_rc'],regs):
        t,data,keep,last,user=completion
        require(previous<request[0]<t<=int(log['t']) and request[1]==base+int(log['address'],16),'serial native register correlation')
        require(keep==15 and last==1 and (data>>32)&2047==1 and (data>>43)&15==0,'successful one-dword native completion')
        require((data>>96)&0xffffffff==int(log['value'],16),'native register data');previous=t
    require([(r[1],r[2]) for r in posted]==[(base+a,v) for _,a,v in writes],'native posted control writes')
    cq,cc=packets['m_axis_cq'],packets['s_axis_cc']
    addresses=[r[1]&((1<<64)-1) for r in cq]
    require(addresses in ([0x100,0x400],[0x100,0x400,0x500]),'bounded host fetch inventory')
    require(len(cc)==2*len(cq),'two CC beats per 64-byte host read')
    desc=bytearray(64)
    for offset,value in [(0,0xad4b0012),(4,64),(8,0x400),(16,0x1000),(24,0x120),
                         (32,0xad4b0013),(36,64),(40,0x500),(48,0x1100)]:
        desc[offset:offset+4]=value.to_bytes(4,'little')
    payloads={0x100:bytes(desc),0x400:patterns[0],0x500:patterns[1]}
    host=[]
    for i,(request,address) in enumerate(zip(cq,addresses)):
        t,data,keep,last,user=request;group=cc[2*i:2*i+2];header=group[0][1]
        require((data>>64)&2047==16 and (data>>75)&15==0 and keep==15 and last==1,'host 64-byte read request')
        require((data>>121)&7==0 and (data>>124)&7==2,'host request TC/attributes')
        require(user&0xffff==0x0f0f,'full first/last DWORD byte enables on native CQ')
        tag=(data>>96)&255;requester=(data>>80)&65535
        require(t<group[0][0]<group[1][0] and (header>>64)&255==tag and (header>>48)&65535==requester,'CC request credit and identity')
        require(header&127==0 and (header>>16)&4095==64 and (header>>32)&2047==16 and
                (header>>43)&31==0 and (header>>31)&1==0 and (header>>88)&1==1 and
                (header>>89)&7==0 and (header>>92)&7==2,'CC header/attributes')
        require(group[0][2:]==[65535,0,1] and group[1][2:]==[7,1,0x240],'CC framing')
        actual=group[0][1].to_bytes(64,'little')[12:]+group[1][1].to_bytes(64,'little')[:12]
        require(actual==payloads[address],'all 64 host payload bytes')
        host.append(dict(address=address,request_ps=t,completion_ps=group[-1][0],tag=tag))
    require(host[0]['completion_ps']<pt,'both adjacent descriptors delivered before first response hold')
    for i in range(count):
        source=next((r for r in host if r['address']==0x400+i*0x100),None)
        require(source is not None and source['completion_ps']<accepted['W'][i][0],'DMA data has completed host read')
    starts,beats,ends=[markers(sim,'NTRPU_HOST_CPL_'+x) for x in ['BEGIN','BEAT','END']]
    require(len(starts)==len(ends)==len(cq) and len(beats)==len(cc),'host stimulus log inventory')
    for log,actual in zip(beats,cc):
        require([int(log['t']),int(log['data'],16),int(log['keep'],16),int(log['last']),int(log['user'],16)]==actual,'host beat log/VCD agreement')
    require([int(x['offset']) for x in beats]==[0,52]*len(cq),'host segmentation')
    for begin,end,item in zip(starts,ends,host):
        require(int(begin['address'],16)==item['address'] and int(begin['tag'],16)==int(end['tag'],16)==item['tag'] and
                int(begin['bytes'])==int(end['bytes'])==64 and int(begin['attr'],16)==2 and int(begin['tc'],16)==0,'host request log identity')
        require(item['request_ps']<=int(begin['t'])<item['completion_ps']<int(end['t']),'host request log lifetime')
    clear_posted=int(next(r for r in markers(sim,'XDMA_CHAIN_WRITE') if int(r['address'],16)==0xc)['t'])
    clear_readback=int(next(r for r in regs if int(r['phase'])==2 and int(r['address'],16)==4)['t'])
    return dict(status='PASS_NATIVE_ADJACENT_TWO_DESCRIPTOR_STOP_OBSERVATION',start_ps=start,finish_ps=finish,
                descriptors_delivered=2,completed_descriptors=count,payload_bytes=count*64,held_b_cycles=held,
                held_b_ps=rt-pt,post_release_observation_ps=finish-rt,register_observations=len(regs),
                native_register_reads=len(reads),native_register_completions=len(reads),native_posted_writes=len(posted),
                accepted=accepted,host_fetches=host,inventory_at_pending=marked[pt],inventory_at_release=marked[rt],
                data_fetches_before_release=[r['address'] for r in host[1:] if r['completion_ps']<rt],
                run_clear_posted_ps=clear_posted,run_clear_readback_ps=clear_readback,
                inventory_at_run_clear={k:sum(r[0]<clear_posted for r in rows) for k,rows in accepted.items()},
                qualification='One native adjacent two-descriptor stop observation; every fetch and accepted/retired AXI write reconstructed.',
                qualified_cancellation=False,qualified_global_drain=False,qualified_c2h=False,
                qualified_endpoint_reset=False,qualified_physical_design=False,
                native_register_correlation='strictly serial one-outstanding register reads; core tag allocation not traced')
