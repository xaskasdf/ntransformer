# This separate checker is for a direct, always-ready response master.
# R2's strict local B/R retention checker remains unchanged and must already pass.
# Report retention is additionally checked in check_ntrpu_xdma_local_master_native.py.
"""Reconstruct local XDMA register transactions and retained public responses."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    names = ['BEGIN', 'BAR_PENDING', 'BAR_RELEASE', 'BAR_RECEIPT', 'H2C_PENDING',
             'H2C_RELEASE', 'C2H_PENDING', 'C2H_RELEASE', 'HOST_BEGIN', 'HOST_END', 'PASS']
    events = {}
    for name in names:
        rows = markers(sim, 'XDMA_LOCAL_' + name)
        require(len(rows) == 1, 'unique ' + name)
        events[name] = rows[0]
    times = {n:int(r['t']) for n,r in events.items()}
    start, finish = times['BEGIN'], times['PASS']
    require(90000000 < start < finish < 240000000 and list(times.values()) == sorted(set(times.values())), 'native causal marker order')
    require('$finish called at time' in sim, 'native finish')
    require({k:events['PASS'][k] for k in ['local_reads','local_writes','host_bytes','dma_bytes','bar_bytes','reset_tested','global_drain_tested']} ==
            dict(local_reads='31',local_writes='13',host_bytes='64',dma_bytes='64',bar_bytes='4',reset_tested='0',global_drain_tested='0'), 'fixture scope')
    ep,tb,rp = '/board/EP/','/board/ntrpu_stop_probe/','/board/RP/'
    lf = {'aw':['awaddr','awprot'], 'w':['wdata','wstrb'], 'b':['bresp'], 'ar':['araddr','arprot'], 'r':['rdata','rresp']}
    af = {'aw':['awid','awaddr','awlen','awsize'], 'w':['wdata','wstrb','wlast'], 'b':['bid','bresp'],
          'ar':['arid','araddr','arlen','arsize'], 'r':['rid','rdata','rresp','rlast']}
    local={c:[] for c in lf}
    accepted={p:{c:[] for c in af} for p in ['m_axi','m_axib']}
    packets={p:[] for p in ['s_axis_rq','m_axis_rc','m_axis_cq','s_axis_cc']}
    retained, stalled, phases = {}, {}, []
    hold_cycles={k:0 for k in ['BAR','H2C','C2H']}
    response_stalls={'b':0,'r':0}
    done={'h2c':[],'c2h':[]}
    states=[]
    memory=bytearray(64);written=0;host_packets=0;host_active=False
    regs=markers(sim,'XDMA_LOCAL_REG')
    query={int(r['t']):r for r in regs}
    qt=sorted(query); qi=0
    require(len(qt)==len(regs)==12,'native register logs')
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
        for label,app,port,channel,hold in [('BAR','bypass','m_axib','b','hold_bypass_b'),('H2C','dma','m_axi','b','hold_dma_b'),('C2H','dma','m_axi','r','hold_dma_r')]:
            release=times[label+'_RELEASE']
            require(after[tb+hold]==int(t<release),'exact hold/release control')
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
                        if prefix=='m_axis_cq':
                            _,data,keep,final,user=row
                            if not host_active and (data>>75)&15==1:
                                require(host_packets==written==0 and data&(2**64-1)==0x600 and (data>>64)&2047==16,'bounded native host write')
                                host_active=True
                            if host_active:
                                first=written==0
                                enable=(user>>16)&(2**64-1)
                                require(keep==(65535 if first else 15) and final==int(not first),'host packet framing')
                                require(enable==((2**48-1)<<16 if first else 65535),'host enabled byte lanes')
                                for lane in range(64):
                                    if enable>>lane&1:
                                        require(written<64,'host byte bound')
                                        memory[written]=(data>>(lane*8))&255;written+=1
                                if final:
                                    require(written==64,'complete host byte count')
                                    host_active=False;host_packets+=1
        require(after[tb+'host_memory']==int.from_bytes(memory,'little') and after[tb+'host_written']==2**written-1 and
                after[tb+'host_bytes']==written and after[tb+'host_packets']==host_packets and after[tb+'host_active']==int(host_active),
                'host memory changes only at accepted native payload bytes')
    require(phases==list(range(1,8)) and finish-1<=last<=finish and qi==len(qt),'complete native VCD')
    require(not stalled and min(hold_cycles.values())>0 and response_stalls=={'b':0,'r':0},'direct master consumes native B/R without response gating')
    require({c:len(v) for c,v in local.items()}==dict(aw=13,w=13,b=13,ar=31,r=31),'local transfer inventory')
    reads=[];writes=[]
    previous=0
    for ar,r in zip(local['ar'],local['r']):
        require(previous<ar[0]<r[0] and ar[1]==r[1] and ar[3]==0 and r[3]==0,'local read lifecycle/protection/response')
        reads.append(dict(request_ps=ar[0],response_ps=r[0],phase=r[1],address=ar[2],value=r[2]))
        previous=r[0]
    previous=0
    for aw,w,b in zip(local['aw'],local['w'],local['b']):
        require(previous<min(aw[0],w[0]) and max(aw[0],w[0])<b[0] and aw[1]==w[1]==b[1] and aw[3]==0 and w[3]==15 and b[2]==0,'local write lifecycle/protection/strobes/response')
        writes.append(dict(aw_ps=aw[0],w_ps=w[0],response_ps=b[0],phase=b[1],address=aw[2],value=w[2]))
        previous=b[0]
    expected_reads=[(1,a,0) for a in [4,0x40,0x48,0x1004,0x1040,0x1048]]+[(1,4,0xfffe7e),
                    (2,0x40,0),(2,0x1040,0),(2,4,0xfffe7c),(2,0x1004,0xfffe7c),(3,0x40,1)]
    for phase,base,busy,count in [(3,0,1,0),(4,0,0,1),(5,0x1000,1,0),(5,0,0,1),(6,0,0,1),(6,0x1000,0,1)]:
        if phase==5 and base==0x1000:expected_reads.append((5,0x1040,1))
        expected_reads += [(phase,base+4,0xfffe7e),(phase,base+0x40,busy),(phase,base+0x48,count)]
    require(len(expected_reads)==31,'reviewer read contract size')
    for actual,(phase,address,value) in zip(reads,expected_reads):
        require((actual['phase'],actual['address'])==(phase,address),'local read sequence')
        require((actual['value']&1==value and actual['value']&~0x47==0) if address in (0x40,0x1040) else actual['value']==value,'local register value')
    expected_writes=[(1,0x1004,0xfffe7e),(2,0xc,2),(2,0x100c,2),(3,0x4080,0x100),(3,0x4084,0),(3,0x4088,0),
                     (3,4,0xfffe7f),(3,0xc,1),(5,0x5080,0x200),(5,0x5084,0),(5,0x5088,0),(5,0x1004,0xfffe7f),(5,0x100c,1)]
    require([(r['phase'],r['address'],r['value']) for r in writes]==expected_writes,'local write sequence')
    for kind,rows in [('READ',reads),('WRITE',writes)]:
        starts,ends=markers(sim,'XDMA_LOCAL_'+kind+'_BEGIN'),markers(sim,'XDMA_LOCAL_'+kind+'_END')
        require(len(starts)==len(ends)==len(rows),'local marker inventory')
        for begin,end,row in zip(starts,ends,rows):
            require(int(begin['t'])<row.get('request_ps',min(row.get('aw_ps',0),row.get('w_ps',0)))<=row['response_ps']==int(end['t']),'local marker causality')
            require(int(begin['address'],16)==row['address']==int(end['address'],16) and int(end['value'],16)==row['value'] and int(end['phase'])==row['phase'],'local marker/VCD identity')
    for label,phase,nread,nwrite in [('BAR',2,4,2),('H2C',3,4,1),('C2H',5,7,1)]:
        a,z=times[label+'_PENDING'],times[label+'_RELEASE']
        rr=[r for r in reads if a<r['request_ps']<r['response_ps']<z]
        ww=[w for w in writes if a<min(w['aw_ps'],w['w_ps'])<=max(w['aw_ps'],w['w_ps'])<w['response_ps']<z]
        require(len(rr)==nread and len(ww)==nwrite and all(r['phase']==phase for r in rr+ww),'local progress while '+label+' actually held')
    # State ports must reflect stopped-but-busy after local W1C, then drained.
    for channel,phase,address in [('h2c',3,0xc),('c2h',5,0x100c)]:
        stop=next(w for w in writes if w['phase']==phase and w['address']==address)
        release=times[channel.upper()+'_RELEASE'];index=1 if channel=='h2c' else 2
        witness=max(r['response_ps'] for r in reads if r['phase']==phase and stop['response_ps']<r['response_ps']<release)
        live=next(row[index] for row in reversed(states) if row[0]<=witness)
        require(live&0x41==1,'stopped-but-busy channel status')
        require(len(done[channel])==1 and done[channel][0]>release,'one completion after held response release')
    require(states[-1][1]&0x41==states[-1][2]&0x41==0,'final live Run/Busy cleared')
    pattern=bytes(((i*13)^0xa5)&255 for i in range(64))
    require(bytes(memory)==pattern and host_packets==1 and not host_active,'native host round-trip payload')
    for port in accepted:
        require(all(len(v)==1 for v in accepted[port].values()),'one complete transaction per direction '+port)
        aw,w,b,ar,r=[accepted[port][c][0] for c in ['aw','w','b','ar','r']]
        require(max(aw[0],w[0])<b[0]<ar[0]<r[0] and b[1:]==[aw[1],0] and r[1]==ar[1] and r[3:]==[0,1] and w[3]==1,'public AXI transaction causality')
        if port=='m_axi':
            require(aw[2:]==ar[2:]==[0x1000,0,6] and w[1].to_bytes(64,'little')==pattern and w[2]==2**64-1 and r[2]==w[1],'all64 DMA data bytes')
            require(times['H2C_RELEASE']<b[0] and times['C2H_RELEASE']<r[0]<times['HOST_BEGIN'],'DMA held response causality')
        else:
            require(aw[2:]==ar[2:]==[0x100002000,0,2] and w[1]&0xffffffff==r[2]&0xffffffff==0xd3c2b1a0 and w[2]==15,'BAR AXI bytes/address')
            require(times['BAR_RELEASE']<b[0]<ar[0]<r[0]<=times['BAR_RECEIPT'],'BAR receipt after release')
    native=review_native_packets(sim,packets,regs,events,accepted,pattern)
    return dict(status='PASS_NATIVE_DIRECT_LOCAL_MANAGEMENT_TRAFFIC',start_ps=start,finish_ps=finish,
                local_reads=reads,local_writes=writes,accepted=accepted,channel_status=states,descriptor_done_ps=done,
                held_cycles=hold_cycles,held_ps={k:times[k+'_RELEASE']-times[k+'_PENDING'] for k in hold_cycles},
                response_stall_cycles=response_stalls,native=native,qualified_global_drain=False,
                qualified_endpoint_reset=False,qualified_physical_design=False,
                scope='Local public XDMA management during three actual retained responses; no OS host memory/coherence, protected global inventory, operational reset or physical throughput approval.')


def review_native_packets(sim, packets, regs, unique, accepted, pattern):
    expected=[(1,4,0xfffe7e),(1,0x1004,0xfffe7e),(2,4,0xfffe7c),(2,0x1004,0xfffe7c),
              (4,0x40,0),(4,0x48,1),(6,0x1004,0xfffe7e),(6,0x1040,0),(6,0x1048,1),
              (6,4,0xfffe7e),(6,0x40,0),(6,0x48,1)]
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
    require(len(read_requests)==len(packets['m_axis_rc'])==len(logs)==13,'exact native read/completion inventory')
    previous=0
    for request,completion,log in zip(read_requests,packets['m_axis_rc'],logs):
        t,data,keep,final,user=completion
        require(previous<request[0]<t<=log['t'] and request[1]==log['address'],'serial native request correlation')
        require(keep==15 and final==1 and (data>>32)&2047==1 and (data>>43)&15==0 and (data>>96)&0xffffffff==log['value'],'native response bytes/status')
        previous=t
    require([(r[1],r[2]) for r in write_requests]==[(0x10004,0xfffe7e),(int(unique['BEGIN']['bypass_base'],16)+0x2000,0xd3c2b1a0)],'native posted control and BAR write')
    controls=markers(sim,'XDMA_LOCAL_WRITE')
    require(len(controls)==1 and int(controls[0]['address'],16)==4 and int(controls[0]['value'],16)==0xfffe7e,'native control log')
    aw,w,b,ar,r=[accepted['m_axi'][n][0] for n in ['aw','w','b','ar','r']]
    host_begin=int(unique['HOST_BEGIN']['t']);host_end=int(unique['HOST_END']['t'])
    host_beats=packets['m_axis_cq'][-2:]
    cq = packets['m_axis_cq']; cc = packets['s_axis_cc']
    require(len(cq) == 5 and len(cc) == 4 and cq[3:] == host_beats, 'exact host read/write inventory')
    desc = {}
    for address, source, destination in [(0x100, 0x400, 0x1000), (0x200, 0x1000, 0x600)]:
        payload = bytearray(32)
        for offset, value in [(0, 0xad4b0013), (4, 64), (8, source), (16, destination)]:
            payload[offset:offset+4] = value.to_bytes(4, 'little')
        desc[address] = bytes(payload)
    payloads = {**desc, 0x400: pattern}; host_reads = []; offset = 0
    for request, address in zip(cq[:3], [0x100, 0x400, 0x200]):
        t, data, keep, final, user = request
        payload = payloads[address]; length = len(payload)
        count = 1 if length == 32 else 2
        group = cc[offset:offset+count]; offset += count; header = group[0][1]
        require(data & ((1 << 64)-1) == address and (data >> 64) & 2047 == length//4 and
                (data >> 75) & 15 == 0 and keep == 15 and final == 1 and user & 65535 == 0xf0f,
                'host descriptor/payload fetch')
        require((data >> 121) & 7 == 0 and (data >> 124) & 7 == 2, 'host fetch attributes')
        tag, requester = (data >> 96) & 255, (data >> 80) & 65535
        require(t < group[0][0] <= group[-1][0] and (header >> 64) & 255 == tag and
                (header >> 48) & 65535 == requester, 'CC request identity')
        require(header & 127 == 0 and (header >> 16) & 4095 == length and (header >> 32) & 2047 == length//4 and
                (header >> 43) & 31 == 0 and (header >> 31) & 1 == 0 and (header >> 88) & 1 == 1 and
                (header >> 89) & 7 == 0 and (header >> 92) & 7 == 2, 'CC header')
        require([g[2:] for g in group] == ([[2047, 1, 0xa41]] if length == 32 else [[65535, 0, 1], [7, 1, 0x240]]), 'CC segmentation')
        actual = group[0][1].to_bytes(64, 'little')[12:12+min(length, 52)]
        if count == 2:
            require(group[0][0] < group[1][0], 'ordered CC beats')
            actual += group[1][1].to_bytes(64, 'little')[:12]
        require(actual == payload, 'all descriptor and seed bytes')
        host_reads.append(dict(address=address, bytes=length, request_ps=t, completion_ps=group[-1][0], tag=tag))
    require(host_reads[0]['completion_ps'] < aw[0] and host_reads[1]['completion_ps'] < w[0] and
            b[0] < host_reads[2]['request_ps'] < host_reads[2]['completion_ps'] < ar[0], 'fetch causality')
    starts, beats, ends = [markers(sim, 'NTRPU_HOST_CPL_'+n) for n in ['BEGIN', 'BEAT', 'END']]
    require(len(starts) == len(ends) == 3 and len(beats) == 4, 'completion driver log inventory')
    require([int(x['offset']) for x in beats] == [0, 0, 52, 0], 'completion log offsets')
    for log, actual in zip(beats, cc):
        require([int(log['t']), int(log['data'], 16), int(log['keep'], 16), int(log['last']), int(log['user'], 16)] == actual,
                'CC log/VCD identity')
    for begin, end, item in zip(starts, ends, host_reads):
        require(int(begin['address'], 16) == item['address'] and int(begin['tag'], 16) == int(end['tag'], 16) == item['tag'] and
                int(begin['bytes']) == int(end['bytes']) == item['bytes'] and int(begin['attr'], 16) == 2 and int(begin['tc'], 16) == 0,
                'completion driver request identity')
        require(item['request_ps'] <= int(begin['t']) < item['completion_ps'] < int(end['t']), 'completion driver lifetime')
    logs = markers(sim, 'XDMA_LOCAL_HOST_BEAT'); require(len(logs) == 2, 'host write log inventory')
    for log, actual, count in zip(logs, host_beats, [48, 64]):
        require([int(log['t']), int(log['data'], 16), int(log['keep'], 16), int(log['last']), int(log['user'], 16)] == actual and
                int(log['bytes']) == count, 'host posted byte accounting')
    require(host_beats[0][0] == host_begin and host_beats[-1][0] == host_end and
            int(unique['HOST_BEGIN']['address'], 16) == 0x600 and unique['HOST_BEGIN']['bytes'] == '64', 'host memory lifetime')

    require(int(unique['HOST_END']['memory'],16)==int.from_bytes(pattern,'little'),'host final logged memory')
    for i,(_,data,keep,final,user) in enumerate(host_beats):
        require((user>>80)&3==int(i==0) and (user>>82)&15==0 and (user>>86)&3==int(i==1) and (user>>96)&1==0,'CQ posted packet boundaries')
        if i==0:
            require(data&(2**64-1)==0x600 and (data>>64)&2047==16 and (data>>75)&15==1 and user&65535==0xf0f,'CQ posted header')
            require((data>>121)&7==0 and (data>>124)&7==0,'C2H posted TC/attributes')
        else:require((user>>88)&15==3,'CQ final DWORD')
    return dict(read_requests=13,posted_writes=2,host_reads=host_reads,host_write_beats=2,host_received_bytes=64,
                register_correlation='serial one-outstanding public RQ/RC; core allocated tags not traced')
