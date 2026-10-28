"""Independent public-VCD review of single-descriptor H2C/BAR response inventory."""
import re


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def vcd_rows(text):
    header, body = text.split('$enddefinitions $end', 1)
    require(re.search(r'\$timescale\s+1ps\s+\$end', header), 'VCD timescale')
    scopes, names, widths = [], {}, {}
    for line in header.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == '$scope':
            scopes.append(f[2])
        elif f[0] == '$upscope':
            scopes.pop()
        elif f[0] == '$var':
            name = '/' + '/'.join(scopes + [f[4]])
            require(name not in names, 'duplicate VCD name')
            names[name] = f[3]
            widths[f[3]] = int(f[2])
    current = {n: None for n in names}
    aliases = {c: [n for n in names if names[n] == c] for c in widths}
    changes, now = {}, 0
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('$'):
            continue
        if line.startswith('#'):
            after = {**current, **changes}
            yield now, current, after
            current, changes = after, {}
            new = int(line[1:]); require(new >= now, 'VCD reversal'); now = new
        else:
            bits, code = line[1:].split() if line.startswith('b') else (line[0], line[1:])
            require(code in widths and set(bits) <= set('01xz') and len(bits) <= widths[code], 'VCD encoding')
            value = None if set(bits) & set('xz') else int(bits, 2)
            for name in aliases[code]:
                changes[name] = value
    yield now, current, {**current, **changes}


def markers(sim, marker):
    return [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in sim.splitlines()
            if line.startswith(marker+' ')]


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    begin, end = markers(sim, 'XDMA_STOP_BEGIN'), markers(sim, 'XDMA_STOP_PASS')
    require(len(begin) == len(end) == 1, 'unique begin/pass')
    begin, end = begin[0], end[0]
    start, finish = int(begin['t']), int(end['t'])
    require(90000000 < start < finish < 180000000, 'bounded native times')
    require(begin['dma_bar'] == '1', 'DMA BAR')
    require({k: end[k] for k in ['dma_bytes','bypass_bytes','reset_tested','chain_stop_tested']} ==
            dict(dma_bytes='64',bypass_bytes='4',reset_tested='0',chain_stop_tested='0'), 'scope')
    require('$finish called at time' in sim, 'simulator finish')
    regs = markers(sim, 'XDMA_STOP_REG')
    expected = [(1,0x40,0),(1,0x48,0),(1,0x40,1),(1,0x48,0)]
    for phase, busy in [(2,1),(2,1),(3,0),(5,0)]:
        expected += [(phase,4,0xfffe7e),(phase,0x40,busy),(phase,0x48,0 if busy else 1),
                     (phase,0x1004,0),(phase,0x1040,0)]
    require(len(regs) == len(expected), 'all register observations')
    for row, (phase,address,value) in zip(regs, expected):
        require(int(row['phase']) == phase and int(row['address'],16) == address, 'register order')
        actual = int(row['value'],16)
        if address == 0x40:
            require(actual & 1 == value and actual & ~0x47 == 0, 'H2C Busy/errors')
        else:
            require(actual == value, 'register value')
    writes = [(int(r['phase']),int(r['address'],16),int(r['value'],16)) for r in markers(sim,'XDMA_STOP_WRITE')]
    require(writes == [(1,0x4080,0x100),(1,0x4084,0),(1,0x4088,0),(1,4,0xfffe7f),(2,0xc,1)], 'native register writes')
    pending = markers(sim,'XDMA_STOP_PENDING'); release = markers(sim,'XDMA_STOP_RELEASE')
    require([int(r['phase']) for r in pending] == [1,4] and [int(r['phase']) for r in release] == [3,5], 'response delay phases')
    require(all([int(r[k]) for k in ['aw','w','b']] == [1,1,0] for r in pending), 'pending inventory markers')
    require(int(pending[0]['t']) < int(release[0]['t']) < int(pending[1]['t']) < int(release[1]['t']), 'response order')
    reads = markers(sim,'XDMA_STOP_BAR_READ')
    require(len(reads)==1 and reads[0]['value']=='d3c2b1a0', 'BAR readback')
    barwrites = markers(sim,'XDMA_STOP_BAR_WRITE')
    require(len(barwrites)==1 and int(barwrites[0]['address'],16)==0x100002000 and barwrites[0]['value']=='d3c2b1a0', 'BAR host request')
    ep, tb = '/board/EP/', '/board/ntrpu_stop_probe/'
    accepted = {p:{c:[] for c in ['AW','W','B','AR','R']} for p in ['DMA','BAR']}
    fields = {'AW':['awid','awaddr','awlen','awsize'], 'W':['wdata','wstrb','wlast'],
              'B':['bid','bresp'],'AR':['arid','araddr','arlen','arsize'],
              'R':['rid','rdata','rresp','rlast']}
    query = {int(r['t']):r for r in regs}
    query_times = sorted(query); query_index = 0
    require(len(query)==len(regs), 'distinct register observation times')
    witnessed = set(); phases = []; cycles = {i:0 for i in range(1,7)}
    held_cycles = {'DMA':0,'BAR':0}; final_time = 0
    requests, completions = [], []
    for t, before, after in vcd_rows(vcd):
        final_time = t
        # Displays need not produce a VCD transition. Use the held value between
        # adjacent timestamps, as well as post-delta values at an exact timestamp.
        while query_index < len(query_times) and query_times[query_index] <= t:
            qt=query_times[query_index]; row=query[qt]
            # The next read can reset P_READ_DATA in the same simulation slot.
            # The read task has already held its completed value for ten clocks.
            samples=[before,after] if qt==t else [before]
            require(any(sample['/board/RP/tx_usrapp/P_READ_DATA'] == int(row['value'],16) and
                        sample[tb+'phase'] == int(row['phase']) for sample in samples),
                    'register log/VCD disagreement')
            witnessed.add(qt); query_index += 1
        if t < start:
            continue
        require(all(after[n] == 1 for n in ['/board/sys_rst_n',ep+'user_resetn',ep+'user_lnk_up','/board/RP/user_lnk_up']), 'continuous native readiness')
        phase = after[tb+'phase']; require(phase in range(1,7), 'phase state')
        if not phases or phases[-1] != phase:
            phases.append(phase)
        require(after[tb+'hold_dma_b'] == int(phase<3), 'DMA delay control')
        require(after[tb+'hold_bypass_b'] == int(phase<5), 'BAR delay control')
        if before['/board/RP/user_clk']==0 and after['/board/RP/user_clk']==1:
            for prefix,destination in [('s_axis_rq',requests),('m_axis_rc',completions)]:
                path='/board/RP/'+prefix+'_'
                if before[path+'tvalid']==1 and before[path+'tready'] is not None and before[path+'tready'] & 1:
                    require(before[path+'tdata'] is not None and before[path+'tlast']==1, 'single native PCIe beat')
                    destination.append((t,before[path+'tdata'],before[path+'tkeep']))
        for port,prefix,app,index in [('DMA','m_axi','dma',0),('BAR','m_axib','bypass',1)]:
            if int(pending[index]['t']) <= t < int(release[index]['t']):
                require(after[ep+'ntrpu_app_'+app+'_bvalid']==1 and
                        after[ep+'ntrpu_app_'+app+'_bready']==0 and
                        after[ep+prefix+'_bvalid']==0, 'continuously retained application B')
                require(len(accepted[port]['AW'])==1 and
                        after[ep+prefix+'_bid']==accepted[port]['AW'][0][1] and
                        after[ep+prefix+'_bresp']==0, 'stable pending B payload')
        if before[ep+'user_clk'] == 0 and after[ep+'user_clk'] == 1:
            cycles[phase] += 1
            for port,prefix,app,hold in [('DMA','m_axi','dma','hold_dma_b'),('BAR','m_axib','bypass','hold_bypass_b')]:
                prefix=ep+prefix+'_'
                if before[ep+'ntrpu_app_'+app+'_bvalid']==1 and before[tb+hold]==1:
                    held_cycles[port]+=1
                    require(before[prefix+'bvalid']==0 and before[ep+'ntrpu_app_'+app+'_bready']==0, 'delayed B handshake')
                for channel in fields:
                    c=channel.lower()
                    if before[prefix+c+'valid']==1 and before[prefix+c+'ready']==1:
                        row=[t]+[before[prefix+f] for f in fields[channel]]
                        require(None not in row, 'unknown accepted payload')
                        accepted[port][channel].append(row)
    require(phases == [1,2,3,4,5,6] and finish-1 <= final_time <= finish, 'complete VCD scope')
    require(witnessed == set(query), 'register timestamps missing from VCD')
    require(min(held_cycles.values()) >= 64, 'observable delayed responses')
    for port in accepted:
        for channel in ['AW','W','B']:
            require(len(accepted[port][channel])==1, 'single '+port+' '+channel)
            rows=[r for r in markers(sim,'XDMA_STOP_'+channel) if r['port']==port]
            require(len(rows)==1, 'unique accepted marker')
            r=rows[0]
            keys={'AW':['id','addr','len','size'],'W':['data','strb','last'],'B':['id','resp']}[channel]
            logged=[int(r['t'])]+[int(r[k], 2 if k=='last' else 16) for k in keys]
            require(logged==accepted[port][channel][0], 'accepted event log/VCD mismatch')
        aw,w,b=[accepted[port][c][0] for c in ['AW','W','B']]
        require(aw[0]<b[0] and w[0]<b[0] and b[1]==aw[1] and b[2]==0 and w[3]==1, 'AXI write causality')
        require(int(pending[0 if port=='DMA' else 1]['t']) < int(release[0 if port=='DMA' else 1]['t']) < b[0], 'B not retired before release')
    aw,w,b=[accepted['DMA'][c][0] for c in ['AW','W','B']]
    require(aw[2:]==[0x1000,0,6], 'DMA burst geometry')
    require(w[2]==(1<<64)-1 and w[1].to_bytes(64,'little')==bytes(((i*13)^0xa5)&255 for i in range(64)), 'all 64 DMA payload bytes')
    require(not accepted['DMA']['AR'] and not accepted['DMA']['R'], 'no C2H exercise claimed')
    aw,w,b=[accepted['BAR'][c][0] for c in ['AW','W','B']]
    require(aw[2:]==[0x100002000,0,2] and w[2]==15 and w[1]&0xffffffff==0xd3c2b1a0, 'BAR write address/data')
    require(len(accepted['BAR']['AR'])==len(accepted['BAR']['R'])==1, 'BAR read inventory')
    ar,r=accepted['BAR']['AR'][0],accepted['BAR']['R'][0]
    require(ar[2:]==[0x100002000,0,2] and r[1]==ar[1] and r[2]&0xffffffff==0xd3c2b1a0 and r[3:]==[0,1], 'BAR AXI read data')
    require(b[0]<ar[0]<r[0]<=int(reads[0]['t']), 'native readback order')
    # Public 512-bit Root Port RQ/RC descriptors, as encoded/decoded by the
    # unchanged AMD TSK_TX_MEMORY_READ_32 and TSK_BUILD_RC_TO_PCIE_PKT tasks.
    read_requests=[]; write_requests=[]
    for t,data,keep in requests:
        kind=(data>>75)&15; length=(data>>64)&0x7ff
        require(kind in [0,1] and length==1 and data&3==0, 'native memory request header')
        address=data&((1<<64)-1);tag=(data>>96)&255
        require(keep==(15 if kind==0 else 31), 'native request dword mask')
        (read_requests if kind==0 else write_requests).append((t,address,tag,(data>>128)&0xffffffff))
    require(len(read_requests)==len(regs)+1 and len(completions)==len(read_requests), 'native completion inventory')
    poll_index=19
    bar_release=int(release[1]['t'])
    require(int(pending[1]['t'])<read_requests[poll_index][0]<bar_release<
            accepted['BAR']['B'][0][0]<completions[poll_index][0],
            'BAR-held register request resumes after independent B release')
    require(held_cycles['BAR']>=1024,'independent BAR release delay')
    dma_base=read_requests[0][1]-0x40
    require(dma_base%0x10000==0 and 0<dma_base<2**32, 'native DMA BAR base')
    observed_tags=[]
    previous_completion=-1
    for index,((qt,address,tag,_),(ct,data,keep)) in enumerate(zip(read_requests,completions)):
        row=regs[index] if index<len(regs) else reads[0]
        expected_address=dma_base+int(row['address'],16) if index<len(regs) else 0x100002000
        require(address==expected_address and previous_completion<qt<ct<=int(row['t']), 'serial native request/completion order')
        previous_completion=ct
        require(keep==15 and (data>>32)&0x7ff==1 and (data>>43)&7==0 and (data>>46)&1==0,
                'successful native one-dword completion')
        require((data>>96)&0xffffffff==int(row['value'],16), 'native completion data')
        observed_tags.append(dict(supplied_rq_tag=tag,returned_rc_tag=(data>>64)&255))
    expected_writes=[(dma_base+address,value) for _,address,value in writes]+[(0x100002000,0xd3c2b1a0)]
    require([(address,value) for _,address,_,value in write_requests]==expected_writes, 'native posted register/BAR writes')
    return dict(status='PASS_NATIVE_SINGLE_H2C_RESPONSE_INVENTORY_AND_BAR_INDEPENDENCE',
                start_ps=start,finish_ps=finish,descriptor_bytes=64,bypass_bytes=4,
                delayed_response_cycles=held_cycles,phase_cycles=cycles,
                held_response_ps={p:int(release[i]['t'])-int(pending[i]['t']) for i,p in enumerate(['DMA','BAR'])},
                register_observations=len(regs),accepted=accepted,
                native_read_requests=len(read_requests),native_posted_writes=len(write_requests),
                native_completions=len(completions),dma_bar_base=dma_base,
                bar_held_register_request_ps=read_requests[poll_index][0],
                bar_held_register_completion_ps=completions[poll_index][0],
                native_register_correlation='strictly serial one-outstanding reads; core allocated tags not traced',
                native_register_tags=observed_tags,
                qualified_chain_stop=False,qualified_c2h=False,qualified_endpoint_reset=False,qualified_physical_design=False)
