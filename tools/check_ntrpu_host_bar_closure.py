"""Reconstruct a bounded producer ledger and native BAR requests/responses independently."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review(sim, vcd):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|Error:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    begin, end = markers(sim, 'HOST_CLOSE_BEGIN'), markers(sim, 'HOST_CLOSE_PASS')
    require(len(begin) == len(end) == 1, 'unique scope markers')
    begin, end = begin[0], end[0]
    start, finish = int(begin['t']), int(end['t'])
    require(90000000 < start < finish < 200000000 and '$finish called at time' in sim, 'bounded complete simulation')
    require(begin['dma_bar'] == '1' and int(begin['bypass_base'], 16) == 0x100000000, 'enumerated BARs')
    require({k: end[k] for k in ['generations','writes','receipts','rejects','certificate_rejects','driver_tested','global_drain_tested','reset_tested']} ==
            dict(generations='2',writes='2',receipts='2',rejects='4',certificate_rejects='10',driver_tested='0',global_drain_tested='0',reset_tested='0'), 'scope claims')
    regs = markers(sim, 'HOST_CLOSE_REG')
    require([(int(r['phase']), int(r['address'],16), int(r['value'],16)) for r in regs] ==
            [(p,a,0) for p in [1,3,5] for a in [4,0x40,0x48,0x1004,0x1040,0x1048]], 'idle SGDMA registers')
    submits = markers(sim, 'HOST_CLOSE_SUBMIT')
    expected = [(7,7,1,0x2000,0xd3c2b1a0,0), (7,7,0,0x2040,0x01234567,1), (7,7,0,0x2080,0x89abcdef,1),
                (8,7,0,0x20c0,0x13579bdf,0), (8,8,1,0x2100,0x4c3d2e1f,0), (8,8,0,0x2140,0x2468ace0,1)]
    require(len(submits) == len(expected), 'all producer requests')
    for row, (gen, req, accepted, address, data, closed) in zip(submits, expected):
        require([int(row[k]) for k in ['generation','requested','accepted','closed']] == [gen,req,accepted,closed] and
                int(row['address'],16) == 0x100000000+address and int(row['data'],16) == data, 'submission scenario')
    opened, closed, sent, pending, released, read_begin, receipts = [markers(sim,'HOST_CLOSE_'+name)
        for name in ['OPEN','CLOSED','SENT','PENDING','RELEASE','READ_BEGIN','RECEIPT']]
    require(all(len(rows) == 2 for rows in [opened,closed,sent,pending,released,read_begin,receipts]), 'two epoch lifetimes')
    certs = markers(sim, 'HOST_CLOSE_CERT')
    require(len(certs) == 12 and sum(int(r['accepted']) for r in certs) == 2, 'certificate attempts')
    for rows in [opened,closed,sent,pending,released,read_begin,receipts]:
        require([int(r['generation']) for r in rows] == [7,8], 'monotonic generation')
    cert_cases = [(7,7,1,0),(7,8,1,0),(7,7,1,0),(7,6,1,0),(7,7,2,0),(7,7,1,1),
                  (8,7,1,0),(8,8,1,0),(8,8,1,0),(8,7,1,0),(8,8,2,0),(8,8,1,1)]
    require([tuple(int(row[k]) for k in ['generation','requested','target','accepted']) for row in certs] == cert_cases,
            'premature, foreign, stale and wrong-target certificate cases')
    ep, rp, tb = '/board/EP/', '/board/RP/', '/board/ntrpu_stop_probe/'
    state = dict(generation=0,admitted=0,retired=0,outstanding=0,close_target=0,producer_closed=1,
                 certificate_valid=0,certificates=0,rejects=0,certificate_rejects=0,hold_bypass_b=1)
    events = []
    for line in sim.splitlines():
        tag = line.split(' ',1)[0]
        if tag in {'HOST_CLOSE_OPEN','HOST_CLOSE_SUBMIT','HOST_CLOSE_CLOSED','HOST_CLOSE_CERT','HOST_CLOSE_RECEIPT','HOST_CLOSE_RELEASE'}:
            row = dict(re.findall(r'(\w+)=([^ ]+)',line))
            events.append((int(row['t']),tag.removeprefix('HOST_CLOSE_'),row))
    require([t for t,_,_ in events] == sorted(t for t,_,_ in events), 'ordered ledger evidence')
    def apply_event(kind, row):
        gen = int(row['generation'])
        if kind == 'OPEN':
            require(gen > state['generation'] and (state['generation'] == 0 or
                    state['producer_closed'] == state['certificate_valid'] == 1 and state['outstanding'] == 0), 'safe model generation transition')
            state.update(generation=gen,admitted=0,retired=0,outstanding=0,close_target=0,producer_closed=0,certificate_valid=0,hold_bypass_b=1)
        else:
            require(gen == state['generation'], 'event current generation')
            if kind == 'SUBMIT':
                accepted = int(not state['producer_closed'] and int(row['requested']) == gen)
                require(accepted == int(row['accepted']), 'admission predicate')
                if accepted:
                    state['admitted'] += 1; state['outstanding'] += 1
                else:
                    state['rejects'] += 1
                require(int(row['closed']) == state['producer_closed'] and int(row['admitted']) == state['admitted'] and
                        int(row['outstanding']) == state['outstanding'], 'submission ledger')
            elif kind == 'CLOSED':
                require(state['producer_closed'] == 0 and state['outstanding'] == 1, 'closing retains obligation')
                state['producer_closed'] = 1; state['close_target'] = state['admitted']
                require(int(row['target']) == state['close_target'] and int(row['outstanding']) == 1, 'close snapshot')
            elif kind == 'RELEASE':
                require(state['outstanding'] == 1 and state['hold_bypass_b'] == 1, 'independent response release')
                state['hold_bypass_b'] = 0
            elif kind == 'RECEIPT':
                require(state['outstanding'] == 1 and state['producer_closed'] == 1, 'receipt lifetime')
                state['outstanding'] -= 1; state['retired'] += 1
                require(int(row['outstanding']) == 0 and int(row['retired']) == 1, 'receipt accounting')
            elif kind == 'CERT':
                accepted = int(state['producer_closed'] and int(row['requested']) == gen and
                    int(row['target']) == state['close_target'] == state['admitted'] == state['retired'] and state['outstanding'] == 0)
                require(accepted == int(row['accepted']) and int(row['outstanding']) == state['outstanding'] and
                        int(row['retired']) == state['retired'], 'certificate predicate')
                if accepted:
                    require(state['certificate_valid'] == 0, 'one certificate per generation')
                    state['certificate_valid'] = 1; state['certificates'] += 1
                else:
                    state['certificate_rejects'] += 1
        require(state['outstanding'] == state['admitted']-state['retired'], 'ledger conservation')
    fields = {'AW':['awid','awaddr','awlen','awsize','awburst'], 'W':['wdata','wstrb','wlast'],
              'B':['bid','bresp'], 'AR':['arid','araddr','arlen','arsize'], 'R':['rid','rdata','rresp','rlast']}
    accepted = {k:[] for k in fields}; packets = {k:[] for k in ['s_axis_rq','m_axis_rc','m_axis_cq','s_axis_cc']}
    checks = sorted([(int(row['t']),row,'reg') for row in regs]+[(int(row['t']),row,'receipt') for row in receipts])
    eindex = qindex = 0; phases = []; last = 0; held = [0,0]; retained = [None,None]
    for t, before, after in vcd_rows(vcd):
        last = t
        while qindex < len(checks) and checks[qindex][0] <= t:
            qt, row, kind = checks[qindex]
            samples = [before,after] if qt == t else [before]
            value = int(row['value' if kind == 'reg' else 'data'],16)
            require(any(s[rp+'tx_usrapp/P_READ_DATA'] == value for s in samples), 'host response log/VCD identity')
            qindex += 1
        if t < start:
            continue
        while eindex < len(events) and events[eindex][0] <= t:
            _, kind, row = events[eindex]; apply_event(kind,row); eindex += 1
        require(all(after[tb+k] == value for k,value in state.items()), 'independent ledger/VCD identity')
        require(after[tb+'hold_dma_b'] == 0, 'no DMA response delay')
        require(all(after[n] == 1 for n in ['/board/sys_rst_n',ep+'user_resetn',ep+'user_lnk_up',rp+'user_lnk_up']), 'continuous native readiness')
        phase = after[tb+'phase']; require(phase in (1,2,3,4,5,6), 'known phase')
        if not phases or phases[-1] != phase:
            phases.append(phase)
        for i in range(2):
            if int(pending[i]['t']) <= t < int(released[i]['t']):
                require(after[ep+'ntrpu_app_bypass_bvalid'] == 1 and after[ep+'ntrpu_app_bypass_bready'] == 0 and
                        after[ep+'m_axib_bvalid'] == 0, 'native B remains retained')
                payload = [after[ep+'m_axib_bid'],after[ep+'m_axib_bresp']]
                require(None not in payload and payload[1] == 0, 'known retained B')
                if retained[i] is None:
                    retained[i] = payload
                require(payload == retained[i], 'retained B stable')
        if before[ep+'user_clk'] == 0 and after[ep+'user_clk'] == 1:
            for i in range(2):
                if int(pending[i]['t']) <= t < int(released[i]['t']):
                    held[i] += 1
            for channel, names in fields.items():
                low = channel.lower()
                require(before[ep+'m_axi_'+low+'valid'] == 0, 'SGDMA transfers absent')
                p = ep+'m_axib_'
                require(before[p+low+'valid'] in (0,1), 'known BAR valid')
                if before[p+low+'valid'] == 1:
                    require(before[p+low+'ready'] in (0,1), 'known BAR ready')
                if before[p+low+'valid'] == before[p+low+'ready'] == 1:
                    row = [t]+[before[p+n] for n in names]
                    require(None not in row, 'known accepted BAR payload'); accepted[channel].append(row)
        if before[rp+'user_clk'] == 0 and after[rp+'user_clk'] == 1:
            for prefix, rows in packets.items():
                p = rp+prefix+'_'
                require(before[p+'tvalid'] in (0,1), 'known PCIe valid')
                if before[p+'tvalid'] == 1:
                    require(before[p+'tready'] is not None, 'known PCIe ready')
                if before[p+'tvalid'] == 1 and before[p+'tready'] is not None and before[p+'tready'] & 1:
                    row = [t]+[before[p+n] for n in ['tdata','tkeep','tlast','tuser']]
                    require(None not in row, 'known PCIe request/response'); rows.append(row)
    require(phases == [1,2,3,4,5,6] and eindex == len(events) and qindex == len(checks) and finish-1 <= last <= finish, 'complete trace scope')
    require(state['certificates'] == 2 and state['certificate_rejects'] == 10 and state['rejects'] == 4 and
            state['outstanding'] == 0 and all(x >= 1024 for x in held), 'final ledger and response holds')
    require(not packets['m_axis_cq'] and not packets['s_axis_cc'], 'no SGDMA fetch or host DMA write')
    for channel, rows in accepted.items():
        logs = markers(sim,'HOST_CLOSE_'+channel)
        require(len(rows) == len(logs) == 2, 'exact BAR channel inventory')
        keys = {'AW':['id','addr','len','size'],'W':['data','strb','last'],'B':['id','resp'],
                'AR':['id','addr','len','size'],'R':['id','data','resp','last']}[channel]
        for actual, log in zip(rows,logs):
            require([int(log['t'])]+[int(log[k],2 if k == 'last' else 16) for k in keys] ==
                    (actual[:-1] if channel == 'AW' else actual), 'AXI log/VCD identity')
    native_reads, native_writes = [], []
    requester = None
    for t,data,keep,final,user in packets['s_axis_rq']:
        kind = (data >> 75) & 15
        require(kind in (0,1) and (data >> 64) & 2047 == 1 and data & 3 == 0 and
                keep == (15 if kind == 0 else 31) and final == 1, 'one-DWORD native request')
        require((data >> 121) & 7 == 0 and (data >> 124) & 7 == 0 and (data >> 79) & 1 == 0,
                'native TC=0 ATTR=0 and unpoisoned request')
        require(user & 65535 == 15, 'one DWORD first/last byte enables')
        rid = (data >> 80) & 65535
        if requester is None:
            requester = rid
        require(rid == requester, 'same host requester scope')
        (native_reads if kind == 0 else native_writes).append([t,data & ((1 << 64)-1),(data >> 128) & 0xffffffff])
    require(len(native_reads) == len(packets['m_axis_rc']) == 20 and len(native_writes) == 2, 'exact RQ/RC inventory')
    expected_reads = sorted([(int(r['t']),0x10000+int(r['address'],16),int(r['value'],16)) for r in regs] +
                            [(int(r['t']),int(r['address'],16),int(r['data'],16)) for r in receipts])
    previous = -1; data_completions = []
    for request, completion, (logged,address,value) in zip(native_reads,packets['m_axis_rc'],expected_reads):
        t,data,keep,final,user = completion
        require(previous < request[0] < t <= logged and request[1] == address, 'serial native read correlation')
        require(keep == 15 and final == 1 and (data >> 32) & 2047 == 1 and (data >> 43) & 15 == 0 and
                (data >> 96) & 0xffffffff == value, 'successful native RC data')
        if address > 0xffffffff:
            data_completions.append((request,completion))
        previous = t
    admitted = [row for row in submits if row['accepted'] == '1']; epochs = []
    for i in range(2):
        row = admitted[i]; address = int(row['address'],16); data = int(row['data'],16)
        aw,w,b,ar,r = [accepted[n][i] for n in fields]
        request,completion = data_completions[i]
        op,ct,pt,rt,receipt = [int(rows[i]['t']) for rows in [opened,closed,pending,released,receipts]]
        native_write = native_writes[i]
        require(native_write[1:] == [address,data] and int(row['t']) <= native_write[0] <= int(sent[i]['t']), 'admission binds exactly one native posted write')
        require(aw[2:] == [address,0,2,1] and w[1] & 0xffffffff == data and w[2:] == [15,1] and b[1:] == [aw[1],0], 'BAR write and response')
        require(ar[2:] == [address,0,2] and r[1] == ar[1] and r[2] & 0xffffffff == data and r[3:] == [0,1], 'same-address BAR readback')
        require(op <= int(row['t']) < ct < rt < b[0] and max(aw[0],w[0]) <= pt < rt and
                native_write[0] < aw[0] and native_write[0] < w[0], 'posted write survives closure and retention')
        require(ct <= int(read_begin[i]['t']) < request[0] < rt < b[0] < ar[0] < r[0] < completion[0] <= receipt,
                'read waits for independent release and actual response')
        success = next(c for c in certs if int(c['generation']) == 7+i and c['accepted'] == '1')
        require(receipt < int(success['t']) < finish, 'certificate follows validated native receipt')
        require([int(pending[i][k]) for k in ['aw','w','b']] == [i+1,i+1,i], 'pending native inventory')
        epochs.append(dict(generation=7+i,closed_ps=ct,submitted_ps=int(row['t']),posted_rq_ps=native_write[0],
            aw_ps=aw[0],w_ps=w[0],b_ps=b[0],read_rq_ps=request[0],ar_ps=ar[0],r_ps=r[0],read_rc_ps=completion[0],
            receipt_ps=receipt,certificate_ps=int(success['t']),held_b_cycles=held[i],held_b_ps=rt-pt,
            write_reached_axi_after_close=aw[0]>ct,address=address,value=data))
    require(int(submits[1]['t']) == int(closed[0]['t'])+2000 and int(submits[2]['t']) == int(closed[0]['t'])+4000,
            'explicit close-before-submit boundary')
    require(int(sent[1]['t']) < int(closed[1]['t']) < int(submits[-1]['t']), 'explicit admit-before-close boundary')
    return dict(status='PASS_NATIVE_HOST_BAR_SUBMISSION_CLOSURE_MODEL',start_ps=start,finish_ps=finish,
        epochs=epochs,admitted_writes=2,rejected_submissions=4,receipts=2,accepted_certificates=2,rejected_certificates=10,
        register_reads=18,native_read_completions=20,native_posted_writes=2,requester_id=requester,
        qualified_bounded_producer_model=True,qualified_driver=False,qualified_universal_host_fence=False,
        qualified_global_drain=False,qualified_endpoint_reset=False,qualified_physical_design=False,
        scope='Two deterministic submission/close orderings in a single-producer testbench. Native BAR response/receipt observed. OS concurrency, all posted traffic, protected XDMA inventory and reset remain unqualified.',
        trace_limit='Zero-time model calls are reconstructed in log order and checked against aggregate VCD state; this is not asynchronous software race coverage.',
        native_register_correlation='Serial one-outstanding native reads; automatic PCIe tag allocation not traced.')
