"""Review the R1 discovery responder in addition to the native management traffic."""
from check_ntrpu_xdma_local_management_direct import review as review_management
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review(sim, vcd):
    result = review_management(sim, vcd)
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
