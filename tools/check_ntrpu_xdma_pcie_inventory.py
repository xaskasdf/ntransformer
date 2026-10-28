"""Cross-correlate native endpoint RQ/RC with independent Root Port CQ/CC."""
from check_ntrpu_xdma_stop_inventory_r4 import vcd_rows, require, markers
from ntrpu_pcie_request_inventory import RequestInventory, AcceptedBeatMonitor
from ntrpu_pcie_vcd import logic_vcd_rows

CORE = '/board/EP/ntrpu_xdma_i/inst/pcie4_ip_i/'
RP = '/board/RP/'


def review_pcie(sim, vcd, native):
    start, finish = native['start_ps'], native['finish_ps']
    beats = []
    host_cq = []
    sequences = []
    monitors = {n:AcceptedBeatMonitor(n) for n in ('rq','rc')}
    for t, before, after in logic_vcd_rows(vcd, {CORE+p+'_tdata' for p in ('s_axis_rq','m_axis_rc')}):
        if t < start:
            continue
        require(all(n in after for n in [CORE+'user_clk', CORE+'user_reset', CORE+'user_lnk_up']),
                'missing declared PCIe observation boundary')
        require(after[CORE+'user_reset'] == 0 and after[CORE+'user_lnk_up'] == 1,
                'continuous subcore readiness')
        require(after[CORE+'user_clk'] == after['/board/EP/user_clk'], 'subcore clock identity')
        if before[CORE+'user_clk'] == 0 and after[CORE+'user_clk'] == 1:
            cycle = dict(time_ps=t)
            for label, prefix in [('rq','s_axis_rq_'),('rc','m_axis_rc_')]:
                p = CORE+prefix
                valid, ready = before[p+'tvalid'], before[p+'tready']
                payload = tuple(before[p+n] for n in ['tdata','tuser'])
                accepted = monitors[label].sample(valid,ready,*payload)
                if accepted is not None:
                    cycle[label] = accepted
            if len(cycle) > 1:
                beats.append(cycle)
            for slot in range(2):
                valid = before[CORE+'pcie_rq_seq_num_vld'+str(slot)]
                require(valid in (0,1), 'known sequence-number valid')
                if valid:
                    number = before[CORE+'pcie_rq_seq_num'+str(slot)]
                    require(number is not None and 0 <= number < 64, 'known sequence number')
                    sequences.append(dict(time_ps=t, slot=slot, number=number))
        if before[RP+'user_clk'] == 0 and after[RP+'user_clk'] == 1:
            if before[RP+'m_axis_cq_tvalid'] == before[RP+'m_axis_cq_tready'] == 1:
                host_cq.append([t]+[before[RP+'m_axis_cq_'+n] for n in ['tdata','tuser','tlast']])
    for monitor in monitors.values():
        monitor.finish()
    require(len(host_cq) == 5, 'native Root Port CQ inventory')
    requesters = {(row[1] >> 80) & 65535 for row in host_cq[:3]}
    require(len(requesters) == 1, 'single generated requester identity')
    requester = requesters.pop()
    model = RequestInventory({requester & 7: requester},ari_enabled=False)
    for cycle in beats:
        model.cycle(**cycle)
    model.finish()
    reads = [r for r in model.requests if r['kind']=='read']
    writes = [r for r in model.requests if r['kind']=='write']
    require([(r['address'],r['requested_bytes']) for r in reads] == [(0x100,32),(0x400,64),(0x200,32)],
            'endpoint descriptor/H2C request set')
    require(len(writes)==1 and (writes[0]['address'],writes[0]['requested_bytes'])==(0x600,64),
            'endpoint C2H request set')
    require(len(model.completed_reads)==len(model.completions)==3, 'native endpoint completion set')
    expected = {}
    for address, src, dst in [(0x100,0x400,0x1000),(0x200,0x1000,0x600)]:
        raw = bytearray(32)
        for offset,value in [(0,0xad4b0013),(4,64),(8,src),(16,dst)]:
            raw[offset:offset+4]=value.to_bytes(4,'little')
        expected[address] = bytes(raw)
    expected[0x400] = native['accepted']['m_axi']['w'][0][1].to_bytes(64,'little')
    correlations = []
    for request, received, host, cq in zip(reads,model.completed_reads,native['native']['host_reads'],host_cq[:3]):
        require(request['serial']==received['serial'] and received['successful'], 'native read transaction identity')
        require(received['payload']==expected[request['address']], 'endpoint all descriptor/H2C bytes')
        require(request['tag']==host['tag']==((cq[1]>>96)&255), 'endpoint/host tag identity')
        require(request['effective_requester_id']==(cq[1]>>80)&65535 and
                request['tc']==(cq[1]>>121)&7 and request['effective_attributes']==(cq[1]>>124)&7,
                'endpoint/host requester and attributes')
        require(request['end_ps']<host['request_ps']<host['completion_ps']<received['payload_end_ps'],
                'endpoint request, host fetch/completion, endpoint receipt causality')
        correlations.append(dict(address=request['address'],tag=request['tag'],bytes=request['requested_bytes'],
            endpoint_request_ps=request['end_ps'],host_request_ps=host['request_ps'],host_completion_ps=host['completion_ps'],
            terminal_header_ps=received['terminal_header_ps'],payload_end_ps=received['payload_end_ps'],
            request_to_payload_ps=received['payload_end_ps']-request['end_ps']))
    host_begin=int(markers(sim,'XDMA_LOCAL_HOST_BEGIN')[0]['t'])
    host_end=int(markers(sim,'XDMA_LOCAL_HOST_END')[0]['t'])
    require(writes[0]['enabled_payload']==expected[0x400], 'C2H endpoint all payload bytes')
    require(writes[0]['end_ps']<host_begin<=host_end, 'posted acceptance distinct from host receipt')
    require(len(sequences)==len(model.requests)==4, 'native sequence notification inventory')
    for request, seq in zip(model.requests,sequences):
        require(request['sequence_number']==seq['number'] and request['start_ps']<=seq['time_ps'],
                'native request sequence correlation')
    return dict(status='PASS_NATIVE_PCIE_MEMORY_INVENTORY_CAPTURE', requester_id=requester,
        requests=model.requests,completions=model.completions,reads=model.completed_reads,
        correlations=correlations,timeline=model.timeline,peak_tags=model.peak_tags,
        peak_outstanding_reads=model.peak_reads,peak_span_bytes=model.peak_span_bytes,
        rq_beats=model.rq.accepted_beats,rc_beats=model.rc.accepted_beats,
        rq_max_sop=model.rq.max_starts_per_beat,rc_max_sop=model.rc.max_starts_per_beat,
        stalled_cycles={n:m.stalled_cycles for n,m in monitors.items()},sequence_observations=sequences,final_empty=model.empty,
        posted_end_ps=writes[0]['end_ps'],host_begin_ps=host_begin,host_end_ps=host_end,
        posted_acceptance_to_host_receipt_ps=host_end-writes[0]['end_ps'],
        production_observation_boundary_qualified=False,global_drain_qualified=False,
        scope='Native hierarchical diagnostic and offline inventory; synthesized provider, closure, full-capacity native stress and physical connection remain required.')
