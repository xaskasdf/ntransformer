"""Audit register-stop evidence using public request, AXI and closure events.

This checker does not read DUT FSM state. It reconstructs each transaction,
each CSR round and every accepted session, and checks the final 89 cases.
The register slave is a fixture, not the AMD XDMA simulation model.
"""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows as raw_vcd_rows

PREFIX = '/tb_ntrpu_xdma_stop_registers/'
ADDRESSES = [0xC, 0x100C, 4, 0x40, 0x48, 0x1004, 0x1040, 0x1048]
ERROR_BITS = [{*range(19,24), *range(9,16), 3,4,5}, {*range(19,24),9,10,3,4,5}]
CASES = {1:(0,1),2:(0,1),3:(32,0),4:(32,0),5:(512,0),6:(512,0),7:(64,0),
         8:(32,0),9:(16,0),10:(16,0),11:(1,0),12:(4,0),13:(8192,0),14:(2048,0),
         15:(2,0),16:(8,0),17:(1,0),18:(1024,0),19:(2,0),20:(8,0),21:(4096,0),
         22:(4096,0),23:(8192,0),24:(1,0),25:(1,0)}
for channel in (0,1):
    for bit in range(32):
        CASES[100+channel*100+bit] = ((1024,0) if bit==0 else (0,1) if bit in (1,2,6)
                                     else (128,0) if bit in ERROR_BITS[channel] else (256,0))




def settled_rows(vcd, netlist):
    outputs = {'start_ready','active','generation','tag','expected_published','faults',
        'snapshot_valid','control_snapshot','status_snapshot','count_snapshot','stop_responses',
        'read_responses','transaction_pending','m_admit','m_req_valid','m_req_write','m_req_address',
        'm_req_data','m_req_strobe','m_req_identity','m_rsp_ready','m_req_ready','master_req_ready',
        'm_rsp_valid','m_rsp_data','m_rsp_code','m_rsp_identity','m_rsp_timeout','m_timeout_latched',
        'master_rsp_valid','master_rsp_identity','master_busy','awaddr','awprot','awvalid',
        'wdata','wstrb','wvalid','bready','araddr','arprot','arvalid','rready'}
    group_time=group_before=group_after=None
    for t,b,a in raw_vcd_rows(vcd):
        if all(v is None for v in a.values()):continue
        delayed=bool(netlist and (t==100 or t%4000==2100))
        if delayed:
            require(group_time==t-100, 'delayed outputs follow their original clock or initialization')
            changed={n.rsplit('/',1)[-1] for n in a if b[n]!=a[n]}
            require(changed<=outputs, 'only sequencer and master outputs may have verified cell delay')
        if not netlist or t>=100:require(all(v is not None for v in a.values()), 'known values after initialization')
        base=t-100 if delayed else t
        if group_time is not None and base!=group_time:
            yield group_time,group_before,group_after
            group_time=None
        if group_time is None:group_time,group_before=base,b
        group_after=a
    if group_time is not None:yield group_time,group_before,group_after


def held_checkpoint_rows(vcd, checkpoints, finish, netlist):
    queries=iter(sorted(checkpoints));query=next(queries,None)
    last=None;last_time=None
    for t,b,a in settled_rows(vcd,netlist):
        while query is not None and query<t:
            require(last is not None and last_time<=query, 'checkpoint inside observed trace')
            yield query,last,last.copy()
            query=next(queries,None)
        yield t,b,a
        if query==t:query=next(queries,None)
        last,last_time=a,t
    # XSIM emits transitions, not display timestamps. The last display is at
    # the bench's known 100ps offset after its last observed falling edge.
    require(last_time<=finish<=last_time+100, 'finish immediately follows last observed transition')
    while query is not None:
        require(last_time<query<=finish, 'trailing checkpoint bounded by native finish')
        yield query,last,last.copy()
        query=next(queries,None)


def review(sim, vcd, *, netlist):
    require(netlist in (0,1), 'declared replay mode')
    require(markers(sim,'STOP_REG_REPLAY')==[dict(netlist=str(netlist),master_netlist=str(netlist))], 'replay mode marker')
    require(markers(sim,'STOP_REG_PASS') == [dict(cases='89',master_wait_limit='32',
            native_xdma='0',global_drain='0',reset_permission='0')], 'complete fixture scope')
    require(not re.search(r'^\s*(?:ERROR:|CRITICAL WARNING:|Fatal:|Failure:)',sim,re.M), 'simulation errors')
    cases=markers(sim,'STOP_REG_CASE')
    require([int(c['phase']) for c in cases] == list(CASES), '89 ordered cases including all64 CSR bits')
    checkpoints={int(c['t']):c for c in cases}
    require(len(checkpoints)==89, 'distinct checkpoint timestamps')
    observations=[];requests=[];responses=[];bus_replies=[];starts=[];captures=[]
    session=None;pending=None;next_index=0;last_clk=None;old_fault=0;last_identity=None
    snapshot_values=[0]*6;stops=0;read_mask=0;previous_counts=[0,0]
    samples=0;stalled_requests=0;revoked_stalls=0;aw_stalls=0;w_stalls=0;ar_stalls=0
    b_late=0;r_late=0;previous=None;last_qualified=False;retired=False

    finish_times=re.findall(r'\$finish called at time : (\d+) ps',sim)
    require(len(finish_times)==1, 'one native simulation finish')
    for t,b,a in held_checkpoint_rows(vcd,checkpoints,int(finish_times[0]),netlist):
        b={n[len(PREFIX):]:v for n,v in b.items() if n.startswith(PREFIX)}
        a={n[len(PREFIX):]:v for n,v in a.items() if n.startswith(PREFIX)}
        if not a or all(v is None for v in a.values()):continue
        require(all(v is not None for v in a.values()),'known public evidence')
        edge=b.get('clk')==0 and a['clk']==1
        if edge:
            if last_clk is not None:require(t-last_clk==4000,'250MHz simulation clock')
            last_clk=t
            if not b['cold_rst_n']:
                session=None;pending=None;next_index=0;last_identity=None;old_fault=0
                snapshot_values=[0]*6;stops=0;read_mask=0;previous_counts=[0,0];retired=False
            else:
                require((a['faults']|old_fault)==a['faults'], 'faults retained until cold configuration')
                old_fault=a['faults']
                if b['start_valid'] and b['start_ready']:
                    require(not b['active'] and not b['faults'], 'exclusive session admission')
                    identity=(b['start_generation'],b['start_tag'])
                    require(all(identity) and (last_identity is None or identity>last_identity),'fresh lexicographic identity')
                    last_identity=identity
                    session=dict(identity=identity,published=b['ledger_published'],phase=b['phase'],captured=False)
                    require(pending is None,'no outstanding request across session admission')
                    next_index=0;snapshot_values=[0]*6;stops=0;read_mask=0;previous_counts=[0,0];retired=False
                    starts.append(dict(t=t,phase=b['phase'],identity=identity,published=b['ledger_published']))
                if b['m_req_valid'] and b['m_req_ready']:
                    require(session is not None and pending is None,'one owned request at a time')
                    i=next_index
                    require(b['m_req_address']==ADDRESSES[i], 'two W1C operations then ordered six-register rounds')
                    require(b['m_req_write']==int(i<2) and b['m_req_data']==int(i<2) and
                            b['m_req_strobe']==(15 if i<2 else 0),'only W1C Run may write with full strobes')
                    identity=(session['identity'][0]<<64)|session['identity'][1]
                    require(b['m_req_identity']==identity,'request belongs to retained identity')
                    if i==2:read_mask=0
                    pending=dict(t=t,phase=b['phase'],index=i,address=b['m_req_address'],write=b['m_req_write'],
                                 identity=identity,aw=None,w=None,ar=None,reply=None)
                    requests.append({k:pending[k] for k in ('t','phase','index','address','write','identity')})
                    next_index=2 if i==7 else i+1
                for v,r,field in [('awvalid','awready','aw'),('wvalid','wready','w'),('arvalid','arready','ar')]:
                    if b[v] and b[r]:
                        require(pending is not None and pending[field] is None,'unique owned AXI request beat')
                        require(bool(pending['write'])==(field!='ar'),'AXI direction matches request')
                        if field=='w':require(b['wdata']==1 and b['wstrb']==15,'actual W1C payload and strobes')
                        else:require(b[field+'addr']==pending['address'] and b[field+'prot']==0,'actual AXI address/protection')
                        pending[field]=t
                for v,r,kind in [('bvalid','bready','B'),('rvalid','rready','R')]:
                    if not (b[v] and b[r]):continue
                    require(pending is not None and pending['reply'] is None,'response has actual pending request')
                    require(bool(pending['write'])==(kind=='B'),'response channel matches request')
                    require((pending['aw'] is not None and pending['w'] is not None) if kind=='B' else pending['ar'] is not None,'response follows accepted address/data')
                    code=b['bresp' if kind=='B' else 'rresp'];data=0 if kind=='B' else b['rdata']
                    pending['reply']=dict(t=t,kind=kind,code=code,data=data)
                    bus_replies.append(dict(t=t,phase=b['phase'],kind=kind,address=pending['address'],
                                            code=code,data=data,latency_ps=t-pending['t']))
                    if b['m_timeout_latched']:
                        if kind=='B':b_late+=1
                        else:r_late+=1
                if b['master_rsp_valid'] and b['m_rsp_ready']:
                    require(pending is not None and pending['reply'] is not None,'master never fabricates a response')
                    reply=pending['reply'];i=pending['index']
                    require((b['m_rsp_data'],b['m_rsp_code'])==(reply['data'],reply['code']),'master preserves actual B/R response')
                    require(b['master_rsp_identity']==pending['identity'],'master response retains accepted identity')
                    require(not b['m_timeout_latched'] or b['m_rsp_timeout'],'late report retains timeout')
                    required_fault=(16 if b['m_rsp_timeout'] else 0)|(32 if b['m_rsp_code'] else 0)|(
                        64 if b['m_rsp_identity']!=pending['identity'] else 0)
                    if not required_fault:
                        if i<2:stops|=1<<i
                        else:
                            read_mask|=1<<(i-2);snapshot_values[i-2]=b['m_rsp_data']
                            if i in (3,6):
                                ch=int(i==6);value=b['m_rsp_data']
                                errors=sum(1<<bit for bit in ERROR_BITS[ch])
                                if value&errors:required_fault|=128
                                if value&~(errors|0x47)&0xffffffff:required_fault|=256
                            if i in (4,7):
                                ch=int(i==7);value=b['m_rsp_data'];expected=(session['published']>>(32*ch))&0xffffffff
                                if value>expected or value<previous_counts[ch]:required_fault|=512
                                previous_counts[ch]=value
                    require((a['faults']&required_fault)==required_fault,'register/transport fault latches at actual report')
                    responses.append(dict(t=t,phase=b['phase'],address=pending['address'],data=b['m_rsp_data'],
                        code=b['m_rsp_code'],identity=b['m_rsp_identity'],timeout=b['m_rsp_timeout']))
                    pending=None
                if b['release_valid'] and b['snapshot_valid']:
                    require(not a['active'],'qualified global-caller retirement releases local session')
                    retired=True

        if session is not None and a['cold_rst_n']:
            require((a['generation'],a['tag'])==session['identity'],'session identity retained')
            require(a['expected_published']==session['published'],'captured publication inventory retained')
            require(a['stop_responses']==stops,'successful actual B response mask')
            expected_snapshots=(snapshot_values[0]|(snapshot_values[3]<<32),
                                snapshot_values[1]|(snapshot_values[4]<<32),
                                snapshot_values[2]|(snapshot_values[5]<<32))
            require((a['control_snapshot'],a['status_snapshot'],a['count_snapshot'])==expected_snapshots,
                    'raw snapshots come only from successful actual register responses')
            context=bool(a['core_ready'] and a['exclusive_owner'] and not a['external_fault'] and not a['wait_expired'] and
                a['close_request'] and a['host_closed'] and (a['host_generation'],a['host_tag'])==session['identity'] and
                a['ledger_active'] and a['ledger_closed'] and (a['ledger_generation'],a['ledger_tag'])==session['identity'] and
                not a['ledger_faults'] and a['ledger_published']==session['published'] and not a['m_timeout_latched'] and
                a['ledger_accounted'] and a['ledger_completed']==session['published'] and not a['start_valid'] and not a['m_rsp_valid'])
            if not context or a['faults'] or retired:require(not a['snapshot_valid'],'immediate fail-closed context revocation')
            if a['snapshot_valid']:
                require(a['active'] and not a['transaction_pending'] and pending is None,'snapshot has no local management obligation')
                require(stops==3 and read_mask==63 and a['read_responses']==63,'snapshot has both real stops and complete current round')
                require(not snapshot_values[0]&1 and not snapshot_values[3]&1 and
                        not snapshot_values[1]&1 and not snapshot_values[4]&1,'both channels observed Run0 and Busy0')
                require(expected_snapshots[2]==session['published'],'all published descriptors accounted by both CSR counts')
                if not last_qualified:captures.append(dict(t=t,phase=a['phase'],published=session['published']))
                session['captured']=True
        elif not a['cold_rst_n']:
            require(not a['snapshot_valid'] and not a['start_ready'],'cold reset never qualifies')

        # Examine consecutive rising-edge samples, independent of internal state.
        if edge and b['cold_rst_n']:
            if previous and previous['cold_rst_n']:
                for valid,ready,payload in [('m_req_valid','m_req_ready',['m_req_write','m_req_address','m_req_data','m_req_strobe','m_req_identity']),
                    ('awvalid','awready',['awaddr','awprot']),('wvalid','wready',['wdata','wstrb']),('arvalid','arready',['araddr','arprot'])]:
                    if previous[valid] and not previous[ready]:
                        require(b[valid] and all(b[n]==previous[n] for n in payload),'offered request remains stable through stall or fault')
                        if valid=='m_req_valid':
                            stalled_requests+=1
                            if previous['faults']:revoked_stalls+=1
                        elif valid=='awvalid':aw_stalls+=1
                        elif valid=='wvalid':w_stalls+=1
                        else:ar_stalls+=1
            previous=b.copy()
        elif edge:previous=None
        if t in checkpoints:
            c=checkpoints[t];phase=int(c['phase'])
            require((a['faults'],a['snapshot_valid'])==CASES[phase],'independent case outcome '+str(phase))
            for field,name,base in [('faults','faults',16),('valid','snapshot_valid',2),('active','active',2),
                                  ('writes','write_count',10),('reads','read_count',10),('pending','transaction_pending',2)]:
                require(int(c[field],base)==a[name],'checkpoint agrees with waveform '+field)
            observations.append(dict(phase=phase,t=t,faults=a['faults'],valid=a['snapshot_valid'],
                                     writes=a['write_count'],reads=a['read_count'],pending=a['transaction_pending']))
        last_qualified=bool(a['snapshot_valid']);samples+=1

    require(len(observations)==89,'all checkpoint timestamps observed')
    require(len(starts)==89 and len(captures)==17,'all sessions and successful/revoked captures exercised')
    require(b_late==1 and r_late==1,'both real delayed responses after master timeout observed')
    require(stalled_requests>=30 and revoked_stalls>=20 and min(aw_stalls,w_stalls,ar_stalls)>0,'real request stalls including revoked offers')
    require(observations[0]['reads']>=24 and observations[0]['writes']==2,'live/lagging inventory retried before qualification')
    require(observations[1]['writes']==4,'fresh second session without cold reset')
    require(observations[2]['writes']==2 and observations[2]['reads']==0,'first B error still attempts both stops before halting')
    require(observations[8]['writes']==1 and observations[8]['pending']==1,'timeout keeps unaccepted second stop offered')
    finish=re.findall(r'\$finish called at time : (\d+) ps',sim)
    require(finish==[str(t)] and 0<t<200000000,'bounded native XSIM finish matches waveform')
    return dict(scenarios=observations,starts=starts,captures=captures,requests=requests,responses=responses,
                axi_responses=bus_replies,waveform_samples=samples,request_stall_cycles=stalled_requests,
                revoked_request_stall_cycles=revoked_stalls,axi_stall_cycles=dict(AW=aw_stalls,W=w_stalls,AR=ar_stalls),
                late_responses=dict(B=b_late,R=r_late),max_request_to_axi_response_ps=max(x['latency_ps'] for x in bus_replies),
                last_evidence_ps=t,scope='Public AXI-Lite register fixture; no native XDMA, driver, host receipt, global drain or operational reset proof')
