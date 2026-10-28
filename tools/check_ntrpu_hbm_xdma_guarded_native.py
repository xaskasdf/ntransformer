"""Reconstruct native dual-C2H payload/writeback from public VCD handshakes."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_xdma_stop_inventory_r4 import require, markers
from ntrpu_pcie_vcd import logic_vcd_rows
from ntrpu_pcie_straddle import LogicWord
from check_ntrpu_xdma_stop_inventory_r4 import vcd_rows
from check_ntrpu_hbm_xdma_guarded import management

LENGTHS = [64, 128, 256, 4096]
DEST = [0x100000800 + ((i % 2) << 32) + (i // 2) * 4096 for i in range(4)]
WB = [0x300001000 + (i % 2) * 128 + (i // 2) * 64 for i in range(4)]
RTL = {
    'ntrpu_bridge_async_fifo': 'be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634',
    'ntrpu_hbm_dma_bridge': 'b4a10f677311bfdf29b00a353bb0d83d6a4e5181ca228c1d5b5da34d86d3848a',
    'ntrpu_hbm_dma_bridge_lane': '7e987b09ae1157251acd460b5511f041f59ab75076ecc62b4dcb0aa5a4926a24',
    'ntrpu_peer_read_pkg': '7d0b994ef8244710d4d2e8ed9ed4960685ad6a89ed5ffa76708439b2dcce603b',
}


GUARDED_RTL = {'fpga/experimental/ntrpu_hbm_xdma_guarded.sv': '32552bf0c620a464aeda5602bf47ae352cdcb51f63bbd10a252ab90ad35f62b2', 'fpga/experimental/ntrpu_xdma_completion_guard.sv': '6d2cfe560b8d582c7edd1ed83b8b6dda24ff7809a587758a8f8caa2e42e3c89e', 'fpga/rtl/ntrpu_xdma_local_master.sv': '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(text, reg_reads):
    tb, ep, rp = '/board/probe/', '/board/EP/', '/board/RP/'
    events = dict(descriptors=[], axis=[], done=[], validated=[], cq=[], results=[], admitted=[], ack=[], clocks=[], phases=[])
    queries={int(r['t']):int(r['value'],16) for r in reg_reads}
    checked_queries=set()
    last_edge = None
    stalled = [None, None]
    words={tb+n for n in ('dsc_src','dsc_dst','dsc_len','dsc_ctl','axis_data','axis_keep','axis_last')}
    words.add(rp+'m_axis_cq_tdata')
    for t, before, after in logic_vcd_rows(text,words):
        phase = after[tb+'phase']
        if phase not in (1, 2, 3, 4):
            continue
        if not events['phases'] or events['phases'][-1] != phase:
            events['phases'].append(phase)
        require(after[ep+'user_resetn'] == after[ep+'user_lnk_up'] == 1, 'link/reset continuity')
        if t in queries:
            require(queries[t] in (before[rp+'tx_usrapp/P_READ_DATA'],after[rp+'tx_usrapp/P_READ_DATA']), 'register log/VCD value')
            checked_queries.add(t)
        if before[ep+'user_clk'] == 0 and after[ep+'user_clk'] == 1:
            if last_edge is not None:
                require(t-last_edge == 4000, 'effective native AXI period')
            last_edge = t
            events['clocks'].append(t)
            for c in range(2):
                def part(name, width=1):
                    v = before[tb+name]
                    require(v is not None, 'unknown public '+name)
                    if isinstance(v,LogicWord):
                        mask=((1<<width)-1)<<(c*width)
                        require(v.known&mask==mask,'unknown active channel '+name)
                        v=v.value
                    return (v >> (c*width)) & ((1 << width)-1)
                if part('native_status',8) & 8:
                    events['done'].append([t,c,part('native_status',8)])
                if part('dsc_load'):
                    require(part('dsc_ready') and part('native_status',8) & 64, 'descriptor Ready/Run')
                    events['descriptors'].append([t,c,part('dsc_src',64),part('dsc_dst',64),part('dsc_len',28),part('dsc_ctl',16)])
                valid, ready = part('axis_valid'), part('axis_ready')
                payload = (part('axis_data',512),part('axis_keep',64),part('axis_last')) if valid else None
                if stalled[c] is not None:
                    require(valid and payload == stalled[c], 'native stream stability')
                stalled[c] = payload if valid and not ready else None
                if valid and ready:
                    events['axis'].append([t,c,*payload])
                if part('dma_done'):
                    require(phase in (2,3) and part('dma_error') == 0, 'autonomous validated Done phase')
                    events['validated'].append([t,c])
        if before[rp+'user_clk'] == 0 and after[rp+'user_clk'] == 1:
            if before[rp+'m_axis_cq_tvalid'] == 1 and before[rp+'m_axis_cq_tready'] == 1:
                values=[before[rp+'m_axis_cq_'+n] for n in ('tdata','tkeep','tlast','tuser')]
                require(None not in values, 'known native CQ payload')
                data,keep,last,user=values
                events['cq'].append([t,data.value,data.known,keep,last,user])
        if before[tb+'clk']==0 and after[tb+'clk']==1:
            if before[tb+'submit_valid']==before[tb+'submit_ready']==1:
                events['admitted'].append([t,before[tb+'submit_channel'],before[tb+'submit_context']])
            for c in range(2):
                if before[tb+'ack_valid'] is not None and before[tb+'ack_valid']>>c&1:
                    events['ack'].append([t,c])
                if before[tb+'result_valid'] is not None and before[tb+'result_valid']>>c&1:
                    if before[tb+'result_ready']>>c&1:
                        values=[before[tb+n] for n in ('result_context','result_status','result_bytes')]
                        require(None not in values,'known result')
                        events['results'].append([t,c,(values[0]>>(c*1024))&((1<<1024)-1),
                                                 (values[1]>>(c*4))&15,(values[2]>>(c*64))&((1<<64)-1)])
        if phase == 4:
            require(after[tb+'contexts_held'] == 0, 'final ownership released')
    require(events['phases'] == [1,2,3,4] and len(events['clocks']) > 32, 'complete native phases')
    require(checked_queries==set(queries),'all native register reads cross-checked')
    return events


def check_events(events):
    descriptors = [[],[]];streams = [bytearray(),bytearray()];packets=[[],[]]
    for t,c,src,dst,length,ctl in events['descriptors']:
        i=c+2*len(descriptors[c]);require(i<4,'descriptor count')
        require([src,dst,length,ctl]==[WB[i],DEST[i],LENGTHS[i],16], 'native descriptor content')
        descriptors[c].append(t)
    for t,c,data,keep,last in events['axis']:
        require(keep==(1<<64)-1,'full stream keep')
        streams[c].extend(data.to_bytes(64,'little'))
        if last:
            i=c+2*len(packets[c]);require(i<4,'packet count')
            expected=bytes(((j*37)^(j>>7)^((i+1)*53))&255 for j in range(LENGTHS[i]))
            require(bytes(streams[c])==expected and descriptors[c][len(packets[c])]<t,'AXIS bytes/order')
            packets[c].append(t);streams[c].clear()
    require([len(x) for x in descriptors]==[2,2] and [len(x) for x in packets]==[2,2] and not any(streams), 'four complete input packets')
    raw=[[],[]];validated=[[],[]]
    for t,c,status in events['done']:
        require(len(raw[c])<2 and t>=packets[c][len(raw[c])], 'attributed native Done')
        raw[c].append(t)
    for t,c in events['validated']:
        require(len(validated[c])<len(raw[c]) and t>raw[c][len(validated[c])], 'stored native Done credit')
        validated[c].append(t)
    require([len(x) for x in raw]==[2,2] and [len(x) for x in validated]==[2,2], 'four native and validated Done events')
    admitted=[[],[]];retired=[0,0];acks=[[],[]]
    for t,c,context in events['admitted']:
        require(c in (0,1) and context is not None,'known admission');admitted[c].append([t,context])
    require([len(x) for x in admitted]==[2,2],'four admissions')
    for t,c in events['ack']:
        require(t>max(x for v in validated for x in v),'ACK after native validation');acks[c].append(t)
    require([len(x) for x in acks]==[2,2],'four ACKs')
    for t,c,context,status,length in events['results']:
        q=retired[c];require(q<2,'no duplicate result')
        at,expected=admitted[c][q]
        require(context==expected and status==0 and length==LENGTHS[c+2*q] and t>max(at,validated[c][q],acks[c][q]), 'attributed result and causality')
        retired[c]+=1
    require(retired==[2,2],'four retired contexts')
    memory={};active=None;packet_count=0;first=0
    for t,data,known,keep,last,user in events['cq']:
        if active is None:
            require(known&((1<<128)-1)==(1<<128)-1,'known CQ header')
            require((user>>80)&3==1 and (data>>75)&15==1 and user&65535==0xf0f,'CQ Memory Write header')
            address=data&((1<<64)-1);length=((data>>64)&2047)*4
            require(length>0,'CQ nonzero length');active=[address,length,0];first=16
        else:
            require((user>>80)&3==0,'no straddled/nested CQ packet');first=0
        require((user>>96)&1==0,'CQ discontinue')
        for lane in range(64):
            if (user>>(16+lane))&1:
                address,length,offset=active
                require(lane>=first and (keep>>(lane//4))&1 and offset<length,'CQ payload byte enable')
                actual=address+offset
                require(actual not in memory,'duplicate native byte')
                require((known>>(lane*8))&255==255,'known enabled CQ byte')
                memory[actual]=(data>>(lane*8))&255;active[2]+=1
        if last:
            require(active[1]==active[2],'CQ payload length');active=None;packet_count+=1
    require(active is None,'complete CQ stream')
    expected={}
    for i,n in enumerate(LENGTHS):
        expected.update({DEST[i]+j:((j*37)^(j>>7)^((i+1)*53))&255 for j in range(n)})
        expected.update({WB[i]+j:v for j,v in enumerate((0x52b40001 | (n<<32)).to_bytes(8,'little'))})
    require(memory==expected,'full-address native memory and metadata writeback')
    return dict(payload_bytes=sum(LENGTHS),writeback_bytes=32,cq_packets=packet_count,
                native_done=4,validated_done=4,descriptors=4,axi_period_ps=4000,
                memory_sha256=hashlib.sha256(b''.join(a.to_bytes(8,'little')+bytes([memory[a]]) for a in sorted(memory))).hexdigest())


def review(stage):
    out=stage/'output';result=json.loads((out/'result.json').read_text())
    for n,h in json.loads((out/'FILES.json').read_text()).items():require(sha(out/n)==h,'changed output '+n)
    require(result['status']=='PENDING_REVIEW' and result['returncode']==0 and not result['timed_out'] and result['seed_unchanged'] and result.get('cache_unchanged',True),'clean native run')
    require(result['sources']==result['sources_after'],'frozen inputs')
    for n,h in result['sources'].items():require(sha(stage/'repo'/n)==h,'changed native source')
    for n,h in RTL.items():require(sha(stage/'repo/fpga/experimental'/(n+'.sv'))==h,'different qualified rev3 RTL')
    for n,h in GUARDED_RTL.items():require(sha(stage/'repo'/n)==h,'different locally verified guarded RTL '+n)
    xci=0
    for n,h in result['seed_files'].items():
        if n.endswith('.xci'):
            require(sha(out/'export/example'/n)==h,'native IP configuration changed');xci+=1
    require(xci>0,'native XCI provenance')
    ep=(out/'export/example/imports/xilinx_dma_pcie_ep.sv').read_text().split('  // XDMA target application')[0]
    for c in range(2):
        for prefix,fields in [('c2h_dsc_byp',{'ready':f'dsc_ready[{c}]','load':f'dsc_load[{c}]',
                              'src_addr':f'dsc_src[{64*c}+:64]','dst_addr':f'dsc_dst[{64*c}+:64]',
                              'len':f'dsc_len[{28*c}+:28]','ctl':f'dsc_ctl[{16*c}+:16]'}),
                              ('s_axis_c2h',{'tdata':f'axis_data[{512*c}+:512]','tkeep':f'axis_keep[{64*c}+:64]',
                               'tvalid':f'axis_valid[{c}]','tready':f'axis_ready[{c}]','tlast':f'axis_last[{c}]'})]:
            for field,signal in fields.items():
                port=f'{prefix}_{field}_{c}'
                require(re.findall(r'\.'+port+r'\s*\(([^)]*)\)',ep)==['board.probe.'+signal], 'native public binding '+port)
        require(re.findall(r'\.c2h_sts_'+str(c)+r'\s*\(([^)]*)\)',ep)==[f'board.probe.native_status[{8*c}+:8]'], 'native status binding')
    for suffix in 'awaddr awprot awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready'.split():
        require(re.findall(r'\.s_axil_'+suffix+r'\s*\(([^)]*)\)',ep)==['board.probe.m_axil_'+suffix],'native management port '+suffix)
    order=(out/'reports/compile_order.rpt').read_text()
    for n in ('ntrpu_hbm_dma_bridge.sv','ntrpu_hbm_dma_bridge_lane.sv','ntrpu_bridge_async_fifo.sv','ntrpu_hbm_xdma_guarded_native.sv','ntrpu_hbm_xdma_guarded.sv','ntrpu_xdma_completion_guard.sv','ntrpu_xdma_local_master.sv'):
        require(n in order,'missing compiled bridge source '+n)
    log=(out/'console.log').read_text();require('HX_SIMULATOR_RETURNED' in log and log.count('HX PASS ')==1,'native completion markers')
    require('Check Max Link Speed = 8.0GT/s - PASSED' in log and
            "Check Negotiated Link Width = 5'h10 - PASSED" in log,'native link capability and negotiated width')
    require(not re.search(r'^\s*(?:Fatal:|FATAL:|Error:|ERROR:|CRITICAL WARNING:)',log,re.M),'native simulator diagnostic')
    warnings=Counter(re.findall(r'^WARNING: \[([^]]+)\]',log,re.M))
    require(warnings=={'Project 1-5713':1,'VRFC 10-3380':1,'XSIM 43-3431':1,'VRFC 10-9543':52,'VRFC 10-5021':7,
                      'VRFC 10-3027':1,'VRFC 10-3705':70,'XSIM 43-4328':1}, 'changed native warning inventory')
    require("[VRFC 10-3380] identifier 'validated_done' is used before its declaration" in log,
            'reviewed fixture declaration-order warning')
    runtime=[l for l in log.splitlines() if l.startswith('WARNING:') and not l.startswith('WARNING: [')]
    require(len(runtime)==1 and runtime[0].startswith('WARNING: 0ns : none of the conditions were true for unique case')
            and runtime[0].endswith('xdma_v4_2_vl_rfs.sv:34099'), 'unreviewed runtime warning')
    regs=markers(log,'HX REG')
    bar=markers(log,'HX BAR');require(len(bar)==1 and bar[0]['selected']=='1' and bar[0]['discovered'] in ('0','1'),'explicit native BAR selection')
    events=extract((out/'reports/native.vcd').read_text(),regs);metrics=check_events(events)
    # Compare every logged handshake with the independently sampled VCD.
    for name,key,fields in [('DESC','descriptors',[('t',10),('channel',10),('src',16),('dst',16),('len',10),('ctl',16)]),
                            ('DONE','done',[('t',10),('channel',10),('status',16)])]:
        actual=[[int(r[k],base) for k,base in fields] for r in markers(log,'HX '+name)]
        require(actual==events[key],'log/VCD identity '+name)
    cq=[]
    for r in markers(log,'HX CQ'):
        raw=''.join(c*4 if c in 'xz' else format(int(c,16),'04b') for c in r['data'].lower())
        word=LogicWord.parse(raw,512)
        cq.append([int(r['t']),word.value,word.known,int(r['keep'],16),int(r['last'],2),int(r['user'],16)])
    require(cq==events['cq'],'log/VCD identity CQ')
    axis=[[int(r['t']),int(r['channel']),int(r['data'],16),(1<<64)-1,int(r['last'],2)] for r in markers(log,'HX AXIS')]
    require(axis==events['axis'],'log/VCD identity AXIS')
    require(len(regs)==5,'host configuration reads only')
    expected=[(0,'identifier'),(0x1040,0),(0x1004,0xf83e7f),(0x1140,0),(0x1104,0xf83e7f)]
    for r,(address,value) in zip(regs,expected):
        actual=int(r['value'],16);require(int(r['address'],16)==address,'management address sequence')
        require(actual>>16==0x1fc0 if value=='identifier' else actual&~0x47==0 if value is None else actual==value,'native management value')
    require(int(regs[-1]['t'])<min(t for t,c,context in events['admitted']),'host configuration before traffic')
    management_result=review_native_management((out/'reports/native.vcd').read_text())
    require(management_result['credits']==events['validated'],'credits originate from autonomous management')
    retired=[(int(r['channel']),int(r['index']),int(r['bytes'])) for r in markers(log,'HX RESULT')]
    require(sorted(retired)==[(0,0,64),(0,1,256),(1,0,128),(1,1,4096)],'all attributed results')
    logged_results=[[int(r['t']),int(r['channel']),int(r['bytes'])] for r in markers(log,'HX RESULT')]
    require(logged_results==[[t,c,length] for t,c,context,status,length in events['results']], 'log/VCD identity result')
    negatives=[]
    for name in ('descriptor_high_address','descriptor_writeback','stream_payload','writeback_payload','lost_done','wrong_result_context','early_ack'):
        bad=copy.deepcopy(events)
        if name=='descriptor_high_address':bad['descriptors'][0][3]&=0xffffffff
        elif name=='descriptor_writeback':bad['descriptors'][0][2]+=64
        elif name=='stream_payload':bad['axis'][0][2]^=1
        elif name=='lost_done':bad['done'].pop()
        elif name=='wrong_result_context':bad['results'][0][2]^=1
        elif name=='early_ack':bad['ack'][0][0]=0
        else:
            index=next(i for i,row in enumerate(bad['cq']) if row[1]&((1<<64)-1) in WB)
            bad['cq'][index][1]^=1<<128
        try:check_events(bad)
        except ValueError as error:negatives.append(dict(name=name,detected=True,reason=str(error)))
        else:raise ValueError('Accepted altered native evidence: '+name)
    return dict(status='REV3_GUARDED_NATIVE_DUAL_C2H_PASS',management=management_result,**metrics,seconds=result['seconds'],negative_controls=negatives,
                unchanged_xci=xci,discovered_dma_bar=int(bar[0]['discovered']),selected_dma_bar=1,
                warnings=dict(warnings),initial_model_warning_count=len(runtime),
                native_hbm_integrated=False,hardware_error_reader_implemented=True,
                host_register_exclusion_enforced=False,remote_ack_wire_protocol_validated=False,reset_fault_matrix_validated=False,
                board_qualified=False)


def review_native_management(vcd):
    tb='/board/probe/';ep='/board/EP/'
    rows=[];discovery_ar=[];discovery_r=[];guard_start=None
    for t,b,a in vcd_rows(vcd):
        if b.get(ep+'user_clk')!=0 or a.get(ep+'user_clk')!=1:continue
        if b.get(ep+'user_resetn')==1:
            require(b[ep+'m_axil_awvalid']==0,'no host BAR0 writes')
            if b[ep+'m_axil_arvalid'] and b[ep+'m_axil_arready']:
                require(b[tb+'phase']==0 and b[ep+'m_axil_araddr']==0,'bounded BAR0 discovery address')
                discovery_ar.append(t)
            if b[ep+'m_axil_rvalid'] and b[ep+'m_axil_rready']:
                require(b[tb+'phase']==0 and b[ep+'m_axil_rdata']==b[ep+'m_axil_rresp']==0,'BAR0 zero response')
                discovery_r.append(t)
            if b[tb+'phase']!=0:require(b[ep+'m_axil_arvalid']==b[ep+'m_axil_rvalid']==0,'BAR0 quiescent after discovery')
        if not (b.get(tb+'guard_start_ready')==1 or b.get(tb+'guard_active')==1):continue
        r={n[len(tb):]:v for n,v in b.items() if n.startswith(tb) and '/' not in n[len(tb):]};r['time']=t
        for name in 'm_req_valid m_req_ready m_req_write m_req_address m_req_strobe m_req_identity m_rsp_valid m_rsp_ready m_rsp_data m_rsp_code m_rsp_identity m_rsp_timeout m_timeout_latched'.split():
            r[name]=b[tb+'dut/'+name]
        for suffix in 'awvalid wvalid arvalid arready araddr arprot rvalid rready rdata rresp'.split():r[suffix]=b[tb+'m_axil_'+suffix]
        native=r['native_status']
        r.update(faults=r['guard_faults'],native_done=((native>>3)&1)|(((native>>11)&1)<<1),
                 native_run=((native>>6)&1)|(((native>>14)&1)<<1),dsc_load=r['dsc_load'],
                 core_ready=r['management_core_ready'] and b[ep+'user_lnk_up'],reset_request=r['management_reset_request'],
                 external_fault=r['management_fault'],wait_expired=r['management_wait_expired'])
        if r['guard_start'] and r['guard_start_ready']:
            require(guard_start is None and r['guard_epoch']==1,'one management epoch');guard_start=t
        rows.append(r)
    result=management(rows,1)
    require(result['faults']==0 and result['published']==result['raw_done']==result['delivered']==[2,2],'native management inventory')
    require(len(discovery_ar)==len(discovery_r)==1 and discovery_ar[0]<discovery_r[0]<guard_start,'one completed BAR0 discovery')
    result['bar0_discovery']={'ar_ps':discovery_ar[0],'r_ps':discovery_r[0],'returned_word':0}
    result['guard_start_ps']=guard_start
    # Negative controls target the native management evidence, independently of payload reconstruction.
    caught=[]
    for name in ['premature_credit','wrong_register','response_corrupt','inventory_erased']:
        bad=copy.deepcopy(rows)
        if name=='premature_credit':next(r for r in bad if r['m_req_valid'])['validated_done']=1
        elif name=='wrong_register':next(r for r in bad if r['m_req_valid'] and r['m_req_ready'])['m_req_address']=0x1044
        elif name=='response_corrupt':next(r for r in bad if r['rvalid'] and r['rready'])['rdata']^=1
        else:next(r for r in bad if r['published_count'])['published_count']=0
        try:management(bad,1)
        except ValueError:caught.append(name)
        else:raise ValueError('undetected native management mutation '+name)
    result['negative_controls']=caught
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',type=Path)
    print(json.dumps(review(ap.parse_args().stage),indent=2))
