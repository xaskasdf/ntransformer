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

LENGTHS = [64, 128, 256, 4096]
DEST = [0x100000800 + ((i % 2) << 32) + (i // 2) * 4096 for i in range(4)]
WB = [0x300001000 + (i % 2) * 128 + (i // 2) * 64 for i in range(4)]
RTL = {
    'ntrpu_bridge_async_fifo': 'be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634',
    'ntrpu_hbm_dma_bridge': 'b4a10f677311bfdf29b00a353bb0d83d6a4e5181ca228c1d5b5da34d86d3848a',
    'ntrpu_hbm_dma_bridge_lane': '7e987b09ae1157251acd460b5511f041f59ab75076ecc62b4dcb0aa5a4926a24',
    'ntrpu_peer_read_pkg': '7d0b994ef8244710d4d2e8ed9ed4960685ad6a89ed5ffa76708439b2dcce603b',
}


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
                    require(phase == 3 and part('dma_error') == 0, 'validated Done policy phase')
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
    require(result['status']=='PENDING_REVIEW' and result['returncode']==0 and not result['timed_out'] and result['seed_unchanged'],'clean native run')
    require(result['sources']==result['sources_after'],'frozen inputs')
    for n,h in result['sources'].items():require(sha(stage/'repo'/n)==h,'changed native source')
    for n,h in RTL.items():require(sha(stage/'repo/fpga/experimental'/(n+'.sv'))==h,'different qualified rev3 RTL')
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
    order=(out/'reports/compile_order.rpt').read_text()
    for n in ('ntrpu_hbm_dma_bridge.sv','ntrpu_hbm_dma_bridge_lane.sv','ntrpu_bridge_async_fifo.sv','ntrpu_hbm_xdma_native.sv'):
        require(n in order,'missing compiled bridge source '+n)
    log=(out/'console.log').read_text();require('HX_SIMULATOR_RETURNED' in log and log.count('HX PASS ')==1,'native completion markers')
    require(not re.search(r'^\s*(?:Fatal:|FATAL:|Error:|ERROR:|CRITICAL WARNING:)',log,re.M),'native simulator diagnostic')
    warnings=Counter(re.findall(r'^WARNING: \[([^]]+)\]',log,re.M))
    require(warnings=={'Project 1-5713':1,'XSIM 43-3431':1,'VRFC 10-9543':52,'VRFC 10-5021':7,
                      'VRFC 10-3027':1,'VRFC 10-3705':70,'XSIM 43-4328':1}, 'changed native warning inventory')
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
    require(len(regs)==11,'complete management reads')
    expected=[(0,'identifier'),(0x1040,0),(0x1004,0xf83e7f),(0x1140,0),(0x1104,0xf83e7f),
              (0x1040,None),(0x1048,2),(0x1004,0xf83e7f),(0x1140,None),(0x1148,2),(0x1104,0xf83e7f)]
    for r,(address,value) in zip(regs,expected):
        actual=int(r['value'],16);require(int(r['address'],16)==address,'management address sequence')
        require(actual>>16==0x1fc0 if value=='identifier' else actual&~0x47==0 if value is None else actual==value,'native management value')
    require(int(regs[-1]['t'])<min(t for t,c in events['validated']),'status checked before validated Done')
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
    return dict(status='REV3_NATIVE_DUAL_C2H_PASS',**metrics,seconds=result['seconds'],negative_controls=negatives,
                unchanged_xci=xci,discovered_dma_bar=int(bar[0]['discovered']),selected_dma_bar=1,
                warnings=dict(warnings),initial_model_warning_count=len(runtime),
                native_hbm_integrated=False,hardware_error_reader_implemented=False,
                remote_ack_wire_protocol_validated=False,reset_fault_matrix_validated=False,
                board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',type=Path)
    print(json.dumps(review(ap.parse_args().stage),indent=2))
