"""Review sampled native host isolation and guard behavior, not universal PCIe security."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from review_native import review as review_baseline, vcd_rows, require


def fields(line):
    return dict(re.findall(r'(\w+)=\s*([^\s]+)',line))


def review(log,vcd):
    baseline=review_baseline(log)
    require(log.count('BRIDGE_ISOLATION_NATIVE_PASS campaigns=3 rq=48 rc=36 bytes=192 guard=1')==1,'Complete isolation campaign')
    guard_lines=re.findall(r'^BI GUARD_PASS (.*)$',log,re.M)
    require(len(guard_lines)==1,'Unique native guard verdict')
    guard_result={k:int(v) for k,v in fields(guard_lines[0]).items()}
    require(tuple(guard_result[k] for k in ('fault','cold','owned','stop','link'))==(1,1,1,1,0) and
            guard_result['code'] in (1,3),'Native PERST guard contract')
    events=[]
    for line in log.splitlines():
        if line.startswith(('BI ','BN ')):
            prefix,kind,*_=line.split();events.append((prefix,kind,fields(line)))
    attacks=[(k,f) for p,k,f in events if p=='BI']
    begin=[f for k,f in attacks if k=='CAMPAIGN_BEGIN']
    end=[f for k,f in attacks if k=='CAMPAIGN_PASS']
    expected_modes=[(0,1,1),(0,0,0),(1,1,1)]
    require([int(f['stage']) for f in begin]==[0,1,2] and [int(f['stage']) for f in end]==[0,1,2],'Campaign order and completeness')
    for i in range(3):
        for marker in (begin[i],end[i]):
            require(tuple(int(marker[k]) for k in ('owned','verified','release_b'))==expected_modes[i],'Campaign ownership/mapping state')
    bar_base=None; summaries=[]
    for stage in range(3):
        trace=[(k,f) for k,f in attacks if f.get('stage')==str(stage)]
        requests=[f for k,f in trace if k=='RQ'];completions=[f for k,f in trace if k=='RC']
        require(len(requests)==16 and len(completions)==12 and sum(k=='TAG' for k,_ in trace)==12,'Host request/completion/tag coverage')
        outstanding=None;request_index=0;completed=0
        expected=[]
        offsets=[0x208,0x20c,0x10208,0x1020c]
        config_offsets=[0x208,0x20c,0xa08,0xa0c]
        for i,offset in enumerate(offsets):
            expected.extend([(1,offset,0xbad00000+stage*16+i),(0,offset,None)])
        for i,offset in enumerate(config_offsets):
            expected.extend([(10,offset,0xfed00000+stage*16+i),(8,offset,None)])
        statuses=[]
        for kind,f in trace:
            if kind=='RQ':
                data=int(f['data'],16);keep=int(f['keep'],16);user=int(f['user'],16)
                typ=(data>>75)&15;address=data&((1<<64)-1);tag=(data>>96)&255
                want_type,offset,payload=expected[request_index]
                require(outstanding is None and typ==want_type,'Serialized expected host request types')
                require((data>>64)&0x7ff==1 and int(f['last'])==1,'One-DWORD request framing')
                require(user&0xffff==15 and keep==(0x1f if payload is not None else 0xf),'Host request byte enables and keep')
                if typ in (0,1):
                    if bar_base is None:
                        bar_base=address-offset
                        require(bar_base>=0 and bar_base<1<<32 and bar_base%0x20000==0,'Aligned 32-bit BAR base inferred from native requests')
                    require(address==bar_base+offset,'BAR target/alias address')
                else:require(address==offset,'Configuration target/alias address')
                if payload is not None:require((data>>128)&0xffffffff==payload,'Host hostile write payload')
                if typ!=1:outstanding=(typ,None,request_index)
                request_index+=1
            elif kind=='TAG':
                require(outstanding is not None and outstanding[1] is None,'Native tag allocation belongs to one pending request')
                outstanding=(outstanding[0],int(f['tag'],16),outstanding[2])
            elif kind=='RC':
                require(outstanding is not None,'Native completion has a request')
                typ,tag,index=outstanding;data=int(f['data'],16)
                status=(data>>43)&7;error=(data>>12)&15;words=(data>>32)&0x7ff
                require(tag is not None and (data>>64)&255==tag and not (data>>46)&1 and (data>>30)&1,'Completion identity, poison and termination')
                require(int(f['last'])==1,'One-beat completion framing')
                if typ==0:
                    i=index//2
                    require(status==error==0 and words==1 and (data>>96)&0xffffffff==0xbad00000+stage*16+i,'Separate BAR memory readback')
                else:
                    require((status,error) in ((0,0),(1,2)),'Configuration request disposition')
                    if status==0:require(words==(0 if typ==10 else 1),'Configuration completion size')
                    statuses.append(dict(type=typ,offset=expected[index][1],status=status,error=error,
                                         words=words,data=((data>>96)&0xffffffff) if words else None))
                completed+=1;outstanding=None
        require(outstanding is None and request_index==16 and completed==12,'Drained host campaign')
        # Native master-to-BRAM evidence must account for each actual BAR write.
        aw=[f for k,f in trace if k=='IN_AW'];wd=[f for k,f in trace if k=='IN_W'];br=[f for k,f in trace if k=='IN_B']
        ar=[f for k,f in trace if k=='IN_AR'];rd=[f for k,f in trace if k=='IN_R']
        require(tuple(map(len,(aw,wd,br,ar,rd)))==(4,4,4,4,4),'Only BAR requests reach incoming AXI target')
        for i,offset in enumerate(offsets):
            address=int(aw[i]['address'],16);strb=int(wd[i]['strb'],16);data=int(wd[i]['data'],16)
            require(address&~63==offset&~63 and int(aw[i]['len'])==0,'Incoming write address and single beat')
            require(strb==15<<(offset%64) and int(wd[i]['last'])==1 and (data>>(offset%64*8))&0xffffffff==0xbad00000+stage*16+i,'Incoming write payload/lanes')
            require(int(br[i]['response'])==0,'Native incoming write response')
            require(int(ar[i]['address'],16)&~63==offset&~63 and int(ar[i]['len'])==0,'Incoming read address and single beat')
            require(int(rd[i]['response'])==0 and int(rd[i]['last'])==1 and (int(rd[i]['data'],16)>>(offset%64*8))&0xffffffff==0xbad00000+stage*16+i,'Native incoming read payload/response')
        private=[(k,f) for k,f in trace if k in ('PRIVATE_AR','PRIVATE_R')]
        destination=0x1234567800001000 if stage==0 else 0x9abcdef000002000
        require([k for k,_ in private]==['PRIVATE_AR','PRIVATE_R']*2,'Private inspection order')
        require([int(f['address'],16) for k,f in private if k=='PRIVATE_AR']==[0x208,0x20c],'Both private words inspected')
        require([int(f['data'],16) for k,f in private if k=='PRIVATE_R']==[destination>>32,destination&0xffffffff] and
                all(int(f['response'])==0 for k,f in private if k=='PRIVATE_R'),'Private translation unchanged after host traffic')
        summaries.append(dict(stage=stage,requests=16,completions=12,bar_words_written_and_read=4,configuration=statuses,private_translation=f'{destination:016x}'))

    # Compare every public native handshake against the console, sampling the
    # values before the clock edge (before same-timestamp sequential updates).
    p='/board/probe/';ep='/board/AXI_PCIE_EP/';rp='/board/RP/'
    specs=[]
    def add(prefix,kind,bus,valid,ready,mapping,inspect=None):
        specs.append((prefix,kind,bus,valid,ready,mapping,inspect))
    control=[('AW','awvalid','awready',{'address':('awaddr',16)}),('W','wvalid','wready',{'data':('wdata',16),'strobe':('wstrb',16)}),
             ('B','bvalid','bready',{'response':('bresp',10)}),('AR','arvalid','arready',{'address':('araddr',16)}),
             ('R','rvalid','rready',{'data':('rdata',16),'response':('rresp',10),'address':None})]
    # R address is reconstructed by the log reviewer; it is not a bus pin.
    for kind,valid,ready,mapping in control:
        mapping={k:v for k,v in mapping.items() if v is not None}
        add('BN',kind,p+'m_axil_',valid,ready,mapping,False if kind in ('AR','R') else None)
    add('BI','PRIVATE_AR',p+'m_axil_','arvalid','arready',{'address':('araddr',16)},True)
    add('BI','PRIVATE_R',p+'m_axil_','rvalid','rready',{'data':('rdata',16),'response':('rresp',10)},True)
    for kind,valid,ready,mapping in [
        ('DMA_AW','awvalid','awready',{'address':('awaddr',16),'len':('awlen',10),'size':('awsize',10),'burst':('awburst',10),'id':('awid',16)}),
        ('DMA_W','wvalid','wready',{'data':('wdata',16),'strb':('wstrb',16),'last':('wlast',10)}),
        ('DMA_B','bvalid','bready',{'response':('bresp',10),'id':('bid',16)})]:add('BN',kind,p+'m_',valid,ready,mapping)
    for kind,valid,ready,mapping in [
        ('IN_AW','awvalid','awready',{'address':('awaddr',16),'len':('awlen',10),'size':('awsize',10)}),
        ('IN_W','wvalid','wready',{'data':('wdata',16),'strb':('wstrb',16),'last':('wlast',10)}),
        ('IN_B','bvalid','bready',{'response':('bresp',10)}),
        ('IN_AR','arvalid','arready',{'address':('araddr',16),'len':('arlen',10),'size':('arsize',10)}),
        ('IN_R','rvalid','rready',{'data':('rdata',16),'response':('rresp',10),'last':('rlast',10)})]:add('BI',kind,ep+'m_axi_',valid,ready,mapping)
    streams={'RQ':rp+'s_axis_rq_','RC':rp+'m_axis_rc_','CQ':rp+'m_axis_cq_'}
    for kind,bus in streams.items():add('BN' if kind=='CQ' else 'BI',kind,bus,'tvalid','tready',
                                      {'data':('tdata',16),'keep':('tkeep',16),'last':('tlast',10),'user':('tuser',16)})
    add('BI','TAG',rp+'pcie_4_0_rport/','pcie_rq_tag_vld0',None,{'tag':('pcie_rq_tag0',16)})
    for kind,valid,ready,mapping in [
        ('CFG','cfg_valid','cfg_ready',{'session':('cfg_session',16),'generation':('cfg_generation',16),'axi':('cfg_axi_base',16),'pcie':('cfg_pcie_base',16)}),
        ('CFG_R','cfg_rsp_valid','cfg_rsp_ready',{'error':('cfg_rsp_error',10)}),
        ('ACK','ack_valid','ack_ready',{'tag':('ack_tag',16),'generation':('ack_generation',16),'bytes':('ack_bytes',10)}),
        ('RESULT','result_valid','result_ready',{'tag':('result_tag',16),'bytes':('result_bytes',10)})]:add('BN',kind,p,valid,ready,mapping)
    expected_trace=defaultdict(list);observed=defaultdict(list)
    for prefix,kind,bus,valid,ready,mapping,inspect in specs:
        key=prefix+' '+kind
        for pr,k,f in events:
            if pr==prefix and k==kind:
                identity=int(f['case' if prefix=='BN' else 'stage'])
                expected_trace[key].append((identity,tuple(int(f[field],base) for field,(_,base) in mapping.items())))
    hold_samples=0;owned_samples=0;guard_seen=False;last_axi=None;clock_edges=0;final_guard_code=None
    for now,before,after in vcd_rows(vcd):
        if before.get(p+'phase') in (1,2):
            require(before.get(p+'cold_rst_n')==1 and before.get('/board/sys_rst_n')==1,'No cold/PCIe reset during campaigns')
            axi_edge=before.get(ep+'axi_aclk_out')==0 and after.get(ep+'axi_aclk_out')==1
            rp_edge=before.get(rp+'user_clk')==0 and after.get(rp+'user_clk')==1
            if rp_edge:require(before.get(rp+'pcie_4_0_rport/pcie_rq_tag_vld1')==0,'Single native tag allocation lane')
            if axi_edge:
                if last_axi is not None:require(now-last_axi==4000,'Native AXI clock at 250 MHz')
                last_axi=now;clock_edges+=1
                require(before.get(p+'link_up')==1 and before.get(p+'stop')==0,'Guard allows healthy native campaigns')
            for prefix,kind,bus,valid,ready,mapping,inspect in specs:
                if not (rp_edge if kind in streams or kind=='TAG' else axi_edge):continue
                if inspect is not None and bool(before.get(p+'inspect_mode'))!=inspect:continue
                if before.get(bus+valid)==1 and (ready is None or before.get(bus+ready)==1):
                    identity=before[p+('case_id' if prefix=='BN' else 'attack_stage')]
                    if prefix=='BI' and identity>2**31:identity-=2**32
                    observed[prefix+' '+kind].append((identity,tuple(before.get(bus+signal) for signal,_ in mapping.values())))
                    if kind in ('RQ','RC'):
                        require(identity in (0,1,2),'Host transaction attributed to campaign')
                        mode=tuple(before.get(p+k) for k in ('owned','map_verified','release_b'))
                        require(mode==expected_modes[identity],'VCD campaign state')
                        if identity==1:
                            require(before.get(p+'m_axil_bvalid')==1 and before.get(p+'m_axil_bready')==0,'VCD mapping truly pending across host accesses');hold_samples+=1
                        if identity==2:
                            require(before.get(p+'src_req_valid')==1 and before.get(p+'src_req_tag')==(9<<64)|2,'VCD pending owned source during attacks');owned_samples+=1
        if after.get(p+'phase')==3 and after.get('/board/sys_rst_n')==0 and after.get(p+'guard_fault')==1:
            require(tuple(after.get(p+k) for k in ('cold_rst_n','owned','stop','link_up'))==(1,1,1,0),'VCD guard retains context after native PERST')
            require(tuple(after.get(p+k) for k in ('src_req_tag','src_req_address','src_req_bytes'))==((9<<64)|3,0x1000,64),
                    'VCD retained source context identity after native PERST')
            require(tuple(after.get(p+k) for k in ('fault','mapper_fault','cmd_ready','cfg_ready','result_valid'))==(1,1,0,0,0),
                    'VCD guard closes admission without false success')
            guard_seen=True
            final_guard_code=after.get(p+'guard_code')
    for key in expected_trace:require(observed[key]==expected_trace[key],'VCD/public log agreement: '+key)
    require(set(observed)==set(expected_trace),'All public handshake classes reviewed')
    require(hold_samples==28 and owned_samples==28 and guard_seen,'VCD concurrency and guard coverage')
    require(final_guard_code==guard_result['code'],'Log/VCD guard cause')
    return dict(status='NATIVE_SAMPLED_HOST_ISOLATION_AND_GUARD_PASS',campaigns=summaries,
                inferred_host_bar0=f'{bar_base:08x}',native_axi_mhz=250,clock_edges=clock_edges,
                baseline=baseline,public_handshakes={k:len(v) for k,v in observed.items()},
                native_perst_pending_context_guarded=True,native_guard_code=final_guard_code,native_clock_loss_tested=False,
                host_isolation_proven_for_all_transactions=False,physical_board_or_cdc_validated=False,
                read_only_inspection_mux_and_b_backpressure_are_fixtures=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log',type=Path);ap.add_argument('vcd',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=review(a.log.read_text(errors='replace'),a.vcd.read_text())
    result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),vcd_sha256=hashlib.sha256(a.vcd.read_bytes()).hexdigest())
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],campaigns=len(result['campaigns']),handshakes=result['public_handshakes'])))
