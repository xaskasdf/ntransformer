"""Independent native HBM -> source join -> XDMA public-handshake review."""
import argparse
from collections import Counter, deque
import copy
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_hbm_xdma_guarded_native import extract, check_events, sha, LENGTHS, RTL
from check_ntrpu_hbm_xdma_guarded import management
from check_ntrpu_xdma_stop_inventory_r4 import require, markers, vcd_rows
from ntrpu_pcie_vcd import logic_vcd_rows
from ntrpu_pcie_straddle import LogicWord

SOURCE_RTL = {'fpga/experimental/ntrpu_hbm_source_join.sv': '4bfc68585e87315ace552d9f431c486cf0beb64fd3bd78d6e06ce9c26f101869', 'fpga/experimental/ntrpu_hbm_xdma_source.sv': '1345a031b20829278c468d1a60b8265e442b9eedd15ffe3b2d9fa8d5ca102e2d', 'fpga/experimental/ntrpu_hbm_xdma_guarded_held.sv': '76d94d7e5c10b81d296bf12c66f82673d3e5bfc782fe44dd3a9e097f0abd63a9', 'fpga/experimental/ntrpu_xdma_completion_guard_held.sv': '8bdfb6944d051ac54ca5031cfc2d82202e23bc735fcd1cb7e844d7ce43f0f1d7', 'fpga/rtl/ntrpu_xdma_local_master.sv': '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc'}

SOURCE = [0x4000 + (i % 2)*0x200000000 + (i//2)*4096 for i in range(4)]


def pattern(i):
    return bytes(((j*37) ^ (j>>7) ^ ((i+1)*53)) & 255 for j in range(LENGTHS[i]))


def check_ack_after_receipt(events):
    require(events['cq'] and events['ack'], 'host receipt and ACK evidence')
    require(max(r[0] for r in events['cq']) < min(r[0] for r in events['ack']),
            'fixture ACK after complete native payload and writeback receipt')


def extract_hbm(text):
    tb = '/board/probe/'; hb = tb+'hbm/'
    specs = {
        'AW': (hb, 'h_aw', [('h_awaddr',34),('h_awlen',4)]),
        'W': (hb, 'h_w', [('h_wdata',256),('h_wlast',1)]),
        'B': (hb, 'h_b', [('h_bresp',2),('h_bid',6)]),
        'AR': (hb, 'h_ar', [('h_araddr',34),('h_arlen',4),('h_arid',6),('h_arsize',3),('h_arburst',2)]),
        'R': (hb, 'h_r', [('h_rdata',256),('h_rlast',1),('h_rresp',2),('h_rid',6)]),
        'CMD': (tb, 'cmd_', [('cmd_addr',64),('cmd_tag',64),('cmd_len',8)]),
        'DATA': (tb, 'r_data_', [('r_data',512),('r_data_tag',64),('r_data_last',1)]),
        'CPL': (tb, 'r_cpl_', [('r_cpl_tag',64),('r_cpl_status',2),('r_cpl_bytes',13)])}
    paths = {prefix+n for prefix, _, fields in specs.values() for n,w in fields}
    events = []; held = {}; last = {}; periods = {'mem':set(), 'reader':set()}
    cattrip = set()
    for t,b,a in logic_vcd_rows(text,paths):
        mem = b.get(hb+'mem_clk')==0 and a.get(hb+'mem_clk')==1 and b.get(hb+'mem_rst_n')==1
        reader = b.get(tb+'clk')==0 and a.get(tb+'clk')==1 and b.get(tb+'reader_started')==1
        if not (mem or reader): continue
        for domain, active in [('mem',mem),('reader',reader)]:
            if active:
                if domain in last: periods[domain].add(t-last[domain])
                last[domain]=t
        if mem:
            require(b[hb+'apb_complete_0']==b[hb+'apb_complete_1']==1,'both native HBM stacks initialized')
            cattrip.add((b[hb+'DRAM_0_STAT_CATTRIP'],b[hb+'DRAM_1_STAT_CATTRIP']))
        if reader:
            require(b[tb+'reader_fault']==b[tb+'source_fault']==b[tb+'reader_stop']==0,'healthy native source')
            events.append(['OWN',t,b[tb+'source_held'],b[tb+'contexts_held'],b[tb+'admission_pending'],b[tb+'preload_done']])
        for kind,(prefix,stem,fields) in specs.items():
            if kind in ('AW','W','B','AR','R') and not mem:continue
            if kind in ('CMD','DATA','CPL') and not reader:continue
            for c in range(2):
                def part(name,width=1):
                    v=b[prefix+name];require(v is not None,'unknown '+name)
                    mask=((1<<width)-1)<<(c*width)
                    if isinstance(v,LogicWord):
                        require(v.known&mask==mask,'unknown active '+name);v=v.value
                    return (v>>(c*width))&((1<<width)-1)
                valid,ready=part(stem+'valid'),part(stem+'ready')
                payload=tuple(part(n,w) for n,w in fields) if valid else None
                key=(kind,c)
                if held.get(key) is not None:require(valid and payload==held[key], 'stable '+kind+' under backpressure')
                held[key]=payload if valid and not ready else None
                if valid and ready:
                    if kind=='AR':require(b[tb+'preload_done']==1,'HBM read after preload publication')
                    events.append([kind,t,c,*payload])
    require(periods=={'mem':{2222},'reader':{5000}},'native HBM/reader clock periods')
    return events, sorted([list(x) for x in cattrip],key=str)


def check_hbm(rows, pcie):
    memory={};aw=[deque(),deque()];ar=[deque(),deque()];pending_b=[0,0]
    commands=[[],[]];data=[bytearray(),bytearray()];completed=[[],[]];last_data={}
    native_read={};counts=Counter();all_cmd=max(r[1] for r in rows if r[0]=='CMD')
    first_result=min(r[0] for r in pcie['results']);last_result=max(r[0] for r in pcie['results'])
    protected=0;final=False;admitted=[[],[]]
    for t,c,ctx in pcie['admitted']:admitted[c].append((t,ctx))
    cmd_by_tag={};cpl_by_tag={}
    for row in rows:
        kind,t,*v=row
        if kind=='OWN':
            source,bridge,pending,preload=v
            if all_cmd<t<first_result:
                require(source==bridge==10 and pending==0 and preload==1,'four source contexts held through native completion until ACK/result');protected+=1
            if t>last_result:
                require(source==bridge==0,'source ownership retired');final=True
            continue
        c,*v=v;counts[kind]+=1
        if kind=='AW':
            address,length=v;require(not aw[c] and address%32==0,'serialized aligned preload')
            aw[c].append([address,length+1,0])
        elif kind=='W':
            word,last=v;require(aw[c],'attributed preload data');a,n,index=aw[c][0]
            require(last==(index==n-1),'preload WLAST')
            for j,byte in enumerate(word.to_bytes(32,'little')):
                key=a+32*index+j;require(key not in memory,'no duplicate preload bytes');memory[key]=byte
            aw[c][0][2]+=1
            if last:aw[c].popleft();pending_b[c]+=1
        elif kind=='B':
            require(v==[0,0] and pending_b[c]>0,'successful attributed native write response');pending_b[c]-=1
        elif kind=='AR':
            a,length,rid,size,burst=v
            require(rid==0 and size==5 and burst==1 and a%32==0,'native AXI3 read format')
            require(any(a>=SOURCE[i] and a+32*(length+1)<=SOURCE[i]+LENGTHS[i] for i in (c,c+2)), 'full-address native read region')
            ar[c].append([a,length+1,0])
        elif kind=='R':
            word,last,resp,rid=v;require(ar[c] and resp==rid==0,'attributed successful native HBM read')
            a,n,index=ar[c][0];require(last==(index==n-1),'native RLAST')
            for j,byte in enumerate(word.to_bytes(32,'little')):
                key=a+32*index+j;require(key in memory and memory[key]==byte and key not in native_read,'native HBM read matches actual write')
                native_read[key]=byte
            ar[c][0][2]+=1
            if last:ar[c].popleft()
        elif kind=='CMD':
            address,tag,length=v;q=len(commands[c]);i=c+2*q
            require(q<2,'bounded source command inventory')
            at,ctx=admitted[c][q];words=[(ctx>>(64*(15-k)))&((1<<64)-1) for k in range(16)]
            require(t>at and tag==words[2]==i+1 and address==SOURCE[i]==0x4000+words[13] and (length+1)*64==words[15]==LENGTHS[i],'source command preserves admitted full identity/address/length')
            commands[c].append((t,tag));cmd_by_tag[tag]=t
        elif kind=='DATA':
            word,tag,last=v;q=len(completed[c]);require(q<len(commands[c]),'reader data has accepted command');i=c+2*q
            require(tag==commands[c][q][1] and t>commands[c][q][0],'full reader data tag')
            offset=len(data[c]);require(offset+64<=LENGTHS[i],'reader frame length')
            actual=word.to_bytes(64,'little')
            require(actual==bytes(native_read[SOURCE[i]+offset+j] for j in range(64)), 'reader bytes came from completed native reads')
            data[c].extend(actual);require(last==(len(data[c])==LENGTHS[i]),'reader data LAST');last_data[tag]=t
        elif kind=='CPL':
            tag,status,length=v;q=len(completed[c]);i=c+2*q
            require(q<len(commands[c]) and tag==commands[c][q][1] and status==0 and length==LENGTHS[i] and data[c]==pattern(i) and t>last_data[tag],'successful CPL after complete attributed payload')
            completed[c].append(tag);data[c].clear();cpl_by_tag[tag]=t
    expected={SOURCE[i]+j:v for i in range(4) for j,v in enumerate(pattern(i))}
    require(memory==native_read==expected,'all 4544 native HBM bytes match distinct source regions')
    require(not any(aw) and not any(ar) and pending_b==[0,0] and not any(data),'all native HBM transactions drained')
    require(counts==Counter(AW=11,W=142,B=11,AR=11,R=142,CMD=4,DATA=71,CPL=4),'native HBM transaction inventory')
    require([len(x) for x in completed]==[2,2] and protected>32 and final,'source completion/retention coverage')
    for t,c,ctx,status,length in pcie['results']:
        tag=(ctx>>(64*13))&((1<<64)-1)
        require(t>cpl_by_tag[tag],'retirement after native local CPL')
    return dict(native_hbm_bytes=len(memory),native_transactions=dict(counts),retention_samples=protected,
        source_sha256=hashlib.sha256(b''.join(a.to_bytes(8,'little')+bytes([memory[a]]) for a in sorted(memory))).hexdigest())


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
            r[name]=b[tb+'dut/guarded/'+name]
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




def review(stage):
    out=stage/'output';run=json.loads((out/'result.json').read_text())
    for name,h in json.loads((out/'FILES.json').read_text()).items():require(sha(out/name)==h,'changed output '+name)
    require(run['status']=='PENDING_REVIEW' and run['returncode']==0 and not run['timed_out'] and run['seed_unchanged'],'bounded clean native run')
    budget=run['timeout']
    override=out/'reports/BUDGET_OVERRIDE.json'
    if override.exists():
        extension=json.loads(override.read_text())
        require(extension['status']=='CHILD_EXITED' and extension['effective_timeout_seconds']==2400 and extension['original_timeout_seconds']==1800,'bounded external supervision completed')
        require(extension['elapsed_seconds']<=2401,'external deadline respected')
        budget=extension['effective_timeout_seconds']
    require(run['seconds']<=budget+1,'execution within effective wall-clock budget')
    require(run['sources']==run['sources_after'],'frozen source inputs')
    if 'snapshot' in run:
        require(run['snapshot_source_unchanged'] and len(run['snapshot'])>=10,'unchanged compiled snapshot')
        require(run.get('compile_result_sha256') and run.get('compile_origin'),'compiled snapshot provenance')
    for name,h in run['sources'].items():require(sha(stage/'repo'/name)==h,'changed source '+name)
    for name,h in RTL.items():require(sha(stage/'repo/fpga/experimental'/(name+'.sv'))==h,'qualified bridge revision')
    for name,h in SOURCE_RTL.items():require(sha(stage/'repo'/name)==h,'locally qualified source/held guard '+name)
    profile=json.loads((stage/'repo/evidence/HEAD_NATIVE_PROFILE.json').read_text())
    for name,h in profile['selected_rtl'].items():
        p=stage/'repo'/name
        if p.exists():require(sha(p)==h,'qualified HBM reader '+name)
    for name,config in profile['selected_read_configs'].items():
        actual=dict(line.split('\t',1) for line in (out/'reports'/name).read_text().splitlines())
        require(actual==config,'native read IP configuration '+name)
    xci=0
    for name,h in run['seed_files'].items():
        if name.endswith('.xci'):
            require(sha(out/'export/example'/name)==h,'unchanged native PCIe IP');xci+=1
    # XCI provenance pins components and generated model parameters, excluding paths.
    for name in ['native_hbm','hbm_read_cdc','hbm_read_width','hbm_read_protocol']:
        original=json.loads((stage/'repo/evidence/ip'/(name+'.xci')).read_text())['ip_inst']['parameters']
        generated=json.loads((out/'export/hbm'/name/(name+'.xci')).read_text())['ip_inst']['parameters']
        for section in ['component_parameters','model_parameters']:
            values=lambda d:{k:[x['value'] for x in v] for k,v in d.items()}
            require(values(original[section])==values(generated[section]),'unchanged HBM IP parameters '+name+'/'+section)
    order=(out/'reports/compile_order.rpt').read_text()
    for name in ['ntrpu_hbm_source_native.sv','ntrpu_hbm_source_native_memory.sv','ntrpu_hbm_xdma_source.sv','ntrpu_hbm_source_join.sv','ntrpu_hbm_xdma_guarded_held.sv','ntrpu_xdma_completion_guard_held.sv','ntrpu_hbm_read_adapter.sv','ntrpu_hbm_read_return.sv','native_hbm','ntrpu_vu47p_xdma']:
        require(name in order,'compiled composition '+name)
    ep=(out/'export/example/imports/xilinx_dma_pcie_ep.sv').read_text().split('  // XDMA target application')[0]
    for c in range(2):
        for prefix,fields in [('c2h_dsc_byp',{'ready':f'dsc_ready[{c}]','load':f'dsc_load[{c}]','src_addr':f'dsc_src[{64*c}+:64]','dst_addr':f'dsc_dst[{64*c}+:64]','len':f'dsc_len[{28*c}+:28]','ctl':f'dsc_ctl[{16*c}+:16]'}),('s_axis_c2h',{'tdata':f'axis_data[{512*c}+:512]','tkeep':f'axis_keep[{64*c}+:64]','tvalid':f'axis_valid[{c}]','tready':f'axis_ready[{c}]','tlast':f'axis_last[{c}]'})]:
            for field,signal in fields.items():
                port=f'{prefix}_{field}_{c}'
                require(re.findall(r'\.'+port+r'\s*\(([^)]*)\)',ep)==['board.probe.'+signal],'public native binding '+port)
    for suffix in 'awaddr awprot awvalid awready wdata wstrb wvalid wready bvalid bresp bready araddr arprot arvalid arready rdata rresp rvalid rready'.split():
        require(re.findall(r'\.s_axil_'+suffix+r'\s*\(([^)]*)\)',ep)==['board.probe.m_axil_'+suffix],'public native management binding')
    log=(out/'console.log').read_text();vcd=(out/'reports/native.vcd').read_text()
    require(log.count('HSN PASS ')==1 and 'HSN_SIMULATOR_RETURNED' in log and '$finish called at time' in log,'completed native simulation')
    require(not re.search(r'^\s*(?:Fatal:|FATAL:|Error:|ERROR:|CRITICAL WARNING:)|HX ASSERT|HX TIMEOUT',log,re.M),'native diagnostics')
    events=extract(vcd,markers(log,'HX REG'));pcie=check_events(events)
    check_ack_after_receipt(events)
    m=review_native_management(vcd);require(m['credits']==events['validated'],'autonomous validated DMA credits')
    rows,cattrip=extract_hbm(vcd);hbm=check_hbm(rows,events)
    # All native-memory log handshakes must equal edge-sampled public VCD ports.
    for kind,fields in [('AW',[('addr',16),('len',16)]),('W',[('data',16),('last',16)]),('B',[('resp',16),('id',16)]),('AR',[('addr',16),('len',16)]),('R',[('data',16),('last',16),('resp',16),('id',16)])]:
        observed=[[int(r['t']),int(r['c']),*[int(r[k],base) for k,base in fields]] for r in markers(log,'NH '+kind)]
        sampled=[r[1:3+len(fields)] for r in rows if r[0]==kind]
        require(observed==sampled,'native memory log/VCD '+kind)
    negatives=[]
    for name in ['source_high_address','native_read_payload','reader_data_tag','cpl_tag','cpl_status','early_source_release']:
        bad=copy.deepcopy(rows)
        if name=='source_high_address':next(r for r in bad if r[0]=='AR' and r[2]==1)[3]&=0xffffffff
        elif name=='native_read_payload':next(r for r in bad if r[0]=='R')[3]^=1
        elif name=='reader_data_tag':next(r for r in bad if r[0]=='DATA')[4]^=1<<32
        elif name=='cpl_tag':next(r for r in bad if r[0]=='CPL')[3]^=1<<32
        elif name=='cpl_status':next(r for r in bad if r[0]=='CPL')[4]=1
        else:
            after=max(r[1] for r in bad if r[0]=='CMD');before=min(r[0] for r in events['results'])
            next(r for r in bad if r[0]=='OWN' and after<r[1]<before)[2]=0
        try:check_hbm(bad,events)
        except (ValueError,KeyError):negatives.append(name)
        else:raise ValueError('undetected native source evidence mutation '+name)
    for name in ['descriptor_high_address','stream_payload','wrong_result_context','early_ack']:
        bad=copy.deepcopy(events)
        if name=='descriptor_high_address':bad['descriptors'][0][3]&=0xffffffff
        elif name=='stream_payload':bad['axis'][0][2]^=1
        elif name=='wrong_result_context':bad['results'][0][2]^=1
        else:bad['ack'][0][0]=0
        try:check_events(bad)
        except ValueError:negatives.append(name)
        else:raise ValueError('undetected native PCIe evidence mutation '+name)
    altered=copy.deepcopy(events)
    altered['cq'][-1][0]=min(r[0] for r in altered['ack'])+1
    try:check_ack_after_receipt(altered)
    except ValueError:negatives.append('ack_before_native_receipt')
    else:raise ValueError('undetected ACK/receipt order mutation')
    warnings=Counter(re.findall(r'^WARNING: \[([^\]]+)\]',log,re.M))
    allowed={'Project 1-5713','XSIM 43-3431','XSIM 43-4328','Vivado 12-13650',
             'VRFC 10-9543','VRFC 10-5021','VRFC 10-3027','VRFC 10-3705',
             'VRFC 10-8862','VRFC 10-8809'}
    require(set(warnings)<=allowed,'unreviewed warning category')
    local_warnings=[line for line in log.splitlines() if line.startswith('WARNING:') and '/repo/' in line]
    require(len(local_warnings)==2 and all('XSIM 43-4328' in line and 'ntrpu_hbm_source_native_memory.sv Line: 93' in line for line in local_warnings),'fixture warning inventory')
    runtime_warnings=[line for line in log.splitlines() if line.startswith('WARNING:') and not line.startswith('WARNING: [')]
    require(len(runtime_warnings)==1 and '0ns : none of the conditions were true for unique case' in runtime_warnings[0] and runtime_warnings[0].endswith('xdma_v4_2_vl_rfs.sv:34099'),'runtime warning inventory')
    return dict(status='NATIVE_HBM_SOURCE_XDMA_PASS',**pcie,**hbm,management=m,
        seconds=run['seconds'],effective_timeout_seconds=budget,unchanged_pcie_xci=xci,negative_controls=negatives,
        warnings=dict(warnings),cattrip_observations=cattrip,
        native_hbm_integrated=True,host_register_exclusion_enforced=False,
        remote_ack_wire_protocol_validated=False,combined_fault_matrix_validated=False,board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',type=Path)
    print(json.dumps(review(ap.parse_args().stage),indent=2))
