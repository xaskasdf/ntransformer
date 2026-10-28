"""Independently match planned PCIe operations, native AXI and captured VCD."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def require(value,why):
    if not value:raise ValueError(why)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fields(line):return dict(re.findall(r'(\w+)=\s*(\S+)',line))


def plan():
    ops=[]
    def rd(address,value=None,resp=0,words=1):ops.append(dict(kind='READ',address=address,value=value,resp=resp,words=words))
    def wr(address,value,be=15,words=1,resp=0):ops.append(dict(kind='WRITE',address=address,value=value,be=be,words=words,resp=resp))
    rd(0,0x4e545243);rd(4,1);rd(0x38,0)
    values=[0xa5100000+i*0x10101 for i in range(8)]
    for i,value in enumerate(values):wr(0x100+4*i,value);rd(0x100+4*i,value)
    wr(0x104,0xdeadbeef,5);values[1]=(values[1]&0xff00ff00)|0x00ad00ef
    rd(0x104,values[1]);rd(0x38,0xffffffff)
    wr(0x10104,0xbad00001,resp=3);rd(0x10104,resp=3)
    wr(0x100,0xbad00002,words=2,resp=3);rd(0x100,resp=3,words=2)
    rd(0x100,values[0]);rd(0x104,values[1]);rd(0xffc,resp=2)
    wr(0x20,1);rd(8,0x40);rd(0x24,1)
    for i in range(16):rd(0x300+i*4,0x5a010000+i)
    wr(0x30,1);rd(8,0);wr(0xc,1);rd(8,1);rd(0x38,0)
    for i,value in enumerate(values):rd(0x400+i*4,value)
    wr(0x100,0xabcd1234);rd(0x400,values[0]);wr(0xc,1,resp=2);rd(0x48,1)
    return ops


def review_log(log):
    require(not re.search(r'(?im)^\s*(?:fatal|error):|BT CHECK|BT WATCHDOG',log),'Simulation failure')
    verdicts=re.findall(r'^BAR_TLP_PASS (.+)$',log,re.M);require(len(verdicts)==1,'Unique complete PASS')
    verdict={k:int(v) for k,v in fields(verdicts[0]).items()}
    require(re.findall(r'^BT GROUP (\w+)$',log,re.M)==['identity','staging_strobes','rejection','snapshot_lanes','commit_retention'],'Group coverage')
    starts=re.findall(r'^BT START base=([0-9a-f]+)$',log,re.M);require(len(starts)==1,'BAR base')
    base=int(starts[0],16);require(base%0x20000==0,'BAR alignment')
    cases=defaultdict(list);events=[]
    for line in log.splitlines():
        if line.startswith('BT ') and 'case=' in line:
            kind=line.split()[1];f=fields(line);events.append((kind,f));cases[int(f['case'])].append((kind,f))
    ops=plan();require(set(cases)==set(range(1,len(ops)+1)),'Exact transaction coverage')
    statuses=[]
    for index,op in enumerate(ops,1):
        ev=cases[index]
        def take(kind,count=1):
            found=[f for k,f in ev if k==kind];require(len(found)==count,f'Case {index}: {kind} count');return found[0] if count==1 else found
        request=take(op['kind']);require(int(request['offset'],16)==op['address'] and int(request['words'])==op['words'],'Requested operation differs')
        rq=take('RQ');d=int(rq['data'],16);w=op['kind']=='WRITE'
        require(d&((1<<64)-1)==base+op['address'] and (d>>75)&15==int(w) and (d>>64)&0x7ff==op['words'],'Native RQ address/type/length')
        require(int(rq['keep'],16)==((1<<(4+op['words']))-1 if w else 15),'Native RQ framing')
        addr=take('AW' if w else 'AR');require(int(addr['addr'],16)==op['address'] and int(addr['len'])==0,'Native AXI address/length')
        require(int(addr['size'])==(2 if op['words']==1 else 6),'Native AXI access size')
        if w:
            require(int(request['value'],16)==op['value'] and int(request['be'],16)==op['be'] and int(request['expected'])==op['resp'],'Write expectation differs')
            wd=take('W');data=int(wd['data'],16);strobe=int(wd['strb'],16);lane=op['address']%64
            expected_mask=(op['be'] if op['words']==1 else op['be']|0xf0)<<lane
            require(strobe==expected_mask and int(wd['last'])==1,'Native write byte lanes')
            for word in range(op['words']):
                require((d>>(128+32*word))&0xffffffff==op['value'],'PCIe write payload')
                for byte in range(4):
                    if expected_mask>>(lane+word*4+byte)&1:
                        require((data>>((lane+word*4+byte)*8))&255==(op['value']>>(byte*8))&255,'AXI write payload')
            require(int(take('B')['resp'])==op['resp'],'Native B response')
            take('RC',0);take('TAG',0)
        else:
            tag=take('TAG');rc=take('RC');d=int(rc['data'],16);rd=take('R')
            require((d>>64)&255==int(tag['tag'],16) and (d>>30)&1 and not (d>>46)&1,'Completion identity/termination')
            status=(d>>43)&7;error=(d>>12)&15;words=(d>>32)&0x7ff
            require(int(rd['resp'])==op['resp'] and int(rd['last'])==1,'Native R response')
            if op['resp']==0:
                require(status==error==0 and words==1 and int(rc['keep'],16)==15,'Successful completion framing')
                require((d>>96)&0xffffffff==op['value'],'Exact host readback')
                require(int(rd['data'],16)==op['value']<<((op['address']%64)*8),'Exact AXI read lane/padding')
            else:
                expected_status={3:1,2:4}[op['resp']]
                require(status==expected_status and error==2 and words==0 and int(rc['keep'],16)==7,'Exact unsuccessful completion')
                require(int(rd['data'],16)==0,'Error payload is zero')
                statuses.append(dict(case=index,offset=op['address'],axi_response=op['resp'],completion_status=status,error=error))
        for kind in ('CSR_AW','CSR_AR'):
            count=int(kind==('CSR_AW' if w else 'CSR_AR') and op['resp']!=3)
            item=take(kind,count)
            if count:require(int(item['addr'],16)==op['address'],'CSR address')
    expected=dict(groups=5,rq=len(ops),rc=49,tags=49,aw=16,b=16,ar=49,r=49,csr_w=14,csr_r=47)
    require(all(verdict[k]==v for k,v in expected.items()) and verdict['checks']>100,'Accounting differs')
    return dict(verdict=verdict,bar_base=base,error_completions=statuses),events


def vcd_rows(text):
    header,body=text.split('$enddefinitions $end',1)
    require(re.search(r'\$timescale\s+1ps\s+\$end',header),'VCD timescale')
    scopes=[];aliases=defaultdict(list)
    for line in header.splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='$scope':scopes.append(f[2])
        elif f[0]=='$upscope':scopes.pop()
        elif f[0]=='$var':aliases[f[3]].append('/'+'/'.join(scopes+[f[4]]))
    current={};changes={};now=0
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            after={**current,**changes};yield now,current,after;current=after;changes={}
            value=int(line[1:]);require(value>=now,'VCD order');now=value
        else:
            raw,code=line[1:].split() if line.startswith('b') else (line[0],line[1:])
            value=None if re.search('[xzXZ]',raw) else int(raw,2);require(code in aliases,'VCD code')
            for name in aliases[code]:changes[name]=value
    yield now,current,{**current,**changes}


def review_vcd(vcd,events):
    ep='/board/AXI_PCIE_EP/';rp='/board/RP/';p='/board/probe/'
    specs=[('AW',ep+'m_axi_','awvalid','awready',{'addr':('awaddr',16),'len':('awlen',10),'size':('awsize',10)}),
           ('W',ep+'m_axi_','wvalid','wready',{'data':('wdata',16),'strb':('wstrb',16),'last':('wlast',10)}),
           ('B',ep+'m_axi_','bvalid','bready',{'resp':('bresp',10)}),
           ('AR',ep+'m_axi_','arvalid','arready',{'addr':('araddr',16),'len':('arlen',10),'size':('arsize',10)}),
           ('R',ep+'m_axi_','rvalid','rready',{'data':('rdata',16),'resp':('rresp',10),'last':('rlast',10)}),
           ('CSR_AW',ep+'csr_','awvalid','awready',{'addr':('awaddr',16)}),
           ('CSR_AR',ep+'csr_','arvalid','arready',{'addr':('araddr',16)}),
           ('RQ',rp+'s_axis_rq_','tvalid','tready',{'data':('tdata',16),'keep':('tkeep',16)}),
           ('RC',rp+'m_axis_rc_','tvalid','tready',{'data':('tdata',16),'keep':('tkeep',16)}),
           ('TAG',rp+'pcie_4_0_rport/','pcie_rq_tag_vld0',None,{'tag':('pcie_rq_tag0',16)})]
    expected={k:[(int(f['case']),tuple(int(f[name],base) for name,(_,base) in mapping.items())) for kind,f in events if kind==k]
              for k,_,_,_,mapping in specs}
    observed=defaultdict(list);edges=0;previous=None
    for now,before,after in vcd_rows(vcd):
        if before.get(p+'phase')!=1:continue
        require(before.get(p+'cold_rst_n')==1 and before.get('/board/sys_rst_n')==1,'No reset during host operations')
        axi=before.get(ep+'axi_aclk_out')==0 and after.get(ep+'axi_aclk_out')==1
        rp_edge=before.get(rp+'user_clk')==0 and after.get(rp+'user_clk')==1
        if axi:
            if previous is not None:require(now-previous==4000,'Native clock period')
            previous=now;edges+=1;require(before.get(ep+'user_lnk_up')==1,'Native link available')
        if rp_edge:require(before.get(rp+'pcie_4_0_rport/pcie_rq_tag_vld1')==0,'Single tag allocation')
        for kind,bus,valid,ready,mapping in specs:
            if not (rp_edge if kind in ('RQ','RC','TAG') else axi):continue
            if before.get(bus+valid)==1 and (ready is None or before.get(bus+ready)==1):
                if kind in ('RQ','RC'):require(before.get(bus+'tlast')==1,'Native single beat framing')
                observed[kind].append((before[p+'case_id'],tuple(before.get(bus+signal) for signal,_ in mapping.values())))
    require(dict(observed)==expected,'VCD/public log agreement')
    return dict(handshakes={k:len(v) for k,v in observed.items()},clock_edges=edges,axi_mhz=250)


def review(root):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Closed inventory')
    for name,digest in files.items():require(sha(root/name)==digest,'Closed hash '+name)
    result=json.loads((root/'RESULT.json').read_text());require(result['status']=='BAR_TLP_EXPORTED' and result['sources_unchanged'],'Successful unchanged native run')
    for name,digest in result['sources'].items():require(sha(root/'source'/name)==digest,'Source '+name)
    binding=json.loads((root/'source/BINDING.json').read_text())
    require(binding['profile_differences']=={'pipe_sim':['true','false']},'Only PIPE profile differs')
    require(binding['frozen_rtl']=={
        'ntrpu_bar32_adapter.sv':'119d1aca6e631d0dcd691e17e7c8872524ff9c864aa2e5efe02c6ab2e1c89855',
        'ntrpu_hbm_control_plane.sv':'b38ab9e1c7d09a99a9143de7216bb4f07eb16cfa4629fd097bdf94cfe2b5e1c7',
        'ntrpu_control_snapshot.sv':'72a096db3944484ac0a2c50677375e42e9a7bfae5bb2f2d233a6fbf5f9608069'},'Frozen donor revision')
    for name,digest in binding['frozen_rtl'].items():require(sha(root/'source'/name)==digest,'Frozen native RTL')
    require(sha(root/'source/xilinx_axi_pcie4_ep.v')==binding['modified_endpoint_sha256'],'Incoming native binding')
    log=(root/'simulate.log').read_text(errors='replace');vcd=(root/'output/native.vcd').read_text()
    summary,events=review_log(log);wave=review_vcd(vcd,events)
    if result.get('requested_bar_base') is not None:
        require(summary['bar_base']==result['requested_bar_base'],'Requested relocated BAR base')
        deposits=re.findall(r'^BT BFM_CONFIG method=ROOT_PORT_ALLOCATOR_DEPOSIT base=([0-9a-fA-F]+)$',log,re.M)
        require(len(deposits)==1 and int(deposits[0],16)==summary['bar_base'],'Root Port allocator configuration')
    if result.get('replay'):
        require(result['replay']['rtl_unchanged'] and [s['name'] for s in result['steps']]==['simulate'],'Identical HDL replay')
    negative=[]
    mutations={'missing_pass':log.replace('BAR_TLP_PASS','REMOVED_PASS'),
        'wrong_readback':log.replace('offset=00000000 data=4e545243','offset=00000000 data=deadbeef'),
        'missing_tag':re.sub(r'^BT TAG .*\n','',log,count=1,flags=re.M),
        'missing_snapshot_group':log.replace('BT GROUP snapshot_lanes','BT GROUP missing'),
        'late_failure':log+'\nFatal: injected\n',
        'wrong_strobe':log.replace('be=5 words=1','be=f words=1'),
        'missing_native_b':re.sub(r'^BT B .*\n','',log,count=1,flags=re.M)}
    # READ_PASS is diagnostic only; corrupt the actual received payload instead.
    first_rc=re.search(r'^BT RC case=1 data=([0-9a-f]+)',log,re.M)
    damaged=int(first_rc[1],16)^(1<<96)
    mutations['wrong_readback']=log[:first_rc.start(1)]+f'{damaged:0{len(first_rc[1])}x}'+log[first_rc.end(1):]
    error_rc=re.search(r'^BT RC case=24 data=([0-9a-f]+)',log,re.M)
    damaged=(int(error_rc[1],16)&~(7<<43))|(4<<43)
    mutations['wrong_completion_status']=log[:error_rc.start(1)]+f'{damaged:0{len(error_rc[1])}x}'+log[error_rc.end(1):]
    start=re.search(r'^BT START base=([0-9a-f]+)$',log,re.M)
    mutations['wrong_bar_base']=log[:start.start(1)]+f'{int(start[1],16)^0x20000:08x}'+log[start.end(1):]
    for name,text in mutations.items():
        try:review_log(text)
        except (ValueError,KeyError):negative.append(name)
        else:raise ValueError('Negative control escaped '+name)
    altered=[(kind,dict(f)) for kind,f in events]
    for kind,f in altered:
        if kind=='W':f['data']=f"{int(f['data'],16)^1:x}";break
    try:review_vcd(vcd,altered)
    except ValueError:negative.append('log_vcd_payload_disagreement')
    else:raise ValueError('VCD negative control escaped')
    return dict(status='NATIVE_BAR_TLP_REVIEWED',**summary,vcd=wave,negative_controls=negative,
        physical_link_simulated=False,hbm_memory_simulated=False,full_native_top_simulated=False,
        downstream_writer_handshake_tested=False,log_sha256=sha(root/'simulate.log'),vcd_sha256=sha(root/'output/native.vcd'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();text=json.dumps(review(a.root),indent=2)+'\n'
    if a.output:a.output.write_text(text)
    else:print(text,end='')
