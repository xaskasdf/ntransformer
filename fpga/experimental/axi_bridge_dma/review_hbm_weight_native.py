"""Independent closed-evidence review of BAR -> HBM -> PCIe -> BAR ACK traffic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from review_bar_tlp import fields,require,review_vcd,vcd_rows

SESSION=0xcafe002112345678
HOST=0x1234567800001000
BAR=0x80000000


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pattern(lane,offset):return ((lane*71+offset*37+91)^(offset>>3))&255
def payload(lane,start,n):return sum(pattern(lane,start+i)<<(8*i) for i in range(n))


def drop_first_assertion(vcd,path):
    header,body=vcd.split('$enddefinitions $end',1);scopes=[];code=None
    for line in header.splitlines():
        words=line.split()
        if not words:continue
        if words[0]=='$scope':scopes.append(words[2])
        elif words[0]=='$upscope':scopes.pop()
        elif words[0]=='$var' and '/'+'/'.join(scopes+[words[4]])==path:
            require(words[2]=='1','Scalar corruption target');code=words[3]
    require(code is not None,'Corruption target present')
    body,count=re.subn(r'^1'+re.escape(code)+r'$',lambda match:'0'+code,body,count=1,flags=re.M)
    require(count==1,'Asserted corruption target')
    return header+'$enddefinitions $end'+body


def plan():
    ops=[]
    def rd(addr,value):ops.append(('READ',addr,value))
    def wr(addr,value):ops.append(('WRITE',addr,value))
    def stage(addr,words,data):
        for i in range(words):wr(addr+4*i,(data>>(32*i))&0xffffffff)
    rd(0x800,0x4e544c44);rd(0x804,1);rd(0x808,0x10)
    for lane,size in enumerate((64,128)):
        for beat in range(size//32):
            address=(lane<<33)+0x1000+beat*32;tag=0xcafe001000000001+lane*256+beat
            stage(0x840,12,(payload(lane,beat*32,32)<<128)|(tag<<64)|address)
            wr(0x80c,1);rd(0x808,0x12);rd(0x820,tag&0xffffffff);rd(0x824,tag>>32)
            rd(0x828,0);rd(0x82c,address&0xffffffff);rd(0x830,address>>32);wr(0x810,1)
    wr(0x814,1);rd(0x808,0x14);rd(0x81c,6)
    rd(0,0x4e545243);rd(4,1)
    stage(0x100,8,(HOST<<192)|(0x10000000<<128)|(1<<64)|SESSION)
    wr(0xc,1);rd(8,9);rd(0x28,0);wr(0x18,1);rd(8,0)
    for lane,size in enumerate((64,128)):
        tag=(SESSION<<64)|(lane+1);source=(lane<<33)+0x1000
        command=(size<<320)|((lane*256)<<256)|(source<<192)|(1<<128)|tag
        stage(0x140,11,command);wr(0x10,1);rd(8,2)
        stage(0x180,7,(size<<192)|(1<<128)|tag);wr(0x14,1)
        rd(8,0x92);rd(0x2c,2)
        for i in range(4):rd(0x200+4*i,(tag>>(32*i))&0xffffffff)
        rd(0x210,size);wr(0x1c,1);rd(8,0x80)
    return ops


def review_log(log):
    require(not re.search(r'(?im)^\s*(?:fatal|error):|HBT WATCHDOG|BT CHECK',log),'Simulation failure')
    verdict=re.findall(r'^HBM_WEIGHT_NATIVE_PASS (.+)$',log,re.M);require(len(verdict)==1,'Unique full PASS')
    verdict={k:int(v) for k,v in fields(verdict[0]).items()}
    expected=dict(transfers=2,bytes=192,requests=201,completions=64,cfg=1,cmds=2,ack=2,results=2,dma_b=2,hbm_beats=6)
    require(all(verdict.get(k)==v for k,v in expected.items()) and verdict['checks']>100,'Integrated accounting')
    require(re.findall(r'^BT START base=([0-9a-f]+)$',log,re.M)==['0000000080000000'],'BAR base')
    require(re.findall(r'^HBT TRANSFER_PASS transfer=(\d+) bytes=(\d+)$',log,re.M)==[('0','64'),('1','128')],'Both transfers')
    events=[];cases=defaultdict(list);path_events=[]
    for line in log.splitlines():
        if line.startswith('BT ') and 'case=' in line:
            kind=line.split()[1];f=fields(line);events.append((kind,f));cases[int(f['case'])].append((kind,f))
        if line.startswith('HBT '):path_events.append((line.split()[1],fields(line)))
    require(set(cases)==set(range(1,202)),'Exact host operation coverage')
    for index,(kind,addr,value) in enumerate(plan(),1):
        def take(k,n=1):
            found=[f for event,f in cases[index] if event==k];require(len(found)==n,f'Host {index}: {k} count')
            return found[0] if n else None
        request=take(kind);rq=take('RQ');d=int(rq['data'],16);w=kind=='WRITE'
        require(int(request['offset'],16)==addr and int(request['words'])==1,'Host requested offset/length')
        require(d&((1<<64)-1)==BAR+addr and (d>>64)&2047==1 and (d>>75)&15==int(w),'Host native RQ')
        require(int(rq['keep'],16)==(31 if w else 15),'Host RQ framing')
        ax=take('AW' if w else 'AR');require(int(ax['addr'],16)==addr and int(ax['len'])==0 and int(ax['size'])==2,'Inbound AXI address')
        csr=take('CSR_AW' if w else 'CSR_AR');require(int(csr['addr'],16)==addr,'CSR address')
        take('CSR_AR' if w else 'CSR_AW',0)
        if w:
            require(int(request['value'],16)==value and int(request['be'],16)==15 and int(request['expected'])==0,'Host write')
            wd=take('W');require(int(wd['strb'],16)==15<<(addr%64) and int(wd['last'])==1,'Inbound write lanes')
            require((int(wd['data'],16)>>((addr%64)*8))&0xffffffff==value and (d>>128)&0xffffffff==value,'Inbound write payload')
            require(int(take('B')['resp'])==0,'Inbound B');take('TAG',0);take('RC',0)
        else:
            tag=int(take('TAG')['tag'],16);rc=take('RC');d=int(rc['data'],16);rd=take('R')
            require((d>>64)&255==tag and (d>>30)&1 and not (d>>46)&1,'Read attribution')
            require((d>>43)&7==0 and (d>>12)&15==0 and (d>>32)&2047==1 and int(rc['keep'],16)==15,'Read completion framing')
            require((d>>96)&0xffffffff==value,'Host readback')
            require(int(rd['data'],16)==value<<((addr%64)*8) and int(rd['resp'])==0 and int(rd['last'])==1,'Inbound R')
    def select(kind,lane=None):return [f for k,f in path_events if k==kind and (lane is None or int(f.get('lane',f.get('transfer',-1)))==lane)]
    require(select('BOOT')==[dict(init='11',locked='1',reader='1',client='1')],'Real readiness before traffic')
    cfg=select('CFG');require(len(cfg)==1,'One configuration')
    require({k:int(v,16) for k,v in cfg[0].items()}==dict(session=SESSION,generation=1,axi=0x10000000,pcie=HOST),'Full mapping identity')
    require(select('CFG_R')==[dict(error='0',verified='1')],'Verified mapping response')
    for kind,key,want,base in [('MAP_AW','address',[0x208,0x20c],16),('MAP_W','data',[HOST>>32,HOST&0xffffffff],16),
                              ('MAP_AR','address',[0x130,0x204,0x208,0x20c],16)]:
        require([int(f[key],base) for f in select(kind)]==want,'Native mapping '+kind)
    require(len(select('MAP_B'))==2 and all(int(f['resp'],16)==0 for f in select('MAP_B')),'Native mapping B')
    reads=select('MAP_R');require(len(reads)==4 and all(int(f['resp'],16)==0 for f in reads),'Native mapping R')
    require(int(reads[0]['data'],16)&0xa==8 and int(reads[1]['data'],16)&0xffff0000==0x03800000,'Native bridge identity')
    require([int(f['data'],16) for f in reads[2:]]==[HOST>>32,HOST&0xffffffff],'Mapping readback')
    for lane,size in enumerate((64,128)):
        tag=(SESSION<<64)|lane+1;source=(lane<<33)+0x1000
        aw=select('HAW',lane);require(len(aw)==size//32,'One HBM write per loaded line')
        for beat,f in enumerate(aw):
            require(int(f['addr'],16)==source+beat*32 and int(f['len'])==0 and int(f['size'])==5,'Loaded line address')
        hw=select('HW',lane);hr=select('HR',lane);require(len(hw)==len(hr)==size//32,'Native HBM beat counts')
        for beat in range(size//32):
            require(int(hw[beat]['data'],16)==payload(lane,beat*32,32) and int(hw[beat]['strb'],16)==0xffffffff and int(hw[beat]['last'])==1,'HBM written bytes')
            require(int(hr[beat]['data'],16)==payload(lane,beat*32,32) and int(hr[beat]['resp'],16)==0 and int(hr[beat]['last'])==int(beat==size//32-1),'HBM read bytes')
        hb=select('HB',lane);require(len(hb)==size//32 and all(int(f['resp'],16)==int(f['id'],16)==0 for f in hb),'HBM load B')
        ar=select('HAR',lane);require(len(ar)==1,'One HBM read burst per transfer')
        require(int(ar[0]['addr'],16)==source and int(ar[0]['len'])==size//32-1 and int(ar[0]['size'])==5 and int(ar[0]['burst'])==1,'HBM requested region')
        require(all(int(f['id'],16)==int(ar[0]['id'],16) for f in hr),'HBM R attribution')
        cmd=select('CMD',lane);require(len(cmd)==1,'One command')
        require(int(cmd[0]['tag'],16)==tag and int(cmd[0]['source'],16)==source and int(cmd[0]['offset'],16)==lane*256 and int(cmd[0]['bytes'])==size and int(cmd[0]['generation'],16)==1,'Command identity')
        aw=select('DMA_AW',lane);require(len(aw)==1,'One outgoing burst')
        require(int(aw[0]['address'],16)==0x10000000+lane*256 and int(aw[0]['len'])==size//64-1 and int(aw[0]['size'])==6 and int(aw[0]['burst'])==1 and int(aw[0]['id'],16)==0,'DMA aperture')
        w=select('DMA_W',lane);require(len(w)==size//64,'DMA beat count')
        for beat,f in enumerate(w):require(int(f['data'],16)==payload(lane,beat*64,64) and int(f['strb'],16)==(1<<64)-1 and int(f['last'])==int(beat==size//64-1),'DMA bytes')
        b=select('DMA_B',lane);require(len(b)==1 and int(b[0]['resp'],16)==int(b[0]['id'],16)==0,'DMA B')
        before=select('BEFORE_ACK',lane);require(len(before)==1 and before[0]['owned']==before[0]['source_owned']=='1' and int(before[0]['results'])==lane and int(before[0]['bytes'])==size,'Ownership before ACK')
        for kind in ('ACK','RESULT'):
            item=select(kind,lane);require(len(item)==1 and int(item[0]['tag'],16)==tag and int(item[0]['bytes'])==size,'Attributed '+kind)
            if kind=='ACK':require(int(item[0]['generation'],16)==1,'ACK generation')
        ordered=[k for k,f in path_events if int(f.get('transfer',-1))==lane and k in ('CMD','DMA_B','BEFORE_ACK','ACK','RESULT','TRANSFER_PASS')]
        require(ordered==['CMD','DMA_B','BEFORE_ACK','ACK','RESULT','TRANSFER_PASS'],'Retirement order')
        received={};active=False;offset=0;address=0;words=0
        for f in select('CQ',lane):
            data=int(f['data'],16);user=int(f['user'],16);keep=int(f['keep'],16);first=0
            if not active:
                require((user>>80)&3==1 and (data>>75)&15==1 and user&0xffff==0x0f0f,'CQ write descriptor')
                address=data&((1<<64)-1);words=((data>>64)&2047)*4;offset=0;active=True;first=16
            else:require((user>>80)&3==0,'No nested packet')
            require(not (user>>96)&1,'No discontinue')
            for byte in range(64):
                if (user>>(16+byte))&1:
                    require(byte>=first and (keep>>(byte//4))&1 and offset<words,'CQ payload framing')
                    dest=address+offset-HOST-lane*256
                    require(0<=dest<size and dest not in received,'Full destination address and unique byte')
                    received[dest]=(data>>(byte*8))&255;offset+=1
            if int(f['last']):require(offset==words,'CQ final length');active=False
        require(not active and received=={i:pattern(lane,i) for i in range(size)},'Exact HBM bytes received over PCIe')
    return verdict,events,path_events


def review_path_vcd(vcd,events):
    ep='/board/AXI_PCIE_EP/';p='/board/probe/';rp='/board/RP/'
    specs=[]
    def add(kind,clock,bus,valid,ready,mapping,identity=None):
        specs.append((kind,clock,bus,valid,ready,mapping,identity))
    add('CFG','axi',ep,'cfg_valid','cfg_ready',dict(session=('cfg_session',16),generation=('cfg_generation',16),axi=('cfg_axi_base',16),pcie=('cfg_pcie_base',16)))
    add('CFG_R','axi',ep,'cfg_rsp_valid','cfg_rsp_ready',dict(error=('cfg_rsp_error',2),verified=('map_verified',2)))
    add('CMD','axi',ep,'cmd_valid','cmd_ready',dict(tag=('cmd_tag',16),source=('cmd_source',16),offset=('cmd_offset',16),bytes=('cmd_bytes',10),generation=('cmd_generation',16)),'transfer')
    add('ACK','axi',ep,'ack_valid','ack_ready',dict(tag=('ack_tag',16),bytes=('ack_bytes',10),generation=('ack_generation',16)),'transfer')
    add('RESULT','axi',ep,'result_valid','result_ready',dict(tag=('result_tag',16),bytes=('result_bytes',10)),'transfer')
    for kind,suffix,valid,ready,mapping in [
        ('MAP_AW','m_axil_','awvalid','awready',dict(address=('awaddr',16))),
        ('MAP_W','m_axil_','wvalid','wready',dict(data=('wdata',16),strb=('wstrb',16))),
        ('MAP_B','m_axil_','bvalid','bready',dict(resp=('bresp',16))),
        ('MAP_AR','m_axil_','arvalid','arready',dict(address=('araddr',16))),
        ('MAP_R','m_axil_','rvalid','rready',dict(data=('rdata',16),resp=('rresp',16))),
        ('DMA_AW','m_','awvalid','awready',dict(address=('awaddr',16),len=('awlen',10),size=('awsize',10),burst=('awburst',10),id=('awid',16))),
        ('DMA_W','m_','wvalid','wready',dict(data=('wdata',16),strb=('wstrb',16),last=('wlast',2))),
        ('DMA_B','m_','bvalid','bready',dict(resp=('bresp',16),id=('bid',16)))]:
        add(kind,'axi',ep+suffix,valid,ready,mapping,'transfer' if kind.startswith('DMA') else None)
    add('CQ','rp',rp+'m_axis_cq_','tvalid','tready',dict(data=('tdata',16),keep=('tkeep',16),last=('tlast',2),user=('tuser',16)),'transfer')
    mem_specs=[
        ('HAW',p+'load_','awvalid','awready',dict(addr=('awaddr',34,16),len=('awlen',4,10),size=('awsize',3,10))),
        ('HW',p+'load_','wvalid','wready',dict(data=('wdata',256,16),strb=('wstrb',32,16),last=('wlast',1,2))),
        ('HB',p+'load_','bvalid','bready',dict(resp=('bresp',2,16),id=('bid',6,16))),
        ('HAR',ep+'h_','arvalid','arready',dict(addr=('araddr',34,16),len=('arlen',4,10),id=('arid',6,16),size=('arsize',3,10),burst=('arburst',2,10))),
        ('HR',ep+'h_','rvalid','rready',dict(data=('rdata',256,16),resp=('rresp',2,16),id=('rid',6,16),last=('rlast',1,2)))]
    observed={kind:[] for kind,*_ in specs+mem_specs};expected={}
    for kind,clock,bus,valid,ready,mapping,identity in specs:
        expected[kind]=[(([int(f[identity])] if identity else [])+[int(f[k],base) for k,(_,base) in mapping.items()]) for knd,f in events if knd==kind]
    for kind,bus,valid,ready,mapping in mem_specs:
        expected[kind]=[[int(f['lane'])]+[int(f[k],base) for k,(_,_,base) in mapping.items()] for knd,f in events if knd==kind]
    clocks=dict(axi=ep+'axi_aclk_out',rp=rp+'user_clk',reader=ep+'reader_clk',mem=ep+'mem_clk',mgmt=ep+'mgmt_clk')
    previous={};periods=defaultdict(set);edges=defaultdict(int);acked=set();completed=set();b_seen=set()
    active_lease=None;received=defaultdict(int);owned_samples=0
    for time,before,after in vcd_rows(vcd):
        if before.get(p+'phase')!=1:continue
        rising={name for name,path in clocks.items() if before.get(path)==0 and after.get(path)==1}
        for name in rising:
            if name in previous:periods[name].add(time-previous[name])
            previous[name]=time;edges[name]+=1
        require(before.get(ep+'hbm_init_complete')==3 and before.get(ep+'fabric_clock_locked')==1,'Native availability in VCD')
        for name in ('fault','mapper_fault','source_fault','guard_fault','reader_fault_raw','availability_fault'):
            require(before.get(ep+name)==0,'Integrated fault in VCD: '+name)
        for name in ('client_cold_rst_n','mem_cold_rst_n','reader_cold_rst_n','management_cold_rst_n'):
            require(before.get(ep+name)==1,'Canonical reset in VCD: '+name)
        if active_lease is not None:
            require(before.get(ep+'owned')==before.get(ep+'source_owned')==1,'Continuous ownership through result acceptance')
            owned_samples+=1
        for kind,clock,bus,valid,ready,mapping,identity in specs:
            if clock not in rising or before.get(bus+valid)!=1 or before.get(bus+ready)!=1:continue
            transfer=before.get(p+'transfer_id')
            values=([transfer] if identity else [])+[before.get(bus+signal) for signal,_ in mapping.values()]
            observed[kind].append(values)
            if kind=='DMA_AW':
                require(active_lease is None and before.get(ep+'owned')==before.get(ep+'source_owned')==1,'Owned outgoing transfer')
                active_lease=transfer
            if kind=='CQ':received[transfer]+=((before[bus+'tuser']>>16)&((1<<64)-1)).bit_count()
            if kind=='DMA_B':b_seen.add(transfer)
            if kind=='ACK':
                require(transfer in b_seen and transfer not in acked,'VCD ACK after B, once')
                require(received[transfer]==(64,128)[transfer],'VCD ACK after complete host receipt')
                acked.add(transfer)
            if kind=='RESULT':
                require(transfer in b_seen and transfer in acked and transfer not in completed and active_lease==transfer,'VCD retirement needs B plus ACK')
                completed.add(transfer);active_lease=None
            if kind in ('MAP_AW','MAP_W','MAP_AR'):require(before.get(ep+'owned')==0,'No mapping access under lease')
        if 'mem' in rising:
            for kind,bus,valid,ready,mapping in mem_specs:
                for lane in (0,1):
                    if ((before.get(bus+valid,0)>>lane)&1) and ((before.get(bus+ready,0)>>lane)&1):
                        observed[kind].append([lane]+[(before[bus+signal]>>(lane*width))&((1<<width)-1) for signal,width,_ in mapping.values()])
    require(observed==expected,'Native path log/VCD agreement')
    require(completed==acked==b_seen=={0,1} and active_lease is None and owned_samples>100,'Waveform completion accounting')
    require(periods['axi']==periods['rp']=={4000} and periods['reader']=={5000} and periods['mgmt']=={10000},'Native clock periods')
    require(periods['mem'] and periods['mem']<={2222,2223} and min(edges.values())>100,'450 MHz native memory clock')
    return dict(handshakes={k:len(v) for k,v in observed.items()},periods_ps={k:sorted(v) for k,v in periods.items()},edges=dict(edges),
                continuous_ownership_samples=owned_samples,host_bytes_before_ack=dict(received))


def review_loader_vcd(vcd):
    ep='/board/AXI_PCIE_EP/';p='/board/probe/';loader=ep+'weight_loader/'
    expected=[]
    for lane,size in enumerate((64,128)):
        for beat in range(size//32):
            address=(lane<<33)+0x1000+beat*32;tag=0xcafe001000000001+lane*256+beat
            expected.append((lane,address,tag,payload(lane,beat*32,32)))
    requests=0;done_count=0;pops=0;seal_count=0;commands=0;held_samples=0
    pending=None;got_aw=False;got_w=False;got_b=False;sealed=False
    for time,before,after in vcd_rows(vcd):
        if before.get(p+'phase')!=1:continue
        require(before.get(loader+'failed')==before.get(loader+'rejected')==0,'Loader healthy in native waveform')
        if sealed:require(before.get(loader+'sealed')==1,'Load fence cannot reopen')
        if pending is not None:
            lane,address,tag,data=pending
            require(before.get(loader+'active')==(data<<128)|(tag<<64)|address,'Retained active line')
            held_samples+=1
        if before.get(loader+'request')!=after.get(loader+'request'):
            require(requests<len(expected) and pending is None and not sealed,'One request at a time before SEAL')
            pending=expected[requests];lane,address,tag,data=pending;requests+=1
            require(after.get(loader+'active')==(data<<128)|(tag<<64)|address,'BAR payload captured atomically')
            require(after.get(loader+'busy')==1 and after.get(loader+'written')==0,'Commit owns and consumes staging mask')
            got_aw=got_w=got_b=False
        if before.get(ep+'mem_clk')==0 and after.get(ep+'mem_clk')==1:
            for lane in (0,1):
                for channel in ('aw','w','b'):
                    bus=p+'load_'
                    if (before.get(bus+channel+'valid',0)&before.get(bus+channel+'ready',0))&(1<<lane):
                        require(pending is not None and pending[0]==lane and not sealed,'Native write belongs to unsealed line')
                        require(before.get(loader+'busy')==1,'Native write retains source ownership')
                        if channel=='aw':
                            require(not got_aw and not got_b,'Single native AW before B');got_aw=True
                            require((before[bus+'awaddr']>>(34*lane))&((1<<34)-1)==pending[1],'Native AW matches active line')
                        if channel=='w':
                            require(not got_w and not got_b,'Single native W before B');got_w=True
                            require((before[bus+'wdata']>>(256*lane))&((1<<256)-1)==pending[3],'Native W matches active line')
                        if channel=='b':
                            require(got_aw and got_w and not got_b,'HBM B follows both independent channels');got_b=True
                if (before.get(ep+'h_arvalid',0)&before.get(ep+'h_arready',0))&(1<<lane):
                    require(before.get(loader+'sealed')==1 and requests==done_count==pops==6,'Reads require all committed lines drained and SEAL')
        if before.get(loader+'done')==0 and after.get(loader+'done')==1:
            require(pending is not None and got_b and after.get(loader+'completion')==0,'Completion requires actual HBM B')
            require(after.get(loader+'busy')==0,'Busy clears only on retained completion');done_count+=1
        if before.get(loader+'done')==1 and after.get(loader+'done')==0:
            require(pending is not None and got_b,'POP follows native completion');pending=None;pops+=1
        if before.get(loader+'sealed')==0 and after.get(loader+'sealed')==1:
            require(requests==done_count==pops==6 and pending is None and not sealed,'SEAL follows six attributed completions and POPs')
            sealed=True;seal_count+=1
        if before.get(ep+'axi_aclk_out')==0 and after.get(ep+'axi_aclk_out')==1:
            if before.get(ep+'cmd_valid')==1 and before.get(ep+'cmd_ready')==1:
                require(before.get(loader+'sealed')==1,'Command fence before reader');commands+=1
    require(requests==done_count==pops==6 and seal_count==1 and commands==2 and pending is None,'Complete loader waveform accounting')
    return dict(lines=requests,bytes=192,completions=done_count,pops=pops,seals=seal_count,
                commands_after_seal=commands,retained_line_samples=held_samples)


def review(root):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Closed inventory')
    for name,digest in files.items():require(sha(root/name)==digest,'Closed hash '+name)
    result=json.loads((root/'RESULT.json').read_text());require(result['status']=='HBM_WEIGHT_NATIVE_EXPORTED' and result['stage']=='traffic' and result['sources_unchanged'],'Successful unchanged native bank')
    for name,digest in result['sources'].items():require(sha(root/'source'/name)==digest,'Source hash '+name)
    binding=json.loads((root/'source/BINDING.json').read_text())
    require(binding['donor_top_sha256']=='7d6fb94faee14a40f1ae3516ac42b0b87bada975dcef80534e7ec0335a34a1da','Frozen native control donor')
    require(hashlib.sha256(json.dumps(binding['frozen_rtl'],sort_keys=True).encode()).hexdigest()==
            '669f286425ded67e7ffdee63afb3a7a4b379c3c584a4d0a79d1505c7cd7b61e0','Pinned 22-module RTL set')
    for name,digest in binding['frozen_rtl'].items():require(sha(root/'source'/name)==digest,'Frozen RTL '+name)
    require(binding['synthesizable_weight_loader'] and not binding['external_preload_ports'],'Internal weight loader')
    require(sha(root/'source/ntrpu_hbm_weight_loader.sv')==binding['loader_sha256'],'Loader binding')
    require(sha(root/'source/xilinx_axi_pcie4_ep.v')==binding['modified_top_sha256'],'Top binding')
    log=(root/'simulate.log').read_text(errors='replace');vcd=(root/'output/native.vcd').read_text()
    verdict,events,path_events=review_log(log)
    bar_wave=review_vcd(vcd,events);path_wave=review_path_vcd(vcd,path_events)
    loader_wave=review_loader_vcd(vcd)
    mutations={'missing_pass':log.replace('HBM_WEIGHT_NATIVE_PASS','REMOVED_PASS'),
        'missing_transfer':log.replace('HBT TRANSFER_PASS transfer=1','REMOVED_TRANSFER transfer=1'),
        'missing_ack':re.sub(r'^HBT ACK .*\n','',log,count=1,flags=re.M),
        'missing_hbm_read':re.sub(r'^HBT HR .*\n','',log,count=1,flags=re.M),
        'missing_dma_b':re.sub(r'^HBT DMA_B .*\n','',log,count=1,flags=re.M),
        'wrong_bar':log.replace('BT START base=0000000080000000','BT START base=0000000000000000'),
        'early_retirement':log.replace('BEFORE_ACK transfer=0 owned=1','BEFORE_ACK transfer=0 owned=0'),
        'late_failure':log+'\nFatal: injected\n'}
    for kind,name in [('HR','wrong_hbm_payload'),('DMA_W','wrong_dma_payload'),('CQ','wrong_pcie_payload')]:
        hit=re.search(r'^HBT '+kind+r' .*?data=([0-9a-f]+)',log,re.M);require(hit is not None,'Payload exists')
        data=int(hit[1],16)^(1<<(128 if kind=='CQ' else 0))
        mutations[name]=log[:hit.start(1)]+format(data,'0'+str(len(hit[1]))+'x')+log[hit.end(1):]
    negative=[]
    for name,damaged in mutations.items():
        try:review_log(damaged)
        except (ValueError,KeyError):negative.append(name)
        else:raise ValueError('Negative control escaped '+name)
    damaged=[(kind,dict(f)) for kind,f in path_events]
    for kind,f in damaged:
        if kind=='DMA_W':f['data']=format(int(f['data'],16)^1,'x');break
    try:review_path_vcd(vcd,damaged)
    except ValueError:negative.append('path_log_vcd_disagreement')
    else:raise ValueError('VCD negative control escaped')
    for name,path,reasons in [
        ('waveform_ownership_loss','/board/AXI_PCIE_EP/owned',{'Owned outgoing transfer','Continuous ownership through result acceptance'}),
        ('waveform_missing_host_receipt','/board/RP/m_axis_cq_tvalid',{'VCD ACK after complete host receipt'})]:
        damaged=drop_first_assertion(vcd,path)
        try:review_path_vcd(damaged,path_events)
        except ValueError as exc:
            require(str(exc) in reasons,'Wrong waveform rejection reason: '+str(exc));negative.append(name)
        else:raise ValueError('Waveform negative control escaped '+name)
    for name,path in [('waveform_missing_seal','/board/AXI_PCIE_EP/weight_loader/sealed'),
                      ('waveform_missing_loader_ownership','/board/AXI_PCIE_EP/weight_loader/busy')]:
        try:review_loader_vcd(drop_first_assertion(vcd,path))
        except ValueError:negative.append(name)
        else:raise ValueError('Loader waveform negative control escaped '+name)
    return dict(status='HBM_WEIGHT_NATIVE_REVIEWED',verdict=verdict,bar_vcd=bar_wave,path_vcd=path_wave,
        negative_controls=negative,bar_base=BAR,source_addresses=[0x1000,0x200001000],
        host_addresses=[HOST,HOST+256],hbm_model='AMD XSIM internal responder',physical_link_simulated=False,
        loader_vcd=loader_wave,native_pcie_weight_loading=True,native_top_with_loader_synthesized=False,
        physical_board_qualified=False,log_sha256=sha(root/'simulate.log'),vcd_sha256=sha(root/'output/native.vcd'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();text=json.dumps(review(a.root),indent=2)+'\n'
    if a.output:a.output.write_text(text)
    else:print(text,end='')
