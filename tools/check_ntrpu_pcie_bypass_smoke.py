"""Independent bounded BAR-to-AXI smoke trace review; not P2P qualification."""
import re
from check_ntrpu_pcie_bypass_native import bits, sample_vcd, review_native


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]+)', line))


def packet_blocks(text):
    blocks=[]
    for match in re.finditer(r'\[\s*(\d+)\]\s*:\s*([^\n]+)\n(.*?)(?=\[\s*\d+\]\s*:|\Z)',text,re.S):
        payload=bytes(int(v,16) for v in re.findall(r'^\s*0x([0-9a-fA-F]{2})\s*$',match[3],re.M))
        values={k.strip():int(v,16) for k,v in re.findall(r'^\s*([^:\n]+): (?:0x)?([0-9a-fA-F]+)',match[3],re.M)}
        blocks.append(dict(ps=int(match[1]),kind=match[2].strip(),values=values,payload=payload))
    return blocks


def review(text, tx_text, rx_text, native_vcd):
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'simulation diagnostic')
    events=[(s.split()[0].removeprefix('NTRPU_BYPASS_'),fields(s)) for s in text.splitlines() if s.startswith('NTRPU_BYPASS_')]
    def rows(name):return [v for k,v in events if k==name]
    def one(name):
        v=rows(name);require(len(v)==1,'one '+name);return v[0]
    enum=one('ENUM');final=one('PASS');memory=one('MEMORY_PASS')
    sampled=sample_vcd(native_vcd)
    native=review_native(sampled,int(enum['ps']),int(final['ps']))
    native_events=[e for e in sampled['events'] if e['ps']>=int(enum['ps'])]
    require(next(i for i,(k,_) in enumerate(events) if k=='ENUM')<next(i for i,(k,_) in enumerate(events) if k=='HOST_WRITE'),'enumeration precedes first write')
    require([enum['type'+str(i)] for i in range(3)]==['10','10','11'],'BAR types')
    require(int(enum['bar2'],16)==0 and int(enum['bar3'],16)==9 and int(enum['dma_bar'])==1,'BAR allocation')
    require(int(enum['range2'],16)&0xfffffff0==0xfff00000,'1 MiB aperture readback')
    require({k:final[k] for k in ['writes','reads','host_bytes']}==dict(writes='3',reads='3',host_bytes='12'),'final accounting')
    require(all(int(memory[k])==3 for k in ['aw','w','b','ar','r']),'complete AXI accounting')
    for name in ['HOST_WRITE','HOST_READ','HOST_VALUE','AW','W','B','AR','R']:
        require(len(rows(name))==3,'three '+name)
    offsets=[0,0x80000,0xfffc0]; words=[0x10234567,0x89abcdef,0xfedcba98]
    tx=packet_blocks(tx_text);rx=packet_blocks(rx_text)
    writes=[p for p in tx if p['kind']=='Memory Write-64 Frame']
    reads=[p for p in tx if p['kind']=='Memory Read-64 Frame']
    require(len(writes)==len(reads)==3,'actual MemWr/MemRd count')
    measured=[]
    for i,(offset,word) in enumerate(zip(offsets,words)):
        hw,hr,hv,aw,w,b,ar,r=[rows(n)[i] for n in ['HOST_WRITE','HOST_READ','HOST_VALUE','AW','W','B','AR','R']]
        require(all(int(v['case'])==i for v in [hw,hr,hv]),'host case order')
        require(int(hw['addr'],16)==int(hr['addr'],16)==(9<<32)+offset,'host address')
        require(int(aw['addr'],16)==int(ar['addr'],16)==(1<<32)+offset,'translated full AXI address')
        require(int(hw['data'],16)==int(hv['data'],16)==word,'host data')
        require(all(int(v['len'])==0 and int(v['size']) in {2,6} and int(v['burst'])==1 for v in [aw,ar]),'single-beat AXI format')
        require(int(w['strb'],16)==15 and w['last']=='1','only requested bytes strobed')
        # Unstrobed W lanes are allowed to be unknown; the four active bytes are not.
        require(int(w['data'][-8:],16)==word,'AXI write bytes')
        require(b['id']==aw['id'] and b['resp']=='0','write response identity')
        require(r['id']==ar['id'] and r['resp']=='0' and r['last']=='1','read response identity')
        expected=(int.from_bytes(bytes([0xa5])*60,'big')<<32)|word
        require(int(r['data'],16)==expected,'read payload and untouched sentinel')
        require(int(enum['ps'])<=int(hw['ps'])<=int(aw['ps'])<int(b['ps']),'write causal order')
        require(int(hw['ps'])<=int(w['ps'])<int(b['ps']),'write data causal order')
        require(int(hr['ps'])<=int(ar['ps'])<int(r['ps'])<int(hv['ps'])<=int(memory['ps']),'read causal order')
        for p,h,kind in [(writes[i],hw,'write'),(reads[i],hr,'read')]:
            v=p['values'];address=(v['Address High']<<32)|v['Address Low']
            require(address==(9<<32)+offset and v['Length']==1,'serialized '+kind+' address/size')
            require(v['Tag']==int(h['tag'],16)&255 and v['Last and First Byte Enables']==15,'serialized request identity')
            require(v['Traffic Class']==0 and v['TD']==0 and v['EP']==0 and v['Attributes']==0,'serialized attributes')
            require(int(h['ps'])<=p['ps']<=int((aw if kind=='write' else ar)['ps']),'serialized request time')
        require(writes[i]['payload']==word.to_bytes(4,'little'),'serialized MemWr payload')
        require(not reads[i]['payload'],'MemRd has no payload')
        witness=native['rows'][i]
        require(witness['requested_tag']==int(hr['tag'],16)&255,'requested native tag')
        tag=witness['allocated_tag']
        completions=[p for p in rx if p['kind']=='Completion With Data Frame' and p['ps']==witness['completion_ps']]
        require(len(completions)==1,'one actual read completion')
        cpl=completions[0];v=cpl['values']
        require(v['Tag']==tag,'completion matches actual core-allocated tag')
        # Native RC proves the DWORD and Byte Count; this public decoder prints
        # bypass payload bytes most significant first, unlike its control path.
        require(v['Completion Status']==0 and v['Length']==1 and len(cpl['payload'])==4,'read completion status/size')
        require(cpl['payload']==word.to_bytes(4,'big'),'decoded read completion payload')
        require(int(r['ps'])<=cpl['ps']<=int(hv['ps']),'completion follows actual AXI response')
        for ch,logged in [('AW',aw),('W',w),('B',b),('AR',ar),('R',r)]:
            observed=[e for e in native_events if e['channel']==ch][i]
            require(int(logged['ps'])==observed['ps'],'log/native handshake time '+ch)
        require(reads[i]['ps']==witness['read_request_ps'],'log/native read request time')
        measured.append(dict(offset=offset,awsize=int(aw['size']),arsize=int(ar['size']),
            write_response_ps=int(b['ps'])-int(hw['ps']),read_result_ps=int(hv['ps'])-int(hr['ps'])))
    require(max(int(v['ps']) for v in rows('B'))<min(int(v['ps']) for v in rows('HOST_READ')),'all writes before readback')
    require(int(final['ps'])==int(memory['ps']),'final drain')
    finish=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns)\s*:(.+)$',text,re.M)
    require(len(finish) in {1,2} and len(set(finish))==1 and int(finish[0][0])*(1000 if finish[0][1]=='ns' else 1)==int(final['ps']),'actual simulator finish/identical Tcl echo')
    require(all(int(memory[k+'_stall'])==sampled['stalls'][k.upper()] for k in ['aw','w','ar']),'independent native stall counts')
    return dict(status='PASS_PCIE_BYPASS_DWORD_NATIVE',qualified=True,pcie_base=9<<32,axi_base=1<<32,
                writes=3,reads=3,host_write_bytes=12,host_read_bytes=12,measurements=measured,
                stalls={k:int(memory[k]) for k in ['aw_stall','w_stall','ar_stall']},final_ps=int(final['ps']),
                native=native,
                scope='One endpoint and bounded behavioral sink; three DWORDs in a 1 MiB window, independently sampled native RQ/RC and AXI. Public decoded TX is pre-core request intent, not an on-wire TLP/ECRC witness. No DDR, P2P, long bursts, reset or throughput qualification.')
