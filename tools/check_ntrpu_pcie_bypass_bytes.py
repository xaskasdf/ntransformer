"""Independent native trace contract for the nine-case byte-mask BAR bank."""
import re
from check_ntrpu_pcie_bypass_native import bits, sample_vcd, require
from check_ntrpu_pcie_bypass_smoke import fields, packet_blocks

OFFSETS=(0,4,60,0x80000,0x80020,0x8003c,0xfffc0,0xfffe0,0xffffc)
MASKS=(15,5,10,15,3,12,15,1,14)
WORDS=tuple(0x10203040 ^ (i*0x11111111) for i in range(9))


def expected_memory():
    banks=[bytearray([0xa5]*64) for _ in range(3)]
    for i,(offset,mask,word) in enumerate(zip(OFFSETS,MASKS,WORDS)):
        for j,value in enumerate(word.to_bytes(4,'little')):
            if mask & (1<<j):banks[i//3][(offset&63)+j]=value
    return banks


def review(sim,tx_text,rx_text,vcd):
    require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',sim,re.M|re.I),'simulation diagnostic')
    lines=[s for s in sim.splitlines() if s.startswith('NTRPU_BYTES_')]
    events=[(s.split()[0].removeprefix('NTRPU_BYTES_'),fields(s)) for s in lines]
    def rows(name):return [v for n,v in events if n==name]
    def one(name):
        items=rows(name);require(len(items)==1,'one '+name);return items[0]
    enum,final,summary=[one(n) for n in ['ENUM','PASS','MEMORY_PASS']]
    require([enum['type'+str(i)] for i in range(3)]==['10','10','11'],'BAR types')
    require(int(enum['bar2'],16)==0 and int(enum['bar3'],16)==9 and int(enum['dma_bar'])==1,'BAR bases')
    require(int(enum['range2'],16)&0xfffffff0==0xfff00000,'BAR window')
    require([int(final[k]) for k in ['writes','reads','write_bytes','read_bytes']]==[9,9,24,36],'host byte accounting')
    require(all(int(summary[k])==9 for k in ['aw','w','b','ar','r']),'AXI counts')
    require(all(len(rows(n))==9 for n in ['HOST_WRITE','HOST_READ','HOST_VALUE','AW','W','B','AR','R']),'nine logged events')
    sampled=sample_vcd(vcd);start=int(enum['ps']);finish=int(final['ps'])
    native=[e for e in sampled['events'] if e['ps']>=start]
    groups={n:[e for e in native if e['channel']==n] for n in ['RQ','RC','TAG','AW','W','B','AR','R']}
    require(len(groups['RQ'])==18 and all(len(groups[n])==9 for n in groups if n!='RQ'),'native transaction counts')
    require(sampled['end_ps']==finish==int(summary['ps']),'native final drain')
    require(all(int(summary[k+'_stall'])==sampled['stalls'][k.upper()] for k in ['aw','w','ar']),'native stall counts')
    tx=packet_blocks(tx_text);rx=packet_blocks(rx_text)
    tx=[p for p in tx if p['ps']>=start];rx=[p for p in rx if p['ps']>=start]
    require(len(tx)==18 and len(rx)==9,'decoded packet counts')
    banks=expected_memory();measurements=[]
    for i,(offset,mask,word) in enumerate(zip(OFFSETS,MASKS,WORDS)):
        hw,hr,hv=[rows(n)[i] for n in ['HOST_WRITE','HOST_READ','HOST_VALUE']]
        require(all(int(v['case'])==i for v in [hw,hr,hv]),'host case ordering')
        require(int(hw['addr'],16)==int(hr['addr'],16)==(9<<32)+offset,'host address')
        require(int(hw['data'],16)==word and int(hw['mask'],16)==mask,'host write data/mask')
        expected_word=int.from_bytes(banks[i//3][offset&63:(offset&63)+4],'little')
        require(int(hv['data'],16)==expected_word,'host masked result')
        write,read=groups['RQ'][i],groups['RQ'][i+9]
        rc,tag,aw,w,b,ar,r=[groups[ch][i] for ch in ['RC','TAG','AW','W','B','AR','R']]
        for req,kind,be,host,packet in [(write,1,mask,hw,tx[i]),(read,0,15,hr,tx[i+9])]:
            data=req['tdata'];user=req['tuser']
            require(bits(data,63,0)==(9<<32)+offset,'native RQ address')
            require(bits(data,74,64)==1 and bits(data,78,75)==kind,'native RQ type/count')
            require(bits(data,79,79)==0 and bits(data,127,120)==0,'native RQ flags and high stimulus bits')
            require(bits(data,95,80)==0xaf and bits(data,119,104)==0x1a0,'native request IDs')
            require(bits(data,103,96)==int(host['tag'],16)==0x40+i+(0 if kind else 9),'requested eight-bit tag')
            require(bits(user,7,0)==be and bits(user,15,8)==0,'native request byte enables')
            require(bits(req['tkeep'])==(31 if kind else 15) and bits(req['tlast'])==1,'RQ keep/last')
            v=packet['values']
            require(packet['kind']==('Memory Write-64 Frame' if kind else 'Memory Read-64 Frame'),'decoded request type')
            require(packet['ps']==req['ps'],'decoded/native request time')
            require((v['Address High']<<32)|v['Address Low']==(9<<32)+offset,'decoded request address')
            require(v['Tag']==int(host['tag'],16) and v['Length']==1 and v['Last and First Byte Enables']==be,'decoded request identity')
            require(packet['payload']==(word.to_bytes(4,'little') if kind else b''),'decoded request payload')
        require(bits(write['tdata'],159,128)==word,'native RQ write data')
        for ch,a in [('AW',aw),('AR',ar)]:
            first=(mask & -mask).bit_length()-1 if ch=='AW' else 0
            require(bits(a['addr'])==(1<<32)+offset+first and bits(a['id'])==0,'AXI first-enabled-byte address/ID')
            require(bits(a['len'])==0 and bits(a['size'])==2 and bits(a['burst'])==1,'AXI request format')
            logged=rows(ch)[i]
            require(all(int(logged[k],16 if k in {'addr','id'} else 10)==bits(a[k]) for k in ['addr','id','len','size','burst']),'logged/native AXI '+ch)
        require(bits(w['strb'])==mask<<(offset&63) and bits(w['last'])==1,'shifted AXI byte strobes')
        for j in range(4):
            if mask&(1<<j):require(bits(w['data'],((offset&63)+j)*8+7,((offset&63)+j)*8)==(word>>(j*8))&255,'active AXI write byte')
        require(bits(b['resp'])==bits(b['id'])==0,'write response')
        require(bits(r['resp'])==bits(r['id'])==0 and bits(r['last'])==1,'read response')
        require(bits(r['data'])==int.from_bytes(banks[i//3],'little'),'full AXI line and untouched sentinels')
        for ch,n in [('AW',aw),('W',w),('B',b),('AR',ar),('R',r)]:
            require(int(rows(ch)[i]['ps'])==n['ps'],'native/log handshake time')
        require(int(rows('W')[i]['strb'],16)==bits(w['strb']) and rows('W')[i]['last']=='1','logged W mask/last')
        require(int(rows('R')[i]['data'],16)==bits(r['data']) and rows('R')[i]['last']=='1','logged R data/last')
        d=rc['tdata']
        require(bits(d,11,0)==offset%4096 and bits(d,15,12)==0,'RC address/error')
        require(bits(d,28,16)==4 and bits(d,29,29)==0 and bits(d,30,30)==1,'RC byte count/completed')
        require(bits(d,42,32)==1 and bits(d,46,43)==0,'RC count/status/poison')
        require(bits(d,71,64)==tag['tag'] and bits(d,63,48)==0xaf and bits(d,87,72)==0x1a0,'RC tag and IDs')
        require(bits(d,94,89)==0 and bits(d,127,96)==expected_word,'RC attributes and payload')
        require(bits(rc['tkeep'])==15 and bits(rc['tlast'])==1,'RC keep/last')
        require(bits(rc['tuser'],63,0)==0xf000 and bits(rc['tuser'],67,64)==1 and bits(rc['tuser'],96,96)==0,'RC byte enables/SOP/discontinue')
        packet=rx[i];v=packet['values']
        require(packet['kind']=='Completion With Data Frame' and packet['ps']==rc['ps'],'decoded completion/time')
        require(v['Tag']==tag['tag'] and v['Length']==1 and v['Completion Status']==0,'decoded completion identity/status')
        require(packet['payload']==expected_word.to_bytes(4,'big'),'decoded completion data')
        require(int(hw['ps'])<=write['ps']<=aw['ps']<b['ps'] and write['ps']<=w['ps']<b['ps'],'write chronology')
        require(int(hr['ps'])<=read['ps']<=tag['ps']<rc['ps']<=int(hv['ps']),'tag/completion chronology')
        require(read['ps']<=ar['ps']<r['ps']<=rc['ps'],'read chronology')
        measurements.append(dict(case=i,offset=offset,mask=mask,word=word,returned_word=expected_word,
                                 allocated_tag=tag['tag'],read_rq_to_rc_ps=rc['ps']-read['ps']))
    require(max(e['ps'] for e in groups['B'])<groups['RQ'][9]['ps'],'all writes drained before reads')
    endings=re.findall(r'^\$finish called at time\s*:\s*(\d+)\s*(ps|ns)\s*:(.+)$',sim,re.M)
    require(len(endings) in {1,2} and len(set(endings))==1 and int(endings[0][0])*(1000 if endings[0][1]=='ns' else 1)==finish,'actual finish and identical Tcl echo')
    return dict(status='PASS_PCIE_BYPASS_BYTE_MASKS',qualified=True,write_bytes=24,read_bytes=36,cases=measurements,
                native_stalls=sampled['stalls'],final_ps=finish,
                scope='Nine DWORD writes/reads, three behavioral 64-byte lines and last DWORD of 1 MiB BAR. Native interfaces only; no DDR, P2P, ECRC, bursts, reset or throughput qualification.')
