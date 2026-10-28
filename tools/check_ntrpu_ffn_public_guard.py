"""Review public-host capacity errors, actual MIG sentinels and subsequent FFNs."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from check_ntrpu_ffn_public_cold import review as review_cold,mutations as cold_mutations

CAPACITY=1<<37
SENTINELS=[0,65536,CAPACITY-256]

def schedule():
    ops=[dict(write=wr,addr=addr,beats=4,tag=i+1,resp=0)
         for wr in (1,0) for i,addr in enumerate(SENTINELS)]
    bad=[(CAPACITY,1),(CAPACITY+65536,4),((1<<48)+65536,4),
         ((1<<63)+CAPACITY-256,4),((1<<64)-256,4),(8*CAPACITY+65536,3)]
    for i,(addr,beats) in enumerate(bad):
        ops += [dict(write=wr,addr=addr,beats=beats,tag=100+i,resp=3) for wr in (1,0)]
        ops += [dict(write=0,addr=addr,beats=4,tag=j+1,resp=0) for j,addr in enumerate(SENTINELS)]
    return ops

def word(tag,beat):
    return int.from_bytes(bytes((tag*31+beat*7+lane*13)&255 for lane in range(64)),'little')

def pack(parts):
    n=0
    for value,width in parts:n=(n<<width)|value
    return n

def review(text,seed=1,host_period_ps=4000):
    result=dict(status='FAIL',errors=[])
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        # The established checker parses all NTRPU fields as its own schema.
        # Probe records have a separate namespace and schema, reviewed below;
        # retain every other line, including all tool/model diagnostics.
        cold_text='\n'.join(line for line in text.splitlines() if not line.startswith('NTRPU_GP_'))+'\n'
        cold=review_cold(cold_text,seed,host_period_ps)
        need(cold['status']=='PASS','subsequent cold FFN bank: '+str(cold['errors']))
        phase=cold['device_phase_ps'];ops=schedule();rows=[];by=defaultdict(list)
        for line_number,line in enumerate(text.splitlines()):
            if not line.startswith('NTRPU_GP_'):continue
            name=line.split()[0].removeprefix('NTRPU_GP_')
            need(name in {'START','BEGIN','END','PASS','STALL','HOLD','ADDR','W','R','B'},'known probe observation')
            r={k:int(v,16 if k in {'addr','data','strb','sample'} else 10)
               for k,v in re.findall(r'(\w+)=([0-9a-f]+)(?=\s|$)',line)}
            rows.append((line_number,name,r));by[name].append(r)
        need(len(by['START'])==len(by['PASS'])==1,'one probe phase')
        start,end=by['START'][0],by['PASS'][0]
        need(start['capacity']==CAPACITY and cold['admission_ps']<start['ps']<end['ps'],'admitted probe phase')
        need((start['ps']-phase)%5000==2500,'probe starts on UI negative edge')
        need({k:v for k,v in end.items() if k!='ps'}==dict(tx=36,invalid=12,sentinel_bytes=768,host_w=32,host_r=104,mig_w=12,mig_r=84),'probe coverage summary')
        need([r['tx'] for r in by['BEGIN']]==list(range(1,37)) and [r['tx'] for r in by['END']]==list(range(1,37)),'ordered36probe transactions')
        first_ffn=re.search(r'^NTRPU_DMA_REC_FFN_BEGIN .*\bps=(\d+)',text,re.M)
        first_dma=re.search(r'^NTRPU_DMA_BEGIN .*\bps=(\d+)',text,re.M)
        need(first_ffn and first_dma and end['ps']<int(first_dma[1])<int(first_ffn[1]),'complete error bank before unchanged FFN loads')
        need(text.index('NTRPU_GP_PASS ')<text.index('NTRPU_DDR_FFN_REGION '),'no original FFN traffic hidden in probe phase')
        tx=defaultdict(lambda:defaultdict(list));events=defaultdict(list);holds=Counter();active=None
        previous_end=start['ps'];boundaries=Counter();physical={};total_edges=Counter()
        for line,name,r in rows:
            if name=='BEGIN':
                need(active is None,'serial probe lifecycle')
                active=r['tx'];spec=ops[active-1]
                need({k:v for k,v in r.items() if k not in {'tx','ps'}}==spec,'independent probe descriptor')
                need(previous_end<r['ps']<end['ps'] and (r['ps']-700)%host_period_ps==0,'probe begins on host negative edge')
                need(r['addr']%64==0 and r['addr']%4096+r['beats']*64<=4096,'AXI framing remains legal')
                permitted=r['addr']<CAPACITY and r['addr']+r['beats']*64<=CAPACITY
                need(permitted==(r['resp']==0),'capacity classification before narrowing')
                if not permitted:
                    alias=r['addr']&(CAPACITY-1)
                    need(any(a<=alias<alias+64*r['beats']<=a+256 for a in SENTINELS),'every alias target has sentinels')
                tx[active]['begin'].append(r)
            elif name=='END':
                need(active==r['tx'] and r['idle']==1,'probe drains before next transaction')
                need(r['ps']>tx[active]['begin'][0]['ps'] and (r['ps']-phase)%5000==2500,'drain on UI negative edge')
                tx[active]['end'].append(r);previous_end=r['ps'];active=None
            elif name in {'HOLD','ADDR','W','R','B'}:
                need(active is not None and active==r['tx'],'event belongs to current transaction')
                spec=ops[active-1];begin=tx[active]['begin'][0]
                need(begin['ps']<r['ps']<end['ps'],'event within probe time')
                if name=='HOLD':
                    ch=r['ch'];need(ch in list(range(5))+list(range(10,15)),'public probe channel');side=ch//5
                    sample=r['sample'];accepted=False;holds[ch]+=1
                else:
                    side=r['side'];need(side in (0,2),'public boundary')
                    if name=='ADDR':
                        need(r['write']==spec['write'],'expected address direction')
                        want=dict(addr=spec['addr'],beats=spec['beats'],size=6,burst=1,aid=9,cache=2,prot=0,lock=0)
                        need(all(r[k]==v for k,v in want.items()),'64bit address and attributes preserved')
                        key='AW' if r['write'] else 'AR';offset=int(r['write'])
                        sample=pack([(r['addr'],64),(r['beats']-1,8),(r['size'],3),(r['burst'],2),(r['aid'],4),(r['cache'],4),(r['prot'],3),(r['lock'],1)])
                    elif name in {'W','R'}:
                        key=name;offset=2 if name=='W' else 3
                        beat=len(tx[active][side,key]);need(beat<spec['beats'] and r['last']==int(beat+1==spec['beats']),'exact beat/LAST coverage')
                        want=0 if name=='R' and spec['resp']==3 else word(spec['tag'],beat)
                        need(r['data']==want,'independent data at boundary '+str(side))
                        if name=='W':
                            need(spec['write']==1 and r['strb']==(1<<64)-1,'write ownership/strobes')
                            sample=pack([(r['data'],512),(r['strb'],64),(r['last'],1)])
                        else:
                            need(spec['write']==0 and r['resp']==spec['resp'] and r['aid']==9,'read DECERR/ID')
                            sample=pack([(r['data'],512),(r['resp'],2),(r['aid'],4),(r['last'],1)])
                        if side==2:
                            addr=spec['addr']+beat*64
                            if name=='W':
                                need(addr not in physical,'sentinel written exactly once')
                                physical[addr]=r['data']
                            else:need(addr in physical and physical[addr]==r['data'],'actual MIG read from accepted sentinel write')
                    else:
                        key='B';offset=4
                        need(spec['write']==1 and r['resp']==spec['resp'] and r['aid']==9,'write DECERR/ID')
                        sample=pack([(r['resp'],2),(r['aid'],4)])
                    ch=side*5+offset;accepted=True
                    tx[active][side,key].append(r);boundaries[side,key]+=1
                need(side!=2 or spec['resp']==0,'rejected request never asserts any downstream VALID')
                period=host_period_ps if side==0 else 5000
                clk_phase=700+host_period_ps//2 if side==0 else phase
                need((r['ps']-clk_phase)%period==0,'event on actual observed clock edge')
                events[active,ch].append((r['ps'],accepted,sample));total_edges[side]+=1
        need(active is None and len(physical)==12,'all probes drained and768bytes really written')
        for ident,spec in enumerate(ops,1):
            group=tx[ident];beats=spec['beats'];wr=spec['write']
            for side in (0,2):
                live=int(side<2 or spec['resp']==0)
                counts={'AW':wr,'W':beats*wr,'B':wr,'AR':1-wr,'R':beats*(1-wr)}
                for key,count in counts.items():need(len(group[side,key])==count*live,'exact public channel coverage')
                if live:
                    if wr:
                        need(group[side,'B'][0]['ps']>max(group[side,'AW'][0]['ps'],group[side,'W'][-1]['ps']),'B after AW/allW')
                    else:need(group[side,'R'][0]['ps']>group[side,'AR'][0]['ps'],'R after AR')
            if spec['resp']==0:
                for key in (['AW','W'] if wr else ['AR']):
                    for a,z in zip(group[0,key],group[2,key]):need(a['ps']<z['ps'],'public forward causality')
                for key in (['B'] if wr else ['R']):
                    for a,z in zip(group[2,key],group[0,key]):need(a['ps']<z['ps'],'public return causality')
            need(group['end'][0]['ps']>max(event[0] for (tid,ch),stream in events.items() if tid==ident for event in stream),'end follows all observed traffic')
        need(by['STALL']==[dict(ch=c,count=holds[c]) for c in list(range(5))+list(range(10,15))],'complete public held-cycle totals')
        need(holds[3]>0 and holds[4]>0,'host R/B backpressure')
        for (ident,ch),stream in events.items():
            stream.sort();period=host_period_ps if ch<5 else 5000
            need(len({t for t,_,_ in stream})==len(stream),'one event per channel and clock edge')
            for i,(ps,accepted,value) in enumerate(stream):
                if not accepted:
                    need(i+1<len(stream) and stream[i+1][0]==ps+period and stream[i+1][2]==value,'continuous VALID/payload until handshake')
        need(boundaries[0,'W']==32 and boundaries[0,'R']==104 and boundaries[2,'W']==12 and boundaries[2,'R']==84,'reconstructed beat summary')
        result.update(cold=cold,probes=36,rejected_descriptors=12,sentinel_bytes=768,
                      sentinel_readbacks=21,held_cycles=sum(holds.values()),
                      boundaries={f'{side}_{key}':n for (side,key),n in sorted(boundaries.items())},
                      probe_start_ps=start['ps'],probe_end_ps=end['ps'],probe_duration_ps=end['ps']-start['ps'],
                      scope='Public hostAXI through AMD CDC and shared arbiter, capacity guard, generated generic128GiB MIG/DRAM BFM; public boundaries only. Cold operation; no runtime reset, exactDIMM, physical board or full inference.')
    except (ValueError,KeyError,IndexError,OverflowError,AssertionError) as exc:
        result['errors'].append(str(exc))
    result['status']='FAIL' if result['errors'] else 'PASS'
    return result

def mutations(text):
    out={'cold_'+k:v for k,v in cold_mutations(text).items()}
    for marker in ['START','BEGIN','END','PASS','STALL','HOLD','ADDR','W','R','B']:
        out['missing_gp_'+marker]=re.sub(r'^NTRPU_GP_'+marker+r' .+\n','',text,count=1,flags=re.M)
    for marker,key in [('BEGIN','addr'),('ADDR','addr'),('ADDR','aid'),('ADDR','beats'),('W','data'),('W','strb'),
                       ('R','data'),('R','resp'),('R','last'),('B','aid'),('HOLD','sample'),('HOLD','ps'),('END','idle')]:
        base=16 if key in {'addr','data','strb','sample'} else 10
        def alter(m,base=base):
            n=int(m[2],base)^1
            return m[1]+(format(n,'x') if base==16 else str(n))
        out['corrupt_gp_'+marker+'_'+key]=re.sub(r'(^NTRPU_GP_'+marker+r' .*?\b'+key+r'=)([0-9a-f]+)',alter,text,count=1,flags=re.M)
    out['decerr_to_okay']=re.sub(r'(^NTRPU_GP_B tx=7 .*?resp=)3',r'\g<1>0',text,count=1,flags=re.M)
    out['host_address_truncation']=re.sub(r'(^NTRPU_GP_ADDR tx=7 side=0 .*?addr=)[0-9a-f]+',r'\g<1>0000000000000000',text,count=1,flags=re.M)
    out['missing_bad_pair']=re.sub(r'^NTRPU_GP_(?:BEGIN|END|ADDR|W|R|B|HOLD) tx=(?:7|8) .*\n','',text,flags=re.M)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--host-period-ps',type=int,default=4000)
    a=p.parse_args();r=review(a.log.read_text(encoding='utf-8'),1,a.host_period_ps)
    print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
