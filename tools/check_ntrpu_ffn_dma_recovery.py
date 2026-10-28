"""Independent byte, transaction, clock and ABI2 review of DMA recovery logs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import struct
from check_ntrpu_ffn_memory import reference
from check_ntrpu_ffn_dma_control import review_control
from check_ntrpu_ffn_irq_async import mutations as irq_mutations

SPANS = [64,1088,1088,1088,256,256,64,128]
BASES = [0x100000000+b*0x20000000 for b in range(8)]


def inputs():
    q = lambda i: 127 if i == 0 else 1 if i == 1 else i*7-113
    out = {0:bytes(q(i)&255 for i in range(32))+struct.pack('<d',1)+bytes(24)}
    for b in range(1,4):
        out[b] = b''.join(struct.pack('<e',257 if b == 1 else 1)+
                         bytes(int(i == (1 if b == 1 else row if b == 2 else row*7%32)) for i in range(32))
                         for row in range(32))
    return out


def parse(text):
    rows=[]
    for line_no,line in enumerate(text.splitlines()):
        if not line.startswith('NTRPU_'):continue
        name=line.split()[0]
        hexkeys={'addr','data','strb','hash','sequence','flags'}
        if not name.startswith('NTRPU_DMA_'):hexkeys.add('epoch')
        row={k:int(v,16 if k in hexkeys else 10) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
        rows.append((line_no,name,row))
    return rows


def review(text,seed,host_period_ps):
    result=dict(status='FAIL',errors=[],seed=seed)
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        need(host_period_ps in (4000,6200),'clock profile')
        need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL\b|CRITICAL WARNING:)',text),'tool error')
        rows=parse(text)
        by=defaultdict(list)
        for line,name,row in rows:by[name].append((line,row))
        def rec(name):return [r for _,r in by['NTRPU_'+name]]
        def one(name):
            values=rec(name);need(len(values)==1,'one '+name);return values[0]
        need(one('FFN_MEMORY_SEED')==dict(seed=seed),'seed')
        need(one('FFN_MEMORY_FORMAT')==dict(fp32=1),'format')
        need(one('DMA_REC_CONFIG')==dict(host_period_ps=host_period_ps,device_period_ps=5000,mgmt_period_ps=10000,dim=32,hidden=32),'configuration')
        need(one('DMA_REC_PASS')==dict(faults=10,host=5,device=5,ffns=10,fp32=320,aborts=10),'coverage')
        ids=list(range(1,11));pauses={'dt':[],'ht':[]};losses=[]
        groups={}
        for name in ['FAULT','STOP','BLOCKED','RESUME','RECOVERED','RETAIN','FFN_BEGIN','FFN_END']:
            values=rec('DMA_REC_'+name)
            need([r.get('id') for r in values]==ids,'ordered ten '+name)
            groups[name]={r['id']:r for r in values}
        for ident in ids:
            fault,stop,blocked,resume,recovered=[groups[n][ident] for n in ['FAULT','STOP','BLOCKED','RESUME','RECOVERED']]
            kind=(ident-1)%5;host=ident<=5
            need(fault==dict(id=ident,kind=kind,stop_host=int(host),ps=fault['ps']),'fault schedule')
            key,period,phase=('ht',host_period_ps,700) if host else ('dt',5000,0)
            need(fault['ps']<stop['ps']<blocked['ps']<resume['ps']<recovered['ps']<groups['FFN_BEGIN'][ident]['ps']<groups['FFN_END'][ident]['ps'],'fault/FFN order')
            need(0<blocked['ps']-stop['ps']<=1000000,'detection deadline')
            need(resume['ps']-blocked['ps']>=1000000,'stopped interval')
            need(0<recovered['ps']-resume['ps']<4000000,'recovery deadline')
            for r in [stop,blocked,resume,recovered]:need(r['fault']==ident and r['ext']==0,'automatic reset')
            for r in [stop,blocked]:need((r['den'],r['hen'])==((1,0) if host else (0,1)),'disabled domain')
            for r in [resume,recovered]:need((r['den'],r['hen'])==(1,1),'resumed domains')
            for r in [stop,recovered]:need(all(r[k]==1 for k in ['rst','hrst','da','ha','run']),'active boundary')
            for r in [blocked,resume]:need(all(r[k]==0 for k in ['rst','hrst','da','ha','run']),'closed boundary')
            for r in [stop,resume]:need((r['ps']-phase-1)%period==0,'clock gate low phase')
            need(stop[key]==blocked[key]==resume[key],'stopped domain edges')
            pauses[key].append((stop['ps'],resume['ps']))
            losses.append(dict(id=ident,kind=kind,domain=key,detection_ps=blocked['ps']-stop['ps'],recovery_ps=recovered['ps']-resume['ps']))
        def edges(ps,key):
            period,phase={'dt':(5000,2500),'ht':(host_period_ps,700+host_period_ps//2),'mt':(10000,5000)}[key]
            full=lambda t:max(0,1+(t-phase)//period)
            return full(ps)-sum(full(min(ps,z))-full(a) for a,z in pauses.get(key,[]) if a<=ps)
        trace=[(n,r) for _,n,r in rows if n in {'NTRPU_DMA_REC_'+v for v in ['M','OUT','STOP','BLOCKED','RESUME','RECOVERED']}]
        need(trace and any(r['ext']==1 for _,r in trace),'initial external reset observed')
        for name,r in trace:
            for key in ['dt','ht','mt']:need(r[key]==edges(r['ps'],key),'observed '+key+' edge count')
            if r['ps']>=640001:need(r['ext']==0,'no external reset rescue')
            for ident in ids:
                b,z=groups['BLOCKED'][ident]['ps'],groups['RESUME'][ident]['ps']
                if b<=r['ps']<=z:need(all(r[k]==0 for k in ['rst','hrst','ha','da','run']),'no release while clock stopped')
        mgmt=rec('DMA_REC_M')
        need([r['ps'] for r in mgmt]==list(range(5001,mgmt[-1]['ps']+1,10000)),'continuous management clock observation')
        # Every host transaction has an explicit epoch, accepted beat trace and end.
        tx={};active=None;counters=Counter();physical=bytearray([0xa5])*(8*16384)
        normal=inputs();golden=reference(32,32,True);core_chunks=defaultdict(list)
        current_fault=0;fault_active=False;commit_bytes=0;fault_commits=0
        def address(addr):
            for b,base in enumerate(BASES):
                if base+4032<=addr<base+4032+SPANS[b]:return b,addr-base-4032
            raise ValueError('address outside registered buffers')
        def payload(row):return row['data'].to_bytes(64,'little')
        def expected(b,off,altered=False):
            value=(normal if b<4 else golden)[b][off:off+64]
            need(len(value)==64,'payload bounds')
            return bytes(v^0x5a for v in value) if altered else value
        def fnv(data):
            h=0xcbf29ce484222325
            for v in data:h=((h^v)*0x100000001b3)&((1<<64)-1)
            return h
        for line,name,r in rows:
            if name=='NTRPU_DMA_REC_FAULT':current_fault=r['id'];fault_active=True
            elif name=='NTRPU_DMA_REC_RECOVERED':fault_active=False
            elif name=='NTRPU_DMA_BEGIN':
                need(active is None and r['epoch']==len(tx)+1,'nonoverlapping unique DMA epoch')
                b,off=address(r['addr']);need(1<=r['beats']<=4 and r['addr']%64==0 and r['addr']%4096+r['beats']*64<=4096,'DMA burst/page')
                active=r['epoch'];tx[active]=dict(begin=r,line=line,events=defaultdict(list),loads=[],fault=current_fault if fault_active else 0)
            elif name in {'NTRPU_DMA_'+n for n in ['AW','W','B_HELD','B','AR','R_HELD','R','END']}:
                need(active is not None and r['epoch']==active,'response belongs to active DMA epoch')
                t=tx[active];kind=name.removeprefix('NTRPU_DMA_');t['events'][kind].append(r)
                need(r['ps']>=t['begin']['ps'],'transaction time')
                if kind in {'AW','W','B','AR','R'}:
                    counters[kind]+=1;need(r['count']==counters[kind],'monotonic '+kind+' handshake count')
                    need((r['ps']-700)%host_period_ps==host_period_ps//2 and not any(a<r['ps']<z for a,z in pauses['ht']),'host handshake on actual clock')
                if kind=='END':active=None
            elif name=='NTRPU_TRANSFER_LOAD':
                need(active is not None,'RAM load belongs to active host transaction')
                t=tx[active];b,off=r['buffer'],r['offset']
                need(0<=b<4 and off%64==0,'host physical destination')
                altered=bool(t['fault']);value=payload(r)
                need(value==expected(b,off,altered),'independent host input bytes')
                need(not altered or (b==1 and off==0 and (t['fault']-1)%5==2),'only B-held fault physically committed')
                t['loads'].append((b,off,value));physical[b*16384+4032+off:b*16384+4096+off]=value
                commit_bytes+=64;fault_commits+=64*altered
            elif name=='NTRPU_FFN_MEMORY_W':
                b,off=r['buffer'],r['offset'];value=payload(r);key=(r['id'],b)
                need(b in golden and r['strb']==(1<<64)-1 and off==len(core_chunks[key])*64,'core physical write sequence/strobe')
                need(value==expected(b,off),'independent FFN intermediate/final byte reference')
                core_chunks[key].append(value);physical[b*16384+4032+off:b*16384+4096+off]=value
            elif name=='NTRPU_DMA_REC_RETAIN':need(r['hash']==fnv(physical),'retained RAM hash from all physical writes')
        need(active is None and len(tx)==350,'complete DMA transaction coverage')
        bursts=[]
        for buf in range(4):
            off=0
            while off<SPANS[buf]:
                addr=BASES[buf]+4032+off
                beats=min(4,(SPANS[buf]-off)//64,(4096-addr%4096)//64)
                bursts.append((0,addr,beats));off+=64*beats
        need(len(bursts)==16,'independent input burst schedule')
        for ident in ids:
            batch=[tx[e] for e in range((ident-1)*35+1,ident*35+1)]
            fault=groups['FAULT'][ident];kind=(ident-1)%5
            expected_ops=bursts+[(int(kind>=3),BASES[1]+4032,1)]+bursts+[(1,BASES[7]+4032+off,1) for off in (0,64)]
            need([(t['begin']['kind'],t['begin']['addr'],t['begin']['beats']) for t in batch]==expected_ops,'per-fault transfer schedule')
            need([t['fault'] for t in batch]==[0]*16+[ident]+[0]*18,'one interrupted DMA per recovery')
            need(batch[15]['events']['END'][0]['ps']<=fault['ps']<=batch[16]['begin']['ps'],'initial load before fault')
            need(groups['RECOVERED'][ident]['ps']<=batch[17]['begin']['ps'] and batch[32]['events']['END'][0]['ps']<=groups['FFN_BEGIN'][ident]['ps'],'complete fresh AXI reload before each FFN')
            done=[r for r in rec('FFN_IRQ_DONE') if r['id']==ident]
            service=[r for r in rec('FFN_IRQ_SERVICE_REQUEST') if r['id']==ident]
            need(len(done)==len(service)==1 and done[0]['time_ps']<batch[33]['begin']['ps']<batch[34]['events']['END'][0]['ps']<service[0]['time_ps'],'host reads final output after completion and before service')
        aborted=[];read_bytes=0
        for epoch,t in tx.items():
            b=t['begin'];ev=t['events'];fault=t['fault'];end=ev['END']
            need(len(end)==1 and end[0]['aborted']==bool(fault),'one end/abort per DMA')
            is_write=b['kind']==0;need(b['kind'] in (0,1),'DMA operation kind')
            need(all(b['ps']<=r['ps']<=end[0]['ps'] for events in ev.values() for r in events),'all beats before transaction end')
            for event in ['AW','AR']:
                for r in ev[event]:need((r['addr'],r['beats'])==(b['addr'],b['beats']),'accepted address/burst')
            for event in ['W','R']:
                for i,r in enumerate(ev[event]):
                    need(r['beat']==i and r['last']==int(i==b['beats']-1),'DMA beat/LAST sequence')
                    buf,off=address(b['addr']+64*i)
                    need(payload(r)==expected(buf,off,bool(fault and is_write)),'independent DMA payload')
                    if event=='W':need(r['strb']==(1<<64)-1,'DMA full write strobe')
            if fault:
                aborted.append(fault);kind=(fault-1)%5
                expected_counts=[dict(AW=1),dict(W=1),dict(AW=1,W=1,B_HELD=1),dict(AR=1),dict(AR=1,R_HELD=1)][kind]
                need({k:len(v) for k,v in ev.items() if v and k!='END'}==expected_counts,'fault accepted prefix/held-response state')
                need(end[0]['phase']==kind+1,'fault phase at abort')
                stop,blocked,resume=[groups[n][fault] for n in ['STOP','BLOCKED','RESUME']]
                need(max(r['ps'] for k,v in ev.items() if k!='END' for r in v)<stop['ps']<end[0]['ps']<=blocked['ps'],'fault trigger before stopped clock and abort before blocked observation')
                need(len(t['loads'])==int(kind==2),'fault physical commit coverage')
            elif is_write:
                need({k:len(v) for k,v in ev.items() if v}==dict(AW=1,W=b['beats'],B_HELD=1,B=1,END=1),'successful write channel coverage')
                need(ev['B'][0]['ps']>max(ev['AW'][0]['ps'],ev['W'][-1]['ps']),'B follows all accepted inputs')
                want=[(*address(b['addr']+64*i),payload(ev['W'][i])) for i in range(b['beats'])]
                need(t['loads']==want,'host W equals committed RAM bytes')
            else:
                need({k:len(v) for k,v in ev.items() if v}==dict(AR=1,R_HELD=b['beats'],R=b['beats'],END=1),'successful read channel coverage')
                need(ev['AR'][0]['ps']<ev['R'][0]['ps'] and not t['loads'],'read order/nonmutating')
                read_bytes+=64*b['beats']
        need(aborted==ids and commit_bytes==66688 and fault_commits==128 and read_bytes==1280,'aggregate host payload/abort counts')
        cases=rec('FFN_MEMORY_CASE');need([c['id'] for c in cases]==ids,'ten complete FFN cases')
        for c in cases:
            ident=c['id'];need(tuple(c[k] for k in ['dim','hidden','mode','target','reset'])==(32,32,0,0,0),'FFN contract')
            need(all(b''.join(core_chunks[(ident,b)])==v for b,v in golden.items()),'all FFN writes complete')
            need((c['ar'],c['r'],c['aw'],c['w'],c['b'],c['bytes'],c['down_bytes'])==(13,62,7,11,7,704,128),'core AXI accounting')
            need(0<c['cycles']<30000 and sum(c['p'+str(i)] for i in range(4))==c['cycles'],'per-stage elapsed cycles')
            need(groups['FFN_END'][ident]['load_bytes']==3328 and groups['FFN_END'][ident]['result_bytes']==128,'per-recovery host bytes')
            for name in ['FFN_BEGIN','FFN_END']:
                t=groups[name][ident]['ps'];need(not any(a<t<z for v in pauses.values() for a,z in v),'FFN runs with clocks active')
        summary=one('FFN_MEMORY_PASS')
        need((summary['cases'],summary['bytes'],summary['resets'],summary['cancels'])==(10,7040,10,0),'FFN summary coverage')
        need(summary['cycles']==sum(c['cycles'] for c in cases),'FFN total cycles')
        need(all(summary[k]>0 for k in ['ar_stalls','aw_stalls','w_stalls','done_stalls']),'backpressure exercised')
        result.update(cases=cases,summary=summary,losses=losses,transactions=len(tx),host_committed_bytes=commit_bytes,
                      fault_committed_bytes=fault_commits,host_result_bytes=read_bytes,verified_core_bytes=7040,fp32_values=320)
        result=review_control(text,seed,host_period_ps,result)
    except (ValueError,KeyError,IndexError,OverflowError,TypeError) as error:
        result['errors'].append(str(error))
    result['status']='FAIL' if result['errors'] else 'PASS'
    result['scope']='Real FFN/HLS and AMD CDC with synthetic D32/H32 data. DMA faults occur with compute idle; each is followed by host reload, full FFN, host readback and ABI2 retirement. Behavioral RAM/protocol DMA; no XDMA, DDR or physical bandwidth.'
    return result


def mutations(text):
    def change(prefix,field,value):return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)\w+',lambda m:m[1]+str(value),text,count=1,flags=re.M)
    old=irq_mutations(text)
    out={k:v for k,v in old.items() if k not in {'reset_duration','reset_missing_start','reset_poison_value','reset_late_ack'}}
    for name,prefix,field,value in [
        ('dma_epoch','NTRPU_DMA_W','epoch',999),('dma_payload','NTRPU_DMA_W','data',0),
        ('memory_input','NTRPU_TRANSFER_LOAD','data',0),('dma_last','NTRPU_DMA_W','last',0),
        ('dma_count','NTRPU_DMA_AW','count',999),('dma_clock','NTRPU_DMA_W','ps',1),
        ('retained_ram','NTRPU_DMA_REC_RETAIN','hash',0),('wrong_domain','NTRPU_DMA_REC_STOP','hen',1),
        ('edge_count','NTRPU_DMA_REC_STOP','dt',0),('external_rescue','NTRPU_DMA_REC_BLOCKED','ext',1),
        ('early_release','NTRPU_DMA_REC_BLOCKED','ha',1),('wrong_fault','NTRPU_DMA_REC_FAULT','kind',4),
        ('readback','NTRPU_DMA_R','data',0),
        ('clock_profile','NTRPU_DMA_REC_CONFIG','host_period_ps',1),('summary','NTRPU_DMA_REC_PASS','aborts',9),
    ]:out[name]=change(prefix,field,value)
    out['abort_hidden']=text.replace('aborted=1 phase=','aborted=0 phase=',1)
    out['missing_mgmt']=re.sub(r'^NTRPU_DMA_REC_M .+\n','',text,count=1,flags=re.M)
    out['missing_dma_end']=re.sub(r'^NTRPU_DMA_END .+\n','',text,count=1,flags=re.M)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--host-period-ps',type=int,choices=[4000,6200],required=True);a=p.parse_args()
    r=review(a.log.read_text(),a.seed,a.host_period_ps);print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
