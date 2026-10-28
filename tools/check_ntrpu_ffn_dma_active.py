"""Qualify simultaneous real FFN/DMA aborts and subsequent full recovery."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
from check_ntrpu_ffn_dma_recovery import parse,inputs,SPANS,BASES
from check_ntrpu_ffn_memory import reference
from check_ntrpu_ffn_dma_active_control import review_control


def review(text,seed,host_period_ps):
    result=dict(status='FAIL',errors=[],seed=seed)
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        need(host_period_ps in (4000,6200),'clock profile')
        need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL\b|CRITICAL WARNING:)',text),'tool error')
        rows=parse(text);by=defaultdict(list)
        for line,name,r in rows:by[name].append((line,r))
        def rec(name):return [r for _,r in by['NTRPU_'+name]]
        def one(name):
            values=rec(name);need(len(values)==1,'one '+name);return values[0]
        need(one('FFN_MEMORY_SEED')==dict(seed=seed),'seed')
        need(one('FFN_MEMORY_FORMAT')==dict(fp32=1),'format')
        need(one('DMA_ACTIVE_CONFIG')==dict(host_period_ps=host_period_ps,device_period_ps=5000,mgmt_period_ps=10000,dim=32,hidden=32),'configuration')
        need(one('DMA_ACTIVE_PASS')==dict(faults=16,host=8,device=8,aborted_ffns=16,ffns=16,fp32=512,aborts=16),'coverage')
        faults=list(range(1,17));odds=list(range(1,33,2));evens=list(range(2,33,2))
        need([r['id'] for r in rec('DMA_REC_FFN_BEGIN')]==list(range(1,33)),'all32 FFNs started')
        need([r['id'] for r in rec('DMA_REC_FFN_END')]==evens,'only16 successful FFNs ended')
        need([r['id'] for r in rec('DMA_REC_RETAIN')]==odds,'retained memory for every abort')
        groups={};pauses={'dt':[],'ht':[]};losses=[]
        names=['DMA_ACTIVE_FAULT','DMA_ACTIVE_ABORT','DMA_ACTIVE_RECOVERY']+['DMA_REC_'+n for n in ['STOP','BLOCKED','RESUME','RECOVERED']]
        for name in names:
            values=rec(name);need([r.get('fault') for r in values]==faults,'ordered sixteen '+name)
            groups[name]={r['fault']:r for r in values}
        for f in faults:
            ident=2*f-1;target=(f-1)%4;kind=(f-1)//8;host=(f-1)//4%2==0
            fault,abort,recovery=[groups[n][f] for n in names[:3]]
            stop,blocked,resume,recovered=[groups['DMA_REC_'+n][f] for n in ['STOP','BLOCKED','RESUME','RECOVERED']]
            need((fault['id'],fault['target'],fault['kind'],fault['stop_host'])==(ident,target,kind,int(host)),'fault schedule')
            need((abort['id'],abort['target'],recovery['id'],recovery['target'])==(ident,target,ident+1,target),'abort/recovery owners')
            need(fault['ps']<stop['ps']<blocked['ps']==abort['ps']<resume['ps']<recovered['ps']<recovery['ps'],'fault timing order')
            need(0<blocked['ps']-stop['ps']<=1000000 and resume['ps']-blocked['ps']>=1000000 and 0<recovered['ps']-resume['ps']<4000000,'detection/hold/recovery limits')
            key,period,phase=('ht',host_period_ps,700) if host else ('dt',5000,0)
            for r in [stop,blocked,resume,recovered]:need(r['id']==ident and r['ext']==0,'automatic recovery owner')
            for r in [stop,blocked]:need((r['den'],r['hen'])==((1,0) if host else (0,1)),'clock disabled')
            for r in [resume,recovered]:need((r['den'],r['hen'])==(1,1),'clocks resumed')
            for r in [stop,recovered]:need(all(r[k]==1 for k in ['rst','hrst','ha','da','run']),'active boundary')
            for r in [blocked,resume]:need(all(r[k]==0 for k in ['rst','hrst','ha','da','run']),'closed boundary')
            for r in [stop,resume]:need((r['ps']-phase-1)%period==0,'clock gate changed low')
            need(stop[key]==blocked[key]==resume[key],'selected clock really stopped')
            pauses[key].append((stop['ps'],resume['ps']))
            losses.append(dict(fault=f,id=ident,target=target,kind=kind,domain=key,detection_ps=blocked['ps']-stop['ps'],recovery_ps=recovered['ps']-resume['ps']))
        def edges(ps,key):
            period,phase={'dt':(5000,2500),'ht':(host_period_ps,700+host_period_ps//2),'mt':(10000,5000)}[key]
            full=lambda t:max(0,1+(t-phase)//period)
            return full(ps)-sum(full(min(ps,z))-full(a) for a,z in pauses.get(key,[]) if a<=ps)
        for _,name,r in rows:
            if name not in {'NTRPU_DMA_REC_'+n for n in ['M','OUT','STOP','BLOCKED','RESUME','RECOVERED']}:continue
            for key in ['dt','ht','mt']:need(r[key]==edges(r['ps'],key),'actual '+key+' edges')
            if r['ps']>=640001:need(r['ext']==0,'no external reset rescue')
            for f in faults:
                if groups['DMA_REC_BLOCKED'][f]['ps']<=r['ps']<=groups['DMA_REC_RESUME'][f]['ps']:
                    need(all(r[k]==0 for k in ['rst','hrst','ha','da','run']),'closed through stopped interval')
        mgmt=rec('DMA_REC_M');need(mgmt and [r['ps'] for r in mgmt]==list(range(5001,mgmt[-1]['ps']+1,10000)),'continuous management observation')
        normal=inputs();golden=reference(32,32,True);physical=bytearray([0xa5])*(8*16384)
        tx={};active=None;pending_fault=0;counters=Counter();chunks=defaultdict(list);loads=0
        def address(addr):
            for b,base in enumerate(BASES):
                if base+4032<=addr<base+4032+SPANS[b]:return b,addr-base-4032
            raise ValueError('registered buffer range')
        def payload(r):return r['data'].to_bytes(64,'little')
        def expected(b,off):
            v=(normal if b<4 else golden)[b][off:off+64];need(len(v)==64,'payload bounds');return v
        def fnv(data):
            h=0xcbf29ce484222325
            for v in data:h=((h^v)*0x100000001b3)&((1<<64)-1)
            return h
        for line,name,r in rows:
            if name=='NTRPU_DMA_ACTIVE_FAULT':pending_fault=r['fault']
            elif name=='NTRPU_DMA_REC_RECOVERED':pending_fault=0
            elif name=='NTRPU_DMA_BEGIN':
                need(active is None and r['epoch']==len(tx)+1,'one unique active DMA epoch')
                address(r['addr']);need(1<=r['beats']<=4 and r['addr']%64==0 and r['addr']%4096+64*r['beats']<=4096,'burst/page boundary')
                active=r['epoch'];tx[active]=dict(begin=r,line=line,events=defaultdict(list),loads=[],fault=pending_fault)
            elif name in {'NTRPU_DMA_'+n for n in ['AW','W','B_HELD','B','AR','R_HELD','R','END']}:
                need(active is not None and r['epoch']==active,'DMA response epoch ownership')
                kind=name.removeprefix('NTRPU_DMA_');t=tx[active];t['events'][kind].append(r)
                if kind in {'AW','W','B','AR','R'}:
                    counters[kind]+=1;need(r['count']==counters[kind],'monotonic accepted '+kind)
                    need((r['ps']-700)%host_period_ps==host_period_ps//2 and not any(a<r['ps']<z for a,z in pauses['ht']),'accepted transfer actual host edge')
                if kind=='END':active=None
            elif name=='NTRPU_TRANSFER_LOAD':
                need(active is not None and not tx[active]['fault'],'physical load belongs to normal DMA')
                b,off=r['buffer'],r['offset'];need(0<=b<4 and off%64==0 and payload(r)==expected(b,off),'host payload reference')
                tx[active]['loads'].append((b,off,payload(r)));physical[b*16384+4032+off:b*16384+4096+off]=payload(r);loads+=64
            elif name=='NTRPU_FFN_MEMORY_W':
                ident,b,off=r['id'],r['buffer'],r['offset'];key=(ident,b)
                need(1<=ident<=32 and b in golden and off==64*len(chunks[key]) and r['strb']==(1<<64)-1,'core write identity/offset/strobe')
                need(payload(r)==expected(b,off),'independent computed byte reference')
                chunks[key].append(payload(r));physical[b*16384+4032+off:b*16384+4096+off]=payload(r)
            elif name=='NTRPU_DMA_REC_RETAIN':need(r['hash']==fnv(physical),'retained RAM reconstructed from all physical writes')
        need(active is None and len(tx)==560 and loads==106496,'complete host loads and560 transactions')
        bursts=[]
        for b in range(4):
            off=0
            while off<SPANS[b]:
                addr=BASES[b]+4032+off;beats=min(4,(SPANS[b]-off)//64,(4096-addr%4096)//64)
                bursts.append((0,addr,beats));off+=64*beats
        starts={r['id']:r for r in rec('DMA_REC_FFN_BEGIN')};ends={r['id']:r for r in rec('DMA_REC_FFN_END')}
        jobs={r['id']:r for r in rec('FFN_IRQ_JOB')};done={r['id']:r for r in rec('FFN_IRQ_DONE')};service={r['id']:r for r in rec('FFN_IRQ_SERVICE_REQUEST')}
        for f in faults:
            ident=2*f-1;batch=[tx[e] for e in range((f-1)*35+1,f*35+1)]
            want=bursts+[(1,BASES[1]+4032,1)]+bursts+[(1,BASES[7]+4032+off,1) for off in (0,64)]
            need([(t['begin']['kind'],t['begin']['addr'],t['begin']['beats']) for t in batch]==want,'complete per-fault operation schedule')
            need([t['fault'] for t in batch]==[0]*16+[f]+[0]*18,'one outstanding interrupted read')
            need(batch[15]['events']['END'][0]['ps']<=starts[ident]['ps']<jobs[ident]['time_ps']<groups['DMA_ACTIVE_FAULT'][f]['ps']<=batch[16]['begin']['ps'],'real compute accepted before interrupted DMA')
            need(groups['DMA_REC_RECOVERED'][f]['ps']<=batch[17]['begin']['ps']<batch[32]['events']['END'][0]['ps']<=starts[ident+1]['ps'],'post-reset fresh AXI loading')
            need(done[ident+1]['time_ps']<batch[33]['begin']['ps']<batch[34]['events']['END'][0]['ps']<service[ident+1]['time_ps']<ends[ident+1]['ps'],'result consumed before retirement')
            need(all(ident not in table for table in [done,service,ends]),'aborted compute has no result/service')
        read_bytes=0;aborted=[]
        for epoch,t in tx.items():
            b,ev,f=t['begin'],t['events'],t['fault'];end=ev['END']
            need(len(end)==1 and end[0]['aborted']==bool(f),'one DMA terminal outcome')
            need(all(b['ps']<=r['ps']<=end[0]['ps'] for v in ev.values() for r in v),'all channels within epoch interval')
            for k in ['AW','AR']:
                for r in ev[k]:need((r['addr'],r['beats'])==(b['addr'],b['beats']),'accepted address identity')
            for k in ['W','R']:
                for i,r in enumerate(ev[k]):
                    need(r['beat']==i and r['last']==int(i==b['beats']-1),'beat and LAST')
                    need(payload(r)==expected(*address(b['addr']+64*i)),'DMA input/readback bytes')
                    if k=='W':need(r['strb']==(1<<64)-1,'all DMA byte lanes')
            counts={k:len(v) for k,v in ev.items() if v}
            if f:
                aborted.append(f);kind=(f-1)//8
                need(counts==(dict(AR=1,END=1) if kind==0 else dict(AR=1,R_HELD=1,END=1)),'interrupted read prefix')
                need(end[0]['phase']==4+kind and not t['loads'],'read abort phase')
                need(max(r['ps'] for k,v in ev.items() if k!='END' for r in v)<groups['DMA_REC_STOP'][f]['ps']<end[0]['ps']<=groups['DMA_REC_BLOCKED'][f]['ps'],'read stopped and aborted on automatic reset')
            elif b['kind']==0:
                need(counts==dict(AW=1,W=b['beats'],B_HELD=1,B=1,END=1),'complete write channels')
                need(ev['B'][0]['ps']>max(ev['AW'][0]['ps'],ev['W'][-1]['ps']),'B after AW/W')
                need(t['loads']==[(*address(b['addr']+64*i),payload(ev['W'][i])) for i in range(b['beats'])],'physical RAM equals accepted host beats')
            else:
                need(b['kind']==1 and counts==dict(AR=1,R_HELD=b['beats'],R=b['beats'],END=1),'complete read channels')
                need(ev['AR'][0]['ps']<ev['R'][0]['ps'] and not t['loads'],'read order');read_bytes+=64*b['beats']
        need(aborted==faults and read_bytes==2048,'16 aborted reads and2048 result bytes')
        verified=0;partial=0
        cases=rec('FFN_MEMORY_CASE');need([c['id'] for c in cases]==evens,'complete FFN records')
        line_index=lambda name,ident:next(i for i,r in by['NTRPU_'+name] if r['id']==ident)
        for ident in range(1,33):
            sizes={b:len(chunks[(ident,b)])*64 for b in golden}
            if ident%2:
                f=(ident+1)//2;target=(f-1)%4;abort=groups['DMA_ACTIVE_ABORT'][f]
                want={b:len(golden[b]) if b<target+4 else 64 if b==target+4 else 0 for b in golden}
                need(sizes==want,'partial prefix held at target phase B')
                need((abort['bytes'],abort['down_bytes'])==(sum(sizes.values()),sizes[7]),'aborted physical accounting')
                write_lines=[i for i,r in by['NTRPU_FFN_MEMORY_W'] if r['id']==ident]
                target_lines=[i for i,r in by['NTRPU_FFN_MEMORY_W'] if r['id']==ident and r['buffer']==target+4]
                need(target_lines and max(target_lines)<line_index('DMA_ACTIVE_FAULT',ident),'target physical output precedes concurrent fault')
                need(max(write_lines)<line_index('DMA_REC_BLOCKED',ident),'no computed writes after reset block')
                partial+=sum(sizes.values())
            else:
                need(sizes=={b:len(v) for b,v in golden.items()},'complete post-recovery numerical coverage')
                c=next(c for c in cases if c['id']==ident)
                need(tuple(c[k] for k in ['dim','hidden','mode','target','reset'])==(32,32,0,0,0),'successful case contract')
                need(tuple(c[k] for k in ['ar','r','aw','w','b','bytes','down_bytes'])==(13,62,7,11,7,704,128),'complete AXI accounting')
                need(0<c['cycles']<30000 and sum(c['p'+str(i)] for i in range(4))==c['cycles'],'successful phase cycle counts')
                need(ends[ident]['load_bytes']==3328 and ends[ident]['result_bytes']==128,'post-recovery transfer totals')
            verified+=sum(sizes.values())
        summary=one('FFN_MEMORY_PASS')
        need(tuple(summary[k] for k in ['cases','bytes','resets','cancels'])==(32,11264,16,0),'summary covers successful bytes separately from aborts')
        need(summary['cycles']==sum(c['cycles'] for c in cases),'successful cycle total')
        need(all(summary[k]>0 for k in ['ar_stalls','aw_stalls','w_stalls','done_stalls']),'backpressure')
        result.update(cases=cases,summary=summary,losses=losses,transactions=len(tx),host_committed_bytes=loads,
                      host_result_bytes=read_bytes,verified_core_bytes=verified,partial_core_bytes=partial,fp32_values=512)
        result=review_control(text,seed,host_period_ps,result)
    except (ValueError,KeyError,IndexError,TypeError,OverflowError,StopIteration) as error:result['errors'].append(str(error))
    result['status']='FAIL' if result['errors'] else 'PASS'
    result['scope']='16 simultaneous FFN+DMA read aborts, four compute phases, both stopped clocks and AR-pending/R-held states. Each followed by full AXI reload, D32/H32 FFN, exact host result and ABI2 retirement. Synthetic weights, real HLS/AMD CDC, behavioral RAM/protocol DMA. No physical PCIe/DDR bandwidth.'
    return result


def mutations(text):
    def change(prefix,key,value):return re.sub(r'(^'+prefix+r'[^\n]*\b'+key+r'=)\w+',lambda m:m[1]+str(value),text,count=1,flags=re.M)
    out={'fatal':text+'\nError: injected\n','missing_pass':text.replace('NTRPU_DMA_ACTIVE_PASS ','REMOVED_PASS ',1)}
    for name,prefix,key,value in [
        ('input_byte','NTRPU_TRANSFER_LOAD','data',0),('core_byte','NTRPU_FFN_MEMORY_W','data',0),
        ('host_byte','NTRPU_DMA_W','data',0),('readback','NTRPU_DMA_R','data',0),('retention','NTRPU_DMA_REC_RETAIN','hash',0),
        ('epoch','NTRPU_DMA_AR','epoch',9999),('count','NTRPU_DMA_AW','count',9999),('dma_edge','NTRPU_DMA_W','ps',1),
        ('last','NTRPU_DMA_W','last',0),('wrong_phase','NTRPU_DMA_ACTIVE_FAULT','target',3),
        ('wrong_domain','NTRPU_DMA_REC_STOP','hen',1),('external_reset','NTRPU_DMA_REC_BLOCKED','ext',1),
        ('early_admission','NTRPU_DMA_REC_BLOCKED','ha',1),('edges','NTRPU_DMA_REC_STOP','dt',0),
        ('partial_bytes','NTRPU_DMA_ACTIVE_ABORT','bytes',1),('completion_owner','NTRPU_FFN_IRQ_EVENT','sequence','ffffffff'),
        ('elapsed','NTRPU_FFN_HOST_CASE id=2','compute_cycles',1),('abi','NTRPU_FFN_IRQ_READ[^\n]*addr=00000004','data','00010000'),
        ('job_clock','NTRPU_FFN_IRQ_JOB','time_ps',1),('event_mode','NTRPU_FFN_IRQ_EVENT','enable',0),
        ('false_counter','NTRPU_FFN_IRQ_READ[^\n]*addr=00000018','data',3),
    ]:out[name]=change(prefix,key,value)
    out['hidden_abort']=text.replace('aborted=1 phase=','aborted=0 phase=',1)
    out['missing_service']=re.sub(r'^NTRPU_FFN_IRQ_SERVICE id=2 .+\n','',text,count=1,flags=re.M)
    out['missing_retire']=re.sub(r'^NTRPU_FFN_IRQ_RETIRE id=2 .+\n','',text,count=1,flags=re.M)
    out['missing_ack']=re.sub(r'^NTRPU_FFN_IRQ_ACK id=2 .+\n','',text,count=1,flags=re.M)
    out['missing_mgmt']=re.sub(r'^NTRPU_DMA_REC_M .+\n','',text,count=1,flags=re.M)
    out['missing_end']=re.sub(r'^NTRPU_DMA_END .+\n','',text,count=1,flags=re.M)
    out['missing_start']=re.sub(r'^NTRPU_FFN_HOST_WRITE id=1 addr=00000010 data=00000001 .+\n','',text,count=1,flags=re.M)
    out['stale_service']=text.replace('data=00000004 resp=2','data=00000004 resp=0',1)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--host-period-ps',type=int,choices=[4000,6200],required=True);a=p.parse_args()
    r=review(a.log.read_text(),a.seed,a.host_period_ps);print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
