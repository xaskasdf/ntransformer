"""Audit public ownership, native reads, payload, DMA and ACK; no internal probes."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from check_ntrpu_peer_shared import sha,require,request,lease,words,bad_config,rejection


def replay(text):
    require('SP PUBLIC_ONLY' in text and text.count('SP ALL_PASS cases=6')==1,'missing public completion')
    require(not any(s in text for s in ('SG_FAIL','SG_WATCHDOG','Fatal:')),'simulation failure')
    require([int(s.split()[2]) for s in text.splitlines() if s.startswith('SP PASS ')]==list(range(1,7)),'case coverage')
    groups=defaultdict(list)
    for line in text.splitlines():
        if not line.startswith(('SP ','SF ')) or line.startswith(('SP PASS ','SP ALL_PASS ','SP PUBLIC_ONLY')):continue
        p=line.split();require(p[1] in {'SAMPLE','REVOKE','CFG','REQ','REJECT','RESULT','AR','DATA','DMA','ACK'},'non-public event')
        groups[(int(p[3]),int(p[2]))].append((p[1],p[4:]))
    require({k[0] for k in groups}==set(range(1,7)),'epoch coverage')
    epoch=None;stats=Counter();endings=[]
    def finish():
        require(not any(rejects),'missing rejection')
        expected=2 if epoch in (4,6) else 0
        require(len(active)==expected and halted==bool(expected),'final ownership/halt state')
        endings.append(dict(epoch=epoch,held=len(active),halted=halted,watermark=watermark))
    for (e,ts),events in sorted(groups.items()):
        if e!=epoch:
            if epoch is not None:finish()
            epoch=e;active={};history={};rejects=[None,None];c=None;watermark=0;halted=False;revoked=False
        rows=defaultdict(list)
        for kind,p in events:rows[kind].append(p)
        if rows['SAMPLE']:
            require(len(rows['SAMPLE'])==len(rows['REVOKE'])==1,'clock observation')
            w,a,b,pending,h,ready=map(int,rows['SAMPLE'][0]);counts=[sum(k[0]==l for k in active) for l in range(2)]
            require([w,a,b]==[watermark,*counts],'public reservation count/watermark')
            require(pending in range(4) and all(not (pending&(1<<l)) or counts[l] for l in range(2)),'pending without reservation')
            require(h in (0,1) and (not halted or h),'halt cleared without reset');halted=bool(h)
            require(ready==int(not active and not halted),'configuration readiness')
            stats['samples']+=1;stats['peak_contexts']=max(stats['peak_contexts'],len(active))
            revoked|=bool(int(rows['REVOKE'][0][0]))
        else:require(not any(rows[k] for k in ('REVOKE','CFG','REQ','REJECT','RESULT')),'missing sample')
        require(len(rows['REQ'])<=1,'double admission')
        for p in rows['REQ']:
            require(c is not None and not halted and not rows['CFG'],'admission while closed')
            lane=int(p[0]);r=request(p[1]);key=(lane,r['identity'])
            require(lane in (0,1) and rejects[lane] is None,'request lane/rejection occupied')
            error=rejection(r,c,watermark,revoked,active);stats['requests']+=1
            if error:rejects[lane]=(error,r['identity'])
            else:
                require(sum(k[0]==lane for k in active)<2 and key not in history,'capacity/identity reused')
                item=dict(r=r,cfg=c.copy(),native=0,beats=0,dma=False,ack=False,result=False)
                active[key]=item;history[key]=item;watermark=r['identity'][2];stats['admitted']+=1
        for p in rows['CFG']:
            require(not active and not halted,'config with active work')
            new=lease(p[0]);require(not bad_config(new,c),'unexpected rejected config')
            c=new;revoked=bool(int(rows['REVOKE'][0][0]));stats['configs']+=1
        for p in rows['REJECT']:
            lane,code=map(int,p[:2]);identity=words(p[2],3)
            require(rejects[lane]==(code,identity),'rejection code/context');rejects[lane]=None;stats['rejections']+=1
        for kind,p in events:
            if kind=='AR':
                lane,address,beats=map(int,p)
                candidates=[v for (l,_),v in history.items() if l==lane and v['native']<v['r']['size']]
                require(candidates,'native read without reservation');item=candidates[0]
                require(1<=beats<=16 and address==item['cfg']['src']+item['r']['src']+item['native'],'native fragment address/length')
                item['native']+=beats*32;require(item['native']<=item['r']['size'],'native overread');stats['native_ar']+=1
            elif kind=='DATA':
                lane,seq,index,last=map(int,p[:4]);item=history.get((lane,(epoch,0,seq)))
                require(item is not None,'payload without reservation');r=item['r'];raw=int(p[4],16).to_bytes(64,'little')
                require(index==item['beats']+1 and index*64<=r['size'] and last==(index*64==r['size']),'payload index/last')
                address=item['cfg']['src']+r['src']+item['beats']*64
                expected=bytes((a^(a>>8)^(a>>16)^0x5b)&255 for a in range(address,address+64))
                require(raw==expected,'payload bytes');item['beats']+=1;stats['payload_bytes']+=64
            elif kind in ('DMA','ACK'):
                lane=int(p[0]);w=words(p[1],9,4);item=active.get((lane,w[:3]));require(item is not None,'unowned completion')
                r=item['r'];require(w[3:7]==r['target'] and w[7:]==(r['window'],r['size']) and int(p[1],16)&15==0,'completion context/count')
                require(item['beats']*64==r['size'] and not item[kind.lower()],'premature/duplicate completion')
                item[kind.lower()]=True;stats[kind.lower()]+=1
        for p in rows['RESULT']:
            lane,status,size=map(int,p[:3]);r=request(p[3]);key=(lane,r['identity']);item=active.get(key)
            require(item is not None and item['r']['raw']==r['raw'] and not item['result'],'result identity/lifetime')
            require(size==(0 if status else r['size']),'result count');item['result']=True
            if status:stats['failed_results_retained']+=1
            else:
                require(item['beats']*64==item['native']==r['size'] and item['dma'] and item['ack'],'release without full payload/DMA/ACK')
                del active[key];stats['successful_results']+=1;stats['successful_payload_bytes']+=size
    finish()
    require(stats['peak_contexts']==4 and stats['successful_results']==27 and stats['successful_payload_bytes']==46848 and
            stats['failed_results_retained']==4 and stats['rejections']==5,'expected transaction coverage')
    return dict(counters=dict(sorted(stats.items())),epoch_endings=endings)


def controls(text):
    lines=text.splitlines();rejected=[]
    variants=[]
    for prefix in ('SP REQ ','SP REJECT ','SP RESULT ','SP CFG ','SP SAMPLE ','SF AR ','SF DATA ','SF DMA ','SF ACK '):
        index=next(i for i,s in enumerate(lines) if s.startswith(prefix))
        variants.append((prefix.strip(),lines[:index]+lines[index+1:]))
    index=next(i for i,s in enumerate(lines) if s.startswith('SF DATA '))
    p=lines[index].split();p[-1]=format(int(p[-1],16)^1,'0128x')
    variants.append(('payload_bit',lines[:index]+[' '.join(p)]+lines[index+1:]))
    for name,changed in variants:
        try:replay('\n'.join(changed))
        except (ValueError,KeyError,IndexError):rejected.append(name)
        else:raise ValueError('negative control accepted: '+name)
    return rejected


def review(out):
    r=json.loads((out/'result.json').read_text());require(r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and
        not r['timed_out'] and r['sources']==r['sources_after'] and r['donor_unchanged'],'runner failed/changed')
    for n,h in json.loads((out/'FILES.json').read_text()).items():require(sha(out/n)==h,'artifact '+n)
    for n,h in r['sources'].items():require(sha(out.parent/'repo'/n)==h,'source '+n)
    prior=json.loads((out/'PHYSICAL_REVIEW.json').read_text());require(prior['status']=='SHARED_ROUTED_INTERNAL_TIMING_PASS' and prior['result_sha256']==r['prior_result_sha256'],'physical binding')
    meta=json.loads((out/'suite/COMPARISON.json').read_text())
    require(meta['netlist_sha256']==meta['isolated_sha256']==r['netlist_sha256']==sha(out/'suite/route_isolated.v'),'netlist binding')
    require(meta['public_ports_only'] and not any(meta[n] for n in ('netlist_edited','sdf','force','cycle_equivalence','internal_forward_and_local_cpl_observed')),'comparison scope')
    text=(out/'suite/simulation.log').read_text()
    return dict(status='SHARED_ROUTED_PUBLIC_REVIEWED',result_sha256=sha(out/'result.json'),netlist_sha256=r['netlist_sha256'],
                **replay(text),negative_controls=controls(text),sdf=False,cycle_equivalence=False,board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
