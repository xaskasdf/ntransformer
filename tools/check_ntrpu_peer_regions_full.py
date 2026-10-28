"""Independent mapped-address, ownership, preload and domain-boundary audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from check_ntrpu_peer_shared import lease,request,bad_config,require
from check_ntrpu_peer_regions_unit import region_map,map_bad,physical,augment


def timeline(text):
    maps={tuple(p[2:4]):region_map(p[4]) for s in text.splitlines()
          if (p:=s.split())[:2]==['RG','MAP']}
    configs=[];previous=None
    for line in text.splitlines():
        p=line.split()
        if p[:2]!=['SG','CFG']:continue
        c=lease(p[4]);c['map']=maps[tuple(p[2:4])]
        if not bad_config(c,previous) and not map_bad(c):
            configs.append((int(p[2]),c));previous=c
    return configs


def active_at(configs,ts):
    return next(c for t,c in reversed(configs) if t<=ts)


def memory_authorized(configs,ts,address,bytes):
    c=active_at(configs,ts);m=c['map']
    return any(m['valid']&(1<<i) and r['physical']<=address and
               address+bytes<=r['physical']+r['size'] for i,r in enumerate(m['regions']))


def ownership_module(root):
    text=(root/'tools/check_ntrpu_peer_shared.py').read_text(encoding='utf-8')
    text=once(text,'cases = 6 if full else 15','cases = 4 if full else 15')
    text=once(text,"set(range(1, cases + 1)), 'epoch coverage'","{1}, 'epoch coverage'")
    text=once(text,'return (not s[0]','return (map_bad(c) or not s[0]')
    text=once(text,"    if any(r['dst']","    if physical(c,r['src'],r['size']) is None:\n        return 8\n    if any(r['dst']")
    text=once(text,"new = lease(sg['CFG'][0][0]); cfg_error = bad_config(new, c)",
              "new = lease(sg['CFG'][0][0]); new['map']=region_map(sg['CFG'][0][1]); cfg_error = bad_config(new, c)")
    text=once(text,"item['cfg']['src'] + item['r']['src']", "physical(item['cfg'],item['r']['src'],item['r']['size'])")
    assert text.count("item['cfg']['src'] + r['src']")==2
    text=text.replace("item['cfg']['src'] + r['src']","physical(item['cfg'],r['src'],r['size'])")
    text=once(text,'(a ^ (a >> 8) ^ (a >> 16) ^ 0x5b)',
              '(a ^ (a >> 8) ^ (a >> 16) ^ (a >> 24) ^ (a >> 32) ^ 0x5b)')
    text=once(text,"stats['successful_retire'] == 27 and stats['successful_payload_bytes'] == 46848",
              "stats['successful_retire'] == 16 and stats['successful_payload_bytes'] == 35200")
    ns=dict(__name__='mapped_ownership',map_bad=map_bad,physical=physical,region_map=region_map)
    exec(compile(text,'frozen_mapped_ownership','exec'),ns);return ns


def mapped_audit(text):
    configs=timeline(text);forwards={};observed={};commands={};completions=set();domain_rows=[];clock_samples=set()
    require([c['map']['generation'] for _,c in configs]==[1,2],'two map generations')
    for line in text.splitlines():
        p=line.split()
        if len(p)<5:continue
        if p[:2] in (['SG','FWD'],['RG','FWD'],['SF','CMD'],['SF','LOCAL']):
            key=tuple(map(int,p[2:5]))
            if p[:2]==['SG','FWD']:
                r=request(p[5]);require(key not in forwards,'duplicate logical forward')
                address=physical(active_at(configs,key[0]),r['src'],r['size'])
                require(address is not None,'forward in a map gap');forwards[key]=(int(p[5],16),address)
            elif p[:2]==['RG','FWD']:
                require(key not in observed,'duplicate physical forward');observed[key]=(int(p[5],16),int(p[6]))
            elif p[:2]==['SF','CMD']:
                require(key not in commands,'duplicate command');commands[key]=int(p[6])
            else:
                require(key not in completions,'duplicate local completion');completions.add(key)
        elif p[:2]==['RG','DOMAIN']:domain_rows.append(list(map(int,p[2:])))
        elif p[:2]==['SG','SAMPLE']:clock_samples.add((int(p[2]),int(p[3])))
    require(forwards==observed and len(forwards)==18,'forward/address binding')
    require(len(domain_rows)==2*len(clock_samples) and {(p[0],p[1],p[2]) for p in domain_rows}==
            {(ts,e,l) for ts,e in clock_samples for l in range(2)},'missing/duplicate domain clock observation')
    require([(p[0],p[1],p[2]) for p in domain_rows]==sorted((p[0],p[1],p[2]) for p in domain_rows),
            'domain observations not in clock/lane order')
    queue=[[],[]];held=[None,None];previous=[None,None];stats=Counter();different_stacks=False
    for ts,epoch,l,v,ready,address,sready,mvalid,complete,busy,fault in domain_rows:
        key=ts,epoch,l;domain=address//(1<<29)
        if l==0 and all(queue) and queue[0][0]//16!=queue[1][0]//16:different_stacks=True
        require(previous[l] is None or ts==previous[l]+5000,'domain clock evidence gap');previous[l]=ts
        allowed=len(queue[l])<2 and (not queue[l] or queue[l][0]==domain)
        require((sready,mvalid,busy,fault)==(int(ready and allowed),int(v and allowed),int(bool(queue[l])),0),
                'domain boundary violated')
        if held[l] is not None:require(mvalid and address==held[l],'domain offered command changed')
        held[l]=address if mvalid and not ready else None
        accepted=mvalid and ready
        stats['cross_domain_wait_cycles']+=bool(v and not allowed and queue[l] and queue[l][0]!=domain)
        require(bool(key in commands)==bool(accepted) and (not accepted or commands.pop(key)==address),'domain/command conservation')
        require(bool(key in completions)==bool(complete),'domain/completion conservation')
        if complete:
            completions.remove(key);require(queue[l],'domain completion without command');queue[l].pop(0)
        if accepted:queue[l].append(domain);stats['accepted']+=1
        stats['samples']+=1;stats['peak_per_port']=max(stats['peak_per_port'],len(queue[l]))
    require(not commands and not completions and different_stacks and stats['cross_domain_wait_cycles']>0 and
            stats['accepted']==18 and stats['peak_per_port']==2,'domain coverage/conservation')
    # Check native AR-to-RLAST overlap too: buffered payload awaiting DMA alone
    # is not evidence of concurrent outstanding reads in the memory fabric.
    from collections import defaultdict,deque
    native=defaultdict(list)
    for line in text.splitlines():
        p=line.split()
        if p[:2] in (['SN','AR'],['SN','NR']) and p[4]=='0':native[int(p[2])].append(p)
    inflight=[deque(),deque()];native_overlap=False
    for ts,rows in sorted(native.items()):
        for p in rows:
            l=int(p[5])
            if p[1]=='AR':inflight[l].append([int(p[7])//(1<<33),int(p[8])])
            else:
                require(inflight[l] and int(p[9])==int(inflight[l][0][1]==1),'native inflight boundary')
                inflight[l][0][1]-=1
                if not inflight[l][0][1]:inflight[l].popleft()
        if all(inflight) and inflight[0][0][0]!=inflight[1][0][0]:native_overlap=True
    require(native_overlap and not any(inflight),'missing cross-stack native overlap/drain')
    return dict(forwards=len(forwards),generations=[1,2],different_stacks_simultaneously_active=different_stacks,
                native_reads_concurrently_outstanding_in_different_stacks=native_overlap,
                domain=dict(stats),final_local_commands=[len(q) for q in queue])


def checker(root):
    text=(root/'tools/check_ntrpu_peer_shared_native.py').read_text(encoding='utf-8')
    text=text.replace('peer_shared_native','peer_regions_full').replace('SHARED_NATIVE','REGIONS_NATIVE')
    text=once(text,'    memory, writes, completed, windows = {}, {}, set(), {}',
              '    configs=timeline(text)\n    memory, writes, completed, windows = {}, {}, set(), {}')
    text=once(text,'publications and address // 8589934592 == publications[-1] // 8589934592',
              'publications and memory_authorized(configs,ts,address,beats*32)')
    text=once(text,'publications == [0,8589934592]','publications == [1,2]')
    text=once(text,'            publications.append(int(p[3]))',
              "            need(int(p[3])==active_at(configs,int(p[2]))['map']['generation'],'publication generation')\n            publications.append(int(p[3]))")
    text=once(text,'from check_ntrpu_shared_native_ledger import audit, negative_controls',
              "model=ownership_module(root)\n        audit=lambda text,full:model['audit'](augment(text),full)\n        negative_controls=lambda text,full:model['negative_controls'](augment(text),full)")
    text=once(text,"'ntrpu_peer_read_shared.sv' in script","'ntrpu_peer_regions.sv' in script")
    ns=dict(__name__='mapped_native',root=root,timeline=timeline,active_at=active_at,
            memory_authorized=memory_authorized,ownership_module=ownership_module,augment=augment)
    exec(compile(text,'frozen_mapped_native','exec'),ns);return ns


def review(out):
    root=out.parent/'repo';ns=checker(root);report=ns['review'](out)
    r=json.loads((out/'result.json').read_text())
    if r['stage']=='compile':return report
    text=(out/('evidence/simulate.log' if r['stage']=='traffic' else 'suite/simulation.log')).read_text(encoding='utf-8')
    report['regions']=mapped_audit(text);negative=[]
    for token in ('RG MAP ','RG FWD ','RG DOMAIN '):
        line=next(s for s in text.splitlines() if s.startswith(token))
        try:mapped_audit(text.replace(line+'\n','',1))
        except (ValueError,KeyError,IndexError,StopIteration):negative.append('drop_'+token.split()[1].lower())
        else:raise ValueError('accepted missing map evidence: '+token)
    line=next(s for s in text.splitlines() if s.startswith('RG FWD ') and int(s.split()[-1])>=1<<33)
    p=line.split();p[-1]=str(int(p[-1])&((1<<33)-1))
    try:mapped_audit(text.replace(line,' '.join(p),1))
    except ValueError:negative.append('truncate_mapped_stack_bit')
    else:raise ValueError('accepted mapped stack truncation')
    report['region_negative_controls']=negative
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output),indent=2))
