"""Audit the domain fence using only public handshakes and completions."""
from collections import Counter
from check_ntrpu_peer_shared import require


def audit(text):
    require(text.count('DM ALL_PASS domains=32')==1 and 'Fatal:' not in text,'domain run failed')
    pending=[];fault=False;previous=0;stats=Counter();domains=set();held=None
    for line in text.splitlines():
        p=line.split()
        if p[:2]!=['DM','SAMPLE']:continue
        cycle,reset,v,ready,complete,address,sready,mvalid,busy,failed=map(int,p[2:])
        require(cycle==previous+1,'domain sample gap');previous=cycle
        allowed=len(pending)<2 and (not pending or pending[0]==address//(1<<29))
        require((sready,mvalid,busy,failed)==(int(ready and allowed),int(v and allowed),int(bool(pending)),int(fault)),
                'domain interface differs from outstanding command queue')
        if not reset:
            pending=[];fault=False;held=None;stats['reset_cycles']+=1;continue
        if held is not None:require(mvalid and address==held,'offered command withdrawn')
        held=address if mvalid and not ready else None
        stats['samples']+=1
        stats['peak_pending']=max(stats['peak_pending'],len(pending))
        stats['different_domain_stalls']+=bool(v and pending and pending[0]!=address//(1<<29))
        stats['output_stalls']+=bool(mvalid and not ready)
        accept=mvalid and ready
        if complete and not pending:
            fault=True;stats['spurious_completions']+=1
            require(not accept,'fault test must not create an untracked command')
        else:
            if complete:pending.pop(0);stats['completions']+=1
            if accept:pending.append(address//(1<<29));domains.add(address//(1<<29));stats['accepted']+=1
        stats['simultaneous_accept_complete']+=bool(accept and complete)
        require(len(pending)<=2 and len(set(pending))<=1,'mixed outstanding physical domains')
    require(not pending and not fault and domains==set(range(32)),'domain completion/reset/coverage')
    require(stats['peak_pending']==2 and stats['spurious_completions']==1 and
            stats['simultaneous_accept_complete']==32 and stats['different_domain_stalls']>=32*8,
            'missing domain boundary cases')
    return dict(counters=dict(stats),physical_domains=sorted(domains))


def negative_controls(text):
    rows=text.splitlines();changes=[]
    i=next(i for i,s in enumerate(rows) if s.startswith('DM SAMPLE ') and s.split()[3]=='1')
    changes.append(('drop_sample',i,None))
    for field,label in ((8,'ready'),(9,'valid'),(10,'pending'),(11,'fault')):
        p=rows[i].split();p[field]=str(1-int(p[field]));changes.append((label,i,' '.join(p)))
    killed=[]
    for label,i,new in changes:
        bad=rows[:i]+([] if new is None else [new])+rows[i+1:]
        try:audit('\n'.join(bad))
        except (ValueError,IndexError):killed.append(label)
        else:raise ValueError('accepted corrupt domain evidence: '+label)
    return killed
