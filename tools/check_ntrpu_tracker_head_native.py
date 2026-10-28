"""Independent native data/ownership audit plus observed cached-head FIFO accounting."""
import argparse
from collections import deque
import json
from pathlib import Path
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_tracker_head_native import verify,SUPPORT
from run_ntrpu_peer_read import sha


def head_audit(text,native):
    queues=[deque(),deque()];stats=[dict(reserves=0,releases=0,peak=0,reloads=0,bypasses=0) for _ in range(2)]
    summaries={};last=[-1,-1]
    for line in text.splitlines():
        p=line.split()
        if p[:1]!=['HN']:continue
        assert len(p)==10 and p[1] in ('EVENT','SUMMARY'),'head trace shape'
        ts,n,l,*v=map(int,p[2:]);assert n==native and l in (0,1) and ts>=last[l] and l not in summaries
        last[l]=ts;q=queues[l];s=stats[l]
        if p[1]=='EVENT':
            count,head,reserve,length,release=v
            assert count==len(q) and (not q or head==q[0]),'head FIFO identity'
            assert reserve in (0,1) and release in (0,1) and (reserve or release) and 0<=length<16
            assert not reserve or count<8
            s['reloads']+=int(bool(release and count>1))
            s['bypasses']+=int(bool(reserve and (count==0 or (release and count==1))))
            if release:assert q;q.popleft();s['releases']+=1
            if reserve:q.append(length);s['reserves']+=1
            s['peak']=max(s['peak'],len(q))
        else:
            reserves,releases,peak,checks,pending=v
            assert (reserves,releases,peak)==(s['reserves'],s['releases'],s['peak'])
            assert reserves>0 and releases>0 and checks>=releases and pending==len(q)==reserves-releases
            assert s['reloads']>0 and s['bypasses']>0 and 1<peak<=8
            summaries[l]=dict(**s,checks=checks,retained_descriptors=pending)
    assert set(summaries)=={0,1},'head summary coverage'
    return {str(k):v for k,v in summaries.items()}


def head_controls(text,native):
    lines=text.splitlines();event=next(s for s in lines if s.startswith('HN EVENT ') and int(s.split()[5])>0)
    summary=next(s for s in lines if s.startswith('HN SUMMARY '));p=event.split();p[6]=str(int(p[6])^1)
    trials={'missing_head_event':text.replace(event+'\n','',1),'wrong_head_length':text.replace(event,' '.join(p),1),
            'missing_head_summary':text.replace(summary+'\n','',1)}
    for name,changed in trials.items():
        try:head_audit(changed,native)
        except (AssertionError,ValueError,IndexError):continue
        raise ValueError('Accepted corrupted cached-head evidence: '+name)
    return list(trials)


def review(out):
    root=out.parent/'repo';profile=verify(root)
    r=json.loads((out/'result.json').read_text())
    assert r['sources']==r['sources_after'] and all(sha(root/n)==h for n,h in r['sources'].items())
    text=(root/SUPPORT).read_text(encoding='utf-8')
    text=once(text,".replace('peer_shared_native','peer_regions_full').replace('SHARED_NATIVE','REGIONS_NATIVE')",
              ".replace('peer_shared_native','tracker_head_native').replace('SHARED_NATIVE','HEAD_NATIVE')")
    ns=dict(__name__='head_native_regions');exec(compile(text,SUPPORT,'exec'),ns)
    report=ns['review'](out)
    report.update(exact_timing_closed_rtl=True,candidate_sha256=profile['candidate_sha256'],
                  native_snapshot_rebuilt=True if r['stage']!='preflight' else False,
                  native_post_route_simulated=False,integrated_timing_qualified=False,board_qualified=False)
    if r['stage']=='compile':
        script=(out/'compile.sh').read_text()
        assert 'tb_ntrpu_tracker_head_native.sv' in script and 'tb_ntrpu_tracker_head_native_behav' in script
        assert 'ntrpu_hbm_read_adapter.sv' in script and 'tb_ntrpu_peer_regions_full_behav' not in script
    else:
        text=(out/('evidence/simulate.log' if r['stage']=='traffic' else 'suite/simulation.log')).read_text(encoding='utf-8')
        native=int(r['stage']=='traffic')
        report['head_observation']=head_audit(text,native)
        report['head_negative_controls']=head_controls(text,native)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output),indent=2))
