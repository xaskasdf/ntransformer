"""Independent conservation/order/occupancy checks at the registered return boundary."""
import argparse
import json
from collections import defaultdict,deque
from pathlib import Path
from check_ntrpu_read_frames import sha,replay,controls


def boundary(lines):
    groups=defaultdict(list)
    for line in lines:
        p=line.split()
        if len(p)>1 and p[0]=='RC':
            assert p[1] in ('STATE','IN','WORD')
            groups[tuple(map(int,p[2:5]))].append((p[1],p[5:]))
    assert groups
    queues=defaultdict(deque);ptrs={};counts=dict(samples=0,push=0,pop=0,held=0,full=0,peak=0)
    seen=defaultdict(set)
    for (ts,epoch,lane),events in sorted(groups.items()):
        assert lane in (0,1);seen[ts,epoch].add(lane)
        state=[p for kind,p in events if kind=='STATE'];assert len(state)==1
        count,rp,wp,sv,sr,mv,mr,pending=map(int,state[0]);q=queues[epoch,lane]
        assert 0<=count<=2 and count==len(q)
        assert (rp,wp)==ptrs.get((epoch,lane),(0,0))
        assert sr==int(count<2) and mv==int(count>0)
        assert pending==1 or count==0
        incoming=[int(p[0],16) for kind,p in events if kind=='IN']
        words=[int(p[0],16) for kind,p in events if kind=='WORD']
        assert len(incoming)==int(bool(sv and sr)) and len(words)==mv
        if mv:
            assert words[0]==q[0]
            if mr:q.popleft();rp^=1;counts['pop']+=1
            else:counts['held']+=1
        if incoming:
            assert 0<=incoming[0]<2**519
            q.append(incoming[0]);wp^=1;counts['push']+=1
        ptrs[epoch,lane]=(rp,wp)
        counts['samples']+=1;counts['peak']=max(counts['peak'],count)
        counts['full']+=int(count==2)
    assert all(lanes=={0,1} for lanes in seen.values())
    assert counts['samples']>100 and counts['push']>0 and counts['pop']>0
    counts['remaining_at_epoch_end']=sum(map(len,queues.values()))
    assert counts['push']==counts['pop']+counts['remaining_at_epoch_end']
    return counts


def negative_controls(lines):
    trials={
        'missing_cut_states':[s for s in lines if not s.startswith('RC STATE ')],
        'missing_cut_input':[s for s in lines if not s.startswith('RC IN ')],
        'missing_cut_output':[s for s in lines if not s.startswith('RC WORD ')]}
    for label,prefix,index,value in [('bad_cut_count','RC STATE ',5,'3'),
                                     ('bad_cut_payload','RC WORD ',-1,'0'),
                                     ('bad_cut_pending','RC STATE ',12,'0')]:
        corrupted=list(lines)
        for i,line in enumerate(corrupted):
            if line.startswith(prefix):
                p=line.split()
                if label=='bad_cut_pending' and p[5]=='0':continue
                if label=='bad_cut_payload' and int(p[-1],16)==0:continue
                p[index]=value;corrupted[i]=' '.join(p);break
        else:raise AssertionError(label+' did not alter evidence')
        trials[label]=corrupted
    for label,corrupted in trials.items():
        try:boundary(corrupted)
        except (AssertionError,KeyError,ValueError,IndexError):continue
        raise AssertionError(label+' was accepted')
    return list(trials)


def review(out,native=0,require_full=False):
    r=json.loads((out/'result.json').read_text())
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out']
    assert r['sources']==r['sources_after']
    lines=(out/'suite/simulation.log').read_text().splitlines()
    result=replay(lines,native=native)
    result.update(boundary=boundary(lines),boundary_negative_controls=negative_controls(lines),
                  negative_controls=controls(lines,native=native),result_sha256=sha(out/'result.json'),
                  status='READ_CUT_FUNCTIONAL_REVIEWED',native_profile=native,
                  checker_sha256=sha(Path(__file__)))
    if require_full:assert result['boundary']['full']>0 and result['boundary']['held']>0
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    ap.add_argument('--native',type=int,default=0);ap.add_argument('--require-full',action='store_true')
    a=ap.parse_args();print(json.dumps(review(a.output,a.native,a.require_full),indent=2))
