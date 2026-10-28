"""Retain all arithmetic/AXI audits and require exact continuation isolation."""
import re
import check_ntrpu_attention_engine_r3 as previous

load_fixture=previous.load_fixture


def state_sentinel_review(text):
    active=None;seen=0;reads=0;completed=0
    for line in text.splitlines():
        if line.startswith('AHE_START '):
            fields=dict(t.split('=') for t in line.split()[1:])
            active=(int(fields['query']),int(fields['start']));seen=0
        elif line.startswith('HLS_STATE_SENTINEL '):
            m=re.fullmatch(r'HLS_STATE_SENTINEL cycle=(\d+) query=(\d+) start=(\d+) index=(\d+) forwarded=(\d+) data=([0-9a-f]{16})',line)
            assert m,'Continuation sentinel schema'
            _,query,start,index,forwarded=map(int,m.groups()[:5])
            assert active==(query,start) and start>0 and index==104 and seen==0,'Continuation sentinel identity/range'
            assert forwarded==0 and int(m[6],16)==0,'Continuation sentinel isolation'
            seen+=1;reads+=1
        elif line.startswith('AHE_DONE '):
            f=dict(t.split('=') for t in line.split()[1:])
            assert active==(int(f['query']),int(f['start']))
            if f['aborting']=='0':
                assert seen==int(active[1]>0),'Continuation sentinel coverage'
                completed+=int(active[1]>0)
            active=None;seen=0
        elif line.startswith('AHE_QUERY_RESET '):active=None;seen=0
    assert completed==reads==4,('Continuation sentinel totals',completed,reads)
    return dict(completed_continuation_tiles=completed,isolated_reads=reads,index=104)


def review(text,fixture,seed=1):
    result=previous.review(text,fixture,seed)
    result['continuation_state_prefetch']=state_sentinel_review(text)
    return result


def controls(text,fixture,seed):
    rejected=previous.controls(text,fixture,seed)
    changes={
        'state_sentinel_data':lambda s:re.sub(r'(HLS_STATE_SENTINEL[^\n]*data=)0000000000000000',r'\g<1>0000000000000001',s,count=1),
        'state_sentinel_forwarded':lambda s:re.sub(r'(HLS_STATE_SENTINEL[^\n]*forwarded=)0',r'\g<1>1',s,count=1),
        'state_sentinel_index':lambda s:re.sub(r'(HLS_STATE_SENTINEL[^\n]*index=)104 ',r'\g<1>105 ',s,count=1),
        'state_sentinel_start':lambda s:re.sub(r'(HLS_STATE_SENTINEL[^\n]*start=)64 ',r'\g<1>0 ',s,count=1),
        'state_sentinel_missing':lambda s:re.sub(r'^HLS_STATE_SENTINEL[^\n]*\n','',s,count=1,flags=re.M),
    }
    for name,change in changes.items():
        bad=change(text);assert bad!=text
        try:state_sentinel_review(bad)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
