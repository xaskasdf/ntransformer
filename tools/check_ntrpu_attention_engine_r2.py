"""Keep the full numerical audit and additionally verify speculative isolation."""
import re
import check_ntrpu_attention_engine_r1 as original

load_fixture = original.load_fixture


def prefetch_review(text):
    active=None;seen=[];completed=0;reads=0
    for line in text.splitlines():
        if line.startswith('AHE_START '):
            fields=dict(t.split('=') for t in line.split()[1:])
            active=(int(fields['query']),int(fields['start']));seen=[]
        elif line.startswith('HLS_PREFETCH '):
            m=re.fullmatch(r'HLS_PREFETCH cycle=(\d+) query=(\d+) start=(\d+) index=(\d+) forwarded=(\d+) data=([0-9a-f]{16})',line)
            assert m,'Prefetch schema'
            _,query,start,index,forwarded=map(int,m.groups()[:5])
            assert active==(query,start) and start==0 and forwarded==0 and int(m[6],16)==0
            assert index in (0,1,2,3,4,5,7) and index not in seen,'Initial header prefetch range/order'
            seen.append(index);reads+=1
        elif line.startswith('AHE_DONE '):
            fields=dict(t.split('=') for t in line.split()[1:])
            if int(fields['start'])==0 and fields['aborting']=='0':
                assert seen==[0,1,2,3,4,5,7],'Missing isolated initial header'
                completed+=1
            active=None;seen=[]
        elif line.startswith('AHE_QUERY_RESET '):active=None;seen=[]
    # 150 successful queries plus the completed result abandoned by reset.
    assert completed==151 and reads>=completed*7,('Prefetch coverage',completed,reads)
    return dict(completed_initial_tiles=completed,isolated_reads=reads,initial_state_value=0)


def review(text,fixture,seed=1):
    result=original.review(text,fixture,seed)
    result['initial_state_prefetch']=prefetch_review(text)
    return result


def controls(text,fixture,seed):
    rejected=original.controls(text,fixture,seed)
    changes={
        'prefetch_data':lambda s:re.sub(r'(HLS_PREFETCH[^\n]*data=)0000000000000000',r'\g<1>0000000000000001',s,count=1),
        'prefetch_forwarded':lambda s:re.sub(r'(HLS_PREFETCH[^\n]*forwarded=)0',r'\g<1>1',s,count=1),
        'prefetch_query':lambda s:re.sub(r'(HLS_PREFETCH[^\n]*query=)1 ',r'\g<1>2 ',s,count=1),
        'prefetch_index':lambda s:re.sub(r'(HLS_PREFETCH[^\n]*index=)0 ',r'\g<1>1 ',s,count=1),
        'prefetch_missing':lambda s:re.sub(r'^HLS_PREFETCH[^\n]*\n','',s,count=1,flags=re.M),
    }
    for name,change in changes.items():
        bad=change(text);assert bad!=text
        try:prefetch_review(bad)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
