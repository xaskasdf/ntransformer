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
            assert index==0 and index not in seen,'Initial header prefetch range/order'
            seen.append(index);reads+=1
        elif line.startswith('AHE_DONE '):
            fields=dict(t.split('=') for t in line.split()[1:])
            if int(fields['start'])==0 and fields['aborting']=='0':
                assert seen==[0],'Missing isolated initial header'
                completed+=1
            active=None;seen=[]
        elif line.startswith('AHE_QUERY_RESET '):active=None;seen=[]
    # 150 successful queries plus the completed result abandoned by reset.
    assert completed==151 and reads>=completed,('Prefetch coverage',completed,reads)
    return dict(completed_initial_tiles=completed,isolated_reads=reads,initial_state_value=0)


def kv_prefetch_review(text):
    active=None;planes=set();reads=0;completed=0
    for line in text.splitlines():
        if line.startswith('AHE_START '):
            f=dict(t.split('=') for t in line.split()[1:])
            active=(int(f['query']),int(f['start']),int(f['count']));planes=set()
        elif line.startswith('HLS_KV_PREFETCH '):
            m=re.fullmatch(r'HLS_KV_PREFETCH cycle=(\d+) query=(\d+) start=(\d+) plane=(\d+) index=(\d+) forwarded=(\d+) data=([0-9a-f]{8})',line)
            assert m,'KV sentinel schema'
            _,query,start,plane,index,forwarded=map(int,m.groups()[:6])
            assert active and active[:2]==(query,start) and 0<active[2]<64
            assert index==active[2]*64 and plane in (0,1) and plane not in planes
            assert forwarded==0 and int(m[7],16)==0,'KV sentinel isolation'
            planes.add(plane);reads+=1
        elif line.startswith('AHE_DONE '):
            f=dict(t.split('=') for t in line.split()[1:])
            if f['aborting']=='0':
                assert active
                assert planes==({0,1} if active[2]<64 else set()),'KV sentinel coverage'
                completed+=1
            active=None;planes=set()
        elif line.startswith('AHE_QUERY_RESET '):active=None;planes=set()
    assert completed==155 and reads>0
    return dict(completed_tiles=completed,isolated_reads=reads)


def review(text,fixture,seed=1):
    result=original.review(text,fixture,seed)
    result['initial_state_prefetch']=prefetch_review(text)
    result['kv_sentinel_prefetch']=kv_prefetch_review(text)
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
    changes={
        'kv_sentinel_data':lambda s:re.sub(r'(HLS_KV_PREFETCH[^\n]*data=)00000000',r'\g<1>00000001',s,count=1),
        'kv_sentinel_forwarded':lambda s:re.sub(r'(HLS_KV_PREFETCH[^\n]*forwarded=)0',r'\g<1>1',s,count=1),
        'kv_sentinel_index':lambda s:re.sub(r'(HLS_KV_PREFETCH[^\n]*index=)64 ',r'\g<1>65 ',s,count=1),
        'kv_sentinel_plane':lambda s:re.sub(r'(HLS_KV_PREFETCH[^\n]*plane=)0 ',r'\g<1>1 ',s,count=1),
        'kv_sentinel_missing':lambda s:re.sub(r'^HLS_KV_PREFETCH[^\n]*\n','',s,count=1,flags=re.M),
    }
    for name,change in changes.items():
        bad=change(text);assert bad!=text
        try:kv_prefetch_review(bad)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
