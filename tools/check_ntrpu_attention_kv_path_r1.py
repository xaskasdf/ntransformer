"""Reconstruct physical writes, leased reads and private RAM consumption."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def query_plan():
    rows=[(1,0,0,0,10),(1,0,0,0,10),(1,0,32,0,11),(1,0,0,0,12)]
    rows += [(1,15 if h%2 else 0,h,0 if h%2 else 2,0) for h in range(32)]
    rows += [(1,0,0,0,13),(1,0,0,2,1),(1,0,0,2,0)]
    rows += [(1,0,0,2,m) for m in range(2,10)]
    rows += [(1,0,31,2,0),(1,2,0,0,0),(1,3,0,0,24),(1,3,0,0,0),(2,15,31,64,0)]
    rows += [(2,15,31,64,m) for m in range(14,19)]+[(2,15,31,64,0)]
    for phase in range(4):
        rows += [(2+phase,15,31,0,20+phase),(3+phase,15,31,0,10),(3+phase,15,31,0,0)]
    assert len(rows)==70
    return rows


def pattern(generation,layer,position,index):
    return sum((0x3f000000|((generation*100003+layer*4099+position*97+index*23+lane*3)&0x7fffff))<<(32*lane)
               for lane in range(16))


def review(text,seed,watchdog):
    assert not re.search(r'(?im)^\s*(?:fatal:|error:|KVP_FAIL)',text),'simulator error'
    events=[];hexkeys={'seq','sequence','epoch','lease','address','base','data','k','v'}
    for line in text.splitlines():
        if not line.startswith('KVP_'):continue
        tag,*parts=line.split();pairs=[p.split('=') for p in parts]
        assert all(len(p)==2 and re.fullmatch('[0-9a-f]+',p[1]) for p in pairs),line
        assert len(dict(pairs))==len(pairs),'duplicate field'
        events.append((tag,{k:int(v,16 if k in hexkeys else 10) for k,v in pairs}))
    assert events[0]==('KVP_START',dict(seed=seed,watchdog=watchdog))
    assert events[-1][0]=='KVP_PASS';summary=events[-1][1]
    fixed=dict(seed=seed,watchdog=watchdog,binds=6,appends=78,queries=70,resets=5,consumers=57,
               reads=33682,completions=66,aw=1319,w=4929,b=1319,ar=216,r=4410)
    assert {k:summary[k] for k in fixed}==fixed and summary['overlap']>0
    counts=Counter(tag for tag,_ in events)
    for tag,key in [('BIND','binds'),('BIND_DONE','binds'),('APPEND','appends'),('APPEND_DONE','appends'),
                    ('QUERY','queries'),('RESET','resets'),('COMPUTE','consumers'),('RAM','reads'),
                    ('QUERY_DONE','completions'),('AW','aw'),('W','w'),('B','b'),('AR','ar'),('R','r')]:
        assert counts['KVP_'+tag]==fixed[key],tag
    assert counts['KVP_QUERY_RESET']==4 and counts['KVP_PASS']==counts['KVP_START']==1
    mem={};prefix=[0]*16;cache=None;append={};queries={};binding=None;aw=None;w=[];ar=None
    lease=None;lease_request=None;current=None;loaded=[];last_read=None;consumer=None;next_start=0
    metrics=[];published=0;overlap_pairs=set();aw_applied=False;observed_reads=0
    epoch=0;last_lease=0;last_completed_lease=0;generation=0
    supported=set('START RESET BIND BIND_DONE APPEND APPEND_DONE AW W B AR R QUERY LEASE LEASE_DONE TILE READ_DONE COMPUTE RAM FINISH CANCEL QUERY_DONE QUERY_RESET PASS'.split())
    for fulltag,r in events:
        tag=fulltag.removeprefix('KVP_');assert tag in supported,tag
        if tag in ('START','PASS'):continue
        if tag=='RESET':
            prefix=[0]*16;cache=None;binding=None;lease=None;lease_request=None;consumer=None;current=None
            loaded=[];last_read=None;next_start=0;aw=None;w=[];ar=None;aw_applied=False;last_lease=0
        elif tag=='BIND':
            generation+=1
            assert r['generation']==generation and not lease
            cap,offset=(4,4032) if generation==1 else ((66,64) if generation==2 else (1,4032))
            assert (r['capacity'],r['base'],r['sequence'],r['epoch'])==(cap,0xa000000000+offset,0x1234567800000000+generation,0xfedcba9800000000+generation)
            binding=r;epoch=r['epoch'];prefix=[0]*16
        elif tag=='BIND_DONE':
            assert binding and r['id']==binding['id'] and r['status']==0
            cache=binding;binding=None
        elif tag=='APPEND':
            assert cache and r['generation']==generation and r['prefix']==prefix[r['layer']]
            assert r['position']==prefix[r['layer']] and r['position']<cache['capacity'] and r['mode'] in (0,1)
            append[r['id']]=dict(r,b_responses=0,physical_words=0)
        elif tag=='APPEND_DONE':
            a=append[r['id']];success=a['mode']==0
            assert r['status']==(0 if success else 3)
            if success:
                assert a['physical_words']==64 and r['bytes']==4096
                prefix[a['layer']]+=1;published+=4096
            else:assert a['b_responses']==1 and a['physical_words']==1 and r['bytes']==64
            assert r['committed']==prefix[a['layer']]
        elif tag=='AW':
            assert aw is None and 1<=r['beats']<=4 and r['address']%64==0 and r['address']%4096+r['beats']*64<=4096
            aw=r;aw_applied=False
        elif tag=='W':
            assert len(w)<4 and (not w or not w[-1]['last']);w.append(r)
        elif tag=='B':
            assert aw and aw_applied and len(w)==aw['beats'] and r['id']==aw['id']
            a=append[r['id']];assert r['ordinal']==a['b_responses'] and r['resp']==(2 if a['mode']==1 else 0)
            a['b_responses']+=1;aw=None;w=[];aw_applied=False
        elif tag=='AR':
            assert ar is None and lease and current and r['id']==current['query']
            assert r['address']%64==0 and 1<=r['beats']<=256 and r['address']%4096+r['beats']*64<=4096
            ar=dict(r,seen=0)
        elif tag=='R':
            assert ar and r['id']==ar['id'] and r['address']==ar['address']+64*ar['seen']
            assert r['last']==int(ar['seen']+1==ar['beats']) and r['resp'] in (0,2)
            assert r['address'] in mem and r['data']==mem[r['address']],'physical RAM readback'
            ar['seen']+=1
            if ar['seen']==ar['beats']:ar=None
        elif tag=='QUERY':
            assert current is None and consumer is None and lease is None
            q=len(queries)+1;assert r['query']==q
            assert tuple(r[k] for k in ('generation','layer','head','position','mode'))==query_plan()[q-1]
            assert r['generation']==generation
            wanted_seq=cache['sequence']^(0x100000000 if r['mode']==12 else 0)
            wanted_lease=last_completed_lease if r['mode']==13 else 0x1122334400000000+q-1
            assert (r['seq'],r['epoch'],r['lease'])==(wanted_seq,cache['epoch'],wanted_lease)
            current=dict(r,reads=0,compute=0,finished=0,cancelled=False,tiles_ok=0,tokens_ok=0)
            queries[q]=current;next_start=0;loaded=[];last_read=None
        elif tag=='LEASE':
            assert current and r['query']==current['query'] and lease_request is None
            assert all(r[k]==current[k] for k in ('layer','head','position','seq','epoch','lease'))
            if r['release']:
                assert lease and consumer is None and ar is None and last_read is not None,'premature lease release'
            else:assert lease is None and not current['cancelled']
            lease_request=r
        elif tag=='LEASE_DONE':
            assert lease_request and current and r['query']==current['query'] and r['release']==lease_request['release']
            if r['release']:
                assert r['status']==0 and r['active']==0 and r['prefix']==lease['prefix'];lease=None
            else:
                expected=2 if current['mode'] in (12,13) else (1 if prefix[current['layer']]<=current['position'] else 0)
                assert r['status']==expected and r['active']==int(expected==0)
                assert r['prefix']==(prefix[current['layer']] if expected==0 else 0)
                if expected==0:
                    assert current['lease']>last_lease;last_lease=current['lease'];lease=dict(current,prefix=r['prefix'])
            lease_request=None
        elif tag=='TILE':
            assert lease and current and not consumer and r['query']==current['query']
            assert all(r[k]==current[k] for k in ('seq','epoch','lease'))
            count=min(64,current['position']+1-next_start);ordinal=len(loaded);plane,index=divmod(ordinal,count*4)
            assert plane<2 and (r['plane'],r['index'],r['last'])==(plane,index,int(ordinal==count*8-1))
            position=next_start+index//4;srcindex=plane*32+(current['head']//4)*4+index%4
            address=cache['base']+(((current['layer']*2+plane)*8+current['head']//4)*cache['capacity']+position)*256+(index%4)*64
            assert r['data']==mem[address]==pattern(generation,current['layer'],position,srcindex),'tile mapping'
            loaded.append(r['data'])
        elif tag=='READ_DONE':
            assert lease and current and ar is None and r['query']==current['query']
            assert r['beats']==len(loaded) and r['bytes']==len(loaded)*64
            # RAM starts its watchdog before issuing the cache read. Its timeout
            # cancels the reader (7); the public query retains timeout status 8.
            mode=current['mode'];wanted={1:3,2:7,9:7}.get(mode,0)
            if mode==15 and next_start==64:wanted=3
            assert r['status']==wanted,('load status',current['query'],next_start,r)
            count=min(64,current['position']+1-next_start)
            if wanted==0:assert len(loaded)==count*8
            last_read=dict(r,start=next_start,tokens=count,data=loaded.copy());loaded=[];current['reads']+=1
        elif tag=='COMPUTE':
            assert lease and current and consumer is None and last_read and last_read['status']==0
            assert (r['start'],r['tokens'])==(next_start,last_read['tokens']) and r['query']==current['query']
            assert r['layer']==current['layer'] and r['head']==current['head'] and r['kv']==current['head']//4 and r['lease']==current['lease']
            consumer=dict(r,read_count=0,data=last_read['data']);current['compute']+=1
        elif tag=='RAM':
            assert consumer and lease and current and r['query']==current['query'] and r['start']==consumer['start']
            index=consumer['read_count'];words=consumer['tokens']*64
            assert r['ka']==index and r['va']==words-1-index
            for plane,key,address in [(0,'k',r['ka']),(1,'v',r['va'])]:
                beat=consumer['data'][plane*(words//16)+address//16]
                assert r[key]==((beat>>((address%16)*32))&0xffffffff),'consumer RAM value'
            consumer['read_count']+=1;observed_reads+=1
        elif tag=='FINISH':
            assert consumer and lease and r['query']==current['query'] and r['start']==consumer['start']
            mode=current['mode'];short=mode in (5,6,7,8) or (mode in (14,18) and next_start==64)
            wanted_count=0 if mode==23 else (3 if short else consumer['tokens']*64)
            assert consumer['read_count']==wanted_count,('missing consumer read',current['query'])
            assert r['status']==(5 if mode==7 or (mode==14 and next_start==64) else 0)
            assert r['abort']==int(mode in (5,6,8) or (mode==18 and next_start==64))
            if r['status']==0 and not r['abort']:
                next_start+=consumer['tokens'];current['tokens_ok']=next_start;current['tiles_ok']+=1
            current['finished']+=1;consumer=None
        elif tag=='CANCEL':
            assert current and r['query']==current['query'] and not current['cancelled']
            assert r['revoke']==int(current['mode'] in (4,6,17));current['cancelled']=True
        elif tag=='QUERY_DONE':
            assert current and r['query']==current['query'] and lease is None and consumer is None and ar is None
            mode=current['mode'];wanted={1:3,2:7,3:7,4:2,5:7,6:2,7:5,8:9,9:8,10:1,11:1,12:2,13:2,14:5,15:3,16:7,17:2,18:7,24:7}.get(mode,0)
            assert r['status']==wanted and r['tokens']==current['tokens_ok'] and r['tiles']==current['tiles_ok']
            if wanted==0:assert r['tokens']==current['position']+1 and r['tiles']==(current['position']//64+1)
            if 14<=mode<=18:assert (r['tokens'],r['tiles'])==(64,1)
            if mode in (2,3,4,5,6,16,17,18,24):assert current['cancelled']
            assert r['elapsed']==r['cycle']-current['cycle'] and r['elapsed']>0
            metrics.append(dict(query=r['query'],mode=mode,status=wanted,tokens=r['tokens'],cycles=r['elapsed']))
            last_completed_lease=current['lease'];current=None
        elif tag=='QUERY_RESET':
            assert current and r['query']==current['query'] and r['phase']==current['mode']-20
            assert (consumer is not None)==(r['phase']==2)
            if r['phase']==3:assert lease is None
        # AXI W may arrive before AW; apply physical bytes once both are complete.
        if aw and not aw_applied and len(w)==aw['beats'] and w[-1]['last']:
            a=append[aw['id']]
            for index,beat in enumerate(w):
                assert beat['id']==aw['id'] and beat['last']==int(index+1==len(w))
                ordinal=a['physical_words'];plane,rest=divmod(ordinal,32);head,word_index=divmod(rest,4)
                address=cache['base']+(((a['layer']*2+plane)*8+head)*cache['capacity']+a['position'])*256+word_index*64
                assert aw['address']+index*64==address
                assert beat['data']==pattern(a['generation'],a['layer'],a['position'],ordinal)
                mem[address]=beat['data'];a['physical_words']+=1
            aw_applied=True
        # Either channel may become active first. Count actual simultaneous
        # outstanding burst pairs, including AW accepted after an earlier AR.
        if aw and ar:overlap_pairs.add((aw['id'],aw['address'],ar['id'],ar['address']))
    assert current is None and lease is None and consumer is None and aw is None and ar is None
    assert published==77*4096 and observed_reads==33682 and overlap_pairs
    return dict(summary=summary,publishable_write_bytes=published,ram_words_compared=observed_reads*2,
                queries=metrics,observed_overlap_burst_pairs=len(overlap_pairs),
                hls_binding_qualified=False,physical_timing_qualified=False)


def controls(text,seed,watchdog):
    changes=[('physical_write','W','data'),('physical_read','R','data'),('write_address','AW','address'),
             ('read_address','AR','address'),('lease_identity','LEASE','lease'),('lease_prefix','LEASE_DONE','prefix'),
             ('tile_identity','TILE','lease'),('tile_data','TILE','data'),('load_status','READ_DONE','status'),
             ('load_bytes','READ_DONE','bytes'),('compute_head','COMPUTE','kv'),('ram_key','RAM','k'),
             ('ram_value','RAM','v'),('finish_status','FINISH','status'),('query_status','QUERY_DONE','status'),
             ('query_tokens','QUERY_DONE','tokens'),('reset_phase','QUERY_RESET','phase')]
    rejected=[];lines=text.splitlines()
    for label,tag,key in changes+[(t.lower()+'_missing',t,None) for t in ('QUERY','TILE','RAM','FINISH','LEASE','READ_DONE','QUERY_DONE','PASS')]:
        bad=lines.copy();i=next(i for i,line in enumerate(bad) if line.startswith('KVP_'+tag+' '))
        if key is None:del bad[i]
        else:
            base=16 if key in ('data','address','lease','k','v') else 10
            bad[i]=re.sub(r'\b'+key+'=([0-9a-f]+)',lambda m:key+'='+format(int(m[1],base)^1,'x' if base==16 else 'd'),bad[i],count=1)
        try:review('\n'.join(bad),seed,watchdog)
        except (AssertionError,KeyError,IndexError,ValueError):rejected.append(label)
        else:raise AssertionError('Accepted altered evidence: '+label)
    return rejected


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--watchdog',type=int,default=32);a=p.parse_args()
    text=a.log.read_text();r=review(text,a.seed,a.watchdog);r['controls']=controls(text,a.seed,a.watchdog)
    print(json.dumps(r,indent=2))
