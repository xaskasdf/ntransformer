"""Review causal memory, all attention state, and the complete real layer result."""
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import re
import struct
import ntrpu_llama32_layer_fixture_r1 as fixture
import check_ntrpu_attention_engine_r1 as attention
import check_ntrpu_attention_axi_reference_r1 as oracle

SEQ=0x1234567800000001;EPOCH=0xfedcba9800000001
SCHEMA={
    'ADMIT':'cycle sequence epoch layer position token','AR':'cycle port index region offset beats',
    'AW':'cycle port index region offset beats','W':'cycle port index last data','B':'cycle port index region offset beats',
    'R':'cycle port region offset data','ROPE_START':'cycle','ROPE_DONE':'cycle','ROPE':'cycle index data',
    'HEAD':'cycle head','STATE':'cycle head index data','ATTENTION':'cycle head index data',
    'SOURCE':'cycle index data','HLS_INPUT':'cycle index data',
    'START':'cycle stage','DONE':'cycle stage','ACK':'cycle stage','PROGRESS':'cycle stage writes',
    'PORT':'port ar r aw w b ar_stalls w_stalls','READ':'port region beats hash',
    'COMPLETE':'cycle status sequence epoch address bytes prefix','PUBLIC_ACK':'cycle',
    'PASS':'seed jobs cycles writes heads source_words',
}

def need(test,message):
    if not test:raise ValueError(message)

def parse(text):
    need(not re.search(r'(?im)(\bfatal\b|^ERROR:|LREAL_FAIL)',text),'Simulator diagnostic')
    rows=[]
    for line in text.splitlines():
        if not line.startswith('LREAL_'):continue
        tag,*words=line[6:].split();need(tag in SCHEMA,'Unknown event')
        pairs=[w.split('=') for w in words];need(all(len(p)==2 for p in pairs),'Malformed event')
        need([p[0] for p in pairs]==SCHEMA[tag].split(),'Event schema')
        values={k:int(v,16 if k in ('sequence','epoch','address','data','hash') else 10) for k,v in pairs}
        need(all(v>=0 for v in values.values()),'Negative event field');rows.append((tag,values))
    need(rows and rows[-1][0]=='PASS','Missing final result');return rows

def load(folder):
    folder=Path(folder);m,data,plans=fixture.load(folder)
    refs=oracle.fixture(folder/'attention-reference')
    rope=[int(s,16) for s in (folder/'rope.hex').read_text().split()]
    outputs=[int(s,16) for s in (folder/'attention.hex').read_text().split()]
    return dict(manifest=m,data=data,plans=plans,refs=refs,rope=rope,attention=outputs)

def review_rows(rows,ref,seed=1):
    m,data,plans=ref['manifest'],ref['data'],ref['plans'];writable={r['index'] for r in m['regions'] if r['writable']}
    produced={};source={};consumed=set();by=defaultdict(list);ar_cursor=[0]*5;aw=[None]*5;pending_r=[None]*5;wq=[[] for _ in range(5)]
    counts=[dict(ar=0,r=0,aw=0,w=0,b=0) for _ in range(5)];commits=defaultdict(int)
    admitted=public=public_ack=None;started={};done={};acked={};last=-1
    heads={};state={h:{} for h in range(32)};out={h:{} for h in range(32)};rope_words={}
    source_requests=[];source_inputs=[]
    def frame(r,off):
        if r in writable:
            need((r,off) in produced,'Read before actual B commit');return produced[r,off][0]
        return data[r][off:off+64]
    for tag,row in rows:
        by[tag].append(row)
        if 'cycle' in row:
            need(last<=row['cycle']<40000000,'Time reversal or deadline');last=row['cycle']
        if tag=='ADMIT':
            need(admitted is None and row==dict(cycle=row['cycle'],sequence=SEQ,epoch=EPOCH,layer=0,position=0,token=0),'Whole-layer admission')
            admitted=row['cycle']
        elif tag=='START':
            s=row['stage'];need(admitted is not None and s in (1,2,3,4) and s not in started,'Stage admission')
            need(row['cycle']>admitted and (s==1 or s-1 in done and done[s-1]<row['cycle']),'Stage dependency')
            started[s]=row['cycle']
        elif tag=='DONE':
            s=row['stage'];need(s in started and s not in done and row['cycle']>started[s],'Stage completion')
            required={1:[2,6,7,8,24],2:[11,12],3:[14,15],4:[19,20,21,22,23]}[s]
            for r in required:need(commits[r]==(64 if r==24 else len(data[r])//64),'Incomplete stage writes')
            if s==1:need(len(rope_words)==2560 and all(len(state[h])==104 and len(out[h])==64 for h in range(32)),'Missing real front arithmetic')
            done[s]=row['cycle']
        elif tag=='ACK':
            s=row['stage'];need(s in done and s not in acked and row['cycle']>done[s],'Stage ACK identity')
            if s==1:
                need(2 in started and 2 not in done and len(source_requests)>=2048 and len(source_requests)==len(source_inputs),'Early front ACK before complete RAM consumption')
                need({r['index'] for r in source_requests}==set(range(2048)),'Source RAM coverage')
            else:
                need(public_ack is not None and row['cycle']>public_ack,'Tail ACK before public retirement')
                if s==2:need(3 in acked and 4 in acked and row['cycle']>max(acked[3],acked[4]),'Projection released before downstream')
            acked[s]=row['cycle']
        elif tag=='AR':
            p=row['port'];need(p in range(5) and 1 in started and not acked.get(4),'Read outside operation')
            need(pending_r[p] is None,'Prior writable R burst incomplete')
            i=ar_cursor[p];need(i<len(plans[p]),'Extra read burst');request=plans[p][i]
            need(row['index']==i and {k:row[k] for k in ('region','offset','beats')}==request,'Complete immutable read plan')
            r,off,n=(request[k] for k in ('region','offset','beats'))
            for beat in range(n):
                raw=frame(r,off+64*beat);need(len(raw)==64,'Read frame size')
                if r in writable:need(produced[r,off+64*beat][1]<row['cycle'],'Read before producing B edge')
            if r in writable:pending_r[p]=dict(region=r,offset=off,left=n,cycle=row['cycle'])
            ar_cursor[p]+=1;counts[p]['ar']+=1;counts[p]['r']+=n
        elif tag=='R':
            p,r,off=(row[k] for k in ('port','region','offset'));need(p in range(5),'Read port')
            pending=pending_r[p];need(pending is not None,'Unexpected writable read')
            need((r,off)==(pending['region'],pending['offset']) and row['cycle']>pending['cycle'],'R location/order')
            need(row['data']==int.from_bytes(frame(r,off),'little'),'Consumer did not receive actual producer bytes')
            pending['offset']+=64;pending['left']-=1
            if pending['left']==0:pending_r[p]=None
        elif tag=='AW':
            p,r,off,n=(row[k] for k in ('port','region','offset','beats'));need(p in range(5) and r in writable,'Write ownership')
            stage=1 if p in (0,3) else 2 if r in (11,12) else 3 if r in (14,15) else 4
            need(stage in started and stage not in done,'Write outside producer lifetime')
            need(aw[p] is None and row['index']==counts[p]['aw'],'AW ordering')
            need(off%64==0 and 0<n<=64 and off+n*64<=len(data[r]) and (4032+off)%4096+n*64<=4096,'AW bounds/page')
            aw[p]=row;counts[p]['aw']+=1
        elif tag=='W':
            p=row['port'];need(p in range(5) and row['index']==counts[p]['w'] and row['last'] in (0,1),'W order')
            need(row['data']<1<<512 and len(wq[p])<64,'W payload bounds');wq[p].append(row);counts[p]['w']+=1
        elif tag=='B':
            p=row['port'];need(p in range(5) and aw[p] is not None,'B without AW');request=aw[p]
            need(row['index']==counts[p]['b'] and all(row[k]==request[k] for k in ('region','offset','beats')),'B ownership')
            need(len(wq[p])==row['beats'] and row['cycle']>request['cycle'],'B before all write data')
            r=row['region']
            for i,w in enumerate(wq[p]):
                off=row['offset']+i*64;raw=w['data'].to_bytes(64,'little')
                need(w['cycle']<row['cycle'] and w['last']==int(i==row['beats']-1),'W/B framing')
                need((r,off) not in produced and raw==data[r][off:off+64],'Duplicate or numerically incorrect intermediate')
                if r==24:need(off//64 in m['cache_written_beats'],'Unexpected KV position/head')
                produced[r,off]=(raw,row['cycle']);commits[r]+=1
            aw[p]=None;wq[p]=[];counts[p]['b']+=1
        elif tag=='ROPE':
            i=row['index'];need(i==len(rope_words) and i<2560 and row['data']==ref['rope'][i],'Actual RoPE arithmetic/order');rope_words[i]=row['data']
        elif tag=='HEAD':
            h=row['head'];need(h==len(heads) and h<32 and len(rope_words)==2560 and commits[24]==64,'Attention before real RoPE/KV');heads[h]=row['cycle']
        elif tag in ('STATE','ATTENTION'):
            h,i=row['head'],row['index'];dest=state if tag=='STATE' else out
            need(h in heads and row['cycle']>heads[h] and i not in dest[h] and i<(104 if tag=='STATE' else 64),'Attention coverage/identity')
            dest[h][i]=row['data']
            if tag=='ATTENTION':need(row['data']==ref['attention'][h*64+i],'Attention bytes do not match downstream reference')
        elif tag in ('SOURCE','HLS_INPUT'):
            need(1 in done and 2 in started and 1 not in acked,'Quantizer read without retained front')
            i=row['index'];need(i<2048 and row['data']==ref['attention'][i],'Quantizer RAM data')
            key=(row['cycle'],i);need(row['cycle']>started[2],'RAM request before O admission')
            if tag=='SOURCE':
                need(key not in source,'Duplicate physical RAM return');source[key]=row['data'];source_requests.append(row)
            else:
                need(key in source and key not in consumed and source[key]==row['data'],'Synchronous HLS RAM correspondence')
                consumed.add(key);source_inputs.append(row)
        elif tag=='COMPLETE':
            need(public is None and set(done)=={1,2,3,4} and set(acked)=={1},'Incomplete retained full layer')
            need(row==dict(cycle=row['cycle'],status=0,sequence=SEQ,epoch=EPOCH,address=fixture.address(23),bytes=8192,prefix=1),'Public result certificate')
            need(row['cycle']>done[4] and all(x is None for x in aw+pending_r) and all(not x for x in wq),'Public response before memory drain')
            public=row['cycle']
        elif tag=='PUBLIC_ACK':
            need(public is not None and public_ack is None and row['cycle']>=public+7,'Held public response');public_ack=row['cycle']
    need(set(acked)=={1,2,3,4} and acked[3]==acked[4],'Complete retained handoff retirement')
    need(ar_cursor==list(map(len,plans)) and len(by['PORT'])==5 and len(by['READ'])==125,'Complete memory observation')
    for p,row in enumerate(by['PORT']):
        need(row['port']==p and all(row[k]==v for k,v in counts[p].items()),'Port accounting')
        if p in (0,3,4):need(row['w_stalls']>0,'Missing write backpressure')
        if p in (0,4):need(row['ar_stalls']>0,'Missing read backpressure on a busy port')
    for i,row in enumerate(by['READ']):
        p,r=divmod(i,25);expected=m['read_stats'][p][str(r)]
        need(row==dict(port=p,region=r,beats=expected['beats'],hash=int(expected['hash'],16)),'Full read data digest/count')
    expected_frames={(r,off) for r in writable if r!=24 for off in range(0,len(data[r]),64)}|{(24,i*64) for i in m['cache_written_beats']}
    need(set(produced)==expected_frames and len(produced)==3264,'Incomplete output images')
    need(len(by['ROPE_START'])==len(by['ROPE_DONE'])==1 and by['ROPE_START'][0]['cycle']<by['ROPE'][0]['cycle']<=by['ROPE'][-1]['cycle']<=by['ROPE_DONE'][0]['cycle'],'RoPE lifecycle')
    max_error=Decimal(0);max_ulp=0
    for h in range(32):
        query=dict(sequence=SEQ,epoch=EPOCH,position=0,layer=0,head=h)
        error,ulp=attention.state_numeric(query,dict(start=0,state=state[h],outputs=out[h]),ref['refs'][h]);max_error=max(max_error,error);max_ulp=max(max_ulp,ulp)
    need(len(by['PASS'])==1 and rows[-1][1]==dict(seed=seed,jobs=1,cycles=rows[-1][1]['cycles'],writes=3264,heads=32,source_words=len(source_requests)),'Final certificate')
    need(rows[-1][1]['cycles']>max(acked.values()) and len(source_requests)==len(source_inputs)>=2048 and {r['index'] for r in source_requests}==set(range(2048)),'Final RAM coverage/retirement')
    return dict(jobs=1,seed=seed,cycles=rows[-1][1]['cycles'],read_beats=sum(c['r'] for c in counts),write_beats=3264,
        source_ram_reads=len(source_requests),unique_source_words=2048,heads=32,rope_words=2560,attention_state_words=3328,
        final_output_bytes=8192,max_attention_scaled_state_error=str(max_error),max_attention_ulp=max_ulp,
        arithmetic_scope='One complete real layer-0/token-0 vector, including actual producer writes and downstream consumers; no full-model generation or physical DDR.')

def review(text,ref,seed=1):return review_rows(parse(text),ref,seed)
