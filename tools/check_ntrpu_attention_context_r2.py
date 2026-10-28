"""Independent inventory and value oracle for the private attention context peer.

This reviews protocol data, not attention arithmetic. The expected case plan is
explicit so omissions cannot acquire a PASS merely by adjusting the summary.
"""
from collections import Counter, defaultdict
import re

def review(text, seed):
    assert not re.search(r'\b(?:FATAL|ERROR|FAIL)\b', text)
    events=[]
    names={'RESET','START','COMMAND','TILE','Q','STATE_READ','STATE_WRITE','OUTPUT_WRITE','FINISH','REJECT_TILE','RESULT','COMPLETE','PASS_ATTENTION_CONTEXT_PROTOCOL_R2'}
    for line in text.splitlines():
        fields=line.split()
        if not fields or fields[0] not in names:continue
        row={}
        for token in fields[1:]:
            k,v=token.split('=');assert k not in row
            row[k]=int(v,16 if k=='data' else 10)
        events.append((fields[0],row))
    summaries=[r for k,r in events if k=='PASS_ATTENTION_CONTEXT_PROTOCOL_R2'];assert len(summaries)==1
    summary=summaries[0]
    expected=dict(seed=seed,cases=69,tiles=77,starts=73,q_reads=4672,state_reads=1976,result_reads=2240,resets=4)
    assert {k:summary[k] for k in expected}==expected and 21000<=summary['cycles']<=24000
    counts=Counter(k for k,r in events)
    for kind,key in [('COMMAND','cases'),('TILE','tiles'),('START','starts'),('Q','q_reads'),('STATE_READ','state_reads'),('RESULT','result_reads'),('RESET','resets')]:
        assert counts[kind]==expected[key],(kind,counts[kind])
    assert counts['COMPLETE']==66 and counts['FINISH']==73 and counts['REJECT_TILE']==3
    assert events[-1][0]=='PASS_ATTENTION_CONTEXT_PROTOCOL_R2'
    expected_status={c:0 for c in range(1,35)}
    modes={35+i:m for i,m in enumerate([1,2,3,4,5,6,7,8,9,10,11,14,15])}
    for c,mode in modes.items():expected_status[c]=7 if mode in (9,11) else 2 if mode==10 else 10 if mode==7 else 4
    expected_status.update({48:10,49:2,50:1,51:2,52:11,53:11,54:11,55:2,56:11,57:3,58:7,59:7,60:4,61:2,62:2,63:1,64:1,65:1,69:0})
    commands={};tiles={};closed={};starts=Counter();state_writes=defaultdict(list);output_writes=defaultdict(list)
    q_reads=defaultdict(list);s_reads=defaultdict(list);results=defaultdict(list);completed={};last_good={}
    active_case=0;reset_rows=[]
    for kind,row in events:
        if kind=='RESET':
            reset_rows.append(row);last_good.clear();continue
        if kind=='PASS_ATTENTION_CONTEXT_PROTOCOL_R2':continue
        c=row['case'];assert 1<=c<=69
        if kind=='COMMAND':
            assert c==len(commands)+1 and c not in commands
            commands[c]=row;active_case=c
            position=0;layer=0;head=0;bad=0
            if c<=32:head=c-1;layer=head%16;position=64 if head%2 else 0
            elif c==33:position=128;layer=15;head=31
            elif c==34:position=63
            elif c==37:position=64
            elif c==48:position=64;layer=7;head=13
            elif c in (60,61,62):bad=c-59
            elif c==63:position=131072
            elif c==64:layer=16
            elif c==65:head=32
            assert [row[k] for k in ('position','layer','head','bad')]==[position,layer,head,bad]
            continue
        assert c==active_case and c in commands
        if kind=='TILE':
            t=row['tile'];assert t==len(tiles)+1
            tiles[t]=row
            count=len([r for r in tiles.values() if r['case']==c])
            assert row['first']==(count-1)*64
            assert row['count']==min(64,commands[c]['position']+1-row['first'])
            assert row['bad']==(c-48 if c in (49,50,51) else 0)
            continue
        if kind in ('RESULT','COMPLETE'):
            assert c not in completed
            if kind=='COMPLETE':
                assert row['status']==expected_status[c] and row['revoke']==(1 if c==32 else 0)
                if row['status']==0:
                    assert len(results[c])==64 and last_good[c][0]==commands[c]['position']+1
                else:assert not results[c]
                completed[c]=row
            else:
                i=row['index'];assert expected_status[c]==0 and i==len(results[c]) and 0<=i<64
                assert row['data']==0x40000000 | (c<<6) | i
                assert last_good[c][0]==commands[c]['position']+1
                last=last_good[c][1];assert dict(output_writes[last])[i]==row['data']
                results[c].append(row['data'])
            continue
        t=row['tile'];assert t in tiles and tiles[t]['case']==c and t not in closed
        tile=tiles[t];mode=modes.get(c,7 if c==48 and tile['first']==64 else 0)
        if kind=='START':starts[t]+=1;assert starts[t]==1
        elif kind=='Q':
            i=row['index'];assert i==len(q_reads[t]) and 0<=i<64 and row['data']==0x3f000000 | (c<<6) | i
            q_reads[t].append(row['data'])
        elif kind=='STATE_READ':
            i=row['index'];assert i==len(s_reads[t]) and 0<=i<104
            assert c in last_good and last_good[c][0]==tile['first']
            prior=last_good[c][1];assert dict(state_writes[prior])[i]==row['data']==(c<<40) | (tile['first']<<8) | i
            s_reads[t].append(row['data'])
        elif kind=='STATE_WRITE':
            i=row['index'];ordinal=len(state_writes[t]);value_index=ordinal
            assert ordinal<104
            expected_index=0 if mode==2 and ordinal==103 else 127 if mode==6 and ordinal==103 else ordinal
            assert i==expected_index
            assert row['data']==(c<<40) | ((tile['first']+tile['count'])<<8) | value_index
            state_writes[t].append((i,row['data']))
        elif kind=='OUTPUT_WRITE':
            i=row['index'];ordinal=len(output_writes[t]);assert ordinal<65
            assert i==(63 if mode==14 and ordinal==64 else ordinal)
            expected_data=0x7fc00001 if mode==4 and i==7 else 0x40000000 | (c<<6) | i
            assert row['data']==expected_data
            output_writes[t].append((i,row['data']))
        elif kind in ('FINISH','REJECT_TILE'):
            if kind=='REJECT_TILE':
                assert c in (49,50,51) and row['status']==expected_status[c]
                assert not starts[t] and not q_reads[t] and not state_writes[t] and not output_writes[t]
            else:
                assert row['mode']==mode and starts[t]==1 and len(q_reads[t])==64
                assert len(s_reads[t])==(104 if tile['first'] else 0)
                assert len(state_writes[t])==(103 if mode==1 else 104)
                final=tile['first']+tile['count']==commands[c]['position']+1
                out_count=(32 if mode==8 else 63 if mode==15 else 65 if mode==14 else 64) if final or mode==3 else 0
                assert len(output_writes[t])==out_count
                status=7 if mode in (9,11) else 2 if mode==10 else 10 if mode==7 else 4 if mode else 0
                assert row['status']==status
                if status==0:
                    assert [i for i,v in state_writes[t]]==list(range(104))
                    last_good[c]=(tile['first']+tile['count'],t)
            closed[t]=row
        else:raise AssertionError(kind)
    assert set(completed)==set(expected_status)
    assert [r['case'] for r in reset_rows]==[0,66,67,68]
    assert len(closed)==76 and [r['case'] for t,r in tiles.items() if t not in closed]==[67]
    assert sum(len(v) for v in results.values())==2240
    return dict(summary=summary,completions=len(completed),reset_aborted_cases=[66,67,68],
        q_words_compared=4672,state_words_compared=1976,result_words_compared=2240,
        state_writes=sum(map(len,state_writes.values())),output_writes=sum(map(len,output_writes.values())),
        arithmetic_qualified=False,generated_hls_bound=False)

def controls(text,seed):
    changes={
        'missing_summary':lambda s:re.sub(r'^PASS_ATTENTION_CONTEXT_PROTOCOL_R2.*\n?', '',s,flags=re.M),
        'wrong_q':lambda s:s.replace('Q case=1 tile=1 index=0 data=3f000040','Q case=1 tile=1 index=0 data=3f000041',1),
        'wrong_state_read':lambda s:re.sub(r'(STATE_READ [^\n]*data=)[0-9a-f]+',r'\g<1>0000000000000000',s,count=1),
        'wrong_state_write':lambda s:re.sub(r'(STATE_WRITE [^\n]*data=)[0-9a-f]+',r'\g<1>0000000000000000',s,count=1),
        'wrong_output_write':lambda s:re.sub(r'(OUTPUT_WRITE [^\n]*data=)[0-9a-f]+',r'\g<1>00000000',s,count=1),
        'wrong_result':lambda s:re.sub(r'(RESULT [^\n]*data=)[0-9a-f]+',r'\g<1>00000000',s,count=1),
        'omit_read':lambda s:re.sub(r'^Q [^\n]*\n','',s,count=1,flags=re.M),
        'omit_tile':lambda s:re.sub(r'^TILE [^\n]*\n','',s,count=1,flags=re.M),
        'omit_start':lambda s:re.sub(r'^START [^\n]*\n','',s,count=1,flags=re.M),
        'omit_finish':lambda s:re.sub(r'^FINISH [^\n]*\n','',s,count=1,flags=re.M),
        'omit_reset':lambda s:re.sub(r'^RESET [^\n]*\n','',s,count=1,flags=re.M),
        'error_claimed_ok':lambda s:s.replace('COMPLETE case=35 status=4','COMPLETE case=35 status=0',1),
        'missing_revocation':lambda s:s.replace('COMPLETE case=32 status=0 revoke=1','COMPLETE case=32 status=0 revoke=0',1),
        'wrong_head':lambda s:s.replace('COMMAND case=32 position=64 layer=15 head=31','COMMAND case=32 position=64 layer=15 head=30',1),
        'extra_pass':lambda s:s+'\n'+next(x for x in s.splitlines() if x.startswith('PASS_ATTENTION_CONTEXT_PROTOCOL_R2'))+'\n',
    }
    rejected=[]
    for name,mutate in changes.items():
        changed=mutate(text);assert changed!=text,name
        try:review(changed,seed)
        except (AssertionError,KeyError,ValueError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted corrupted evidence: '+name)
    return rejected
