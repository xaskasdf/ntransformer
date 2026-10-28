"""Independently review FFN ownership traces; this is not arithmetic qualification."""
import copy
import re

PLAN=[v for m in range(48) for v in ((m,0) if 37<=m<=40 else (m,))]
RESET_MODES={37,38,39,40}
NO_START=set(range(2,8))|set(range(20,33))|set(range(41,45))|{37,38,51}
STATUS={**{m:2 for m in (2,3,4,5,10,11,44,45)},**{m:7 for m in (6,7,8,9)},
    1:6,12:11,13:11,**{m:5 for m in (14,15,16,17,18,19,47)},
    **{m:1 for m in list(range(20,33))+[41,42,43]},51:10}
HEX={'sequence','epoch','tag','ffn','residual','hls'}
SCHEMA={'RESET':'cycle job mode','BEGIN':'cycle job mode sequence epoch expected',
    'START':'cycle job mode tag','ACK':'cycle job mode',
    'COMPLETE':'cycle job mode status child stages phase bytes stage_bytes hls sequence epoch valid ffn residual ffn_bytes residual_bytes launched tag',
    'PASS':'config seed jobs completed starts acks resets held cycles'}

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|Fatal:|ERROR:|FOWNER_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('FOWNER_'):continue
        words=line.split();kind=words[0][7:];assert kind in SCHEMA
        pairs=[v.split('=') for v in words[1:]]
        assert all(len(v)==2 for v in pairs) and [k for k,v in pairs]==SCHEMA[kind].split()
        row={k:int(v,16 if k in HEX else 10) for k,v in pairs}
        assert all(v>=0 for v in row.values());rows.append((kind,row))
    return rows

def review_rows(rows,seed,config):
    assert config in (0,1,2) and rows and rows[0][0]=='RESET' and rows[-1][0]=='PASS'
    assert sum(k=='PASS' for k,r in rows)==1
    assert rows[0][1]['job']==0 and rows[0][1]['mode']==0 and 0<rows[0][1]['cycle']<10
    last=rows[0][1]['cycle'];cursor=1;starts=acks=completed=0;resets=1
    plan=PLAN if config==0 else [0,51,51] if config==1 else [51,51]
    initial=0xfffffffc if config==0 else 0xfffffffffffffffc if config==1 else 3
    next_tag=initial
    def consume(kind,job,mode):
        nonlocal cursor,last
        tag,r=rows[cursor];cursor+=1
        assert tag==kind and r['job']==job and r['mode']==mode and last<r['cycle']
        last=r['cycle'];return r
    for job,mode in enumerate(plan,1):
        begin=consume('BEGIN',job,mode);status=STATUS.get(mode,0)
        assert begin['sequence']==0x1234567800000000+job
        assert begin['epoch']==0xfedcba9800000000+job*3 and begin['expected']==status
        launched=mode not in NO_START;tag=next_tag
        if launched:
            start=consume('START',job,mode);assert start['tag']==tag and tag%4==0;starts+=1
            if tag!=0xfffffffffffffffc:next_tag+=4
        if mode in RESET_MODES:
            consume('RESET',job,mode);resets+=1;next_tag=initial;continue
        finish=consume('COMPLETE',job,mode);completed+=1
        valid=int(status==0 and mode not in (33,34,35,36,46))
        diag=dict(child=0,stages=4,phase=3,bytes=8192,stage_bytes=155648,hls=0) if launched else dict(child=0,stages=0,phase=0,bytes=0,stage_bytes=0,hls=0)
        if mode==1:diag.update(child=8,stages=2,phase=2,bytes=128,stage_bytes=123,hls=0x55)
        if mode==14:diag['stages']=3
        if mode==15:diag['bytes']=8191
        if mode==16:diag['stage_bytes']=155647
        if mode==17:diag['phase']=2
        if mode==18:diag['hls']=1
        if mode==19:diag['child']=1
        if mode==47:diag['stage_bytes']=155649
        expected=dict(cycle=finish['cycle'],job=job,mode=mode,status=status,**diag,
            sequence=begin['sequence'],epoch=begin['epoch'],valid=valid,
            ffn=finish['ffn'] if valid else 0,residual=finish['residual'] if valid else 0,
            ffn_bytes=8192 if valid else 0,residual_bytes=8192 if valid else 0,launched=int(launched),tag=tag)
        assert finish==expected,(job,mode,finish,expected)
        if valid:
            ffn_base=0x1000000000+7*0x4000000+job*0x200000000
            residual_base=0x700000000000+job*0x200000
            for name,base in [('ffn',ffn_base),('residual',residual_base)]:
                assert base<=finish[name]<=base+512 and finish[name]%64==0
        if launched:consume('ACK',job,mode);acks+=1
    assert cursor==len(rows)-1
    summary=rows[-1][1]
    assert summary==dict(config=config,seed=seed,jobs=len(plan),completed=completed,starts=starts,acks=acks,resets=resets,held=summary['held'],cycles=summary['cycles'])
    assert summary['held']>=3*completed and last<summary['cycles']<40000
    assert (completed,starts,acks,resets)==((48,27,25,5) if config==0 else (3,1,1,1) if config==1 else (2,0,0,1))
    return dict(config=config,seed=seed,jobs=len(plan),completed=completed,starts=starts,acks=acks,resets=resets,cycles=summary['cycles'],
        scope='Directed ownership protocol with controlled FFN peer; no FFN arithmetic, AXI transfers, full inference, or physical timing qualification.')

def review(text,seed,config=0):return review_rows(parse(text),seed,config)

def controls(text,seed,config=0):
    rows=parse(text);rejected=[]
    changes=[('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<40),('BEGIN','expected',1),
        ('COMPLETE','status',1),('COMPLETE','child',1),('COMPLETE','stages',1),('COMPLETE','phase',1),
        ('COMPLETE','bytes',1),('COMPLETE','stage_bytes',1),('COMPLETE','hls',1),
        ('COMPLETE','sequence',1<<40),('COMPLETE','epoch',1<<40),('COMPLETE','valid',1),
        ('COMPLETE','ffn',1),('COMPLETE','residual',1),('COMPLETE','ffn_bytes',1),('COMPLETE','residual_bytes',1),
        ('COMPLETE','launched',1),('COMPLETE','tag',4),('PASS','jobs',1),('PASS','completed',1),('PASS','acks',1)]
    if config!=2:changes.append(('START','tag',4))
    for kind,key,mask in changes:
        bad=copy.deepcopy(rows);i=next(i for i,(k,r) in enumerate(bad) if k==kind);bad[i][1][key]^=mask
        try:review_rows(bad,seed,config)
        except (AssertionError,ValueError,KeyError,IndexError):rejected.append(kind+'.'+key)
        else:raise AssertionError('Accepted mutation '+kind+'.'+key)
    bad=copy.deepcopy(rows);i=next(i for i,(k,r) in enumerate(bad) if k=='COMPLETE');del bad[i]
    try:review_rows(bad,seed,config)
    except (AssertionError,ValueError,KeyError,IndexError):rejected.append('missing-completion')
    else:raise AssertionError('Accepted missing completion')
    return rejected
