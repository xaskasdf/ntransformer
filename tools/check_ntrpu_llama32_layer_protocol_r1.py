"""Review complete layer-control transcripts; arithmetic peers are synthetic."""
import re

SCHEMA={'LAYER_ADMIT':{'case','mode','target','cycle','sequence','epoch','layer','position'},
    'LAYER_RESULT':{'case','status','stage','revoked','result','front','projection','post','ffn','words','cycle'},
    'LAYER_RELEASE':{'case','drain','cycle'},'LAYER_PASS':{'seed','jobs','cycles','arithmetic'}}

def require(value,message):
    if not value:raise ValueError(message)

def plan():
    rows=[(0,0,0)]*8+[(1,i,1) for i in range(24)]+[(2,i,5) for i in range(1,5)]
    rows += [(m,i,7 if m==3 else 2) for m in (3,4) for i in range(1,5)]
    return rows+[(5,i,8) for i in range(1,4)]+[(6,i,0) for i in range(4)]+[(7,i,11) for i in range(1,5)]

def parse(text):
    require(not re.search(r'(?im)(\bfatal\b|^ERROR:|LAYER_FAIL)',text),'Diagnostic in transcript')
    rows=[]
    for line in text.splitlines():
        if not line.startswith('LAYER_'):continue
        words=line.split();kind=words.pop(0);require(kind in SCHEMA,'Unknown event')
        row={'kind':kind}
        for word in words:
            k,v=word.split('=');require(k not in row,'Duplicate field')
            row[k]=v if k=='arithmetic' else int(v,16 if k in ('sequence','epoch') else 10)
        require(set(row)==SCHEMA[kind]|{'kind'},'Schema mismatch');rows.append(row)
    return rows

def review(text,seed):
    events=parse(text);cases=plan();require(len(cases)==55 and len(events)==166,'Incomplete event inventory')
    previous=-1;cycles=[];full_words=0
    for job,(mode,target,status) in enumerate(cases):
        admit,result,release=events[job*3:job*3+3]
        require([e['kind'] for e in (admit,result,release)]==['LAYER_ADMIT','LAYER_RESULT','LAYER_RELEASE'],'Event ordering')
        require(all(e['case']==job for e in (admit,result,release)),'Case identity')
        require((admit['mode'],admit['target'])==(mode,target),'Case plan')
        require(admit['sequence']==0x1234567887654321 and admit['epoch']==0xfedcba9876543210,'64-bit identity')
        require(admit['position']==(job%2 if job<8 else 0),'Committed prefix reuse')
        require(admit['layer']==(job-job%2 if job<8 else job%16),'Layer selection')
        require(previous<admit['cycle']<result['cycle']<release['cycle'],'Job overlap or time reversal')
        require(result['status']==status and result['result']==int(status==0),'Completion status/result')
        require(0<=result['stage']<=4 and all(result[k] in (0,1) for k in ('front','projection','post','ffn','revoked')),'Invalid counters')
        if mode in (0,6):
            require(result['stage']==4 and all(result[k]==1 for k in ('front','projection','post','ffn')),'Missing successful stage')
            require(result['words']==2048 and result['revoked']==0,'Source RAM transfer');full_words+=2048
        if mode==1:require(result['stage']==0 and all(result[k]==0 for k in ('front','projection','post','ffn')),'Preflight launched arithmetic')
        if mode in (2,3,4,7):require(result['stage']==target,'Unexpected stage progression')
        if mode==5:require(result['stage']==target-1,'Timeout stage progression')
        if mode in (3,4,5,7):require(result['revoked']==1,'Fault not retained')
        require(0<release['drain']<=100,'Drain latency');previous=release['cycle'];cycles.append(result['cycle']-admit['cycle'])
    final=events[-1]
    require(final==dict(kind='LAYER_PASS',seed=seed,jobs=55,cycles=previous,arithmetic='synthetic'),'Final certificate')
    return dict(seed=seed,jobs=55,cycles=previous,full_result_words=full_words,max_response_cycles=max(cycles),arithmetic_qualified=False)

def controls(text,seed):
    edits={
        'missing_result':lambda t:'\n'.join(l for l in t.splitlines() if not l.startswith('LAYER_RESULT case=0 ')),
        'false_success':lambda t:t.replace('status=1 stage=0','status=0 stage=0',1),
        'identity_high_word':lambda t:t.replace('sequence=1234567887654321','sequence=0000000087654321',1),
        'wrong_prefix':lambda t:t.replace('position=1','position=0',1),
        'short_source':lambda t:t.replace('words=2048','words=2047',1),
        'missing_ffn':lambda t:t.replace('ffn=1','ffn=0',1),
        'unexpected_case':lambda t:t.replace('mode=1 target=0','mode=1 target=1',1),
        'wrong_count':lambda t:t.replace('jobs=55','jobs=54'),
        'wrong_seed':lambda t:t.replace(f'LAYER_PASS seed={seed}',f'LAYER_PASS seed={seed+1}'),
        'false_arithmetic_claim':lambda t:t.replace('arithmetic=synthetic','arithmetic=real'),
        'duplicate_pass':lambda t:t+'\n'+next(l for l in t.splitlines() if l.startswith('LAYER_PASS')),
        'fatal_after_pass':lambda t:t+'\nFATAL: late error\n',
    }
    rejected=[]
    for name,edit in edits.items():
        changed=edit(text);require(changed!=text,'Inactive trace control')
        try:review(changed,seed)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Accepted corrupted trace: '+name)
    return rejected
