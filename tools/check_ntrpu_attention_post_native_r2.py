"""Review the root/post-O controller protocol bank; arithmetic peers are not real."""
import copy
import re

PLAN=[0]+[v for m in range(1,28) for v in (m,0)]
SCHEMA={'RESET':'cycle job mode','BEGIN':'cycle job mode sequence epoch',
 'COMPLETE':'cycle job mode status valid root post sequence epoch hls post_hls read residual abi',
 'PASS':'seed jobs publics resets root_starts post_starts root_acks post_acks cycles'}
STATUS={1:3,2:5,3:11,4:11,5:7,6:2,7:7,8:2,9:4,10:5,11:5,12:11,19:5,20:7,21:2,22:1,23:1,24:7,25:11,26:1,27:1}
RESETS={15,16,17,18}
REJECTED={20,21,22,23,26,27}
NO_POST=REJECTED|{1,2,3,4,5,6,15,16}

def parse(text):
    assert not re.search(r'(?im)^\s*(FATAL:|Fatal:|ERROR:|APATH_FAIL)',text)
    rows=[]
    for line in text.splitlines():
        if not line.startswith('APATH_'):continue
        fields=line.split();tag=fields[0][6:];assert tag in SCHEMA
        pairs=[v.split('=') for v in fields[1:]]
        assert all(len(v)==2 for v in pairs) and [k for k,v in pairs]==SCHEMA[tag].split()
        row={k:int(v,16 if k in ('sequence','epoch') else 10) for k,v in pairs}
        assert all(v>=0 for v in row.values());rows.append((tag,row))
    return rows

def review_rows(rows,seed):
    assert rows[0]==('RESET',dict(cycle=6,job=0,mode=0))
    assert rows[-1][0]=='PASS' and sum(k=='PASS' for k,r in rows)==1
    summary=rows[-1][1]
    assert summary==dict(seed=seed,jobs=55,publics=51,resets=5,root_starts=49,post_starts=41,root_acks=45,post_acks=39,cycles=summary['cycles'])
    previous=6;cursor=0;completions=0;reset_count=1
    for j,mode in enumerate(PLAN,1):
        begin,finish=rows[1+(j-1)*2:1+j*2]
        assert begin[0]=='BEGIN';b=begin[1];k,f=finish
        epoch=0xfedcba9800000000+j
        if mode==21:epoch^=1<<40
        assert b==dict(cycle=b['cycle'],job=j,mode=mode,sequence=0x1234567800000000+j,epoch=epoch)
        assert previous<=b['cycle']<f['cycle']<summary['cycles'];previous=f['cycle']
        if mode in RESETS:
            assert k=='RESET' and f==dict(cycle=f['cycle'],job=j,mode=mode);reset_count+=1
        else:
            status=STATUS.get(mode,0);post=mode not in NO_POST
            assert k=='COMPLETE' and f==dict(cycle=f['cycle'],job=j,mode=mode,status=status,
                valid=int(status==0 and mode not in (13,14)),root=int(mode not in REJECTED),post=int(post),
                sequence=b['sequence'],epoch=epoch,hls=17 if mode==1 else 0,post_hls=5 if mode==19 else 0,
                read=384 if post else 0,residual=128 if post else 0,abi=64 if post else 0)
            completions+=1
    assert len(rows)==112 and completions==51 and reset_count==5
    return dict(seed=seed,jobs=55,completions=completions,resets=reset_count,cycles=summary['cycles'],
        scope='Controller and bus ownership with explicit protocol peers. No HLS arithmetic, actual AXI memory, physical timing or complete inference qualification.')

def review(text,seed):return review_rows(parse(text),seed)

def controls(text,seed):
    rows=parse(text);rejected=[]
    for tag,key,mask in [('BEGIN','mode',1),('BEGIN','sequence',1<<40),('BEGIN','epoch',1<<40),
        ('COMPLETE','status',1),('COMPLETE','valid',1),('COMPLETE','root',1),('COMPLETE','post',1),
        ('COMPLETE','read',1),('COMPLETE','hls',1),('COMPLETE','post_hls',1),('PASS','root_acks',1)]:
        bad=copy.deepcopy(rows);i=next(i for i,(k,r) in enumerate(bad) if k==tag);bad[i][1][key]^=mask
        try:review_rows(bad,seed)
        except (AssertionError,ValueError,KeyError,IndexError):rejected.append(tag+'.'+key)
        else:raise AssertionError('Accepted '+tag+'.'+key)
    try:review_rows(rows[:2]+rows[3:],seed)
    except (AssertionError,ValueError,KeyError,IndexError):rejected.append('missing-completion')
    else:raise AssertionError('Accepted missing completion')
    return rejected
