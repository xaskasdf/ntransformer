"""Check the fixed authority/arbitration scenario plan against every layer."""
import re


def expected():
    rows=[];prefix=[0]*17
    def add(tag,valid=1,fault=0):
        rows.append((tag,valid,fault,prefix.copy() if valid else [0]*17))
    add(0,0);add(1)
    for layer in range(16):
        for pos in range(2):
            add(10+layer*2+pos);prefix[layer]=pos+1;add(50+layer*2+pos)
    for tag in range(100,104):add(tag)
    add(104,0);add(105,0);prefix=[0]*17;add(106)
    prefix[5]=1;add(107);add(108);add(109,0,1);prefix=[0]*17
    for tag in range(110,130):
        valid=tag not in (111,113,115,117,119,120,122,124,126,128)
        fault=tag in (111,113,115,117,120,122)
        add(tag,int(valid),int(fault))
    return rows


def review(text,seed):
    assert not re.search(r'(?mi)^\s*(?:FATAL|Fatal:|ERROR|ROOTGUARD_FAIL)',text)
    assert text.splitlines().count(f'ROOTGUARD_CONFIG seed={seed} arithmetic_peer=0')==1
    checks=re.findall(r'^ROOTGUARD_CHECK step=(\d+) tag=(\d+) layer=(\d+) valid=(\d+) fault=(\d+) prefix=(\d+)$',text,re.M)
    want=[]
    for step,(tag,valid,fault,prefix) in enumerate(expected()):
        want.extend((step,tag,layer,valid,fault,prefix[layer]) for layer in range(17))
    assert [tuple(map(int,r)) for r in checks]==want,'Authority/coverage/order mismatch'
    assert re.findall(r'^ROOTGUARD_RESET count=(\d+)$',text,re.M)==['1','2','3','4']
    summary=re.findall(r'^ROOTGUARD_PASS steps=(\d+) resets=(\d+) cycles=(\d+)$',text,re.M)
    assert len(summary)==1
    steps,resets,cycles=map(int,summary[0]);assert steps==96 and resets==4 and 700<cycles<2000
    return dict(seed=seed,steps=steps,resets=resets,layer_checks=len(checks),cycles=cycles,arithmetic_qualified=False)


def controls(text,seed):
    variants={
        'prefix':lambda s:s.replace('prefix=1','prefix=2',1),
        'authority':lambda s:s.replace('valid=0 fault=0','valid=1 fault=0',1),
        'poison':lambda s:s.replace('fault=1','fault=0',1),
        'layer':lambda s:s.replace('layer=16','layer=15',1),
        'order':lambda s:s.replace('tag=10','tag=11',1),
        'missing':lambda s:re.sub(r'^ROOTGUARD_CHECK[^\n]*\n','',s,count=1,flags=re.M),
        'reset':lambda s:s.replace('RESET count=2','RESET count=3',1),
        'diagnostic':lambda s:s+'\nROOTGUARD_FAIL injected\n',
    }
    rejected=[]
    for name,mutate in variants.items():
        changed=mutate(text);assert changed!=text
        try:review(changed,seed)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected
