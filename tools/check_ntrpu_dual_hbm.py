"""Replay independent source/native/readback ledgers; reject partial simulations."""
import argparse,copy,hashlib,json,re
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def pattern(tag,beat):
    return sum((((tag*0x9e3779b9)^(beat*0x10203)^(j*0x1010101))&0xffffffff)<<(32*j) for j in range(16))

def replay(lines,native=0):
    records=[line.split()[1:] for line in lines if line.startswith('DH ')]
    config=[r for r in records if r[0]=='CONFIG'];assert len(config)==1
    assert f'native={native}' in config[0]
    finish=[r for r in records if r[0]=='ALL_PASS'];assert len(finish)==1,'Missing full completion'
    wanted=3 if native else 11
    assert finish[0][1]==f'cases={wanted}'
    passes=[r for r in records if r[0]=='PASS'];assert [int(r[1]) for r in passes]==list(range(1,wanted+1)),'Missing scenarios'
    commands={};queues={};stages={};leases={};generations={};bursts={};barriers={};reads=set();early=0;dual={}
    command_total=byte_total=read_total=0
    keys={'CMD','IN','AW','W','NAW','NW','NB','B','CPL','RETURN','FENCE','FENCED','DRAIN','R'}
    events=sorted((r for r in records if r[0] in keys),key=lambda r:int(r[1]))
    for r in events:
        kind,ts,epoch=r[0],int(r[1]),int(r[2]);lane=int(r[3]) if kind not in ('FENCE','FENCED','DRAIN') else -1
        key=(epoch,lane);q=queues.setdefault(key,[]);stage=stages.setdefault(key,dict(AW=0,W=0,B=0,CPL=0))
        if kind=='CMD':
            t,n,address,bank,gen=int(r[4]),int(r[5]),int(r[6],16),int(r[7]),int(r[8]);ck=(epoch,t)
            assert ck not in commands and 1<=n<=64 and t%2==lane,'Command identity/length'
            assert address==(lane<<33)+0x10000+(t//2)*4096,'Command address'
            assert 0<=bank<8 and (bank>=4 or bank//2==lane),'Bank ownership'
            assert (epoch,bank) not in leases,'Double allocation'
            assert gen>generations.get((epoch,bank),0),'Stale generation'
            assert epoch not in barriers,'Admission across fence'
            leases[epoch,bank]=(lane,t,gen);generations[epoch,bank]=gen;q.append(t)
            commands[ck]=dict(lane=lane,n=n,addr=address,bank=bank,gen=gen,IN=0,W=0,NW=0,NAW=0,NB=0,AW=None,B=None,CPL=None,returned=None)
            command_total+=1;byte_total+=n*64
        elif kind in ('IN','W','NW','R'):
            t,k,value=int(r[4]),int(r[5]),int(r[6],16);c=commands[epoch,t]
            assert c['lane']==lane,'Payload lane'
            if kind in ('IN','W'):
                assert k==c[kind] and k<c['n'],'Source payload sequence'
                assert value==pattern(t,k),'Source payload data'
                if kind=='W':
                    assert c['AW'] is not None and k<c['IN'],'Source W causal'
                    assert stage['W']<len(q) and q[stage['W']]==t,'Source W order'
                    c['last_W']=ts
                    if k+1==c['n']:stage['W']+=1
                else:dual.setdefault(ts,set()).add(lane)
                c[kind]+=1
            else:
                assert k<c['n']*2 and value==(pattern(t,k//2)>>(256*(k%2)))&((1<<256)-1),'Native/readback data'
                if kind=='NW':
                    assert k==c['NW'] and k//2<c['W'],'Native W before source data'
                    b=next((b for b in bursts.get(key,[]) if b['W']<b['n']),None)
                    assert b and b['tag']==t and k==b['off']+b['W'],'Native W order'
                    b['W']+=1;c['NW']+=1
                else:
                    assert c['CPL'] is not None and ts>c['CPL'],'Read before completion'
                    reads.add((epoch,t,k));read_total+=1
        elif kind=='AW':
            t=int(r[4]);assert stage['AW']<len(q) and q[stage['AW']]==t,'AW order'
            c=commands[epoch,t];assert c['AW'] is None;c['AW']=ts;stage['AW']+=1
        elif kind=='NAW':
            t,off,n,address=int(r[4]),int(r[5]),int(r[6]),int(r[7],16);c=commands[epoch,t]
            assert c['AW'] is not None and c['AW']<ts,'Native AW before source'
            assert off==c['NAW'] and 1<=n<=16 and off+n<=c['n']*2,'Native burst coverage'
            assert address==c['addr']+off*32 and address//4096==(address+n*32-1)//4096,'Native address'
            bursts.setdefault(key,[]).append(dict(tag=t,off=off,n=n,W=0,B=None));c['NAW']+=n
        elif kind=='NB':
            t,off,status=map(int,r[4:7]);c=commands[epoch,t]
            pending=next((b for b in bursts.get(key,[]) if b['B'] is None),None)
            assert pending and (pending['tag'],pending['off'])==(t,off) and pending['W']==pending['n'],'Native B causal/order'
            assert status==(2 if t==1000 else 0),'Native B status';pending['B']=ts;c['NB']+=pending['n']
        elif kind in ('B','CPL'):
            t,status=map(int,r[4:6]);c=commands[epoch,t]
            assert stage[kind]<len(q) and q[stage[kind]]==t,kind+' order'
            assert c['IN']==c['W']==c['n'] and c['NW']==c['NB']==c['n']*2,kind+' before native completion'
            assert status==(2 if t==1000 else 0),kind+' status'
            if kind=='CPL':assert c['B'] is not None and c['B']<ts,'CPL before source B'
            assert c[kind] is None;c[kind]=ts;stage[kind]+=1
        elif kind=='RETURN':
            bank,gen=map(int,r[4:6]);assert (epoch,bank) in leases,'Unowned return'
            owner,t,generation=leases.pop((epoch,bank));c=commands[epoch,t]
            assert owner==lane and gen==generation,'Wrong return owner/generation'
            assert c['W']==c['n'] and ts==c['last_W']+5000,'Return before complete source payload'
            assert c['returned'] is None;c['returned']=ts
            if c['NB']<c['n']*2:early+=1
        elif kind=='FENCE':
            assert epoch not in barriers;barriers[epoch]=int(r[3])
        elif kind=='FENCED':
            assert barriers.pop(epoch,None)==int(r[3]),'Fence identity'
            assert all(c['CPL'] is not None for (e,t),c in commands.items() if e==epoch),'Early fence'
        elif kind=='DRAIN':
            assert epoch not in barriers,'Drain with fence pending'
            assert all(c['CPL'] is not None and c['returned'] is not None for (e,t),c in commands.items() if e==epoch),'Early drain'
    assert not leases and not barriers,'Live leases/fences at finish'
    for (epoch,t),c in commands.items():
        assert c['IN']==c['W']==c['n'] and c['NW']==c['NAW']==c['n']*2 and c['returned'] is not None,'Incomplete payload'
        if t==800:
            assert not native and c['B'] is None and c['CPL'] is None and c['NB']==0,'Quarantine escaped'
        else:
            assert c['CPL'] is not None and c['NB']==c['n']*2,'Lost completion'
            assert all((epoch,t,k) in reads for k in range(c['n']*2)),'Missing readback'
    counts=dict(x.split('=') for x in finish[0][1:])
    assert command_total==int(counts['commands'])==(58 if native else 68),'Command coverage'
    assert byte_total==int(counts['bytes'])==(36224 if native else 242112),'Payload coverage'
    assert read_total==int(counts['reads'])==(1460 if native else 7878),'Readback coverage'
    assert max(int(p[5]) for p in passes)==8 and early>0 and any(len(x)==2 for x in dual.values()),'Missing pressure/dual/CDC coverage'
    assert [int(p[-1]) for p in passes]==([0]*3 if native else [0]*5+[1]*6),'Fault scenario coverage'
    return dict(commands=command_total,payload_bytes=byte_total,readback_beats=read_total,
                cases=wanted,early_returns=early,simultaneous_input_cycles=sum(len(x)==2 for x in dual.values()),
                quarantined_commands=0 if native else 1)


def negative_controls(lines,native):
    controls=[]
    for kind,index,label in [('IN',7,'input_data'),('W',7,'source_data'),('NW',7,'native_data'),('R',7,'readback_data'),
                              ('CMD',7,'address'),('CMD',9,'generation'),('RETURN',6,'return_generation'),
                              ('NAW',7,'native_length'),('NB',7,'native_status'),('B',6,'source_status'),('CPL',6,'completion_status')]:
        damaged=list(lines);i=next(i for i,x in enumerate(damaged) if x.startswith('DH '+kind+' '));fields=damaged[i].split()
        # Columns here include the leading DH token; increment a decimal field
        # or corrupt a payload/address without changing trace syntax.
        if label in ('input_data','source_data','native_data','readback_data','address'):fields[index]=format(int(fields[index],16)^1,'x')
        else:fields[index]=str(int(fields[index])+1)
        damaged[i]=' '.join(fields)
        try:replay(damaged,native)
        except (AssertionError,KeyError,ValueError,IndexError):controls.append(label)
        else:raise AssertionError('Undetected negative control '+label)
    for kind in ('ALL_PASS','CPL','RETURN','R','NB','NAW'):
        damaged=list(lines);i=next(i for i,x in enumerate(damaged) if x.startswith('DH '+kind+' '));del damaged[i]
        try:replay(damaged,native)
        except (AssertionError,KeyError,ValueError,IndexError):controls.append('missing_'+kind)
        else:raise AssertionError('Undetected missing '+kind)
    return controls


def review(out):
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items()),'Changed exported evidence'
    result=json.loads((out/'result.json').read_text());assert result['status']=='EXPORTED_PENDING_REVIEW'
    assert result['returncode']==0 and not result['timed_out'] and result['sources']==result['sources_after']
    log=out/'suite/simulation.log';lines=log.read_text().splitlines()
    assert not re.search(r'(?im)^\s*(Fatal:|ERROR:|FAIL )','\n'.join(lines))
    native=result.get('native_profile',0);counts=replay(lines,native)
    return dict(status='DUAL_HBM_PREFLIGHT_REVIEWED',result_sha256=sha(out/'result.json'),simulation_sha256=sha(log),
                policy=result['prefer_local'],**counts,negative_controls=negative_controls(lines,native),native_profile=native,native_controller=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
