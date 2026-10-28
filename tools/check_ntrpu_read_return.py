"""Independent write/read/fragment/consumer ledger; no success from exit code alone."""
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def pattern(tag,beat):
    words=[((tag*0x9e3779b9)^(beat*0x10203)^(j*0x1010101))&0xffffffff for j in range(16)]
    return int.from_bytes(b''.join(w.to_bytes(4,'little') for w in words),'little')

def replay(lines,native=0):
    events=[s.split()[1:] for s in lines if s.startswith('RR ')]
    config=[r for r in events if r[0]=='CONFIG'];assert len(config)==1 and f'native={native}' in config[0]
    versions=[int(v.split('=')[1]) for v in config[0] if v.startswith('revision=')]
    assert len(versions)<=1
    revision=versions[0] if versions else 0
    assert revision in (0,1,2,3)
    ready_checks=[r for r in events if r[0]=='READY_CUT']
    assert len(ready_checks)==(1 if revision and not native else 0)
    if ready_checks:assert ready_checks[0][2]=='1'
    cases=4 if native else 20
    assert [int(r[1]) for r in events if r[0]=='CASE']==list(range(1,cases+1))
    assert [int(r[1]) for r in events if r[0]=='PASS']==list(range(1,cases+1))
    finish=[r for r in events if r[0]=='ALL_PASS'];assert len(finish)==1
    total={k:int(v) for k,v in (f.split('=') for f in finish[0][1:])};assert total['cases']==cases
    boots=[r for r in events if r[0]=='BOOT'];assert [int(r[2]) for r in boots]==list(range(1,(1 if native else 12)+1))
    assert all(r[3:5]==['1','1'] for r in boots)
    writes={};wactive={};memory={};commands={};active={};fragments={};counts=Counter();samples={};quarantines=[];case=0
    for r in events:
        kind=r[0]
        if kind=='CASE':case=int(r[1])
        if kind in ('WCMD','WI','WCPL','CMD','AR','NR','DATA','CPL'):
            ts,epoch,lane=map(int,r[1:4]);key=(epoch,lane);assert lane in (0,1)
        if kind=='WCMD':
            tag,n,addr=int(r[4]),int(r[5]),int(r[6],16);ck=(epoch,tag)
            assert ck not in writes and key not in wactive and 1<=n<=64 and tag%2==lane
            assert addr==(lane<<33)+0x10000+(tag//2)*4096
            writes[ck]=dict(lane=lane,n=n,addr=addr,beats=0,complete=False);wactive[key]=ck;counts['writes']+=1
        elif kind=='WI':
            assert key in wactive;ck=wactive[key];w=writes[ck];value=int(r[4],16)
            assert w['beats']<w['n'] and value==pattern(ck[1],w['beats'])
            for half in (0,1):memory[epoch,w['addr']+w['beats']*64+half*32]=(value>>(256*half))&((1<<256)-1)
            w['beats']+=1
        elif kind=='WCPL':
            assert key in wactive and wactive[key]==(epoch,int(r[4]));w=writes[wactive.pop(key)]
            assert w['beats']==w['n'] and not w['complete'];w['complete']=True
        elif kind=='CMD':
            tag,n,addr=int(r[4]),int(r[5]),int(r[6],16);ck=(epoch,tag)
            assert ck not in commands and key not in active and tag%2==lane
            invalid=not(1<=n<=64 and addr%64==0 and 0<=addr<2**34 and addr+n*64<=2**34 and addr//4096==(addr+n*64-1)//4096)
            if not invalid:assert ck in writes and writes[ck]['complete'] and writes[ck]['addr']==addr and writes[ck]['n']==n
            commands[ck]=dict(lane=lane,n=n,addr=addr,invalid=invalid,issued=0,received=0,delivered=0,error=0,complete=False,quarantine=False,case=case,protocol_bad=False)
            active[key]=ck;counts['commands']+=1
        elif kind=='AR':
            assert key in active and key not in fragments;c=commands[active[key]];addr,n=int(r[4],16),int(r[5])
            assert not c['invalid'] and 1<=n<=16 and addr==c['addr']+c['issued']*32 and c['issued']+n<=2*c['n']
            fragments[key]=dict(addr=addr,n=n,received=0,bad=False);c['issued']+=n;counts['native_addresses']+=1
        elif kind=='NR':
            assert key in active and key in fragments;c=commands[active[key]];f=fragments[key]
            rid,resp,last,value=int(r[4]),int(r[5]),int(r[6]),int(r[7],16)
            assert f['received']<f['n']
            malformed=rid!=0 or resp not in (0,2,3) or last!=int(f['received']+1==f['n'])
            if malformed:
                assert revision in (1,2,3) and lane==0
                assert ((c['case']==13 and rid==32 and f['received']==0) or
                        (c['case']==14 and last==1 and f['received']==0) or
                        (c['case']==15 and last==0 and f['received']+1==f['n']) or
                        (c['case']==16 and resp==1 and f['received']==0))
                f['bad']=True;c['protocol_bad']=True;counts['captured_malformed']+=1
            assert value==memory[epoch,f['addr']+f['received']*32]
            f['received']+=1;c['received']+=1;c['error']=max(c['error'],resp if resp in (2,3) else 0);counts['native_reads']+=1
            if f['received']==f['n'] and not f['bad']:del fragments[key]
        elif kind=='DATA':
            assert key in active and active[key]==(epoch,int(r[4]));c=commands[active[key]]
            beat,last,value=int(r[5]),int(r[6]),int(r[7],16)
            assert not c['invalid'] and not c['protocol_bad'] and c['error']==0 and c['received']==c['issued']==2*c['n'] and key not in fragments
            assert beat==c['delivered']<c['n'] and last==int(beat+1==c['n']) and value==pattern(int(r[4]),beat)
            c['delivered']+=1;counts['output_beats']+=1;samples.setdefault(ts,set()).add(lane)
        elif kind=='CPL':
            assert key in active and active[key]==(epoch,int(r[4])) and key not in fragments
            c=commands[active.pop(key)];assert not c['protocol_bad'];status,nbytes=int(r[5]),int(r[6])
            expected=1 if c['invalid'] else 2 if c['error'] else 0
            assert status==expected and nbytes==c['delivered']*64
            assert c['delivered']==(c['n'] if expected==0 else 0)
            assert (c['issued'],c['received'])==((0,0) if c['invalid'] else (2*c['n'],2*c['n']))
            assert not c['complete'];c['complete']=True;counts['completions']+=1
        elif kind=='QUARANTINE':
            epoch,tag,case=map(int,r[2:5]);ck=(epoch,tag);c=commands[ck];key=(epoch,c['lane'])
            assert case in range(13,18) and active[key]==ck and key in fragments and not c['invalid']
            assert not c['complete'] and not c['quarantine'] and c['delivered']==0 and c['received']<2*c['n']
            c['quarantine']=True;quarantines.append(case)
            if revision and case<17:assert c['protocol_bad']
    assert not wactive and all(v['complete'] for v in writes.values())
    assert all(c['complete']!=c['quarantine'] for c in commands.values())
    assert len(active)==len(fragments)==len(quarantines)==(0 if native else 5)
    assert quarantines==([] if native else list(range(13,18)))
    for marker in ('FULL_RESERVED','CPL_HELD'):assert sum(r[0]==marker for r in events)==1
    counts['dual']=sum(len(v)==2 for v in samples.values());assert counts['dual']>0
    for name in ('commands','writes','native_reads','output_beats','completions','dual'):assert total[name]==counts[name],name
    assert sum(c['invalid'] for c in commands.values())==(0 if native else 5)
    assert sum(c['error']!=0 for c in commands.values())==(0 if native else 2)
    assert counts['captured_malformed']==(4 if revision and not native else 0)
    return dict(revision=revision,ready_cut_checks=len(ready_checks),cases=cases,**counts,quarantined_commands=len(quarantines),invalid_commands=sum(c['invalid'] for c in commands.values()),
                read_error_commands=sum(c['error']!=0 for c in commands.values()),boot_ps=int(boots[0][1]))

def controls(lines,native):
    passed=[]
    edits=[('data','RR DATA ',7),('native_data','RR NR ',7),('write_data','RR WI ',4),('address','RR AR ',4),
           ('command_length','RR CMD ',5),('native_id','RR NR ',4),('native_last','RR NR ',6),('completion_status','RR CPL ',5),('completion_bytes','RR CPL ',6)]
    for label,prefix,column in edits:
        mutated=list(lines);index=next(i for i,s in enumerate(mutated) if s.startswith(prefix));fields=mutated[index].split()
        # fields include the RR prefix, whereas column is numbered after it.
        j=column+1;radix=16 if label in ('data','native_data','write_data','address') else 10
        fields[j]=format(int(fields[j],radix)^1,'x') if radix==16 else str(int(fields[j])^1);mutated[index]=' '.join(fields)
        try:replay(mutated,native)
        except (AssertionError,KeyError):passed.append(label)
        else:raise AssertionError('Undetected corruption '+label)
    for prefix in ('WCMD','WI','WCPL','CMD','AR','NR','DATA','CPL','ALL_PASS','BOOT','FULL_RESERVED','CPL_HELD')+(() if native else ('QUARANTINE',))+ (('READY_CUT',) if any(s.startswith('RR READY_CUT ') for s in lines) else ()):
        mutated=list(lines);index=next(i for i,s in enumerate(mutated) if s.startswith('RR '+prefix+' '));del mutated[index]
        try:replay(mutated,native)
        except (AssertionError,KeyError):passed.append('missing_'+prefix)
        else:raise AssertionError('Undetected deletion '+prefix)
    return passed

def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='read_return'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    log=(out/'suite/simulation.log').read_text();assert not re.search(r'(?im)^\s*(ERROR:|Fatal:|FAIL )',log)
    lines=log.splitlines();counts=replay(lines,r['native_profile'])
    configs={}
    for name in ('hbm_read_cdc','hbm_read_width','hbm_read_protocol'):
        configs[name]=dict(s.split('\t',1) for s in (out/'suite'/(name+'_config.tsv')).read_text().splitlines() if s.startswith('CONFIG.'))
        assert configs[name]['CONFIG.READ_WRITE_MODE']=='READ_ONLY'
    assert configs['hbm_read_cdc']['CONFIG.ACLK_ASYNC']=='1' and configs['hbm_read_cdc']['CONFIG.SYNCHRONIZATION_STAGES']=='3'
    assert configs['hbm_read_width']['CONFIG.SI_DATA_WIDTH']=='512' and configs['hbm_read_width']['CONFIG.MI_DATA_WIDTH']=='256'
    assert configs['hbm_read_width']['CONFIG.MAX_SPLIT_BEATS']=='16'
    assert configs['hbm_read_protocol']['CONFIG.MI_PROTOCOL']=='AXI3'
    return dict(status='READ_RETURN_PREFLIGHT_REVIEWED',result_sha256=sha(out/'result.json'),seconds=r['seconds'],
                common=r['common'],native_profile=r['native_profile'],**counts,negative_controls=controls(lines,r['native_profile']),
                read_return_synthesizable=True,amd_conversion_and_cdc=True,native_hbm_controller=False,physical_timing_qualified=False,board_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
