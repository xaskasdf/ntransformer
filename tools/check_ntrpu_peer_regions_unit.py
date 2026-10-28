"""Independent two-extent integer model plus the existing ownership auditor."""
import argparse
import importlib.util
import json
from pathlib import Path
from run_ntrpu_peer_read import sha
from check_ntrpu_peer_shared import words,lease,request,require


def region_map(text):
    w=words(text,7,2)
    return dict(generation=w[0],valid=int(text,16)&3,
                regions=[dict(logical=w[1],physical=w[2],size=w[3]),dict(logical=w[4],physical=w[5],size=w[6])])


def map_bad(c):
    m=c['map'];regions=[r for i,r in enumerate(m['regions']) if m['valid']&(1<<i)]
    if not regions or not m['generation'] or m['generation']!=c['source'][3] or c['src'] or not 0<c['src_size']<=1<<34:
        return True
    for r in regions:
        if not r['size'] or any(r[k]%4096 for k in ('logical','physical','size')):
            return True
        if r['logical']+r['size']>c['src_size'] or r['physical']+r['size']>1<<34:
            return True
    if len(regions)==2:
        a,b=regions
        for k in ('logical','physical'):
            if max(a[k],b[k])<min(a[k]+a['size'],b[k]+b['size']):
                return True
    return False


def physical(c,offset,size):
    if map_bad(c) or not size:
        return None
    m=c['map']
    matches=[r['physical']+offset-r['logical'] for i,r in enumerate(m['regions'])
             if m['valid']&(1<<i) and r['logical']<=offset and offset+size<=r['logical']+r['size']]
    return matches[0] if len(matches)==1 else None


def ledger_module(root):
    # Adapt the scope, not the reservation/event checks, of the reviewed model.
    text=(root/'tools/check_ntrpu_peer_shared.py').read_text(encoding='utf-8')
    def once(old,new):
        nonlocal text
        assert text.count(old)==1,old;text=text.replace(old,new)
    once('cases = 6 if full else 15','cases = 6 if full else 8')
    once('return (not s[0]', 'return (map_bad(c) or not s[0]')
    once("    if any(r['dst']", "    if physical(c,r['src'],r['size']) is None:\n        return 8\n    if any(r['dst']")
    once("new = lease(sg['CFG'][0][0]); cfg_error = bad_config(new, c)",
         "new = lease(sg['CFG'][0][0]); new['map']=region_map(sg['CFG'][0][1]); cfg_error = bad_config(new, c)")
    namespace=dict(__name__='regions_ledger',map_bad=map_bad,physical=physical,region_map=region_map)
    exec(compile(text,'frozen_regions_ledger','exec'),namespace)
    return namespace


def augment(text):
    maps={}
    for line in text.splitlines():
        p=line.split()
        if p[:2]==['RG','MAP']:
            key=tuple(p[2:4]);require(key not in maps,'duplicate map config trace');maps[key]=p[4]
    rows=[]
    for line in text.splitlines():
        p=line.split()
        if p[:2]==['SG','CFG']:
            key=tuple(p[2:4]);require(key in maps,'config without map');line+=' '+maps.pop(key)
        rows.append(line)
    require(not maps,'map without config')
    return '\n'.join(rows)


def extra_audit(text):
    vectors=0;expected_vector=0;maps={};cfgs={};pending_cfg={};forwards={};observed={}
    for line in text.splitlines():
        p=line.split()
        if p[:2]==['RG','LOOKUP']:
            index=int(p[2]);require(index==expected_vector,'lookup sequence');expected_vector+=1
            c=lease(p[3]);c['map']=region_map(p[4]);offset,size,bad,hit,address=map(int,p[5:])
            resolved=physical(c,offset,size)
            require((bad,hit,address)==(int(map_bad(c)),int(resolved is not None),resolved or 0),'lookup model mismatch')
            vectors+=1
        elif p[:2]==['RG','MAP']:
            maps[int(p[2]),int(p[3])]=region_map(p[4])
    # Configuration and forward traces may originate from independent SV blocks
    # on the same edge; group by timestamp before applying configuration.
    from collections import defaultdict
    groups=defaultdict(list)
    for line in text.splitlines():
        p=line.split()
        if p[:2] in (['SG','CFG'],['SG','FWD'],['RG','FWD']):groups[int(p[3]),int(p[2])].append(p)
    original_bad=__import__('check_ntrpu_peer_shared').bad_config
    for (epoch,ts),rows in sorted(groups.items()):
        for p in rows:
            if p[:2]==['SG','CFG']:
                c=lease(p[4]);c['map']=maps[ts,epoch]
                if not original_bad(c,cfgs.get(epoch)) and not map_bad(c):cfgs[epoch]=c
            elif p[:2]==['SG','FWD']:
                r=request(p[5]);key=epoch,ts,int(p[4]);require(key not in forwards,'duplicate forward')
                address=physical(cfgs[epoch],r['src'],r['size']);require(address is not None,'forward through unmapped gap')
                forwards[key]=int(p[5],16),address
            elif p[:2]==['RG','FWD']:
                key=epoch,ts,int(p[4]);require(key not in observed,'duplicate physical forward')
                observed[key]=int(p[5],16),int(p[6])
    require(vectors==278,'lookup vector coverage')
    require(forwards and forwards==observed,'resolved address/context mismatch')
    return dict(lookup_vectors=vectors,resolved_forwards=len(forwards))


def audit(text,root):
    require('RG ALL_PASS unit cases=8 vectors=278' in text,'region end marker')
    model=ledger_module(root);report=model['audit'](augment(text),False)
    return dict(ownership=report,**extra_audit(text))


def review(out):
    root=out.parent/'repo';r=json.loads((out/'result.json').read_text())
    require(r['status']=='EXPORTED_PENDING_REVIEW' and not r['timed_out'] and r['returncode']==0,'runner status')
    require(r['sources']==r['sources_after'] and all(sha(root/n)==h for n,h in r['sources'].items()),'source binding')
    for n,h in json.loads((out/'FILES.json').read_text()).items():require(sha(out/n)==h,'artifact hash '+n)
    text=(out/'simulation.log').read_text();report=audit(text,root)
    model=ledger_module(root);negative=model['negative_controls'](augment(text),False)
    for token in ('RG MAP ','RG FWD ','RG LOOKUP '):
        line=next(s for s in text.splitlines() if s.startswith(token))
        try:audit(text.replace(line+'\n','',1),root)
        except (ValueError,KeyError,AssertionError,IndexError):negative.append('drop_'+token.split()[1].lower())
        else:raise ValueError('missing region evidence accepted')
    line=next(s for s in text.splitlines() if s.startswith('RG FWD ') and int(s.split()[-1])>=1<<33)
    p=line.split();p[-1]=str(int(p[-1])&((1<<33)-1))
    try:audit(text.replace(line,' '.join(p),1),root)
    except (ValueError,KeyError,AssertionError,IndexError):negative.append('truncate_physical_stack_bit')
    else:raise ValueError('truncated stack bit accepted')
    return dict(status='REGIONS_UNIT_REVIEWED',result_sha256=sha(out/'result.json'),seconds=r['seconds'],
                **report,negative_controls=negative,physical_timing_qualified=False,native_hbm_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output),indent=2))
