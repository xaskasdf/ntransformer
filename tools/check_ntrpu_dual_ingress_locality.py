"""Independent data, lease and completion replay for the experimental receiver."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pattern(tag,beat):return sum((((tag*0x9e3779b9)^(beat*0x10203)^(j*0x1010101))&0xffffffff)<<(j*32) for j in range(16))
def specs(pressure_beats=64):
    return {
      1:{t:64 for t in range(100,116)},
      2:{t:pressure_beats for first in (200,220) for t in range(first,first+16)},
      3:dict([(300+2*j,9) for j in range(8)]+[(301,64)]),
      4:{400:8,401:8,402:1,403:1},5:{500:64,501:64},
      6:{600+j:(1,9,64)[(j//2)%3] for j in range(24)},7:{700:8,701:8},8:{},
      9:dict([(900+2*j,64) for j in range(6)]+[(901,64),(903,64)]),
      10:dict([(1001+2*j,64) for j in range(6)]+[(1000,64),(1002,64)]),
      11:dict([(1100+2*j,64) for j in range(6)]+[(1101,64),(1103,64)]),
      12:dict([(1201+2*j,64) for j in range(6)]+[(1200,64),(1202,64)]),
      13:{1300+j:(9 if (j//2)%4==0 else 64) for j in range(64)},
      14:{1500+j:(9 if (j//2)%4==0 else 64) for j in range(64)}}


def replay(lines,mode,pressure_beats=64):
    assert not any(re.search(r'^\s*(Fatal:|ERROR:)',x) for x in lines)
    groups={};case=None;final=None
    for line in lines:
        a=line.split()
        if len(a)<2 or a[0]!='DI':continue
        if a[1]=='CASE':case=int(a[3]);assert case not in groups;groups[case]=[]
        if a[1]=='ALL_PASS':assert final is None;final=a[2:]
        else:assert case is not None;groups[case].append(a[1:])
    assert list(groups)==list(range(1,15)) and final and final[:2]==[f'mode={mode}','cases=14']
    report={};total_simultaneous=0;rate=None
    for case,spec in specs(pressure_beats).items():
        records=groups[case];ev=defaultdict(list)
        for a in records:ev[a[0]].append(a[1:])
        cmds={int(v[2]):v for v in ev['CMD']};assert set(cmds)==set(spec) and len(ev['CMD'])==len(spec)
        for v in ev['CMD']:assert int(v[1])==int(v[2])%2 and int(v[3])==spec[int(v[2])]
        bytype={name:{int(v[2]):v for v in ev[name]} for name in ('AW','B','CPL')}
        for name,bytag in bytype.items():
            assert set(bytag)==set(spec) and len(ev[name])==len(spec)
            for lane in (0,1):assert [int(v[2]) for v in ev[name] if int(v[1])==lane]==[t for t in spec if t%2==lane]
        all_in=defaultdict(list);all_w=defaultdict(list)
        for name,dest in [('IN',all_in),('W',all_w)]:
            assert len(ev[name])==sum(spec.values())
            for v in ev[name]:dest[int(v[2])].append(v)
        for tag,beats in spec.items():
            assert len(all_in[tag])==len(all_w[tag])==beats
            for i,(inp,w) in enumerate(zip(all_in[tag],all_w[tag])):
                assert int(inp[1])==int(w[1])==tag%2 and int(inp[3])==int(w[3])==i
                assert int(inp[4],16)==int(w[4],16)==pattern(tag,i)
                assert int(cmds[tag][0])<int(inp[0])<=int(w[0])
            aw,b,cpl=[bytype[n][tag] for n in ('AW','B','CPL')]
            assert int(all_in[tag][-1][0])<int(aw[0])<=int(all_w[tag][0][0])
            assert int(all_w[tag][-1][0])<int(b[0])<int(cpl[0])
            assert int(b[3])==int(cpl[3])==(2 if tag==700 else 0)
        live={};generation=[0]*8;usage=[0]*8;peak=0
        assert len(ev['RETURN'])==len(spec)
        for a in records:
            name,v=a[0],a[1:]
            if name=='CMD':
                ts,lane,tag,beats,bank,gen=map(int,v)
                assert bank not in live and (bank>=4 or bank//2==lane) and gen==generation[bank]+1
                live[bank]=(lane,tag,gen);generation[bank]=gen;usage[bank]+=1;peak=max(peak,len(live))
            elif name=='RETURN':
                ts,lane,bank,gen=map(int,v);assert bank in live
                owner,tag,want=live[bank];assert (owner,want)==(lane,gen)
                assert ts==int(all_w[tag][-1][0])+5000;del live[bank]
        assert not live
        inputs_by_time=Counter(int(v[0]) for v in ev['IN']);simultaneous=sum(n==2 for n in inputs_by_time.values())
        assert all(n<=2 for n in inputs_by_time.values()) and (mode==0 or simultaneous==0)
        total_simultaneous+=simultaneous
        assert len(ev['PASS'])==1
        wanted=[case,len(spec),len(spec),peak,simultaneous,int(case in (7,8))]
        assert list(map(int,ev['PASS'][0]))==wanted,(case,ev['PASS'],wanted)
        if case==1:
            assert simultaneous>400 if mode==0 else simultaneous==0
            assert len(ev['RATE'])==1;control,beats,fill_cycles,drain_cycles=map(int,ev['RATE'][0])
            assert control==mode and beats==1024 and 0<fill_cycles<drain_cycles
            elapsed=max(inputs_by_time)-min(inputs_by_time)+5000
            rate=dict(fill_cycles=fill_cycles,drain_cycles=drain_cycles,payload_span_ps=elapsed,
                      payload_gb_s=1024*64*1000/elapsed,completion_gb_s=1024*64/(drain_cycles*5),simultaneous_cycles=simultaneous)
        if case==2:
            assert peak==8 and min(usage)>=2 and len(ev['FULL'])==len(ev['OPEN0'])==2
            for full,opening in zip(ev['FULL'],ev['OPEN0']):assert int(full[0])<int(opening[0]) and int(opening[1])>int(opening[2])
        if case==3:
            assert len(ev['INDEPENDENT'])==1;time=int(ev['INDEPENDENT'][0][0])
            assert all(int(bytype['CPL'][t][0])<=time for t in spec if t%2==0)
            assert not any(int(v[0])<time and int(v[1])==1 for v in ev['IN'])
        if case==4:
            assert len(ev['FENCE'])==len(ev['FENCED'])==len(ev['FENCE_BLOCKED'])==1
            ft=int(ev['FENCE'][0][0]);done=int(ev['FENCED'][0][0]);assert ft<int(ev['FENCE_BLOCKED'][0][0])<done
            assert ev['FENCE'][0][1]==ev['FENCED'][0][1]=='77'
            assert all(int(bytype['CPL'][t][0])<done for t in (400,401))
            assert all(int(cmds[t][0])>done for t in (402,403))
        if case==5:
            assert len(ev['STOP'])==1;time=int(ev['STOP'][0][0])
            assert all(int(v[0])<time for v in ev['CMD'])
            assert all(int(bytype['CPL'][t][0])>time and int(all_in[t][-1][0])>time for t in spec)
        report[case]=dict(descriptors=len(spec),beats=sum(spec.values()),peak_banks=peak,bank_allocations=usage,simultaneous_cycles=simultaneous)
    assert final[2]==f'simultaneous={total_simultaneous}'
    return dict(cases=report,descriptors=249,verified_bytes=sum(sum(s.values()) for s in specs(pressure_beats).values())*64,
                pressure_frame_beats=pressure_beats,serial=mode,rate=rate,total_simultaneous_cycles=total_simultaneous)


def ports(lines,mode=0,locality=1):
    turns={};stats={};last={};case=None;events=defaultdict(Counter);inputs=defaultdict(dict);leases={}
    for line in lines:
        a=line.split()
        if a[:2]==['DI','CASE']:case=int(a[3])
        elif a[:2]==['DI','CMD']:
            leases[(case,int(a[4]))]=int(a[6])
        elif a[:2] in (['DI','IN'],['DI','W']):
            ts,lane,tag=map(int,a[2:5]);events[case][(a[1],lane)]+=1
            if a[1]=='IN':inputs[(case,ts)][lane]=leases[(case,tag)]
    for line in lines:
        if not line.startswith('CP PORT '):continue
        ts,case,wr,wg,wa,rr,rg,ra,used=map(int,line.split()[2:])
        assert ts>last.get(case,-1);last[case]=ts
        accepted_inputs=inputs.pop((case,ts),{})
        assert wa==sum(1<<c for c in accepted_inputs)
        for lane,bank in accepted_inputs.items():
            assert (wg>>lane&1)==(bank//2 if bank<4 else bank%2)
        turn=turns.setdefault(case,[[0,0],[0,0]])
        st=stats.setdefault(case,dict(write_conflicts=0,read_conflicts=0,write_grants=[0,0],read_grants=[0,0]))
        for kind,(request,groups,accepted) in enumerate(((wr,wg,wa),(rr,rg,ra))):
            expected=0
            for group in (0,1):
                eligible=[c for c in (0,1) if request>>c&1 and (groups>>c&1)==group]
                if len(eligible)==2:st[('write_conflicts','read_conflicts')[kind]]+=1
                if eligible:
                    chosen=turn[kind][group] if turn[kind][group] in eligible else eligible[0]
                    expected |= 1<<chosen;turn[kind][group]=1-chosen
                    st[('write_grants','read_grants')[kind]][chosen]+=1
            assert accepted==expected,(ts,case,kind,accepted,expected)
    assert not inputs
    assert set(stats)==set(range(1,8))|set(range(9,15))
    for case,st in stats.items():
        for lane in (0,1):
            assert st['write_grants'][lane]==events[case][('IN',lane)]
            assert st['read_grants'][lane]==events[case][('W',lane)]
    for case in (9,10):
        assert (stats[case]['write_conflicts']==0 if mode else stats[case]['write_conflicts']>=0)
        assert stats[case]['read_conflicts']>=0
        assert all(n>0 for n in stats[case]['write_grants']+stats[case]['read_grants'])
    return stats


def allocation(lines,locality):
    assert [x for x in lines if x.startswith('LP POLICY ')]==[f'LP POLICY {locality}']
    events=defaultdict(lambda:defaultdict(list));case=None
    for line in lines:
        a=line.split()
        if a[:2]==['DI','CASE']:case=int(a[3])
        elif a[:2]==['LP','ARB']:
            ts,c,*v=map(int,a[2:]);assert c==case
            events[c][ts].append(('ARB',v))
        elif a[:2] in (['DI','CMD'],['DI','RETURN']):
            ts,*v=map(int,a[2:]);events[case][ts].append((a[1],v))
    reports={}
    for case,timeline in events.items():
        live={};generation=[0]*8;turn=0;counts=dict(allocations=0,local_reserves=0,foreign_reserves=0,bank_allocations=[0]*8)
        for ts,records in sorted(timeline.items()):
            arb=[v for kind,v in records if kind=='ARB'];cmds=[v for kind,v in records if kind=='CMD'];returns=[v for kind,v in records if kind=='RETURN']
            assert len(arb)<=1 and len(cmds)<=1 and not(cmds and returns)
            expected=[]
            if arb:
                request,granted,used,retired,blocked,chosen=arb[0]
                assert used==sum(1<<b for b in live) and retired==0
                if not blocked:
                    for lane in (turn,1-turn):
                        own=[lane*2,lane*2+1]
                        preferred=[b for b in range(4,8) if b%2==lane] if locality else []
                        order=list(dict.fromkeys(own+preferred+list(range(4,8))))
                        candidates=[b for b in order if b not in live]
                        if request>>lane&1 and candidates:
                            expected=[lane,candidates[0]];break
                assert granted==(1<<expected[0] if expected else 0),(case,ts,'channel',granted,expected)
                assert chosen==(expected[1] if expected else 0),(case,ts,'bank',chosen,expected)
            assert bool(cmds)==bool(expected),(case,ts,'command without reviewed grant')
            for lane,tag,beats,bank,gen in cmds:
                assert [lane,bank]==expected and bank not in live and gen==generation[bank]+1
                generation[bank]=gen;live[bank]=(lane,gen);turn=1-lane;counts['allocations']+=1;counts['bank_allocations'][bank]+=1
                if bank>=4:counts['local_reserves' if bank%2==lane else 'foreign_reserves']+=1
            for lane,bank,gen in returns:
                assert live[bank]==(lane,gen);del live[bank]
        assert not live;reports[case]=counts
    assert sum(v['allocations'] for v in reports.values())==249
    for case in (9,10,11,12):
        assert min(reports[case]['bank_allocations'])==1 and reports[case]['foreign_reserves']>=2
    return reports


def workloads(lines):
    starts={};fills={};rates={}
    for line in lines:
        a=line.split()
        if a[:2]==['LP','BEGIN']:
            ts,case=map(int,a[2:]);assert case not in starts;starts[case]=ts
        elif a[:2]==['LP','FILLED']:
            ts,case=map(int,a[2:]);assert case not in fills;fills[case]=ts
        elif a[:2]==['LP','RATE']:
            ts,case,beats,fill,drain=map(int,a[2:]);assert case not in rates
            assert 0<fill<drain and fills[case]-starts[case]==fill*5000 and ts-starts[case]==drain*5000
            assert beats==(512 if case<=12 else 3216)
            rates[case]=dict(beats=beats,bytes=beats*64,fill_cycles=fill,drain_cycles=drain,completion_gb_s=beats*64/(drain*5))
    assert set(starts)==set(fills)==set(rates)=={11,12,13,14}
    return rates


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='dual_ingress_locality' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['sources']==r['sources_after'] and r['returncode']==0 and not r['timed_out']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    if r['stage']=='synth':
        table={k:int(v) for line in (out/'suite/structure.tsv').read_text().splitlines() for k,v in [line.split('\t')]}
        assert table['LD*']==table['blackboxes']==0 and table['RAMB36*']+table['RAMB18*']>0
        return dict(status='LOCALITY_DUAL_INGRESS_SYNTHESIS_REVIEWED',serial=r['serial'],prefer_local=r['prefer_local'],structure=table,
                    physical_timing_qualified=False,result_sha256=sha(out/'result.json'),seconds=r['seconds'])
    lines=(out/'suite/simulation.log').read_text().splitlines();result=replay(lines,r['serial']);result['ports']=ports(lines,r['serial'],r['prefer_local']);result['allocation']=allocation(lines,r['prefer_local']);result['workloads']=workloads(lines);controls=[]
    for prefix in ('DI CMD ','DI IN ','DI AW ','DI W ','DI B ','DI CPL ','DI RETURN ','DI FULL ',
                   'DI INDEPENDENT ','DI FENCE ','DI FENCED ','DI STOP ','DI PASS ','DI ALL_PASS '):
        i=next(i for i,v in enumerate(lines) if v.startswith(prefix))
        try:replay(lines[:i]+lines[i+1:],r['serial'])
        except (AssertionError,KeyError,IndexError,ValueError):controls.append('remove '+prefix.strip())
        else:raise AssertionError('Insensitive checker '+prefix)
    for prefix,field in (('DI IN ',6),('DI W ',6),('DI CMD ',7),('DI RETURN ',5)):
        i=next(i for i,v in enumerate(lines) if v.startswith(prefix));a=lines[i].split();a[field]=format(int(a[field],16)^1,'x')
        mutant=list(lines);mutant[i]=' '.join(a)
        try:replay(mutant,r['serial'])
        except (AssertionError,KeyError,IndexError,ValueError):controls.append('corrupt '+prefix.strip())
        else:raise AssertionError('Insensitive checker '+prefix)
    for change in ('remove_port','corrupt_write_grant','corrupt_read_grant'):
        i=next(i for i,v in enumerate(lines) if v.startswith('CP PORT ') and int(v.split()[3])==9)
        mutant=list(lines)
        if change=='remove_port':del mutant[i]
        else:
            a=mutant[i].split();field=6 if change=='corrupt_write_grant' else 9
            a[field]=str(int(a[field])^1);mutant[i]=' '.join(a)
        try:ports(mutant,r['serial'],r['prefer_local'])
        except (AssertionError,KeyError,IndexError,ValueError):controls.append(change)
        else:raise AssertionError('Insensitive port checker '+change)
    for change in ('remove_allocation','wrong_bank','wrong_policy'):
        mutant=list(lines)
        if change=='wrong_policy':
            i=next(i for i,v in enumerate(mutant) if v.startswith('LP POLICY '));mutant[i]=f"LP POLICY {1-r['prefer_local']}"
        else:
            i=next(i for i,v in enumerate(mutant) if v.startswith('LP ARB ') and int(v.split()[5])!=0)
            if change=='remove_allocation':del mutant[i]
            else:
                a=mutant[i].split();a[9]=str(int(a[9])^1);mutant[i]=' '.join(a)
        try:allocation(mutant,r['prefer_local'])
        except (AssertionError,KeyError,IndexError,ValueError):controls.append(change)
        else:raise AssertionError('Insensitive allocation checker '+change)
    result.update(prefer_local=r['prefer_local'],status='LOCALITY_DUAL_INGRESS_FUNCTIONAL_PASS',negative_trace_controls=controls,
                  result_sha256=sha(out/'result.json'),seconds=r['seconds'],physical_timing_qualified=False)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
