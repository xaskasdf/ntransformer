"""Replay pressure, lease ownership and complete data independently of TB PASS."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_hbm_native_transport import address, descriptor, pattern, configuration


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def replay(lines,beats=8):
    assert beats in (8,64)
    assert not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )',x,re.I) for x in lines)
    events=defaultdict(list)
    for line in lines:
        a=line.split()
        if len(a)>2 and a[0] in ('TE','NC','NR'): events[' '.join(a[:2])].append(a[2:])
    spec={tag:beats for first in (100,200) for tag in range(first,first+24)}
    order={lane:[t for t in spec if t%2==lane] for lane in (0,1)}
    bytag={}
    for name in ('SUB','FETCH','AW','B','CPL'):
        values=events['TE '+name];assert len(values)==48,name
        for lane in (0,1): assert [int(v[2]) for v in values if int(v[1])==lane]==order[lane]
        bytag[name]={int(v[2]):v for v in values};assert len(bytag[name])==48
    parent_writes={}
    for tag,beats in spec.items():
        for name in ('SUB','FETCH'): assert int(bytag[name][tag][3],16)==descriptor(tag,beats)
        assert int(bytag['AW'][tag][3],16)==address(tag) and int(bytag['AW'][tag][4])==beats
        assert int(bytag['B'][tag][3])==int(bytag['CPL'][tag][3])==0
        times=[int(bytag[n][tag][0]) for n in ('SUB','FETCH','AW','B','CPL')];assert times==sorted(times)
        ins=[v for v in events['TE IN'] if int(v[1])==tag]
        writes=[v for v in events['TE W'] if int(v[2])==tag]
        parent_writes[tag]=writes
        assert len(ins)==len(writes)==beats
        for i,(inp,w) in enumerate(zip(ins,writes)):
            assert int(inp[2])==int(w[3])==i and int(w[1])==tag%2
            assert int(inp[3],16)==int(w[5],16)==pattern(tag,i)
            assert int(w[4])==(i==beats-1) and int(inp[0])<=int(w[0])<=int(bytag['B'][tag][0])
    assert len(events['TE IN'])==len(events['TE W'])==48*beats
    memory={};rtime={}
    for lane in (0,1):
        aw=[v for v in events['NC AW'] if int(v[1])==lane]
        w=[v for v in events['NC W'] if int(v[1])==lane]
        b=[v for v in events['NC B'] if int(v[1])==lane]
        ar=[v for v in events['NC AR'] if int(v[1])==lane]
        r=[v for v in events['NC R'] if int(v[1])==lane]
        fragments=beats//8
        assert len(aw)==len(b)==len(ar)==24*fragments and len(w)==len(r)==24*beats*2
        for j,tag in enumerate(order[lane]):
            for f in range(fragments):
                av,bv,rv=aw[j*fragments+f],b[j*fragments+f],ar[j*fragments+f]
                assert int(av[2],16)==int(rv[2],16)==address(tag)+f*512
                assert int(av[3])==int(rv[3])==16
                assert int(av[4])==int(bv[2])==int(bv[3])==0 and int(rv[4])==tag%64
                assert int(av[0])>=int(bytag['AW'][tag][0])
                assert int(bv[0])<=int(bytag['B'][tag][0]) and int(rv[0])>int(bytag['CPL'][tag][0])
            for beat in range(beats*2):
                av,bv,rv=aw[j*fragments+beat//16],b[j*fragments+beat//16],ar[j*fragments+beat//16]
                ww,rr=w[j*beats*2+beat],r[j*beats*2+beat]
                wanted=(pattern(tag,beat//2)>>(256*(beat%2)))&((1<<256)-1)
                assert int(ww[2],16)==int(rr[2],16)==wanted
                assert int(ww[3],16)==0xffffffff and int(ww[4])==int(rr[5])==(beat%16==15)
                assert int(rr[3])==tag%64 and int(rr[4])==0
                parent_w=parent_writes[tag][beat//2]
                assert int(parent_w[0])<=int(ww[0])<=int(bv[0])
                assert int(rr[0])>int(rv[0]);rtime[tag]=int(rr[0])
                addr=address(tag)+beat*32;assert addr not in memory;memory[addr]=wanted
        read_times=[int(x[0]) for x in r]
        assert read_times==sorted(read_times)
        for v in [v for v in events['NC RS'] if int(v[1])==lane]:
            after=r[bisect_right(read_times,int(v[0]))];assert after[2:]==v[2:]
    assert len(events['NC VERIFIED'])==48
    assert {int(v[2]):int(v[3]) for v in events['NC VERIFIED']}=={t:beats*64 for t in spec}
    assert all(int(v[1])==int(v[2])%2 and int(v[0])>=rtime[int(v[2])] for v in events['NC VERIFIED'])

    # Reconstruct ownership from actual allocator commits, not aggregate counters.
    leases={};generation=[0]*8;allocation_counts=[0]*8;allocated_tags=set();lasts={}
    waves={1:set(),2:set()};peak=0
    timeline=[]
    for name in ('NR ALLOC','NR LAST','NR RETURN','TE S'):
        timeline += [(int(v[0]),name,v) for v in events[name]]
    assert len(events['NR ALLOC'])==len(events['NR LAST'])==len(events['NR RETURN'])==48
    # TE S samples pre-NBA state on the edge; allocator events commit afterward.
    for stamp,name,v in sorted(timeline,key=lambda x:(x[0],0 if x[1]=='TE S' else 1,x[1])):
        if name=='TE S':
            assert int(v[6],16)==sum(1<<bank for bank in leases),('ownership mask',stamp)
            if int(v[4]): assert not leases
            continue
        if name=='NR ALLOC':
            _,lane,tag,bank,gen=map(int,v)
            assert bank not in leases and tag in spec and tag not in allocated_tags
            assert lane==tag%2 and (bank>=4 or bank//2==lane)
            assert gen==generation[bank]+1
            assert int(bytag['SUB'][tag][0])<=stamp<=int(bytag['FETCH'][tag][0])
            generation[bank]=gen;allocation_counts[bank]+=1;allocated_tags.add(tag)
            leases[bank]=(lane,tag,gen);waves[1 if tag<200 else 2].add(bank)
            peak=max(peak,len(leases))
        else:
            _,lane,bank,gen=map(int,v);assert bank in leases
            owner,tag,generation_value=leases[bank];assert (lane,gen)==(owner,generation_value)
            key=(bank,gen)
            if name=='NR LAST':
                assert key not in lasts;lasts[key]=stamp
                assert max(int(x[0]) for x in events['TE IN'] if int(x[1])==tag)<=stamp
            else:
                assert key in lasts and stamp-lasts[key]==5000
                del leases[bank]
    assert not leases and peak==8 and min(allocation_counts)>=2
    assert all(banks==set(range(8)) for banks in waves.values())

    for name in ('CLOSED','FULL','OPEN0','ISOLATED','OPEN1','EMPTY'):
        assert len(events['NR '+name])==2,name
        assert [int(v[1]) for v in events['NR '+name]]==[1,2]
    for i in (0,1):
        chain=[events['NR '+n][i] for n in ('CLOSED','FULL','OPEN0','ISOLATED','OPEN1','EMPTY')]
        times=[int(v[0]) for v in chain];assert times==sorted(times)
        closed,full,opened0,isolated,opened1,empty=times
        assert int(chain[1][2],16)==255 and opened0-full>=495000
        assert int(chain[3][2])>0 and int(chain[3][3])==0
        samples=[int(v[6],16) for v in events['TE S'] if full<int(v[0])<opened0]
        assert len(samples)>=99 and all(v==255 for v in samples)
        for lane,opened in ((0,opened0),(1,opened1)):
            assert not any(closed<=int(v[0])<opened and int(v[1])==lane for v in events['NC W'])
            assert not any(closed<=int(v[0])<opened and int(v[1])==lane for v in events['TE CPL'])
        assert any(opened0<int(v[0])<opened1 and int(v[1])==0 for v in events['TE CPL'])
    assert [list(map(int,v)) for v in events['NC CASE_PASS']]==[[1,12,12,12,12],[2,24,24,24,24]]
    assert len(events['TE DRAIN'])==2
    for name in ('NC RELEASE','NC BOOT','NC THERMAL','NC COUNTS','NR ALL_PASS'):assert len(events[name])==1,name
    assert events['NC BOOT'][0][1:]==['1','1'] and '1' not in events['NC THERMAL'][0][1:]
    assert events['NR ALL_PASS'][0][0]=='cases=2'
    counts={k:int(v) for token in events['NC COUNTS'][0] for k,v in [token.split('=')]}
    for k in ('sub','cpl'):assert counts[k]==48
    for k in ('aw','b','ar'):assert counts[k]==48*beats//8
    assert counts['w']==counts['r']==len(memory)==48*beats*2
    assert counts['peak_banks']==8 and counts['used_union']==255
    assert counts['r_stalls']==len(events['NC RS'])>0 and counts['prestart']>0
    assert counts['thermal_unknown']>0
    final=events['TE S'][-1];assert int(final[3])==0 and int(final[4])==1 and int(final[5],16)==int(final[6],16)==0
    return dict(cases=2,descriptors=48,frame_beats=beats,write_beats=48*beats*2,read_beats=48*beats*2,verified_bytes=48*beats*64,
                pool_peak=peak,bank_allocations=allocation_counts,reserve_banks=[4,5,6,7],
                saturation_windows=2,bank_reuse_without_reset=True,counts=counts,
                finish_ps=int(events['NR ALL_PASS'][0][1].split('=')[1]),
                native_hbm=False,thermal_validated=False,hardware_validated=False)


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='hbm_reserve' and r['stage']=='preflight' and not r['native_hbm']
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out']
    assert r['sources']==r['sources_after']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    comp=json.loads((out/'suite/COMPOSITION.json').read_text())
    assert comp['bench_sha256']==sha(out/'suite/generated/tb_ntrpu_hbm_reserve.sv')
    assert comp['testbench_write_gate'] and not comp['production_rtl_modified']
    assert all(sha(out/'suite/generated'/(n+'.sv'))==h for n,h in comp['frozen_logic'].items())
    beats=comp.get('frame_beats',8);assert r.get('frame_beats',8)==beats
    lines=(out/'suite/simulation.log').read_text().splitlines();result=replay(lines,beats)
    controls=[]
    for prefix in ('TE SUB ','TE FETCH ','TE IN ','TE AW ','TE W ','TE B ','TE CPL ',
                   'NC AW ','NC W ','NC B ','NC AR ','NC R ','NC VERIFIED ',
                   'NR ALLOC ','NR LAST ','NR RETURN ','NR CLOSED ','NR FULL ','NR OPEN0 ',
                   'NR ISOLATED ','NR OPEN1 ','NR EMPTY ','NR ALL_PASS '):
        index=next(i for i,v in enumerate(lines) if v.startswith(prefix))
        try: replay(lines[:index]+lines[index+1:],beats)
        except (AssertionError,StopIteration,KeyError,IndexError,ValueError):controls.append('remove '+prefix.strip())
        else: raise AssertionError('Insensitive checker: '+prefix)
    for prefix,field in (('NC W ',4),('NC R ',4),('NR ALLOC ',6),('NR RETURN ',5)):
        i=next(i for i,v in enumerate(lines) if v.startswith(prefix));parts=lines[i].split()
        parts[field]=format(int(parts[field],16)^1,'x');mutant=list(lines);mutant[i]=' '.join(parts)
        try:replay(mutant,beats)
        except (AssertionError,StopIteration,KeyError,IndexError,ValueError):controls.append('corrupt '+prefix.strip())
        else:raise AssertionError('Insensitive checker: '+prefix)
    return dict(result,status='RESERVE_PRESSURE_PREFLIGHT_PASS',seconds=r['seconds'],
                negative_trace_controls=controls,rtl_fault_injections=0,result_sha256=sha(out/'result.json'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);a=ap.parse_args()
    print(json.dumps(review(a.output),indent=2))
