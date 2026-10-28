"""Independently replay descriptor-to-native-HBM write/readback evidence."""
import argparse, collections, hashlib, json, re
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def expected_cases():
    return [(10+j,(1,8,9,64)[j//2]) for j in range(8)]+[(20,8),(21,9)]+[(30+j,8) for j in range(12)]+[(50,8),(51,8)]
def address(tag): return (0x200000000 if tag%2 else 0)+0x10000+(tag//2)*4096
def pattern(tag,beat):
    return sum((((tag*0x9e3779b9)^(beat*0x10203)^(lane*0x01010101))&0xffffffff)<<(lane*32) for lane in range(16))
def descriptor(tag,beats):
    return (0x76543210<<200)|(tag<<136)|(address(tag)<<72)|((0x800000000+tag*4096)<<8)|(beats-1)

def base(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='hbm_native_transport' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert not r['timed_out'] and r['returncode']==0 and r['sources']==r['sources_after']
    files=json.loads((out/'FILES.json').read_text())
    assert files and all(sha(out/n)==h for n,h in files.items())
    return r

def configuration(path):
    return {a:b for line in path.read_text().splitlines() if '\t' in line for a,b in [line.split('\t',1)] if a.startswith('CONFIG.')}

def compile_review(out,r):
    assert r['stage']=='compile'
    if r.get('operation')=='verify_completed_snapshot_after_parent_timeout':
        recovery=json.loads((out/'RECOVERY.json').read_text());parent=json.loads((out/'TIMED_OUT_PARENT.json').read_text())
        assert recovery['status']=='SNAPSHOT_LOAD_VERIFIED' and not recovery['original_parent_accepted']
        assert recovery['donor_unchanged'] and recovery['snapshot_unchanged']
        assert parent['status']=='FAILED_OR_CHANGED' and parent['timed_out'] and parent['sources']==r['sources']
        assert sha(out/'TIMED_OUT_PARENT.json')==r['prior_result_sha256']==recovery['prior_result_sha256']
        assert sha(out/'SNAPSHOT_INPUTS.json')==r['prior_manifest_sha256']==recovery['snapshot_manifest_sha256']
        log=(out/'load_evidence/load.log').read_text()
        assert 'NTRPU_NATIVE_SNAPSHOT_LOAD_COMPLETE time=0' in log
        assert not re.search(r'^\s*(ERROR:|Fatal:)',log,re.M)
    else:
        assert 'NTRPU_NATIVE_TRANSPORT_COMPLETE compile' in (out/'console.log').read_text()
    c=json.loads((out/'suite/COMPOSITION.json').read_text());b=json.loads((out/'suite/BINDING.json').read_text())
    assert c['cold_epochs']==1 and c['expected_descriptors']==24 and c['cases']==4
    assert c['native_binding_sha256']==b['binding_sha256']==sha(out/'suite/ntrpu_hbm_native_rw.sv')
    assert c['bench_sha256']==sha(out/'suite/generated/tb_ntrpu_hbm_native_transport.sv')
    assert all(sha(out/'suite/generated'/(n+'.sv'))==h for n,h in c['frozen_logic'].items())
    cfg=configuration(out/'suite/native_hbm_config.tsv')
    assert cfg['CONFIG.USER_HBM_DENSITY']=='16GB' and cfg['CONFIG.USER_AXI_CLK_FREQ']=='450'
    assert cfg['CONFIG.USER_AXI_CLK1_FREQ']=='450'
    assert cfg['CONFIG.USER_SWITCH_ENABLE_00']=='TRUE' and cfg['CONFIG.USER_SWITCH_ENABLE_01']=='TRUE'
    assert cfg['CONFIG.USER_HBM_STACK']=='2'
    assert all(cfg[f'CONFIG.USER_MC{i}_USER_PARITY_EN']=='false' for i in range(16))
    assert all(cfg[f'CONFIG.USER_MC{i}_{key}']=='true' for i in range(16) for key in ('DM_EN','EN_DATA_MASK'))
    snapshot=json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
    assert len(snapshot)>=15 and sum(v['bytes'] for v in snapshot.values())>100_000_000
    for i in (0,1):
        name=f'xpm_internal_config_file_sim_{i}.mem'
        assert snapshot[name]['sha256']==sha(out/'suite/model_inputs'/name)
    return dict(status='NATIVE_TRANSPORT_SNAPSHOT_REVIEWED',native_hbm=True,simulation_executed=False,
                frozen_logic_count=len(c['frozen_logic']),snapshot_files=len(snapshot),configuration_sha256=sha(out/'suite/native_hbm_config.tsv'),
                result_sha256=sha(out/'result.json'),snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'))

def replay(lines):
    assert not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )',x,re.I) for x in lines)
    records=[x.split() for x in lines if x.startswith(('TE ','NC '))]
    te=collections.defaultdict(list);nc=collections.defaultdict(list)
    for a in records:(te if a[0]=='TE' else nc)[a[1]].append(a[2:])
    spec=dict(expected_cases());assert len(spec)==24
    order={lane:[t for t in spec if t%2==lane] for lane in (0,1)}
    events={}
    for name in ('SUB','FETCH','AW','B','CPL'):
        values=te[name];assert len(values)==24,(name,len(values))
        for lane in (0,1):assert [int(v[2]) for v in values if int(v[1])==lane]==order[lane],name
        events[name]={int(v[2]):v for v in values}
        assert len(events[name])==24
    for tag,beats in spec.items():
        for name in ('SUB','FETCH'):assert int(events[name][tag][3],16)==descriptor(tag,beats)
        aw=events['AW'][tag];assert int(aw[3],16)==address(tag) and int(aw[4])==beats
        assert int(events['B'][tag][3])==int(events['CPL'][tag][3])==0
        times=[int(events[k][tag][0]) for k in ('SUB','FETCH','AW','B','CPL')]
        assert times==sorted(times)
        ins=[v for v in te['IN'] if int(v[1])==tag]
        writes=[v for v in te['W'] if int(v[2])==tag]
        assert len(ins)==len(writes)==beats
        for beat,(inp,w) in enumerate(zip(ins,writes)):
            assert int(inp[2])==int(w[3])==beat and int(w[1])==tag%2
            assert int(inp[3],16)==int(w[5],16)==pattern(tag,beat)
            assert int(w[4])==(beat==beats-1) and int(inp[0])<=int(w[0])<=int(events['B'][tag][0])
    memory={};last_b={};last_w={};read_times={};write_fragments=read_fragments=0
    for lane in (0,1):
        aw=[v for v in nc['AW'] if int(v[1])==lane];w=[v for v in nc['W'] if int(v[1])==lane];b=[v for v in nc['B'] if int(v[1])==lane]
        ar=[v for v in nc['AR'] if int(v[1])==lane];rr=[v for v in nc['R'] if int(v[1])==lane]
        assert len(aw)==len(b)==len(ar)==sum((spec[t]*2+15)//16 for t in order[lane])
        assert len(w)==len(rr)==sum(spec[t]*2 for t in order[lane])
        wi=ri=0;written=collections.Counter();read=collections.Counter()
        base_tag={address(t):t for t in order[lane]}
        # Write and read may partition a parent burst differently. Verify exact
        # contiguous byte coverage independently, never zip AW against AR.
        for av,bv in zip(aw,b):
            addr=int(av[2],16);tag=base_tag[addr&~4095];off=written[tag];count=int(av[3])
            assert addr==address(tag)+off*32 and 1<=count<=16 and off+count<=spec[tag]*2
            assert int(av[4])==int(bv[2])==0 and int(bv[3])==0
            assert int(av[0])>=int(events['AW'][tag][0])
            for beat in range(count):
                ww=w[wi];half=off+beat
                wanted=(pattern(tag,half//2)>>(256*(half%2)))&((1<<256)-1)
                assert int(ww[2],16)==wanted and int(ww[3],16)==0xffffffff
                assert int(ww[4])==(beat+1==count)
                assert int(ww[0])<=int(bv[0]) and int(av[0])<=int(bv[0])
                source_w=[v for v in te['W'] if int(v[2])==tag][half//2]
                assert int(ww[0])>=int(source_w[0])
                assert addr+beat*32 not in memory
                memory[addr+beat*32]=int(ww[2],16)
                last_w[tag]=int(ww[0]);wi+=1
            written[tag]+=count;last_b[tag]=int(bv[0]);write_fragments+=1
        for rv in ar:
            addr=int(rv[2],16);tag=base_tag[addr&~4095];off=read[tag];count=int(rv[3])
            assert addr==address(tag)+off*32 and 1<=count<=16 and off+count<=spec[tag]*2
            assert int(rv[4])==tag%64 and int(rv[0])>int(events['CPL'][tag][0])
            for beat in range(count):
                rrr=rr[ri];half=off+beat
                wanted=(pattern(tag,half//2)>>(256*(half%2)))&((1<<256)-1)
                assert int(rrr[2],16)==memory[addr+beat*32]==wanted
                assert int(rrr[3])==tag%64 and int(rrr[4])==0 and int(rrr[5])==(beat+1==count)
                assert int(rrr[0])>int(rv[0])
                read_times[tag]=int(rrr[0]);ri+=1
            read[tag]+=count;read_fragments+=1
        assert dict(written)==dict(read)=={t:spec[t]*2 for t in order[lane]}
        for stalled in [v for v in nc['RS'] if int(v[1])==lane]:
            after=next(v for v in rr if int(v[0])>int(stalled[0]))
            assert after[2:]==stalled[2:]
    assert all(last_b[t]<=int(events['B'][t][0]) for t in spec)
    verified=nc['VERIFIED'];assert len(verified)==24
    assert {int(v[2]):int(v[3]) for v in verified}=={t:n*64 for t,n in spec.items()}
    assert all(int(v[1])==int(v[2])%2 for v in verified)
    assert all(int(v[0])>=read_times[int(v[2])] for v in verified)
    assert len(te['FENCE'])==len(te['FENCED'])==len(nc['FENCE_BLOCKED'])==1
    ft=int(te['FENCE'][0][0]);done=int(te['FENCED'][0][0]);assert te['FENCE'][0][1]==te['FENCED'][0][1]=='77'
    assert ft<int(nc['FENCE_BLOCKED'][0][0])<done
    assert all(int(events['CPL'][t][0])<=done for t in spec if int(events['SUB'][t][0])<ft)
    assert [list(map(int,v)) for v in nc['CASE_PASS']]==[[1,4,4,4,4],[2,5,5,5,5],[3,11,11,11,11],[4,12,12,12,12]]
    assert len(nc['RELEASE'])==len(nc['BOOT'])==len(nc['THERMAL'])==len(nc['ALL_PASS'])==len(nc['COUNTS'])==1
    assert nc['BOOT'][0][1:]==['1','1'] and '1' not in nc['THERMAL'][0][1:]
    assert nc['ALL_PASS'][0][0]=='cases=4'
    counts={k:int(v) for token in nc['COUNTS'][0] for k,v in [token.split('=')]}
    assert counts['sub']==counts['cpl']==24 and counts['aw']==counts['b']==counts['ar']==write_fragments==41
    assert counts['w']==len(nc['W'])==counts['r']==len(nc['R'])==586
    assert counts['r_stalls']==len(nc['RS'])>0 and counts['cpl_stalls']>0 and counts['prestart']>0
    assert counts['thermal_unknown']>0
    assert len(te['DRAIN'])==4 and len(memory)==586
    masks=[int(v[6],16) for v in te['S']];peak=max(v.bit_count() for v in masks);used=0
    for value in masks:used|=value
    if 'peak_banks' in counts:assert counts['peak_banks']==peak and counts['used_union']==used
    return dict(descriptors=24,cases=4,native_write_fragments=41,native_read_fragments=41,
                write_beats=586,read_beats=586,verified_bytes=586*32,counts=counts,
                start_ps=int(nc['BOOT'][0][0]),finish_ps=int(nc['ALL_PASS'][0][1].split('=')[1]),
                pool_peak_banks=peak,pool_used_mask=used,
                thermal_validated=False,hardware_validated=False)

def traffic_review(out,r):
    path=out/('suite/simulation.log' if r['stage']=='preflight' else 'evidence/simulate.log')
    lines=path.read_text().splitlines();result=replay(lines)
    controls=[]
    for prefix in ('TE SUB ','TE FETCH ','TE IN ','TE AW ','TE W ','TE B ','TE CPL ','TE FENCE ','TE FENCED ',
                   'NC AW ','NC W ','NC B ','NC AR ','NC R ','NC VERIFIED ','NC CASE_PASS ','NC BOOT ','NC ALL_PASS ','NC COUNTS '):
        index=next(i for i,v in enumerate(lines) if v.startswith(prefix));mutant=lines[:index]+lines[index+1:]
        try:replay(mutant)
        except (AssertionError,StopIteration,KeyError,IndexError,ValueError):controls.append('remove '+prefix.strip())
        else:raise AssertionError('Insensitive checker: '+prefix)
    for prefix,field in [('NC W ',4),('NC R ',4),('NC AR ',4),('TE SUB ',5),('NC THERMAL ',3)]:
        index=next(i for i,v in enumerate(lines) if v.startswith(prefix));parts=lines[index].split();parts[field]='1' if prefix=='NC THERMAL ' else format(int(parts[field],16)^1,'x')
        mutant=list(lines);mutant[index]=' '.join(parts)
        try:replay(mutant)
        except (AssertionError,StopIteration,KeyError,IndexError,ValueError):controls.append('corrupt '+prefix.strip())
        else:raise AssertionError('Insensitive checker corruption: '+prefix)
    if r['stage']=='traffic':
        assert r['donor_unchanged'] and r['runtime_inputs_unchanged']
        import csv
        rows=list(csv.DictReader((out/'evidence/progress.tsv').open(),delimiter='\t'));assert rows
        last=rows[-1]
        for name in ('apb_complete_0','apb_complete_1','init_seq_complete_0','init_seq_complete_1','started','drained'):assert int(last[name],16)==1,name
        for name in ('fault','inflight','banks_used','queued','pending_counts'):assert int(last[name],16)==0,name
        for name,wanted in [('native_aw',41),('native_b',41),('native_ar',41),('native_w',586),('native_r',586)]:assert int(last[name],16)==wanted,name
        for name in ('DRAM_0_STAT_CATTRIP','DRAM_1_STAT_CATTRIP'):assert last[name].lower()=='x'
        assert int(last['thermal_unknown'],16)==result['counts']['thermal_unknown']
        assert int(last['native_r_stalls'],16)==result['counts']['r_stalls']
        assert int(last['cpl_stalls'],16)==result['counts']['cpl_stalls']
        assert any(row['started']=='0' and row['apb_complete_0']=='0' for row in rows)
        result['progress_samples']=len(rows)
    result.update(status='NATIVE_HBM_TRANSPORT_READBACK_PASS' if r['stage']=='traffic' else 'BENCH_PREFLIGHT_PASS',
                  native_hbm=r['stage']=='traffic',negative_trace_controls=controls,rtl_fault_injections=0)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();r=base(a.output)
    print(json.dumps(compile_review(a.output,r) if r['stage']=='compile' else traffic_review(a.output,r),indent=2))
