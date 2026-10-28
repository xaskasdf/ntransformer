"""Tie a startup-only observation to the identical full native traffic snapshot."""
import json,re
from check_ntrpu_dual_hbm import sha

NAMES=['simulation_time','wall_ms','apb_complete_0','apb_complete_1','mem_rst_n','apb_rst_n',
       'started','scenario','epoch','fault','drained','inflight','banks_used','pending_counts',
       'total_cmds','total_bytes','native_read_beats','DRAM_0_STAT_CATTRIP','DRAM_1_STAT_CATTRIP']

def time_ps(value):
    m=re.fullmatch(r'([0-9.]+)(ps|ns|us|ms)',value)
    assert m is not None
    return round(float(m[1])*{'ps':1,'ns':1000,'us':1_000_000,'ms':1_000_000_000}[m[2]])

def rows_review(text,boot_ps):
    rows=[line.split('\t') for line in text.splitlines()];assert rows[0]==NAMES
    assert len(rows)>40 and all(len(row)==len(NAMES) for row in rows)
    data=[dict(zip(NAMES,row)) for row in rows[1:]]
    times=[time_ps(v['simulation_time']) for v in data]
    assert times==list(range(1_000_000,times[-1]+1,1_000_000)) and 40_000_000<boot_ps<=times[-1]<=55_000_000
    assert all('UNAVAILABLE' not in v.values() for v in data)
    ready=[v for v in data if v['started']=='1'];assert len(ready)==1 and ready[0] is data[-1]
    assert all(v['started']=='0' and int(v['total_cmds'],16)==0 for v in data[:-1])
    assert any(v['apb_complete_0']==v['apb_complete_1']=='0' for v in data)
    v=ready[0]
    assert v['apb_complete_0']==v['apb_complete_1']==v['mem_rst_n']==v['apb_rst_n']=='1' and v['fault']=='0'
    assert int(v['epoch'],16)==1
    return dict(samples=len(data),started_at_sample_ps=times[-1],
                thermal_observed=[v['DRAM_0_STAT_CATTRIP'],v['DRAM_1_STAT_CATTRIP']],thermal_qualified=False)

def controls(text,boot_ps):
    done=[]
    for name,value in [('started','0'),('apb_complete_0','0'),('apb_complete_1','0'),('mem_rst_n','0'),
                       ('apb_rst_n','0'),('fault','1'),('epoch','2'),('simulation_time','1us')]:
        rows=[line.split('\t') for line in text.splitlines()];rows[-1][NAMES.index(name)]=value
        try:rows_review('\n'.join('\t'.join(row) for row in rows),boot_ps)
        except AssertionError:done.append(name)
        else:raise AssertionError('Undetected observation corruption '+name)
    return done

def review(out,traffic):
    r=json.loads((out/'result.json').read_text());tr=json.loads((traffic/'result.json').read_text())
    assert r['experiment']=='dual_hbm_native_observe' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert r['donor_unchanged'] and r['runtime_inputs_unchanged']
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
    assert prior['status']=='DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED'
    assert prior['result_sha256']==r['prior_result_sha256']==tr['prior_result_sha256']
    assert sha(out/'SNAPSHOT_INPUTS.json')==sha(traffic/'SNAPSHOT_INPUTS.json')==prior['snapshot_manifest_sha256']
    log=(out/'evidence/simulate.log').read_text();script=(out/'evidence/run.tcl').read_text()
    assert sha(out/'evidence/run.tcl')==r['sources']['tools/ntrpu_dual_hbm_native_observe.tcl']
    assert re.search(r'^DUAL_HBM_NATIVE_STARTUP_OBSERVED$',log,re.M) and '\nDH_PROBES 17\n' in log
    assert not re.search(r'(?im)^\s*(ERROR:|Fatal:)|while executing|UNAVAILABLE',log)
    assert not re.search(r'\b(add_force|force|deposit)\b',script)
    observed=[s for s in log.splitlines() if s.startswith('DH ')]
    full=[s for s in (traffic/'evidence/simulate.log').read_text().splitlines() if s.startswith('DH ')]
    assert len(observed)>5 and observed==full[:len(observed)]
    boot=re.findall(r'^DH BOOT (\d+) (\d+)$',log,re.M);assert len(boot)==1 and boot[0][1]=='1'
    progress=(out/'evidence/progress.tsv').read_text();info=rows_review(progress,int(boot[0][0]))
    return dict(status='DUAL_HBM_NATIVE_STARTUP_REVIEWED',result_sha256=sha(out/'result.json'),seconds=r['seconds'],
                snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),matching_prefix_events=len(observed),
                boot_ps=int(boot[0][0]),negative_observation_controls=controls(progress,int(boot[0][0])),**info)
