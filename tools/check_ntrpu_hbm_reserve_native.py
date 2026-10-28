"""Verify native provenance and replay the frozen reserve bench's full trace."""
import argparse,csv,json,re
from pathlib import Path
from check_ntrpu_hbm_reserve import replay,sha
from check_ntrpu_hbm_native_transport import configuration


def review(out):
    r=json.loads((out/'result.json').read_text())
    assert r['experiment']=='hbm_reserve_native' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['sources']==r['sources_after'] and r['returncode']==0 and not r['timed_out'] and r['donor_unchanged']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    snapshot=json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
    assert len(snapshot)>=15 and sum(x['bytes'] for x in snapshot.values())>100_000_000
    if r['stage']=='compile':
        log=(out/'console.log').read_text()
        assert 'Built simulation snapshot tb_ntrpu_hbm_reserve_behav' in log
        assert not re.search(r'^\s*(ERROR:|Fatal:)',log,re.M)
        p=json.loads((out/'CACHE_PROVENANCE.json').read_text())
        c=json.loads((out/'suite/RESERVE_COMPOSITION.json').read_text())
        assert p['bench_sha256']==c['bench_sha256']==sha(out/'suite/generated/tb_ntrpu_hbm_reserve.sv')
        assert c['expected_descriptors']==48 and c['cases']==2 and c.get('frame_beats',8)==8
        assert all(sha(out/'suite/generated'/(n+'.sv'))==h for n,h in c['frozen_logic'].items())
        assert c['native_binding_sha256']==sha(out/'suite/ntrpu_hbm_native_rw.sv')
        assert sha(out/'CACHE_REVIEW.json')==p['native_snapshot_review_sha256']
        cache=json.loads((out/'CACHE_REVIEW.json').read_text())
        pre=json.loads((out/'PREFLIGHT_REVIEW.json').read_text())
        assert cache['status']=='NATIVE_TRANSPORT_SNAPSHOT_REVIEWED' and pre['status']=='RESERVE_PRESSURE_PREFLIGHT_PASS'
        assert cache['configuration_sha256']==sha(out/'suite/native_hbm_config.tsv')
        assert pre['result_sha256']==p['preflight_result_sha256']
        cfg=configuration(out/'suite/native_hbm_config.tsv')
        for key,value in [('USER_HBM_DENSITY','16GB'),('USER_HBM_STACK','2'),('USER_AXI_CLK_FREQ','450'),('USER_AXI_CLK1_FREQ','450')]:
            assert cfg['CONFIG.'+key]==value
        for i in (0,1):
            name=f'xpm_internal_config_file_sim_{i}.mem';assert snapshot[name]['sha256']==sha(out/'suite/model_inputs'/name)
        return dict(status='NATIVE_RESERVE_SNAPSHOT_REVIEWED',simulation_executed=False,
                    bench_sha256=p['bench_sha256'],native_config_sha256=cache['configuration_sha256'],
                    snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),result_sha256=sha(out/'result.json'),seconds=r['seconds'])
    assert r['stage']=='traffic' and r['runtime_inputs_unchanged']
    prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
    assert prior['status']=='NATIVE_RESERVE_SNAPSHOT_REVIEWED' and prior['result_sha256']==r['prior_result_sha256']
    assert prior['snapshot_manifest_sha256']==sha(out/'SNAPSHOT_INPUTS.json')
    lines=(out/'evidence/simulate.log').read_text().splitlines();result=replay(lines,8)
    rows=list(csv.DictReader((out/'evidence/progress.tsv').open(),delimiter='\t'));assert rows
    last=rows[-1]
    for name in ('apb_complete_0','apb_complete_1','init_seq_complete_0','init_seq_complete_1','started','drained'):assert int(last[name],16)==1,name
    for name in ('fault','inflight','banks_used','queued','pending_counts'):assert int(last[name],16)==0,name
    for name,value in [('native_aw',48),('native_b',48),('native_ar',48),('native_w',768),('native_r',768)]:assert int(last[name],16)==value
    assert any(row['started']=='0' and row['apb_complete_0']=='0' for row in rows)
    controls=[]
    for prefix in ('TE SUB ','TE IN ','TE CPL ','NC W ','NC R ','NC VERIFIED ',
                   'NR ALLOC ','NR LAST ','NR RETURN ','NR FULL ','NR OPEN0 ','NR ISOLATED ','NR OPEN1 ','NR ALL_PASS '):
        index=next(i for i,v in enumerate(lines) if v.startswith(prefix))
        try:replay(lines[:index]+lines[index+1:],8)
        except (AssertionError,KeyError,IndexError,ValueError,StopIteration):controls.append('remove '+prefix.strip())
        else:raise AssertionError('Insensitive checker '+prefix)
    for prefix,field in (('NC W ',4),('NC R ',4),('NR ALLOC ',6),('NR RETURN ',5)):
        i=next(i for i,v in enumerate(lines) if v.startswith(prefix));parts=lines[i].split();parts[field]=format(int(parts[field],16)^1,'x')
        mutant=list(lines);mutant[i]=' '.join(parts)
        try:replay(mutant,8)
        except (AssertionError,KeyError,IndexError,ValueError,StopIteration):controls.append('corrupt '+prefix.strip())
        else:raise AssertionError('Insensitive checker '+prefix)
    result.update(status='NATIVE_HBM_RESERVE_PRESSURE_PASS',native_hbm=True,progress_samples=len(rows),
                  negative_trace_controls=controls,seconds=r['seconds'],result_sha256=sha(out/'result.json'),
                  final_cattrip=[last['DRAM_0_STAT_CATTRIP'],last['DRAM_1_STAT_CATTRIP']])
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
