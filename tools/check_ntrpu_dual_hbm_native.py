"""Separate native snapshot review from traffic completion and readback review."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_dual_hbm import sha,replay,negative_controls

def review(out,observation=None):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='dual_hbm_native'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out']
    assert r['sources']==r['sources_after'] and r['donor_unchanged']
    files=json.loads((out/'FILES.json').read_text());assert all(sha(out/n)==h for n,h in files.items())
    result=dict(result_sha256=sha(out/'result.json'),seconds=r['seconds'],snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'))
    inputs=json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
    assert any('tb_ntrpu_dual_hbm_behav/xsimk' in n for n in inputs) and len(inputs)>=10
    assert sum(v['bytes'] for v in inputs.values())>100_000_000
    if r['stage']=='compile':
        pre=json.loads((out/'PREFLIGHT_REVIEW.json').read_text());cache=json.loads((out/'CACHE_REVIEW.json').read_text())
        assert pre['status']=='DUAL_HBM_PREFLIGHT_REVIEWED' and pre['native_profile']==1
        assert cache['status']=='NATIVE_RESERVE_SNAPSHOT_REVIEWED'
        provenance=json.loads((out/'CACHE_PROVENANCE.json').read_text())
        assert provenance['native_config_sha256']==cache['native_config_sha256']==sha(out/'suite/native_hbm_config.tsv')
        assert provenance['native_binding_sha256']==sha(out/'suite/ntrpu_hbm_native_rw.sv')
        script=(out/'compile.sh').read_text();assert 'native_stub' not in script
        assert '-generic_top NATIVE=1' in script and 'ntrpu_hbm_native_rw.sv' in script
        assert not re.search(r'(?im)^\s*(ERROR:|Fatal:)',(out/'console.log').read_text())
        result.update(status='DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED',native_config_sha256=provenance['native_config_sha256'],
                      simulation_executed=False,bench_sha256=r['sources']['fpga/tb/tb_ntrpu_dual_hbm.sv'],
                      runtime_files=len(inputs),runtime_bytes=sum(v['bytes'] for v in inputs.values()))
    else:
        prior=json.loads((out/'PRIOR_REVIEW.json').read_text());assert prior['status']=='DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED'
        assert prior['result_sha256']==r['prior_result_sha256'] and prior['snapshot_manifest_sha256']==result['snapshot_manifest_sha256']
        assert r['runtime_inputs_unchanged']
        log=(out/'evidence/simulate.log').read_text();assert not re.search(r'(?im)^\s*(ERROR:|Fatal:|FAIL )',log)
        counts=replay(log.splitlines(),1);controls=negative_controls(log.splitlines(),1)
        boot=re.findall(r'^DH BOOT (\d+) (\d+)$',log,re.M);assert len(boot)==1 and int(boot[0][0])>40_000_000
        rows=(out/'evidence/progress.tsv').read_text().splitlines();header=rows[0].split('\t')
        values=[dict(zip(header,row.split('\t'))) for row in rows[1:]]
        ready=[v for v in values if v.get('started')=='1']
        if observation is None:
            assert ready and all(v['apb_complete_0']==v['apb_complete_1']=='1' for v in ready), 'Startup observation missing; supply the separately reviewed identical-snapshot observation'
            thermal=sorted({(v['DRAM_0_STAT_CATTRIP'],v['DRAM_1_STAT_CATTRIP']) for v in ready})
        else:
            from check_ntrpu_dual_hbm_observe import review as observe_review
            startup=observe_review(observation,out);assert startup['boot_ps']==int(boot[0][0])
            thermal=[startup['thermal_observed']]
            result.update(startup_observation=startup,startup_observed_in_separate_window=True)
        result.update(status='DUAL_HBM_NATIVE_TRAFFIC_REVIEWED',**counts,negative_controls=controls,
                      native_controller=True,boot_ps=int(boot[0][0]),thermal_observed=thermal,thermal_qualified=False,
                      readback_via_testbench=True,read_return_cdc=False,pcie_integrated=False,board_qualified=False)
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--observation',type=Path);a=ap.parse_args();print(json.dumps(review(a.output,a.observation),indent=2))
