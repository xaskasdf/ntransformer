"""Review real HBM elaboration and the synthesizable read-return traffic separately."""
import argparse,json,re
from pathlib import Path
from check_ntrpu_read_return import sha,replay,controls

def review(out):
    r=json.loads((out/'result.json').read_text());assert r['experiment']=='read_return_native'
    assert r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out']
    assert r['sources']==r['sources_after'] and r['donor_unchanged']
    threads=r.get('threads',2)
    assert threads in (2,4,8) and len(set(r['cpus']))==threads and r['nice']>=10
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    inputs=json.loads((out/'SNAPSHOT_INPUTS.json').read_text());assert len(inputs)>=10 and sum(v['bytes'] for v in inputs.values())>100_000_000
    assert any('tb_ntrpu_read_return_behav/xsimk' in n for n in inputs)
    result=dict(result_sha256=sha(out/'result.json'),seconds=r['seconds'],snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),
                execution_cpus=r['cpus'],threads=threads)
    if r['stage']=='compile':
        pre=json.loads((out/'PREFLIGHT_REVIEW.json').read_text());cache=json.loads((out/'CACHE_REVIEW.json').read_text())
        assert pre['status']=='READ_RETURN_PREFLIGHT_REVIEWED' and pre['common']==0 and pre['native_profile']==1
        assert cache['status'] in ('DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED','READ_RETURN_NATIVE_SNAPSHOT_REVIEWED')
        provenance=json.loads((out/'CACHE_PROVENANCE.json').read_text())
        assert provenance['native_config_sha256']==cache['native_config_sha256']==sha(out/'suite/native_hbm_config.tsv')
        assert provenance['native_binding_sha256']==sha(out/'suite/ntrpu_hbm_native_rw.sv')
        script=(out/'compile.sh').read_text();assert 'native_stub' not in script and '-generic_top NATIVE=1' in script and '-generic_top COMMON=0' in script
        assert re.findall(r'--mt\s+(\d+)',script)==[str(threads)]
        for name in ('hbm_read_cdc','hbm_read_width','hbm_read_protocol'):
            assert name+'_wrapper.v' in script
            config=dict(line.split('\t',1) for line in (out/'suite'/(name+'_config.tsv')).read_text().splitlines() if line.startswith('CONFIG.'))
            assert config['CONFIG.READ_WRITE_MODE']=='READ_ONLY'
        log=(out/'console.log').read_text();assert not re.search(r'(?im)^\s*(ERROR:|Fatal:)',log)
        for name in ('ntrpu_hbm_read_return','ntrpu_hbm_read_frame','ntrpu_hbm_read_adapter','native_hbm'):
            assert 'Compiling module xil_defaultlib.'+name in log,name
        result.update(status='READ_RETURN_NATIVE_SNAPSHOT_REVIEWED',revision=pre.get('revision',0),native_config_sha256=provenance['native_config_sha256'],
                      simulation_executed=False,bench_sha256=r['sources']['fpga/tb/tb_ntrpu_read_return.sv'],runtime_files=len(inputs),runtime_bytes=sum(v['bytes'] for v in inputs.values()))
    else:
        prior=json.loads((out/'PRIOR_REVIEW.json').read_text());assert prior['status']=='READ_RETURN_NATIVE_SNAPSHOT_REVIEWED'
        assert prior['result_sha256']==r['prior_result_sha256'] and prior['snapshot_manifest_sha256']==result['snapshot_manifest_sha256'] and r['runtime_inputs_unchanged']
        log=(out/'evidence/simulate.log').read_text();assert not re.search(r'(?im)^\s*(ERROR:|Fatal:|FAIL )|while executing|UNAVAILABLE',log)
        assert re.search(r'^NTRPU_READ_RETURN_NATIVE_WINDOW_COMPLETE$',log,re.M)
        counts=replay(log.splitlines(),1);assert counts['boot_ps']>40_000_000
        rows=[s.split('\t') for s in (out/'evidence/progress.tsv').read_text().splitlines()];assert all(len(s)==len(rows[0]) for s in rows)
        values=[dict(zip(rows[0],v)) for v in rows[1:]];ready=[v for v in values if v['started']=='1']
        assert ready and all(v['apb_complete_0']==v['apb_complete_1']==v['mem_rst_n']==v['apb_rst_n']=='1' and v['fault']=='0' for v in ready)
        thermal=sorted({(v['DRAM_0_STAT_CATTRIP'],v['DRAM_1_STAT_CATTRIP']) for v in ready})
        result.update(status='READ_RETURN_NATIVE_TRAFFIC_REVIEWED',**counts,negative_controls=controls(log.splitlines(),1),
                      native_hbm_controller=True,read_return_synthesizable=True,thermal_observed=thermal,thermal_qualified=False,
                      pcie_integrated=False,board_qualified=False)
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
