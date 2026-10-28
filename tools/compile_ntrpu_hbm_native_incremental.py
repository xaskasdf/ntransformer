"""Re-elaborate only our probe using a private copy of reviewed native IP cache."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
suite=Path(sys.argv[1]);prior=Path(sys.argv[2]);review=Path(sys.argv[3])
r=json.loads(review.read_text());result=json.loads((prior/'result.json').read_text())
assert r['status']=='NATIVE_TRAFFIC_SNAPSHOT_REVIEWED' and r['sources']==result['sources']==result['sources_after']
assert sha(prior/'SNAPSHOT_INPUTS.json')==r['snapshot_manifest_sha256']
for n in ('tools/prepare_ntrpu_hbm_native_link.py','tools/prepare_ntrpu_hbm_cdc.py',
          'tools/prepare_ntrpu_hbm_native_traffic.py','tools/ntrpu_hbm_native_traffic.tcl'):
    assert sha(ROOT/n)==r['sources'][n]
original=prior/'suite';suite.mkdir(parents=True,exist_ok=False)
assert sha(original/'native_hbm_config.tsv')==r['native_config_sha256']
donor={p.relative_to(original).as_posix():sha(p) for p in original.rglob('*') if p.is_file()}
shutil.copytree(original/'project',suite/'project')
shutil.copytree(original/'model_inputs',suite/'model_inputs')
for n in ('native_hbm.veo','native_hbm.xci','native_hbm_config.tsv'):shutil.copy2(original/n,suite/n)
copied={p.relative_to(suite).as_posix():sha(p) for p in suite.rglob('*') if p.is_file()}
assert all(h==donor[n] for n,h in copied.items())
subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_native_traffic.py'),str(suite)],check=True)
sim=suite/'project/hbm_native_traffic.sim/sim_1/behav/xsim'
os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
subprocess.run(['xvlog','--incr','--relax','--work','xil_defaultlib','--sv',
                str(suite/'ntrpu_hbm_native_rw.sv'),str(suite/'tb_ntrpu_hbm_native_traffic.sv'),'-log','incremental_compile.log'],cwd=sim,check=True)
cmd=['xelab','--incr','--debug','off','--relax','--mt','2']
for lib in ('xil_defaultlib','hbm_v1_0_20','uvm','unisims_ver','unimacro_ver','secureip','xpm'):cmd+=['-L',lib]
cmd+=['--snapshot','tb_ntrpu_hbm_native_traffic_behav','xil_defaultlib.tb_ntrpu_hbm_native_traffic','xil_defaultlib.glbl','-log','incremental_elaborate.log']
subprocess.run(cmd,cwd=sim,check=True)
assert all(sha(original/n)==h for n,h in donor.items()),'Donor modified'
record=dict(status='INCREMENTAL_NATIVE_CACHE_REVIEWED',prior=str(prior),prior_review_sha256=sha(review),
            native_config_sha256=r['native_config_sha256'],donor_unchanged=True,
            copied_files=len(copied),donor_hashes=donor,changes_limited_to_testbench=True)
(suite/'INCREMENTAL.json').write_text(json.dumps(record,indent=2)+'\n')
print('NTRPU_NATIVE_TRAFFIC_COMPILED',flush=True)
