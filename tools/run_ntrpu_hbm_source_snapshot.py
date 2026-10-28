"""Run a frozen private native snapshot without rebuilding vendor models."""
import argparse, hashlib, json, os, shutil, signal, subprocess, time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def run(donor,stage,timeout):
 os.nice(10);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
 stage.mkdir(exist_ok=False);out=stage/'output';out.mkdir();(out/'reports').mkdir()
 original=json.loads((donor/'output/result.json').read_text())
 assert original['seed_unchanged'] and original['sources']==original['sources_after']
 for n,h in original['sources'].items():assert sha(donor/'repo'/n)==h,n
 shutil.copytree(donor/'repo',stage/'repo')
 shutil.copytree(donor/'output/export',out/'export')
 for p in (donor/'output/reports').iterdir():
  if p.is_file() and p.suffix in ['.rpt','.tsv']:shutil.copy2(p,out/'reports'/p.name)
 sim=donor/'output/work/example/ntrpu_vu47p_xdma_ex.sim/sim_1/behav/xsim'
 cwd=out/'work/sim';cwd.mkdir(parents=True)
 files={p.relative_to(sim).as_posix():p for p in (sim/'xsim.dir/board_behav').iterdir() if p.is_file()}
 files.update({p.name:p for p in sim.iterdir() if p.is_file() and (p.suffix=='.mem' or p.name=='xsim.ini')})
 snapshot={n:sha(p) for n,p in files.items()}
 for n,p in files.items():
  q=cwd/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==snapshot[n]
 source=(stage/'repo/tools/ntrpu_hbm_source_native.tcl').read_text()
 body=source[source.index('    set names '):source.index('    close_vcd;close_sim;close_project')]
 tcl=cwd/'review.tcl';tcl.write_text('set out {'+str(out/'reports')+'}\n'+body+'\nclose_vcd\nputs HSN_SIMULATOR_RETURNED\nquit\n')
 command=['xsim','board_behav','-tclbatch',str(tcl),'-log','simulate.log','-testplusarg','TESTNAME=ntrpu_hbm_xdma']
 result={**original,'status':'RUNNING','seconds':None,'returncode':None,'timed_out':False,
  'command':command,'timeout':timeout,'compile_origin':str(donor),
  'compile_result_sha256':sha(donor/'output/result.json'),'snapshot':snapshot,'snapshot_source_unchanged':False}
 (out/'result.json').write_text(json.dumps(result,indent=2))
 # Original compile diagnostics are preserved; runtime diagnostics belong only to this execution.
 compile_log=(donor/'output/console.log').read_text().split('Built simulation snapshot board_behav')[0]+'Built simulation snapshot board_behav\nSNAPSHOT_REUSED_WITH_IDENTICAL_INPUTS\n'
 start=time.monotonic();expired=False
 with (out/'console.log').open('w') as f:
  f.write(compile_log);f.flush()
  p=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   expired=True;os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
 result.update(returncode=rc,timed_out=expired,seconds=round(time.monotonic()-start,3),
  sources_after={n:sha(stage/'repo'/n) for n in original['sources']},snapshot_source_unchanged=all(sha(p)==snapshot[n] for n,p in files.items()))
 result['status']='PENDING_REVIEW' if rc==0 and not expired and result['snapshot_source_unchanged'] and result['sources']==result['sources_after'] else 'FAILED'
 for p in cwd.iterdir():
  if p.is_file() and p.suffix in ['.log','.tcl','.ini']:shutil.copy2(p,out/'reports'/('snapshot_'+p.name))
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 (out/'FILES.json').write_text(json.dumps({p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.relative_to(out).parts[0]!='work' and p.name!='FILES.json'},indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['status','seconds','returncode']}),flush=True)
 return result

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('donor',type=Path);ap.add_argument('stage',type=Path);ap.add_argument('--timeout',type=int,required=True)
 a=ap.parse_args();r=run(a.donor.resolve(),a.stage.resolve(),a.timeout);raise SystemExit(0 if r['status']=='PENDING_REVIEW' else 1)
