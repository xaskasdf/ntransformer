"""Native HBM validation in a private copy of the previously reviewed IP cache."""
import argparse,json,os,shlex,shutil,signal,subprocess,time
from pathlib import Path
from run_ntrpu_dual_hbm import ROOT,SOURCES as BASE_SOURCES,sha,save
SOURCES=BASE_SOURCES+['tools/run_ntrpu_dual_hbm_native.py','tools/ntrpu_dual_hbm_native_run.tcl']
SIMDIR=Path('suite/project/hbm_adapter.sim/sim_1/behav/xsim')
TOP='tb_ntrpu_dual_hbm'
def checked(out):
    for n,h in json.loads((out/'FILES.json').read_text()).items():
        assert (out/n).resolve().is_relative_to(out.resolve()) and sha(out/n)==h
def config(p):return dict(line.split('\t',1) for line in p.read_text().splitlines() if line.startswith('CONFIG.') and '\t' in line)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('compile','traffic'),required=True)
    for name in ('cache','cache-review','preflight','preflight-review','prior','prior-review'):ap.add_argument('--'+name,type=Path)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
    sources={n:sha(ROOT/n) for n in SOURCES};r=dict(status='PREPARING',experiment='dual_hbm_native',stage=a.stage,
        sources=sources,cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=1800,load_before=os.getloadavg())
    save(out/'result.json',r);runtime={}
    if a.stage=='compile':
        cache=a.cache.resolve();pre=a.preflight.resolve();checked(cache);checked(pre)
        cr=json.loads(a.cache_review.read_text());pr=json.loads(a.preflight_review.read_text())
        assert cr['status']=='NATIVE_RESERVE_SNAPSHOT_REVIEWED'
        assert cr['result_sha256']==sha(cache/'result.json') and cr['snapshot_manifest_sha256']==sha(cache/'SNAPSHOT_INPUTS.json')
        assert cr['native_config_sha256']==sha(cache/'suite/native_hbm_config.tsv')
        assert pr['status']=='DUAL_HBM_PREFLIGHT_REVIEWED' and pr['native_profile']==1 and pr['policy']==1
        assert pr['result_sha256']==sha(pre/'result.json')
        prior_result=json.loads((pre/'result.json').read_text());assert prior_result['sources']==prior_result['sources_after']
        assert all(sources[n]==h for n,h in prior_result['sources'].items())
        assert sha(pre/'suite/native_binding.sv')==sha(cache/'suite/ntrpu_hbm_native_rw.sv')
        for n in ('hbm_width_config.tsv','hbm_protocol_config.tsv','hbm_cdc_config.tsv'):
            assert config(pre/'suite'/n)==config(cache/'suite'/n)
        donor=cache/'suite';original={p.relative_to(donor).as_posix():sha(p) for p in donor.rglob('*') if p.is_file()}
        shutil.copytree(donor,out/'suite');assert all(sha(out/'suite'/n)==h for n,h in original.items())
        shutil.copy2(pre/'suite/ntrpu_dual_hbm_binding.svh',out/'suite/ntrpu_dual_hbm_binding.svh')
        shutil.copy2(a.cache_review,out/'CACHE_REVIEW.json');shutil.copy2(a.preflight_review,out/'PREFLIGHT_REVIEW.json')
        save(out/'CACHE_PROVENANCE.json',dict(cache=str(cache),donor_hashes=original,preflight_result_sha256=sha(pre/'result.json'),
             native_binding_sha256=sha(out/'suite/ntrpu_hbm_native_rw.sv'),native_config_sha256=sha(cache/'suite/native_hbm_config.tsv')))
        cwd=out/SIMDIR
        command=next(line for line in (cwd/'elaborate.sh').read_text().splitlines() if line.startswith('xelab '))
        for old in ('tb_ntrpu_hbm_native_transport','tb_ntrpu_hbm_reserve'):command=command.replace(old,TOP)
        command=command.replace('xelab ','xelab -generic_top PREFER_LOCAL=1 -generic_top NATIVE=1 ',1)
        sv=[str(ROOT/n) for n in SOURCES if n.endswith('.sv')]
        compile_cmd=['xvlog','--incr','--relax','--work','xil_defaultlib','--sv','-i',str(out/'suite'),*sv,
                     str(out/'suite/ntrpu_hbm_native_rw.sv'),'-log','dual_compile.log']
        (out/'compile.sh').write_text('set -e\n'+shlex.join(compile_cmd)+'\n'+command+'\n')
        command=['bash',str(out/'compile.sh')]
    else:
        prior=a.prior.resolve();checked(prior);pr=json.loads(a.prior_review.read_text())
        assert pr['status']=='DUAL_HBM_NATIVE_SNAPSHOT_REVIEWED' and pr['result_sha256']==sha(prior/'result.json')
        assert pr['snapshot_manifest_sha256']==sha(prior/'SNAPSHOT_INPUTS.json')
        previous=json.loads((prior/'result.json').read_text());assert previous['sources']==previous['sources_after']==sources
        donor=prior/SIMDIR;runtime=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text());original={n:item['sha256'] for n,item in runtime.items()}
        cwd=out/'runtime';cwd.mkdir()
        for n,item in runtime.items():
            assert (donor/n).resolve().is_relative_to(donor.resolve()) and sha(donor/n)==item['sha256']
            dest=cwd/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,dest);assert sha(dest)==item['sha256']
        shutil.copy2(ROOT/'tools/ntrpu_dual_hbm_native_run.tcl',cwd/'run.tcl')
        shutil.copy2(a.prior_review,out/'PRIOR_REVIEW.json');shutil.copy2(prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json')
        r['prior_result_sha256']=sha(prior/'result.json');command=['xsim',TOP+'_behav','-tclbatch','run.tcl','-log','simulate.log']
    r.update(status='RUNNING',command=command);save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES},
             donor_unchanged=all(sha(donor/n)==h for n,h in original.items()))
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['donor_unchanged'] and sources==r['sources_after'] else 'FAILED_OR_CHANGED'
    if a.stage=='compile':
        selected=[p for p in [*cwd.glob('*.mem'),cwd/'xsim.ini',*(cwd/'xsim.dir'/(TOP+'_behav')).glob('xsim*')] if p.is_file()]
        save(out/'SNAPSHOT_INPUTS.json',{p.relative_to(cwd).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in selected})
        for p in cwd.glob('*.log'):
            dest=out/'tool_logs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    else:
        r['runtime_inputs_unchanged']=all(sha(cwd/n)==h for n,h in original.items())
        if not r['runtime_inputs_unchanged']:r['status']='INPUTS_CHANGED'
        for n in ('simulate.log','run.tcl','progress.tsv'):
            if (cwd/n).exists():
                dest=out/'evidence'/n;dest.parent.mkdir(exist_ok=True);shutil.copy2(cwd/n,dest)
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
        and not set(p.relative_to(out).parts)&{'project','runtime'} and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}),flush=True)
if __name__=='__main__':main()
