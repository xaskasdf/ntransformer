"""Bounded native HBM preflight, elaboration and traffic; frozen private sources."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
from run_ntrpu_peer_read import sha, save

ROOT=Path(__file__).resolve().parents[1]
SIMDIR=Path('suite/project/hbm_adapter.sim/sim_1/behav/xsim')
TOP='tb_ntrpu_peer_shared_native'


def checked(out):
    for n,h in json.loads((out/'FILES.json').read_text()).items():
        assert (out/n).resolve().is_relative_to(out.resolve()) and sha(out/n)==h,n


def config(path):
    return dict(s.split('\t',1) for s in path.read_text().splitlines() if s.startswith('CONFIG.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('preflight','compile','traffic'),required=True)
    ap.add_argument('--threads',type=int,choices=(2,4,8),default=2)
    for n in ('cache','cache-review','preflight','preflight-review','prior','prior-review'):
        ap.add_argument('--'+n,type=Path)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=a.threads
    os.sched_setaffinity(0,cpus[:a.threads]);os.nice(10);os.environ['OMP_NUM_THREADS']=str(a.threads)
    os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    profile=json.loads((ROOT/'SHARED_NATIVE_PROFILE.json').read_text())
    assert all(sources[n]==h for n,h in profile['selected_rtl'].items())
    shutil.copyfile(ROOT/'SHARED_NATIVE_PROFILE.json',out/'SHARED_NATIVE_PROFILE.json')
    r=dict(status='PREPARING',experiment='peer_shared_native',stage=a.stage,sources=sources,
           cpus=sorted(os.sched_getaffinity(0)),threads=a.threads,nice=os.nice(0),timeout=1800,load_before=os.getloadavg())
    save(out/'result.json',r);original={};donor=None
    if a.stage=='preflight':
        assert a.threads==2
        command=['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_peer_shared_native.tcl'),'-tclargs',str(out/'suite')]
        cwd=out
    elif a.stage=='compile':
        cache=a.cache.resolve();pre=a.preflight.resolve();checked(cache);checked(pre)
        cr=json.loads(a.cache_review.read_text());pr=json.loads(a.preflight_review.read_text())
        assert cr['status'] in ('PEER_NATIVE_SNAPSHOT_REVIEWED',)
        assert cr['result_sha256']==sha(cache/'result.json') and cr['snapshot_manifest_sha256']==sha(cache/'SNAPSHOT_INPUTS.json')
        assert profile['native_config_sha256']==sha(cache/'suite/native_hbm_config.tsv')
        assert pr['status']=='SHARED_NATIVE_PREFLIGHT_REVIEWED' and pr['result_sha256']==sha(pre/'result.json')
        previous=json.loads((pre/'result.json').read_text())
        assert previous['sources']==previous['sources_after'] and set(previous['sources'])==set(sources)
        changed={n for n,h in sources.items() if previous['sources'][n]!=h}
        assert changed <= {'tools/run_ntrpu_peer_shared_native.py'},changed
        if changed:
            r['preflight_driver_delta']=dict(reason='Recognize original top retained in the inherited HBM cache elaborate.sh',
                previous_sha256=previous['sources']['tools/run_ntrpu_peer_shared_native.py'],
                current_sha256=sources['tools/run_ntrpu_peer_shared_native.py'])
        assert sha(ROOT/'fixtures/native.sv')==sha(cache/'suite/ntrpu_hbm_native_rw.sv')
        for name in ('hbm_width','hbm_protocol','hbm_cdc','hbm_read_width','hbm_read_protocol','hbm_read_cdc'):
            assert config(pre/'suite'/(name+'_config.tsv'))==config(cache/'suite'/(name+'_config.tsv')),name
        donor=cache/'suite';original={p.relative_to(donor).as_posix():sha(p) for p in donor.rglob('*') if p.is_file()}
        shutil.copytree(donor,out/'suite');assert all(sha(out/'suite'/n)==h for n,h in original.items())
        for n in profile['selected_read_configs']:
            shutil.copyfile(pre/'suite'/n,out/'suite'/n)
            assert config(out/'suite'/n)==profile['selected_read_configs'][n],n
        shutil.copyfile(a.cache_review,out/'CACHE_REVIEW.json');shutil.copyfile(a.preflight_review,out/'PREFLIGHT_REVIEW.json')
        save(out/'CACHE_PROVENANCE.json',dict(cache=str(cache),donor_hashes=original,
             native_binding_sha256=sha(cache/'suite/ntrpu_hbm_native_rw.sv'),native_config_sha256=profile['native_config_sha256']))
        cwd=out/SIMDIR
        elaborate=next(s for s in (cwd/'elaborate.sh').read_text().splitlines() if s.startswith('xelab '))
        for old in ('tb_ntrpu_peer_native','tb_ntrpu_hbm_native_transport','tb_ntrpu_hbm_reserve','tb_ntrpu_dual_hbm','tb_ntrpu_read_return'):
            elaborate=elaborate.replace(old,TOP)
        elaborate=re.sub(r' -generic_top (PREFER_LOCAL|COMMON|NATIVE|FRAMES|FULL|PAIR)=\d+','',elaborate)
        elaborate=elaborate.replace('xelab ','xelab -generic_top NATIVE=1 ',1)
        elaborate,count=re.subn(r'--mt\s+\d+','--mt '+str(a.threads),elaborate);assert count==1
        assert 'xil_defaultlib.'+TOP in elaborate and TOP+'_behav' in elaborate
        from run_ntrpu_read_return import SOURCES
        sv=[str(ROOT/n) for n in SOURCES if n.endswith('.sv') and not n.startswith('fpga/tb/')]
        sv += [str(ROOT/'fpga/experimental'/(n+'.sv')) for n in ('ntrpu_peer_read_pkg','ntrpu_peer_read_lane','ntrpu_peer_read_admission','ntrpu_peer_read_shared')]
        sv += [str(ROOT/'fpga/tb/tb_ntrpu_peer_shared_native.sv')]
        for name in ('hbm_read_cdc','hbm_read_width','hbm_read_protocol'):
            shutil.copyfile(pre/'suite'/(name+'_wrapper.v'),out/'suite'/(name+'_wrapper.v'))
            sv.append(str(out/'suite'/(name+'_wrapper.v')))
        compile_cmd=['xvlog','--incr','--relax','--work','xil_defaultlib','--sv','-i',str(ROOT/'fixtures'),'-i',str(ROOT/'fpga/tb'),
                     *sv,str(out/'suite/ntrpu_hbm_native_rw.sv'),'-log','peer_native_compile.log']
        (out/'compile.sh').write_text('set -e\n'+shlex.join(compile_cmd)+'\n'+elaborate+'\n')
        command=['bash',str(out/'compile.sh')]
    else:
        prior=a.prior.resolve();checked(prior);pr=json.loads(a.prior_review.read_text())
        assert pr['status']=='SHARED_NATIVE_SNAPSHOT_REVIEWED' and pr['result_sha256']==sha(prior/'result.json')
        assert pr['snapshot_manifest_sha256']==sha(prior/'SNAPSHOT_INPUTS.json')
        previous=json.loads((prior/'result.json').read_text());assert previous['sources']==previous['sources_after']
        assert set(previous['sources'])==set(sources)
        changed={n for n,h in sources.items() if previous['sources'][n]!=h}
        assert changed in (set(),{'tools/run_ntrpu_peer_shared_native.py'}),changed
        if changed:
            r['driver_recovery']=dict(reason='Preserve executable mode when copying the reviewed runtime',
                source='tools/run_ntrpu_peer_shared_native.py',previous_sha256=previous['sources']['tools/run_ntrpu_peer_shared_native.py'],
                current_sha256=sources['tools/run_ntrpu_peer_shared_native.py'])
        donor=prior/SIMDIR;inputs=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text())
        original={n:item['sha256'] for n,item in inputs.items()};cwd=out/'runtime';cwd.mkdir()
        for n,h in original.items():
            assert (donor/n).resolve().is_relative_to(donor.resolve()) and sha(donor/n)==h
            dest=cwd/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,dest);assert sha(dest)==h
            assert (dest.stat().st_mode & 0o777)==((donor/n).stat().st_mode & 0o777)
        shutil.copyfile(ROOT/'tools/ntrpu_peer_shared_native_run.tcl',cwd/'run.tcl')
        shutil.copyfile(a.prior_review,out/'PRIOR_REVIEW.json');shutil.copyfile(prior/'SNAPSHOT_INPUTS.json',out/'SNAPSHOT_INPUTS.json')
        r['prior_result_sha256']=sha(prior/'result.json')
        command=['xsim',TOP+'_behav','-tclbatch','run.tcl','-log','simulate.log']
    r.update(status='RUNNING',command=command);save(out/'result.json',r);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in sources},
             donor_unchanged=not donor or all(sha(donor/n)==h for n,h in original.items()))
    if a.stage=='compile':
        selected=[p for p in [*cwd.glob('*.mem'),cwd/'xsim.ini',*(cwd/'xsim.dir'/(TOP+'_behav')).glob('xsim*')] if p.is_file()]
        save(out/'SNAPSHOT_INPUTS.json',{p.relative_to(cwd).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in selected})
        for p in cwd.glob('*.log'):
            dest=out/'tool_logs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    elif a.stage=='traffic':
        r['runtime_inputs_unchanged']=all(sha(cwd/n)==h for n,h in original.items())
        for n in ('simulate.log','run.tcl','progress.tsv'):
            if (cwd/n).exists():
                dest=out/'evidence'/n;dest.parent.mkdir(exist_ok=True);shutil.copyfile(cwd/n,dest)
    if a.stage=='preflight':
        for n,h in profile['selected_read_configs'].items():
            if (out/'suite'/n).exists():assert config(out/'suite'/n)==h,n
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['donor_unchanged'] and sources==r['sources_after'] and r.get('runtime_inputs_unchanged',True) else 'FAILED_OR_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
         and not set(p.relative_to(out).parts)&{'project','runtime'} and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}),flush=True)


if __name__=='__main__':main()
