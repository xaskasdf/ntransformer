"""Compile a frozen reserve bench against a private, verified native HBM cache."""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from run_ntrpu_hbm_native_transport import ROOT, SOURCES as BASE_SOURCES, SIMDIR, sha, save

SOURCES=BASE_SOURCES+['tools/run_ntrpu_hbm_reserve_native.py']
TOP='tb_ntrpu_hbm_reserve'


def checked_files(path):
    manifest=json.loads((path/'FILES.json').read_text())
    assert all((path/n).resolve().is_relative_to(path.resolve()) and sha(path/n)==h for n,h in manifest.items())
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--stage',choices=('compile','traffic'),required=True)
    for n in ('cache','cache-review','preflight','preflight-review','prior','prior-review'):
        ap.add_argument('--'+n,type=Path)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
    sources={n:sha(ROOT/n) for n in SOURCES}
    r=dict(status='PREPARING',experiment='hbm_reserve_native',stage=a.stage,sources=sources,
           cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),load_before=os.getloadavg(),timeout=900)
    save(out/'result.json',r);donor_hashes={};runtime_inputs={}
    if a.stage=='compile':
        cache=a.cache.resolve();pre=a.preflight.resolve();checked_files(cache);checked_files(pre)
        cr=json.loads(a.cache_review.read_text());pr=json.loads(a.preflight_review.read_text())
        assert cr['status']=='NATIVE_TRANSPORT_SNAPSHOT_REVIEWED'
        assert cr['snapshot_manifest_sha256']==sha(cache/'SNAPSHOT_INPUTS.json')
        assert cr['configuration_sha256']==sha(cache/'suite/native_hbm_config.tsv')
        assert pr['status']=='RESERVE_PRESSURE_PREFLIGHT_PASS' and pr.get('frame_beats',8)==8
        assert pr['result_sha256']==sha(pre/'result.json')
        cache_result=json.loads((cache/'result.json').read_text())
        assert cache_result['sources']==cache_result['sources_after']
        assert all(sources[n]==h for n,h in cache_result['sources'].items())
        c=json.loads((pre/'suite/COMPOSITION.json').read_text())
        assert c['bench_sha256']==sha(pre/'suite/generated'/f'{TOP}.sv')
        for n,h in c['frozen_logic'].items():assert sha(cache/'suite/generated'/(n+'.sv'))==h
        assert sha(pre/'suite/ntrpu_hbm_native_rw.sv')==sha(cache/'suite/ntrpu_hbm_native_rw.sv')
        for n in ('hbm_width_config.tsv','hbm_protocol_config.tsv','hbm_cdc_config.tsv','ip_config.tsv'):
            # Paths in metadata differ; compare configuration values only.
            cfg=lambda p:{x.split('\t',1)[0]:x.split('\t',1)[1] for x in p.read_text().splitlines() if x.startswith('CONFIG.') and '\t' in x}
            assert cfg(pre/'suite'/n)==cfg(cache/'suite'/n)
        donor=cache/'suite'
        donor_hashes={p.relative_to(donor).as_posix():sha(p) for p in donor.rglob('*') if p.is_file()}
        shutil.copytree(donor,out/'suite')
        assert all(sha(out/'suite'/n)==h for n,h in donor_hashes.items())
        shutil.copy2(pre/'suite/generated'/f'{TOP}.sv',out/'suite/generated'/f'{TOP}.sv')
        shutil.copy2(pre/'suite/COMPOSITION.json',out/'suite/RESERVE_COMPOSITION.json')
        shutil.copy2(a.cache_review,out/'CACHE_REVIEW.json');shutil.copy2(a.preflight_review,out/'PREFLIGHT_REVIEW.json')
        save(out/'CACHE_PROVENANCE.json',dict(donor_hashes=donor_hashes,cache=str(cache),cache_original_status=cache_result['status'],
            native_snapshot_review_sha256=sha(a.cache_review),preflight_result_sha256=sha(pre/'result.json'),
            bench_sha256=c['bench_sha256'],original_timed_out_parent_accepted=False))
        cwd=out/SIMDIR
        elaborate=(cwd/'elaborate.sh').read_text()
        command=next(line for line in elaborate.splitlines() if line.startswith('xelab '))
        command=command.replace('tb_ntrpu_hbm_native_transport',TOP)
        compile_cmd=['xvlog','--incr','--relax','--work','xil_defaultlib','--sv',str(out/'suite/generated'/f'{TOP}.sv'),'-log','reserve_compile.log']
        # Both commands and their exit codes remain visible in the batch log.
        shell='set -e\n'+shlex.join(compile_cmd)+'\n'+command+'\n'
        (out/'compile.sh').write_text(shell);command=['bash',str(out/'compile.sh')]
    else:
        prior=a.prior.resolve();checked_files(prior);review=json.loads(a.prior_review.read_text())
        previous=json.loads((prior/'result.json').read_text())
        assert review['status']=='NATIVE_RESERVE_SNAPSHOT_REVIEWED'
        assert review['result_sha256']==sha(prior/'result.json')
        assert previous['sources']==previous['sources_after']==sources
        assert review['snapshot_manifest_sha256']==sha(prior/'SNAPSHOT_INPUTS.json')
        runtime_inputs=json.loads((prior/'SNAPSHOT_INPUTS.json').read_text());donor=prior/SIMDIR
        cwd=out/'runtime';cwd.mkdir()
        for n,v in runtime_inputs.items():
            assert (donor/n).resolve().is_relative_to(donor.resolve()) and sha(donor/n)==v['sha256']
            target=cwd/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor/n,target);assert sha(target)==v['sha256']
        tcl=(ROOT/'tools/ntrpu_hbm_native_transport_run.tcl').read_text().replace('tb_ntrpu_hbm_native_transport',TOP)
        (cwd/'run.tcl').write_text(tcl)
        for name in ('SNAPSHOT_INPUTS.json',):shutil.copy2(prior/name,out/name)
        shutil.copy2(a.prior_review,out/'PRIOR_REVIEW.json')
        command=['xsim',TOP+'_behav','-tclbatch','run.tcl','-log','simulate.log']
        r.update(prior_result_sha256=sha(prior/'result.json'))
    r.update(status='RUNNING',command=command);save(out/'result.json',r)
    start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        job=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=job.wait(timeout=900)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(job.pid,signal.SIGTERM)
            try:code=job.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
    r.update(seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after={n:sha(ROOT/n) for n in SOURCES})
    r['status']='EXPORTED_PENDING_REVIEW' if code==0 and not expired and r['sources']==r['sources_after'] else 'FAILED_OR_CHANGED'
    if a.stage=='compile':
        r['donor_unchanged']=all(sha(donor/n)==h for n,h in donor_hashes.items())
        selected=[p for p in [*cwd.glob('*.mem'),cwd/'xsim.ini',*(cwd/'xsim.dir'/(TOP+'_behav')).glob('xsim*')] if p.is_file()]
        save(out/'SNAPSHOT_INPUTS.json',{p.relative_to(cwd).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in selected})
        for p in cwd.glob('*.log'):
            dest=out/'tool_logs'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest)
    else:
        r['donor_unchanged']=all(sha(donor/n)==v['sha256'] for n,v in runtime_inputs.items())
        r['runtime_inputs_unchanged']=all(sha(cwd/n)==v['sha256'] for n,v in runtime_inputs.items())
        for n in ('simulate.log','run.tcl','progress.tsv'):
            dest=out/'evidence'/n;dest.parent.mkdir(exist_ok=True)
            if (cwd/n).exists():shutil.copy2(cwd/n,dest)
    if not r['donor_unchanged'] or r.get('runtime_inputs_unchanged') is False:r['status']='INPUTS_CHANGED'
    save(out/'result.json',r)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
        and not set(p.relative_to(out).parts)&{'project','runtime'} and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:r[k] for k in ('stage','status','seconds')}),flush=True)


if __name__=='__main__':main()
