"""Bounded shared-origin admission experiment, always in a private source tree."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from run_ntrpu_peer_read import sha,save
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--full',action='store_true');a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10);os.environ['OMP_NUM_THREADS']='2'
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    if a.full:
        commands=[['vivado','-mode','batch','-notrace','-source',str(ROOT/'tools/ntrpu_peer_shared.tcl'),'-tclargs',str(out/'suite')]]
    else:
        commands=[['xvlog','-sv','-i',str(ROOT/'fpga/tb'),*[str(ROOT/n) for n in (
            'fpga/experimental/ntrpu_peer_read_pkg.sv','fpga/experimental/ntrpu_peer_read_admission.sv','fpga/tb/tb_ntrpu_peer_admission.sv')]],
            ['xelab','tb_ntrpu_peer_admission','-s','shared','-debug','off','--mt','2'],
            ['xsim','shared','-runall','-log','simulation.log']]
    result=dict(status='RUNNING',experiment='peer_shared',full=a.full,sources=sources,commands=commands,
                cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),timeout=900)
    save(out/'result.json',result);start=time.monotonic();expired=False
    with (out/'console.log').open('w') as log:
        for command in commands:
            job=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=job.wait(timeout=max(1,900-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                expired=True;os.killpg(job.pid,signal.SIGTERM)
                try:code=job.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(job.pid,signal.SIGKILL);code=job.wait()
            if code or expired:break
    after={n:sha(ROOT/n) for n in sources};p=out/('suite/simulation.log' if a.full else 'simulation.log')
    logtext=p.read_text(errors='replace') if p.exists() else ''
    marker='SG ALL_PASS mode='+('full cases=6' if a.full else 'unit cases=15')
    passed=marker in logtext and not any(s in logtext for s in ('Fatal:','SG_FAIL','SG_WATCHDOG'))
    result.update(status='EXPORTED_PENDING_REVIEW' if code==0 and not expired and sources==after and passed else 'FAILED_OR_CHANGED',
                  seconds=round(time.monotonic()-start,3),returncode=code,timed_out=expired,sources_after=after,functional_marker=passed)
    save(out/'result.json',result)
    save(out/'FILES.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()
         and not set(p.relative_to(out).parts)&{'project','xsim.dir'} and not any(n.startswith('.') for n in p.relative_to(out).parts)})
    print(json.dumps({k:result[k] for k in ('status','seconds','returncode')}))


if __name__=='__main__':main()
