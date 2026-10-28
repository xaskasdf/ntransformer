"""Bounded XSim microdiagnostic for conditional assertion helper semantics."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert not a.output.exists();os.sched_setaffinity(0,{0,1})
    hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    assert all(hashlib.sha256((a.source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    a.output.mkdir();out=a.output/'output';out.mkdir();commands=[];started=time.monotonic()
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    inv=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,cpu_affinity=[0,1],commands=commands,status='running')
    path=a.output/'invocation.json';path.write_text(json.dumps(inv,indent=2)+'\n')
    def run(cmd,log):
        with (out/log).open('wb') as f:r=subprocess.run(['timeout','--kill-after=10s','180s']+cmd,cwd=out,stdout=f,stderr=subprocess.STDOUT)
        commands.append(dict(command=cmd,log=log,exit_code=r.returncode));return r.returncode
    tb=str(a.source/'fpga/tb/tb_ntrpu_assertion_guard_probe_r1.sv')
    native=run(['iverilog','-g2012','-s','tb_ntrpu_assertion_guard_probe_r1','-o','guard.vvp',tb],'iverilog-compile.log')
    if native==0:
        for mode in range(6):run(['vvp','guard.vvp','+MODE='+str(mode)],f'iverilog-mode-{mode}.log')
    code=run(['xvlog','--sv',tb],'compile.log')
    if code==0:code=run(['xelab','--debug','off','--mt','2','tb_ntrpu_assertion_guard_probe_r1','--snapshot','guard'],'elaborate.log')
    if code==0:
        (out/'run.tcl').write_text('run all\nquit\n')
        for mode in range(6):run(['xsim','guard','-tclbatch','run.tcl','-testplusarg','MODE='+str(mode)],f'mode-{mode}.log')
    inv.update(finished_utc=now(),duration_seconds=time.monotonic()-started,exit_code=code,status='awaiting_independent_review')
    path.write_text(json.dumps(inv,indent=2)+'\n');(a.output/'console.log').write_text(json.dumps(inv,indent=2)+'\n');print(json.dumps(inv))
if __name__=='__main__':main()
