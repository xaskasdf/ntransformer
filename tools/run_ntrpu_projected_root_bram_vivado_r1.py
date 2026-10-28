"""Bounded synthesis/XSim, independent arithmetic gate, then physical measurement."""
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    affinity=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,affinity)
    for name,sha in json.loads((a.source/'SNAPSHOT_FILES.json').read_text()).items():
        assert hashlib.sha256((a.source/name).read_bytes()).hexdigest()==sha,name
    a.output.mkdir(parents=True);started=time.monotonic();now=lambda:datetime.now(timezone.utc).isoformat()
    def vivado(script,log,args,seconds):
        return ['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s','vivado','-mode','batch','-source',str(a.source/'tools'/script),'-log',str(a.output/(log+'.log')),'-journal',str(a.output/(log+'.jou')),'-tclargs',*map(str,args)]
    commands=[('console.log',vivado('ntrpu_board_projected_root_bram_real_r1.tcl','vivado',[a.output/'output'],7200)),
        ('arithmetic-gate.log',['timeout','--signal=TERM','--kill-after=30s','600s','python3',str(a.source/'tools/check_ntrpu_projected_root_bram_gate_r1.py'),'--source',str(a.source),'--run',str(a.output)]),
        ('physical-console.log',vivado('ntrpu_board_projected_root_bram_route_r1.tcl','physical',[a.output/'physical',a.output/'output/attention_projected_root_real_synth.dcp'],3600))]
    info=dict(stage='V7-projected-root-bram-vivado-r1',started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=affinity,max_threads=2,timeout_seconds=11400,part='xcvu5p-flva2104-2-i',clock_ns=5,
        commands=[],status='running',scope='Candidate BRAM enable, full actual attention/quantization/O RTL followed by OOC placement, route and post-route physopt. No board or post-route simulation qualification.')
    inv=a.output/'invocation.json';inv.write_text(json.dumps(info,indent=2)+'\n')
    with (a.output/'environment.log').open('wb') as f:subprocess.run(['vivado','-version'],stdout=f,stderr=subprocess.STDOUT,check=True)
    for name,cmd in commands:
        t=time.monotonic()
        with (a.output/name).open('wb') as f:r=subprocess.run(cmd,cwd=a.output,stdout=f,stderr=subprocess.STDOUT)
        info['commands'].append(dict(log=name,command=cmd,exit_code=r.returncode,duration_seconds=time.monotonic()-t))
        inv.write_text(json.dumps(info,indent=2)+'\n')
        if r.returncode:break
    info.update(exit_code=r.returncode,duration_seconds=time.monotonic()-started,finished_utc=now(),status='awaiting_independent_review')
    inv.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
