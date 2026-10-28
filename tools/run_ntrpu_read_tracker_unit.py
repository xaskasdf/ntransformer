"""Bounded tracker stress tests and three private RTL negative controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    root=Path(__file__).resolve().parents[1]
    names=['fpga/experimental/ntrpu_hbm_read_adapter.sv',
           'fpga/experimental/ntrpu_hbm_read_tracker.sv',
           'fpga/tb/tb_ntrpu_read_tracker.sv','tools/run_ntrpu_read_tracker_unit.py']
    hashes={n:sha(root/n) for n in names}
    output.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);os.nice(10)
    os.environ['RDI_DATADIR']=str(Path(os.environ['XILINX_VIVADO'])/'data')
    variants=[('depth4',4,None),('depth8',8,None),
              ('early_credit',4,('wire release_credit=validate_fire&&terminal;',
                                'wire release_credit=capture_fire&&h_rlast;','EARLY_CREDIT_RELEASE')),
              ('rid_mask',4,('q_rid!=0','q_rid[3:0]!=0','QUARANTINE_FAULT')),
              ('ready_path',4,('assign h_rready=q_space&&capture_enable;',
                              'assign h_rready=q_space&&capture_enable&&(h_rid==0);','READY_DEPENDENCE'))]
    rows=[]
    for label,depth,mutation in variants:
        folder=output/label;folder.mkdir()
        for name in names:
            text=(root/name).read_text()
            if name.endswith('ntrpu_hbm_read_tracker.sv') and mutation:
                old,new,_=mutation;assert text.count(old)==1;text=text.replace(old,new)
            (folder/Path(name).name).write_text(text)
        commands=[['xvlog','--sv','ntrpu_hbm_read_adapter.sv','ntrpu_hbm_read_tracker.sv','tb_ntrpu_read_tracker.sv'],
                  ['xelab','--debug','off','--mt','2','-generic_top','DEPTH='+str(depth),
                   '--snapshot','unit','work.tb_ntrpu_read_tracker'],
                  ['xsim','unit','-runall']]
        steps=[]
        for stage,cmd in zip(('compile','elaborate','simulate'),commands):
            before=time.monotonic();expired=False
            with (folder/(stage+'.log')).open('w') as log:
                job=subprocess.Popen(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=job.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    expired=True;os.killpg(job.pid,signal.SIGKILL);code=job.wait()
            steps.append(dict(stage=stage,command=cmd,returncode=code,timed_out=expired,seconds=round(time.monotonic()-before,3)))
            if expired or (code!=0 and stage!='simulate'):break
        rows.append(dict(label=label,depth=depth,mutation=mutation,steps=steps))
    result=dict(experiment='read_tracker_unit',sources=hashes,
                sources_after={n:sha(root/n) for n in names},cpus=sorted(os.sched_getaffinity(0)),nice=os.nice(0),variants=rows)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    (output/'FILES.json').write_text(json.dumps({p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and len(p.relative_to(output).parts)<=2
        and p.suffix in ('.json','.log','.sv','.py')},indent=2))


def review(output):
    result=json.loads((output/'result.json').read_text())
    assert result['experiment']=='read_tracker_unit' and result['sources']==result['sources_after']
    assert all(sha(output/n)==h for n,h in json.loads((output/'FILES.json').read_text()).items())
    reviewed=[]
    assert [v['label'] for v in result['variants']]==['depth4','depth8','early_credit','rid_mask','ready_path']
    for v in result['variants']:
        folder=output/v['label'];steps=v['steps']
        assert len(steps)==3 and not any(s['timed_out'] for s in steps)
        assert all(s['returncode']==0 for s in steps[:2])
        for stage in ('compile','elaborate'):
            assert not re.search(r'(?im)^\s*(ERROR:|Fatal:)',(folder/(stage+'.log')).read_text())
        log=(folder/'simulate.log').read_text()
        if v['mutation']:
            cause=v['mutation'][2]
            assert 'TRACKER_UNIT_FAIL' in log and cause in log
            assert 'TU ALL_PASS' not in log and 'TRACKER_UNIT_WATCHDOG' not in log
            status='REJECTED_'+cause
        else:
            assert steps[2]['returncode']==0 and not re.search(r'(?im)^\s*(ERROR:|Fatal:)',log)
            assert re.findall(r'^TU PASS case=(\d+)',log,re.M)==list('123456')
            assert 'TU ALL_PASS depth='+str(v['depth'])+' cases=6' in log
            status='PASS'
        reviewed.append(dict(label=v['label'],status=status))
    return dict(status='READ_TRACKER_UNIT_REVIEWED',result_sha256=sha(output/'result.json'),variants=reviewed,
                native_hbm=False,physical_timing_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('run','review'));ap.add_argument('output',type=Path)
    a=ap.parse_args()
    if a.action=='run':run(a.output.resolve())
    else:print(json.dumps(review(a.output),indent=2))
