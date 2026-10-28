"""Temporarily transfer timeout supervision without stopping the HLS job.

Only the identified Python waiting process is stopped. Its independently
sessioned tool group keeps running. Resume the waiter after its child becomes
a zombie, or at the new deadline so the original timeout cleanup applies.
Linux CPython waitpid ordering is checked and tested before this tool is used.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process(pid):
    root=Path('/proc')/str(pid)
    try:
        raw=(root/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        cmd=(root/'cmdline').read_bytes().replace(b'\x00',b' ').decode().strip()
        return dict(pid=pid,state=fields[0],parent=int(fields[1]),group=int(fields[2]),
                    session=int(fields[3]),start_ticks=int(fields[19]),cmdline=cmd)
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=int,required=True)
    p.add_argument('--child',type=int,required=True);p.add_argument('--effective-seconds',type=int,required=True)
    p.add_argument('--plan-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and 2400<a.effective_seconds<=5400
    run=a.run.resolve(strict=True)
    assert run.parent==Path('/lambda/nfs/vivado/ntrpu/board-vivado/20261002T085705Z/runs')
    inv_path=run/'invocation.json';inv_raw=inv_path.read_bytes();inv=json.loads(inv_raw)
    assert inv['stage']=='V7-norm-q8-hls-r1' and inv['timeout_seconds']==2400 and 'finished_utc' not in inv
    assert inv['source_sha256']=='fb81afdb4704a0a0f4f1206750ddb86eeb88dec7162ff942f5091f9404341edb'
    start=datetime.fromisoformat(inv['started_utc']).timestamp();original_deadline=start+2400
    assert time.time()<original_deadline-120, 'Not enough time for safe transfer'
    parent=process(a.parent);child=process(a.child)
    assert parent and child and child['parent']==a.parent and child['group']==child['session']==a.child
    assert parent['state']!='T' and child['state']!='Z' and parent['group']!=child['group']
    expected='python3 '+inv['source']+'/tools/run_ntrpu_norm_q8_hls_r2.py'
    assert parent['cmdline'].startswith(expected+' ') and '--output '+str(run) in parent['cmdline']
    assert child['cmdline']=='/bin/bash /lambda/nfs/vivado/AMDDesignTools/2026.1/Vitis/bin/vitis-run --mode hls --tcl source/run_hls_r2.tcl'
    assert (Path('/proc')/str(a.child)/'cwd').resolve()==run/'output'
    wait_source=inspect.getsource(subprocess.Popen._wait)
    assert wait_source.index('if pid == self.pid:')<wait_source.index('remaining = self._remaining_time(endtime)')
    path=run/'budget-extension-r1.json';assert not path.exists()
    state=dict(status='transferring_timeout_supervision',utc=datetime.now(timezone.utc).isoformat(),
        guardian_pid=os.getpid(),parent=parent,child=child,original_timeout_seconds=2400,
        effective_timeout_seconds=a.effective_seconds,effective_deadline_epoch=start+a.effective_seconds,
        original_invocation_sha256=hashlib.sha256(inv_raw).hexdigest(),plan_sha256=a.plan_sha,
        guardian_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python_wait_source_sha256=hashlib.sha256(wait_source.encode()).hexdigest(),tool_group_never_stopped=True)
    def save(status,**extra):
        state.update(status=status,utc=datetime.now(timezone.utc).isoformat(),**extra)
        temp=path.with_suffix('.pending');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
    def same(pid,identity):
        now=process(pid)
        return now if now and now['start_ticks']==identity['start_ticks'] else None
    def interrupted(signum,frame):
        raise RuntimeError('Guardian interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    stopped=False;reason='guardian_exception'
    save('transferring_timeout_supervision')
    try:
        os.kill(a.parent,signal.SIGSTOP);stopped=True
        for _ in range(50):
            current=same(a.parent,parent)
            assert current, 'Original waiter disappeared'
            if current['state']=='T':break
            time.sleep(.02)
        assert current['state']=='T' and same(a.child,child)['state']!='T'
        save('tool_running_under_extended_deadline')
        last_progress=None;last_save=0
        while True:
            current=same(a.child,child);owner=same(a.parent,parent)
            assert owner and owner['state']=='T', 'Waiter ownership changed'
            assert current and current['parent']==a.parent, 'Tool child identity changed'
            if current['state']=='Z':reason='tool_completed_before_extended_deadline';break
            if time.time()>=start+a.effective_seconds:reason='extended_deadline_reached';break
            log=run/'console.log'
            progress=[s for s in log.read_text(errors='replace').splitlines() if 'RTL Simulation :' in s]
            latest=progress[-1] if progress else None
            if latest!=last_progress or time.monotonic()-last_save>=30:
                save('tool_running_under_extended_deadline',latest_progress=latest,child_state=current['state'])
                last_progress=latest;last_save=time.monotonic()
            time.sleep(2)
    except BaseException as exc:
        save('guardian_failure_resuming_original_waiter',error=repr(exc))
        raise
    finally:
        if stopped and same(a.parent,parent):
            os.kill(a.parent,signal.SIGCONT)
        save('original_waiter_resumed',resume_reason=reason)
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        final=json.loads(inv_path.read_text())
        if 'finished_utc' in final:
            save('extension_complete',resume_reason=reason,actual_exit_code=final['exit_code'],
                actual_duration_seconds=final['duration_seconds'],final_invocation_sha256=hashlib.sha256(inv_path.read_bytes()).hexdigest())
            return
        time.sleep(1)
    save('waiter_resumed_terminal_observation_pending',resume_reason=reason)


if __name__=='__main__':main()
