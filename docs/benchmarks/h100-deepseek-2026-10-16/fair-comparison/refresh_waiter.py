"""Replace only our idle waiter, preserving the original frozen preparation."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time
import common as b

ROOT=b.ROOT
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016')
REVISION='paused-admission-20261016'

def main():
    b.verify()
    current=json.loads((ROOT/'queue-status.json').read_text())
    pid=current['pid'];process=Path('/proc',str(pid))
    assert (process/'cwd').resolve()==ROOT
    assert str(ROOT/'queue.py').encode() in (process/'cmdline').read_bytes().split(b'\0')
    assert not list(ROOT.glob('pilot-*-command.json')), 'GPU experiments already started'
    os.kill(pid,signal.SIGSTOP)
    stopped=True
    try:
        for _ in range(50):
            if (process/'stat').read_text().split(') ',1)[1].startswith('T'):break
            time.sleep(.02)
        else:raise RuntimeError('waiter did not stop')
        state=json.loads((ROOT/'status.json').read_text())
        assert state['state']=='waiting' and state['pid']==pid,state
        assert not (process/'task'/str(pid)/'children').read_text().strip(), 'waiter has active child processes'
        assert not list(ROOT.glob('pilot-*-command.json'))
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped=False
        for _ in range(100):
            if not process.exists() or (process/'stat').read_text().split(') ',1)[1].startswith('Z'):break
            time.sleep(.05)
        else:raise RuntimeError('old waiter did not exit')
    finally:
        if stopped and process.exists():os.kill(pid,signal.SIGCONT)
    backup=ROOT/'revisions'/REVISION;backup.mkdir(parents=True,exist_ok=False)
    names=['admission.py','frozen-inputs.json','preparation.json','preparation.tar.gz',
        'expected-upload.json','launch.json','queue-status.json','status.json']
    for name in names:shutil.copy2(ROOT/name,backup/name)
    persistent_backup=PERSIST/'revisions'/REVISION
    persistent_backup.mkdir(parents=True,exist_ok=False)
    for name in names:
        shutil.copy2(backup/name,persistent_backup/name)
        assert b.sha(backup/name)==b.sha(persistent_backup/name)
    shutil.copy2(ROOT/'admission-next.py',ROOT/'admission.py')
    validation=dict(tests_passed=8,scope='CPU-only admission fixtures; no GPU inference',
        candidate_sha256=b.sha(ROOT/'admission.py'),test_sha256=b.sha(ROOT/'test_admission.py'),
        coverage=['paused without PID','paused dead PID','paused live PID','busy GPU','held lock','running stage','complete','superseded'])
    b.save('admission-validation.json',validation)
    frozen=json.loads((ROOT/'frozen-inputs.json').read_text())
    for name in ['admission.py','admission-next.py','test_admission.py','admission-validation.json','refresh_waiter.py','PROTOCOL.md']:
        frozen[name]=b.sha(ROOT/name)
    b.save('frozen-inputs.json',frozen)
    expected=json.loads((ROOT/'expected-upload.json').read_text());expected['admission.py']=frozen['admission.py']
    b.save('expected-upload.json',expected);b.verify()
    with tarfile.open(ROOT/'preparation.tar.gz','w:gz') as archive:
        for name in [*frozen,'frozen-inputs.json']:archive.add(ROOT/name,arcname=name,recursive=False)
    info=dict(sha256=b.sha(ROOT/'preparation.tar.gz'),bytes=(ROOT/'preparation.tar.gz').stat().st_size,
        frozen_files=len(frozen),revision=REVISION,scope='Runnable preparation and CPU validation; GPU benchmark pending')
    b.save('preparation.json',info)
    for name in ['preparation.tar.gz','preparation.json','frozen-inputs.json','admission-validation.json','PROTOCOL.md']:
        shutil.copy2(ROOT/name,PERSIST/name);assert b.sha(ROOT/name)==b.sha(PERSIST/name)
    with (ROOT/'queue.log').open('a') as log:
        log.write('\nAdmission-only revision: inactive paused stages accepted after locks and idle-GPU checks.\n');log.flush()
        worker=subprocess.Popen([str(b.PYTHON),str(ROOT/'queue.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    b.save('launch.json',dict(pid=worker.pid,replaces_pid=pid,preparation=info,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(pid=worker.pid,replaces_pid=pid,preparation=info)),flush=True)

if __name__=='__main__':main()
