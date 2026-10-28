"""Preserve the failed attempt, freeze the tested log reader, and restart the full plan."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import common as b

ROOT=b.ROOT
PERSIST=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-fair-20261016')
REVISION='log-decode-20261016'

def main():
    b.verify()
    current=json.loads((ROOT/'queue-status.json').read_text())
    assert current['state']=='failed' and 'UnicodeDecodeError' in current['error'],current
    pid=current['pid'];process=Path('/proc',str(pid))
    assert not process.exists() or (process/'stat').read_text().split(') ',1)[1].startswith('Z'), 'worker still live'
    assert not list(ROOT.glob('nt-*-command.json')), 'main comparison already started'
    backup=ROOT/'revisions'/REVISION;backup.mkdir(parents=True,exist_ok=False)
    names=['run.py','frozen-inputs.json','preparation.json','preparation.tar.gz',
        'expected-upload.json','launch.json','queue-status.json','status.json','queue.log']
    for name in names:shutil.copy2(ROOT/name,backup/name)
    # Move only artifacts of this failed benchmark attempt, never external jobs.
    move_names=[p.name for p in ROOT.glob('pilot-cpu*')]
    move_names+=['telemetry.csv','compute-processes.jsonl','admission.json']
    for name in move_names:
        source=ROOT/name
        if source.exists():source.rename(backup/name)
    target=PERSIST/'revisions'/REVISION
    shutil.copytree(backup,target)
    for source in backup.rglob('*'):
        if source.is_file():assert b.sha(source)==b.sha(target/source.relative_to(backup))
    shutil.copy2(ROOT/'run-next.py',ROOT/'run.py')
    frozen=json.loads((ROOT/'frozen-inputs.json').read_text())
    for name in ['run.py','run-next.py','test_log_decode.py','log-decode-validation.log','restart_after_log_decode.py']:
        frozen[name]=b.sha(ROOT/name)
    b.save('frozen-inputs.json',frozen)
    expected=json.loads((ROOT/'expected-upload.json').read_text());expected['run.py']=frozen['run.py']
    b.save('expected-upload.json',expected);b.verify()
    with tarfile.open(ROOT/'preparation.tar.gz','w:gz') as archive:
        for name in [*frozen,'frozen-inputs.json']:archive.add(ROOT/name,arcname=name,recursive=False)
    info=dict(sha256=b.sha(ROOT/'preparation.tar.gz'),bytes=(ROOT/'preparation.tar.gz').stat().st_size,
        frozen_files=len(frozen),revision=REVISION,
        scope='Runnable preparation and CPU validation; full GPU comparison restarts after archived log-decoding failure')
    b.save('preparation.json',info)
    for name in ['preparation.tar.gz','preparation.json','frozen-inputs.json','log-decode-validation.log']:
        shutil.copy2(ROOT/name,PERSIST/name);assert b.sha(ROOT/name)==b.sha(PERSIST/name)
    with (ROOT/'queue.log').open('a') as log:
        log.write('\nLog-reader revision: tolerate truncated UTF-8 diagnostics, preserve raw bytes and strict numeric JSON; restarting all pilots and main cases.\n');log.flush()
        worker=subprocess.Popen([str(b.PYTHON),str(ROOT/'queue.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    b.save('launch.json',dict(pid=worker.pid,replaces_pid=pid,preparation=info,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(pid=worker.pid,replaces_pid=pid,preparation=info)),flush=True)

if __name__=='__main__':main()
