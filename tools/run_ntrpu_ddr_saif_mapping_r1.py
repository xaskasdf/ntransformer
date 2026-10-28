"""One bounded mapping diagnostic using frozen SAIFs and checkpoint; never retry."""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);a=p.parse_args()
    assert os.name=='posix' and not a.output.exists()
    os.sched_setaffinity(0,{0,1})
    source_hashes=json.loads((a.source/'SNAPSHOT_FILES.json').read_text())
    assert all(digest(a.source/n)==h for n,h in source_hashes.items())
    external=json.loads((a.source/'MAPPING_INPUTS.json').read_text())
    assert all(digest(row['remote'])==row['sha256'] for row in external['inputs'].values())
    a.output.mkdir(parents=True)
    command=['timeout','--signal=TERM','--kill-after=30s','1200s','vivado','-mode','batch',
        '-source',str(a.source/'tools/ntrpu_board_ddr_saif_mapping_r1.tcl'),
        '-log',str(a.output/'vivado.log'),'-journal',str(a.output/'vivado.jou'),'-tclargs',
        external['inputs']['dcp']['remote'],external['inputs']['prbs_0']['remote'],external['inputs']['prbs_1']['remote'],
        str(a.output/'output')]
    now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    started=time.monotonic();info=dict(started_utc=now(),source=str(a.source),source_sha256=a.source_sha,
        cpu_affinity=[0,1],timeout_seconds=1200,command=command,inputs=external['inputs'],
        automatic_retry=False,power_estimate_executed=False,board_power_qualified=False)
    (a.output/'invocation.started.json').write_text(json.dumps(info,indent=2)+'\n')
    exit_code=125
    try:
        with (a.output/'console.log').open('xb') as f:
            exit_code=subprocess.run(command,cwd=a.output,stdout=f,stderr=subprocess.STDOUT).returncode
    except Exception as exc:info['runner_error']=repr(exc)
    finally:
        info.update(finished_utc=now(),exit_code=exit_code,wall_seconds=time.monotonic()-started,
            frozen_source_unchanged=all((a.source/n).is_file() and digest(a.source/n)==h for n,h in source_hashes.items()),
            external_inputs_unchanged=all(Path(row['remote']).is_file() and digest(row['remote'])==row['sha256'] for row in external['inputs'].values()))
        (a.output/'invocation.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    return exit_code if info['frozen_source_unchanged'] and info['external_inputs_unchanged'] else 126

if __name__=='__main__':raise SystemExit(main())
