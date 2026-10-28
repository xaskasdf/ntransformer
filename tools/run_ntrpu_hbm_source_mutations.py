"""Require targeted incorrect source joins to fail the public integration bench."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(snapshot, out, chain):
    out.mkdir(parents=True, exist_ok=False)
    bins = chain/'mingw64/bin'
    env = dict(os.environ)
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str, [bins, windows/'System32', windows]))
    manifest = json.loads((snapshot/'result.json').read_text())['sources']
    assert all(sha(snapshot/'repo'/n) == h for n, h in manifest.items())
    variants = [
        ('release_during_fault', ' &&\n                   !stop_out)active[result_index]<=0;',
         ')active[result_index]<=0;', 'error reports retain all reservations'),
        ('reserve_rejected', 'wire reserve=pending && !b_reject_valid;', 'wire reserve=pending;',
         'both contexts reserved before command'),
        ('truncate_data_tag', 'reading.identity.sequence_id==r_data_tag[lane*64+:64]',
         'reading.identity.sequence_id[31:0]==r_data_tag[lane*64+:32]', 'invalid 64-bit read tag held'),
    ]
    rows = []
    for label, old, new, marker in variants:
        case = out/label
        repo = case/'repo'
        repo.mkdir(parents=True)
        for name in manifest:
            target = repo/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot/'repo'/name, target)
        path = repo/'fpga/experimental/ntrpu_hbm_source_join.sv'
        text = path.read_text(encoding='utf-8')
        assert text.count(old) == 1
        path.write_text(text.replace(old, new), encoding='utf-8', newline='\n')
        rtl = [repo/n for n in manifest if n.endswith('.sv')]
        steps = []
        for command, log in [([bins/'iverilog.exe', '-g2012', '-s', 'tb_ntrpu_hbm_xdma_source',
                              '-o', case/'mutant.vvp', *rtl], 'compile.log'),
                             ([bins/'vvp.exe', case/'mutant.vvp', '+seed=0'], 'simulation.log')]:
            job = subprocess.run(list(map(str, command)), cwd=case, env=env, text=True, capture_output=True, timeout=30)
            output = job.stdout+job.stderr
            (case/log).write_text(output, encoding='utf-8')
            steps.append(dict(log=log, returncode=job.returncode))
        assert steps[0]['returncode'] == 0
        assert steps[1]['returncode'] != 0 and marker in output and 'watchdog' not in output and 'SJ PASS' not in output
        hashes = {n: sha(repo/n) for n in manifest}
        assert {n for n in manifest if hashes[n] != manifest[n]} == {'fpga/experimental/ntrpu_hbm_source_join.sv'}
        rows.append(dict(label=label, marker=marker, steps=steps, sources=hashes))
    assert all(sha(snapshot/'repo'/n) == h for n, h in manifest.items())
    result = dict(status='SOURCE_JOIN_RTL_MUTATIONS_REJECTED', variants=rows, sources=manifest,
                  toolchain={p.relative_to(chain).as_posix(): sha(p) for p in chain.rglob('*') if p.is_file()})
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (out/'FILES.json').write_text(json.dumps({p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], variants=len(rows))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('snapshot', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--toolchain', type=Path, required=True)
    a = ap.parse_args()
    run(a.snapshot.resolve(), a.output.resolve(), a.toolchain.resolve())
