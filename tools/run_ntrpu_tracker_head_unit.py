"""Bounded cached-head equivalence, quarantine tests and deliberate RTL faults."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]


def variants():
    rows = [(f'diff{d}_seed{s}', d, s, 'diff', None)
            for d in (1, 2, 4, 8) for s in (1, 23)]
    rows += [(f'faults{d}', d, 1, 'unit', None) for d in (1, 2, 4, 8)]
    rows += [
        ('stale_reuse', 4, 1, 'diff', (
            'reserved_count==0||(release_credit&&reserved_count==1)',
            'reserved_count==0', 'HEAD_INVARIANT')),
        ('wrong_next', 4, 1, 'diff', (
            'head_length<=lengths[next_validate_ptr];',
            'head_length<=lengths[validate_ptr];', 'HEAD_INVARIANT')),
        ('early_credit', 4, 1, 'unit', (
            'wire release_credit=validate_fire&&terminal;',
            'wire release_credit=capture_fire&&h_rlast;', 'EARLY_CREDIT_RELEASE')),
        ('rid_mask', 4, 1, 'unit', ('q_rid!=0', 'q_rid[3:0]!=0', 'QUARANTINE_FAULT')),
        ('ready_path', 4, 1, 'unit', (
            'assign h_rready=q_space&&capture_enable;',
            'assign h_rready=q_space&&capture_enable&&(h_rid==0);', 'READY_DEPENDENCE'))]
    return rows


def run(out):
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*') if p.is_file()}
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2]); os.nice(10)
    os.environ['RDI_DATADIR'] = str(Path(os.environ['XILINX_VIVADO']) / 'data')
    rows = []; start = time.monotonic()
    for label, depth, seed, kind, mutation in variants():
        folder = out / label; folder.mkdir()
        adapter = (ROOT / 'fpga/experimental/ntrpu_hbm_read_adapter.sv').read_text()
        queue = '`timescale 1ns/1ps\nmodule ntrpu_hbm_read_response_queue' + adapter.split(
            'module ntrpu_hbm_read_response_queue', 1)[1]
        head = (ROOT / 'fpga/experimental/ntrpu_hbm_read_head_tracker.sv').read_text()
        if mutation:
            old, new, _ = mutation; assert head.count(old) == 1; head = head.replace(old, new)
        reference = (ROOT / 'HEAD_REFERENCE.sv').read_text()
        if kind == 'diff':
            bench = (ROOT / 'fpga/tb/tb_ntrpu_tracker_head.sv').read_text()
            top = 'tb_ntrpu_tracker_head'
        else:
            bench = (ROOT / 'fpga/tb/tb_ntrpu_read_tracker.sv').read_text().replace(
                'ntrpu_hbm_read_tracker #', 'ntrpu_hbm_read_head_tracker #')
            top = 'tb_ntrpu_read_tracker'
        texts = dict(queue=queue, reference=reference, head=head, bench=bench)
        for n, text in texts.items(): (folder / (n + '.sv')).write_text(text, encoding='utf-8', newline='\n')
        commands = [['xvlog', '--sv', 'queue.sv', 'reference.sv', 'head.sv', 'bench.sv'],
                    ['xelab', '--debug', 'off', '--mt', '2', '-generic_top', 'DEPTH=' + str(depth),
                     *(['-generic_top', 'SEED=' + str(seed)] if kind == 'diff' else []),
                     '--snapshot', 'unit', 'work.' + top], ['xsim', 'unit', '-runall']]
        steps = []
        for stage, cmd in zip(('compile', 'elaborate', 'simulate'), commands):
            before = time.monotonic(); expired = False
            with (folder / (stage + '.log')).open('w') as log:
                job = subprocess.Popen(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try: code = job.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    expired = True; os.killpg(job.pid, signal.SIGKILL); code = job.wait()
            steps.append(dict(stage=stage, command=cmd, returncode=code, timed_out=expired,
                              seconds=round(time.monotonic() - before, 3)))
            if expired or (code and stage != 'simulate'): break
        rows.append(dict(label=label, depth=depth, seed=seed, kind=kind, mutation=mutation, steps=steps))
    save(out / 'result.json', dict(experiment='tracker_head_unit', sources=sources,
         sources_after={n: sha(ROOT/n) for n in sources}, variants=rows,
         seconds=round(time.monotonic()-start, 3), cpus=sorted(os.sched_getaffinity(0)), nice=os.nice(0)))
    save(out / 'FILES.json', {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*')
         if p.is_file() and len(p.relative_to(out).parts)<=2 and p.suffix in ('.json', '.log', '.sv')})


def review(out):
    r = json.loads((out/'result.json').read_text()); repo=out.parent/'repo'
    assert r['experiment']=='tracker_head_unit' and r['sources']==r['sources_after']
    assert all(sha(repo/n)==h for n,h in r['sources'].items())
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    assert [v['label'] for v in r['variants']]==[v[0] for v in variants()]
    reviewed=[]
    for v, expected in zip(r['variants'], variants()):
        label,depth,seed,kind,mutation=expected
        assert (v['depth'],v['seed'],v['kind'])==(depth,seed,kind)
        assert v['mutation']==(list(mutation) if mutation else None)
        folder=out/label; steps=v['steps']; log=(folder/'simulate.log').read_text()
        assert len(steps)==3 and not any(s['timed_out'] for s in steps)
        assert all(s['returncode']==0 for s in steps[:2])
        assert all(not re.search(r'(?im)^\s*(ERROR:|Fatal:)',(folder/(s+'.log')).read_text())
                   for s in ('compile','elaborate'))
        row=dict(label=label)
        if mutation:
            assert mutation[2] in log and ('HEAD_DIFF_FAIL' if kind=='diff' else 'TRACKER_UNIT_FAIL') in log
            assert 'WATCHDOG' not in log and 'ALL_PASS' not in log and 'HEAD_DIFF_PASS' not in log
            row['status']='REJECTED_'+mutation[2]
        else:
            assert steps[2]['returncode']==0 and not re.search(r'(?im)^\s*(ERROR:|Fatal:)',log)
            if kind=='diff':
                m=re.search(r'^HEAD_DIFF_PASS (.+)$',log,re.M); assert m,label
                row['coverage']={k:int(n) for k,n in re.findall(r'(\w+)=(\d+)',m.group(1))}
                assert row['coverage']['depth']==depth and row['coverage']['requests']==384
            else:
                assert re.findall(r'^TU PASS case=(\d+)',log,re.M)==list('123456')
                assert f'TU ALL_PASS depth={depth} cases=6' in log
            row['status']='PASS'
        reviewed.append(row)
    return dict(status='TRACKER_HEAD_UNIT_REVIEWED', result_sha256=sha(out/'result.json'),
                seconds=r['seconds'], variants=reviewed, cycle_equivalent_tested=True,
                native_hbm_controller=False, physical_timing_qualified=False, board_qualified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('action',choices=('run','review')); ap.add_argument('output',type=Path)
    a=ap.parse_args()
    if a.action=='run':run(a.output.resolve())
    else:print(json.dumps(review(a.output.resolve()),sort_keys=True))
