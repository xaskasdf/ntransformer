"""Review the closed short loader bank; reject missing or altered traffic evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern(identifier):
    return sum((((identifier*71+i*37+91)^(i>>3))&255)<<(8*i) for i in range(32))


def log_review(log, seed):
    require(not re.search(r'(?im)^\s*(?:error|fatal):|WL FAIL|WL WATCHDOG', log), 'Failed simulation')
    groups = ['staging_stalls_retention', 'bus_reset_and_attribution', 'clock_pause_boundary_seal',
              'invalid_address_and_unavailable', 'bad_response_blocks_seal', 'bad_id',
              'lost_availability', 'cold_reset_recovery']
    require(re.findall(r'^WL GROUP (\d+) (\w+)$', log, re.M) ==
            [(str(i+1), name) for i, name in enumerate(groups)], 'Named groups')
    verdict = re.findall(r'^WL PASS seed=(\d+) groups=(\d+) checks=(\d+) aw=(\d+) w=(\d+) b=(\d+)$', log, re.M)
    require(len(verdict) == 1, 'Unique PASS')
    values = tuple(map(int, verdict[0]))
    require(values[:2] == (seed, 8) and values[2] >= 400 and values[3:] == (8, 8, 7), 'Accounting')
    aw = [(int(lane), int(address, 16)) for lane, address in re.findall(r'^WL AW lane=(\d+) addr=(\w+)$', log, re.M)]
    require(aw == [(0, 0x1000), (1, 0x200001000), (1, 0x3ffffffe0)] + [(0, 0x1000)]*5, 'Full HBM addresses')
    w = [(int(lane), int(data, 16)) for lane, data in re.findall(r'^WL W lane=(\d+) data=(\w+)$', log, re.M)]
    require(w == [(0, pattern(0)), (1, pattern(1)), (1, pattern(2))] +
            [(0, pattern(i)) for i in range(4, 9)], 'Exact independent HBM payloads')
    b = [tuple(int(x, 16) for x in entry) for entry in re.findall(r'^WL B lane=(\d+) resp=(\w+) id=(\w+)$', log, re.M)]
    require(b == [(0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0)], 'Response fault campaign')
    return dict(seed=seed, groups=8, checks=values[2], aw=8, w=8, b=7)


def review(root):
    files = json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(files)|{'FILES.json'}, 'Closed inventory')
    for name, digest in files.items():
        require(sha(root/name) == digest, 'Closed file hash: '+name)
    result = json.loads((root/'RESULT.json').read_text())
    require(result['status'] == 'WEIGHT_LOADER_PREFLIGHT_EXPORTED' and result['sources_unchanged'], 'Closed successful bank')
    for name, digest in result['sources'].items():
        require(sha(root/'source'/name) == digest, 'Source hash: '+name)
    require(result['seeds'] == [0, 1, 3], 'Selected seed coverage')
    summaries=[];negative=[]
    for seed in result['seeds']:
        log=(root/f'xsim-seed-{seed}.log').read_text()
        summaries.append(log_review(log, seed))
        mutations = {'missing_pass': log.replace('WL PASS', 'REMOVED_PASS'),
                     'missing_group': log.replace('WL GROUP 7', 'REMOVED_GROUP 7'),
                     'missing_b': re.sub(r'^WL B .*\n', '', log, count=1, flags=re.M),
                     'wrong_address': log.replace('addr=3ffffffe0', 'addr=1ffffffe0'),
                     'late_failure': log+'\nFatal: injected\n'}
        hit=re.search(r'^WL W .*data=([0-9a-f]+)',log,re.M)
        mutations['wrong_payload']=log[:hit.start(1)]+f'{int(hit[1],16)^1:064x}'+log[hit.end(1):]
        for name, damaged in mutations.items():
            try:
                log_review(damaged, seed)
            except ValueError:
                negative.append(f'{seed}:{name}')
            else:
                raise ValueError('Negative control escaped: '+name)
    return dict(status='WEIGHT_LOADER_PREFLIGHT_REVIEWED', simulations=summaries,
                groups=sum(s['groups'] for s in summaries), checks=sum(s['checks'] for s in summaries),
                negative_controls=negative, native_pcie=False, native_hbm=False,
                loader_sha256=sha(root/'source/ntrpu_hbm_weight_loader.sv'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();value=json.dumps(review(a.root),indent=2)+'\n'
    if a.output:a.output.write_text(value)
    else:print(value,end='')
