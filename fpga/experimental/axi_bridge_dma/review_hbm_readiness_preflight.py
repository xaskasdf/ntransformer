"""Review registered-readiness traffic and an actual RTL register-bypass control."""
import argparse
import hashlib
import json
from pathlib import Path
import re


LOADER_SHA = '14aca9b0b01fbb575671e4399a76a35613010b7a2e16b1e0637b9dbb3bda1103'
SOURCE_SHA = '165e12bfa07e588de337c25a1f37b2766790c5bd0b54ba711f906c5b17041162'
GROUPS = ['staging_stalls_retention', 'bus_reset_and_attribution', 'clock_pause_boundary_seal',
          'invalid_address_and_unavailable', 'bad_response_blocks_seal', 'bad_id',
          'lost_availability', 'cold_reset_recovery', 'registered_staggered_startup',
          'management_clock_pause', 'memory_halt_loss_and_late_b', 'no_rearm_after_relock',
          'cold_reset_with_stopped_clocks', 'registered_source_recovery_and_seal']


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closed(root):
    files = json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(files)|{'FILES.json'}, 'Closed inventory')
    for name, digest in files.items():
        require(sha(root/name) == digest, 'Closed file hash: '+name)
    result = json.loads((root/'RESULT.json').read_text())
    require(result['sources_unchanged'], 'Sources changed during execution')
    for name, digest in result['sources'].items():
        require(sha(root/'source'/name) == digest, 'Source hash: '+name)
    return result


def pattern(identifier):
    return sum((((identifier*71+i*37+91)^(i>>3))&255)<<(8*i) for i in range(32))


def log_review(log, seed):
    require(not re.search(r'(?im)^\s*(?:error|fatal):|WL FAIL|WL WATCHDOG', log), 'Failed simulation')
    require(re.findall(r'^WL GROUP (\d+) (\w+)$', log, re.M) ==
            [(str(i+1), name) for i, name in enumerate(GROUPS)], 'Named groups')
    verdict = re.findall(r'^WL PASS seed=(\d+) groups=(\d+) checks=(\d+) aw=(\d+) w=(\d+) b=(\d+)$', log, re.M)
    require(len(verdict) == 1, 'Unique PASS')
    values = tuple(map(int, verdict[0]))
    require(values[:2] == (seed, 14) and values[2] >= 1800 and values[3:] == (11, 11, 9), 'Accounting')
    aw = [(int(lane), int(address, 16)) for lane, address in re.findall(r'^WL AW lane=(\d+) addr=(\w+)$', log, re.M)]
    require(aw == [(0, 0x1000), (1, 0x200001000), (1, 0x3ffffffe0)] + [(0, 0x1000)]*5 +
            [(0, 0x2000), (0, 0x2020), (0, 0x2040)], 'Full HBM addresses')
    w = [(int(lane), int(data, 16)) for lane, data in re.findall(r'^WL W lane=(\d+) data=(\w+)$', log, re.M)]
    require(w == [(0, pattern(0)), (1, pattern(1)), (1, pattern(2))] +
            [(0, pattern(i)) for i in range(4, 12)], 'Exact independent HBM payloads')
    b = [tuple(int(x, 16) for x in entry) for entry in re.findall(r'^WL B lane=(\d+) resp=(\w+) id=(\w+)$', log, re.M)]
    require(b == [(0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 1)] + [(0, 0, 0)]*4, 'Response fault campaign')
    samples = [tuple(map(int, r)) for r in re.findall(r'^WR SAMPLE t=(\d+) expect=([01]) actual=([01])$', log, re.M)]
    counts = re.findall(r'^WR SOURCE samples=(\d+)$', log, re.M)
    require(len(counts) == 1 and int(counts[0]) == len(samples) > 1000, 'Source sample coverage')
    require(all(a == b for _, a, b in samples) and {a for _, a, _ in samples} == {0, 1}, 'Registered source mismatch')
    require(all(a[0] < b[0] for a, b in zip(samples, samples[1:])), 'Source sample time order')
    return dict(seed=seed, groups=14, checks=values[2], aw=11, w=11, b=9, source_samples=len(samples))


def review(root, bypass):
    result = closed(root)
    require(result['status'] == 'WEIGHT_LOADER_PREFLIGHT_EXPORTED', 'Closed successful bank')
    require(sha(root/'source/ntrpu_hbm_weight_loader.sv') == LOADER_SHA, 'Loader changed')
    require(sha(root/'source/ntrpu_hbm_loader_readiness.sv') == SOURCE_SHA, 'Readiness source changed')
    require(result['seeds'] == [0, 1, 3], 'Selected seed coverage')
    require(all(r['returncode'] == 0 and not r['timed_out'] for r in result['steps']), 'Failed or timed-out step')
    summaries = []; negative = []
    for seed in result['seeds']:
        log = (root/f'xsim-seed-{seed}.log').read_text()
        summaries.append(log_review(log, seed))
        mutations = {'missing_pass': log.replace('WL PASS', 'REMOVED_PASS'),
                     'missing_group': log.replace('WL GROUP 11', 'REMOVED_GROUP 11'),
                     'missing_b': re.sub(r'^WL B .*\n', '', log, count=1, flags=re.M),
                     'wrong_address': log.replace('addr=3ffffffe0', 'addr=1ffffffe0'),
                     'missing_source_sample': re.sub(r'^WR SAMPLE .*\n', '', log, count=1, flags=re.M),
                     'wrong_source_sample': log.replace('expect=1 actual=1', 'expect=1 actual=0', 1),
                     'late_failure': log+'\nFatal: injected\n'}
        hit = re.search(r'^WL W .*data=([0-9a-f]+)', log, re.M)
        mutations['wrong_payload'] = log[:hit.start(1)]+f'{int(hit[1],16)^1:064x}'+log[hit.end(1):]
        for name, damaged in mutations.items():
            try:
                log_review(damaged, seed)
            except ValueError:
                negative.append(f'{seed}:{name}')
            else:
                raise ValueError('Negative evidence control escaped: '+name)
    failed = closed(bypass)
    require(failed['status'] == 'FAILED' and failed['error'] == 'Failed step: xsim-seed-0', 'RTL bypass must fail in simulation')
    require(failed['completed'] == [] and len(failed['steps']) == 3, 'Unexpected RTL bypass campaign')
    require(all(not step['timed_out'] for step in failed['steps']), 'RTL bypass timeout is not a detected fault')
    require(all(step['returncode'] == 0 for step in failed['steps'][:2]), 'RTL bypass did not compile/elaborate')
    positive = result['sources']; altered = failed['sources']
    require(set(positive) == set(altered), 'Bypass source inventory differs')
    require({n for n in positive if positive[n] != altered[n]} == {'ntrpu_hbm_loader_readiness.sv'}, 'Unrelated RTL bypass changes')
    expected = (root/'source/ntrpu_hbm_loader_readiness.sv').read_text().replace(
        'always @(posedge clk or negedge cold_rst_n) begin', 'always @* begin')
    require((bypass/'source/ntrpu_hbm_loader_readiness.sv').read_text() == expected, 'Wrong RTL bypass mutation')
    log = (bypass/'xsim-seed-0.log').read_text()
    require('WL PASS' not in log and re.search(r'Fatal: WL FAIL group=\d+ registered readiness latency', log), 'RTL bypass escaped the register assertion')
    return dict(status='LOADER_REGISTERED_READINESS_PREFLIGHT_REVIEWED', simulations=summaries,
                groups=sum(s['groups'] for s in summaries), checks=sum(s['checks'] for s in summaries),
                source_samples=sum(s['source_samples'] for s in summaries),
                negative_evidence_controls=negative, rtl_bypass_rejected=True,
                loader_sha256=LOADER_SHA, readiness_sha256=SOURCE_SHA,
                native_pcie=False, native_hbm=False, native_traffic_rerun=False, board_qualified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path); p.add_argument('bypass', type=Path); p.add_argument('--output', type=Path)
    a = p.parse_args(); value = json.dumps(review(a.root, a.bypass), indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f: f.write(value)
    else:
        print(value, end='')
