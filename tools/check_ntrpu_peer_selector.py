"""Audit selector lineage, differential fault coverage, and shared behavior."""
import argparse
import json
from pathlib import Path
import re
from run_ntrpu_peer_read import sha
from prepare_ntrpu_peer_selector import LANE, CANDIDATE, REFERENCE, variant
from check_ntrpu_peer_shared import audit, negative_controls, require


def lineage(repo):
    profile = json.loads((repo / 'SELECTOR_PROFILE.json').read_text())
    native = json.loads((repo / 'NATIVE_PROFILE.json').read_text())
    require(sha(repo / 'NATIVE_PROFILE.json') == profile['reference_profile_sha256'], 'native donor profile changed')
    require(native['selected_rtl'][LANE] == profile['reference_lane_sha256'], 'reference lane binding')
    original = (repo / REFERENCE).read_text().replace('module ntrpu_peer_read_lane_reference #(',
                                                    'module ntrpu_peer_read_lane #(')
    require(original == (repo / 'SELECTOR_REFERENCE.sv').read_text(), 'reference alias differs')
    require(sha(repo / 'SELECTOR_REFERENCE.sv') == profile['reference_lane_sha256'], 'original reference hash')
    require((repo / CANDIDATE).read_text() == variant(original), 'unexpected candidate change')
    require((repo / LANE).read_text() == variant(original, alias=True), 'private alias differs')
    require(sha(repo / LANE) == profile['selected_lane_sha256'] and
            sha(repo / CANDIDATE) == profile['candidate_sha256'], 'candidate profile hash')
    for name, digest in profile['reference_sources'].items():
        if name != LANE:
            require(sha(repo / name) == digest, 'donor source changed: ' + name)
    require(not profile['native_hbm_qualified'] and not profile['physical_timing_qualified'] and
            not profile['board_qualified'], 'candidate inherited qualification')
    return profile


def differential(text):
    require('SEL ALL_PASS cases=48' in text and not any(x in text for x in ('SEL_FAIL', 'SEL_WATCHDOG', 'Fatal:')), 'differential failed')
    rows = re.findall(r'^SEL CASE (\d+) PASS event=(\d+) seq=(\d+) fault=(\d+) held=(\d+) halted=(\d+)$', text, re.M)
    require(len(rows) == 48 and {int(v[0]) for v in rows} == set(range(1, 49)), 'case coverage')
    require({tuple(map(int, v[1:4])) for v in rows} == {(e, s, f) for e in range(2) for s in (1, 2) for f in range(12)}, 'event/context/fault coverage')
    for _, event, seq, fault, held, halted in rows:
        require((int(held), int(halted)) == ((0, 0) if int(fault) == 0 else (2, 1)), 'reservation retention')
    retire = re.findall(r'^SEL RETIRE (\d+) (\d+) (\d+) (\d+)$', text, re.M)
    require(len(retire) == 96, 'result count')
    faults = {int(v[0]): int(v[3]) for v in rows}
    for case in range(1, 49):
        events = [tuple(map(int, v[1:])) for v in retire if int(v[0]) == case]
        require(events == ([(1, 0, 2), (2, 0, 1)] if faults[case] == 0 else [(1, 1, 2), (2, 1, 2)]), 'retirement order/status/ownership')
    differences = list(map(int, re.findall(r'^SEL SELECTOR_DIVERGENCE (\d+)$', text, re.M)))
    expected = [int(v[0]) for v in rows if int(v[2]) == 2 and int(v[3]) in (1, 2)]
    require(differences == expected, 'partial identity collision coverage')
    m = re.search(r'^SEL COMPARED cycles=(\d+) divergences=4$', text, re.M)
    require(m is not None and int(m[1]) > 1000, 'missing public differential samples')
    return dict(cases=48, fault_cases=44, success_cases=4, result_handshakes=96,
                public_comparison_cycles=int(m[1]), selector_divergences=4)


def controls(text):
    changes = [('success_marker', text.replace('SEL ALL_PASS', 'SEL NOT_PASS', 1)),
               ('case', re.sub(r'^SEL CASE .*\n', '', text, count=1, flags=re.M)),
               ('result', re.sub(r'^SEL RETIRE .*\n', '', text, count=1, flags=re.M)),
               ('collision', re.sub(r'^SEL SELECTOR_DIVERGENCE .*\n', '', text, count=1, flags=re.M)),
               ('retention', text.replace('held=2 halted=1', 'held=0 halted=1', 1))]
    for name, bad in changes:
        try: differential(bad)
        except ValueError: pass
        else: raise ValueError('accepted negative control: ' + name)
    return [name for name, _ in changes]


def review(out):
    result = json.loads((out / 'result.json').read_text()); repo = out.parent / 'repo'
    require(result['status'] == 'EXPORTED_PENDING_REVIEW' and result['returncode'] == 0 and not result['timed_out'], 'run incomplete')
    require(result['sources'] == result['sources_after'], 'source changed during run')
    for name, digest in result['sources'].items(): require(sha(repo / name) == digest, 'source hash: ' + name)
    for name, digest in json.loads((out / 'FILES.json').read_text()).items(): require(sha(out / name) == digest, 'artifact hash: ' + name)
    profile = lineage(repo)
    if result.get('experiment') == 'peer_shared':
        require(result['full'] and result['functional_marker'], 'not a full shared run')
        native = json.loads((repo / 'NATIVE_PROFILE.json').read_text())
        for name, config in native['selected_read_configs'].items():
            observed = dict(line.split('\t', 1) for line in (out / 'suite' / name).read_text().splitlines() if line.startswith('CONFIG.'))
            require(observed == config, 'read IP configuration changed')
        log = out / 'suite/simulation.log'; text = log.read_text()
        checked = dict(status='SHARED_TRACE_REVIEWED', mode='full_behavioral',
                       **audit(text, True), negative_controls=negative_controls(text, True))
    else:
        require(result['experiment'] == 'peer_selector', 'unexpected experiment')
        log = out / 'simulation.log'; text = log.read_text()
        checked = dict(status='SELECTOR_DIFFERENTIAL_REVIEWED', **differential(text), negative_controls=controls(text))
    checked.update(result_sha256=sha(out / 'result.json'), log_sha256=sha(log),
                   selector_profile_sha256=sha(repo / 'SELECTOR_PROFILE.json'),
                   native_profile_preserved=True, native_hbm_qualified=False,
                   physical_timing_qualified=False, board_qualified=False)
    return checked


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()), sort_keys=True))
