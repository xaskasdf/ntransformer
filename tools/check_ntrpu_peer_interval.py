"""Audit cached interval bounds, public differential tests and shared behavior."""
import argparse
import json
from pathlib import Path
import re
from run_ntrpu_peer_read import sha
from prepare_ntrpu_peer_interval import GUARD, CANDIDATE, REFERENCE, variant
from check_ntrpu_peer_shared import audit, negative_controls, require


def lineage(repo):
    profile = json.loads((repo / 'INTERVAL_PROFILE.json').read_text())
    original = (repo / 'INTERVAL_REFERENCE.sv').read_text()
    require(sha(repo / 'INTERVAL_REFERENCE.sv') == profile['reference_guard_sha256'] ==
            profile['reference_sources'][GUARD], 'reference guard binding')
    require((repo / REFERENCE).read_text() == original.replace('module ntrpu_peer_read_admission #(',
            'module ntrpu_peer_admission_reference #('), 'reference alias differs')
    require((repo / CANDIDATE).read_text() == variant(original), 'unexpected interval variant')
    require((repo / GUARD).read_text() == variant(original, alias=True), 'private guard differs')
    require(sha(repo / CANDIDATE) == profile['candidate_sha256'] and
            sha(repo / GUARD) == profile['selected_guard_sha256'], 'candidate hashes')
    for name, digest in profile['reference_sources'].items():
        if name != GUARD:
            require(sha(repo / name) == digest, 'donor source changed: ' + name)
    require(not any(profile[n] for n in ('native_hbm_qualified', 'physical_timing_qualified',
                                       'board_qualified')), 'inherited qualification')
    return profile


def differential(text):
    require('INT ALL_PASS cases=324' in text and not any(x in text for x in
            ('INT_FAIL', 'INT_WATCHDOG', 'Fatal:')), 'differential failed')
    cases = list(map(int, re.findall(r'^INT CASE (\d+) PASS$', text, re.M)))
    require(cases == list(range(1, 325)), 'case coverage/order')
    offers = [tuple(map(int, r)) for r in re.findall(r'^INT OFFER case=(\d+) lane=(\d+) code=(\d+)$', text, re.M)]
    # Values from ntrpu_peer_read_pkg: RANGE=3, UNPUBLISHED=4, SPAN=5, REPLAY=6.
    expected = [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 1, 2, 6,
                0, 0, 4, 4, 4, 3, 4, 4, 0, 5, 3, 4, 5, 1, 2, 6]
    require(offers[:64] == [(1 + 32 * lane + kind, lane, code)
                           for lane in range(2) for kind, code in enumerate(expected)], 'boundary/error priorities')
    require(offers[64:71] == [(65, 0, 0), (66, 1, 3), (66, 1, 0),
                             (67, 0, 2), (67, 0, 0), (68, 0, 0), (68, 1, 0)], 'configuration lifetime')
    require(len(offers) == 327 and [(case, lane) for case, lane, _ in offers[71:]] ==
            [(69 + i, i % 2) for i in range(256)], 'deterministic random coverage')
    require(all(0 <= code <= 7 for _, _, code in offers), 'invalid error code')
    m = re.search(r'^INT COMPARED cycles=(\d+) offered=(\d+) accepted=(\d+) rejected=(\d+)$', text, re.M)
    require(m is not None, 'missing counters')
    cycles, offered, accepted, rejected = map(int, m.groups())
    require(cycles > 3000 and offered == 327 and accepted == sum(code == 0 for _, _, code in offers)
            and rejected == offered - accepted, 'counter mismatch')
    return dict(cases=324, directed_boundary_cases=64, configuration_cases=4, random_cases=256,
                public_comparison_cycles=cycles, public_output_bits=2519,
                offered=offered, accepted=accepted, rejected=rejected)


def controls(text):
    changes = [('marker', text.replace('INT ALL_PASS', 'INT NOT_PASS', 1)),
               ('case', re.sub(r'^INT CASE .*\n', '', text, count=1, flags=re.M)),
               ('offer', re.sub(r'^INT OFFER .*\n', '', text, count=1, flags=re.M)),
               ('priority', text.replace('case=14 lane=0 code=1', 'case=14 lane=0 code=4', 1)),
               ('invalid_cfg', text.replace('case=65 lane=0 code=0', 'case=65 lane=0 code=4', 1)),
               ('counters', text.replace('offered=327', 'offered=326', 1))]
    for name, bad in changes:
        require(bad != text, 'ineffective negative control: ' + name)
        try: differential(bad)
        except ValueError: pass
        else: raise ValueError('accepted negative control: ' + name)
    return [name for name, _ in changes]


def review(out):
    r = json.loads((out / 'result.json').read_text()); repo = out.parent / 'repo'
    require(r['status'] == 'EXPORTED_PENDING_REVIEW' and r['returncode'] == 0 and not r['timed_out'], 'run incomplete')
    require(r['sources'] == r['sources_after'], 'source changed')
    for name, digest in r['sources'].items(): require(sha(repo / name) == digest, 'source hash: ' + name)
    for name, digest in json.loads((out / 'FILES.json').read_text()).items(): require(sha(out / name) == digest, 'artifact hash: ' + name)
    lineage(repo)
    if r['experiment'] == 'peer_shared':
        require(r['functional_marker'], 'shared functional marker')
        if r['full']:
            native = json.loads((repo / 'NATIVE_PROFILE.json').read_text())
            for name, config in native['selected_read_configs'].items():
                observed = dict(line.split('\t', 1) for line in (out / 'suite' / name).read_text().splitlines() if line.startswith('CONFIG.'))
                require(observed == config, 'read IP configuration changed')
        log = out / ('suite/simulation.log' if r['full'] else 'simulation.log'); text = log.read_text()
        checked = dict(status='SHARED_TRACE_REVIEWED', mode='full_behavioral' if r['full'] else 'unit',
                       **audit(text, r['full']), negative_controls=negative_controls(text, r['full']))
    else:
        require(r['experiment'] == 'peer_interval', 'unexpected experiment')
        log = out / 'simulation.log'; text = log.read_text()
        checked = dict(status='INTERVAL_DIFFERENTIAL_REVIEWED', **differential(text), negative_controls=controls(text))
    checked.update(result_sha256=sha(out / 'result.json'), log_sha256=sha(log),
                   interval_profile_sha256=sha(repo / 'INTERVAL_PROFILE.json'),
                   native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False)
    return checked


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()), sort_keys=True))
