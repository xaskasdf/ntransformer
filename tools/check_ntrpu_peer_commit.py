"""Audit the context-commit stage with ordered public transaction scoreboards."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from run_ntrpu_peer_read import sha
from prepare_ntrpu_peer_commit import LANE, CANDIDATE, REFERENCE, variant
from check_ntrpu_peer_shared import audit, negative_controls, require


def lineage(repo):
    p = json.loads((repo / 'COMMIT_PROFILE.json').read_text())
    original = (repo / 'COMMIT_REFERENCE.sv').read_text()
    require(sha(repo / 'COMMIT_REFERENCE.sv') == p['reference_lane_sha256'] ==
            p['reference_sources'][LANE], 'reference lane binding')
    require((repo / REFERENCE).read_text() == original.replace('module ntrpu_peer_read_lane #(',
            'module ntrpu_peer_commit_reference #('), 'reference alias differs')
    require((repo / CANDIDATE).read_text() == variant(original), 'unexpected commit variant')
    require((repo / LANE).read_text() == variant(original, alias=True), 'private lane differs')
    require(sha(repo / CANDIDATE) == p['candidate_sha256'] and
            sha(repo / LANE) == p['selected_lane_sha256'], 'candidate hashes')
    for name, digest in p['reference_sources'].items():
        if name != LANE: require(sha(repo / name) == digest, 'donor source changed: ' + name)
    require(not any(p[n] for n in ('native_hbm_qualified', 'physical_timing_qualified', 'board_qualified')),
            'inherited qualification')
    return p


def differential(text):
    require(text.count('CMT ALL_PASS cases=80') == 1 and not any(s in text for s in
            ('CMT_FAIL', 'CMT_WATCHDOG', 'Fatal:')), 'differential failed')
    require(list(map(int, re.findall(r'^CMT CASE (\d+) PASS', text, re.M))) == list(range(1,81)), 'case coverage')
    streams = defaultdict(list)
    for case, dut, event, payload in re.findall(r'^CMT TX (\d+) ([01]) (REQ|CMD|DATA|REJECT|RESULT) ([0-9a-fA-F]+)$', text, re.M):
        streams[int(case), int(dut), event].append(int(payload, 16))
    require(len(re.findall(r'^CMT TX ', text, re.M)) == sum(map(len, streams.values())), 'unparsed transaction')
    totals = defaultdict(int)
    for case in range(1,81):
        require(streams[case,0,'REQ'] and streams[case,1,'REQ'], 'no input transactions')
        for event in ('REQ','CMD','DATA','REJECT','RESULT'):
            a, b = streams[case,0,event], streams[case,1,event]
            require(a == b, f'ordered public transactions differ: case={case} event={event}')
            totals[event] += len(a)
    require(all(totals[e] > 0 for e in ('REQ','CMD','DATA','REJECT','RESULT')), 'missing channel coverage')
    require(len(streams[76,1,'CMD']) == len(streams[76,1,'RESULT']) == 2, 'consecutive capture')
    require((streams[73,1,'REJECT'][0] & 15) == 7 and (streams[74,1,'REJECT'][0] & 15) == 6, 'pending overlap/replay')
    require((streams[75,1,'REJECT'][0] & 15) == 2 and (streams[75,1,'RESULT'][0] >> 64 & 15) == 0, 'revoke lifetime')
    for case in range(57,73):
        require(len(streams[case,1,'RESULT']) == (1 if case < 65 else 2) and
                all(v >> 64 & 15 for v in streams[case,1,'RESULT']), 'staged fault result')
    m = re.search(r'^CMT COMPARED cycles=(\d+) paired=(\d+)$', text, re.M)
    require(m is not None and int(m[1]) > 1000 and int(m[2]) == 9, 'coverage counters')
    return dict(cases=80, observed_cycles=int(m[1]), paired_event_cycles=int(m[2]),
                transactions_per_variant=dict(totals), latency_sensitive_public_checks=True)


def controls(text):
    candidate = re.search(r'^CMT TX \d+ 1 CMD [0-9a-fA-F]+$', text, re.M)
    require(candidate is not None, 'missing candidate command')
    changes = [('marker',text.replace('CMT ALL_PASS','CMT NOT_PASS',1)),
               ('case',re.sub(r'^CMT CASE .*\n','',text,count=1,flags=re.M)),
               ('drop_command',text[:candidate.start()] + text[candidate.end()+1:]),
               ('corrupt_command',text[:candidate.start()] + candidate[0][:-1] + ('1' if candidate[0][-1]!='1' else '2') + text[candidate.end():]),
               ('paired',text.replace('paired=9','paired=8',1)),
               ('fatal',text+'\nFatal: CMT_FAIL\n')]
    for name, bad in changes:
        require(bad != text, 'ineffective control: '+name)
        try: differential(bad)
        except ValueError: pass
        else: raise ValueError('accepted negative control: '+name)
    return [name for name,_ in changes]


def review(out):
    r=json.loads((out/'result.json').read_text());repo=out.parent/'repo'
    require(r['status']=='EXPORTED_PENDING_REVIEW' and r['returncode']==0 and not r['timed_out'], 'incomplete run')
    require(r['sources']==r['sources_after'], 'sources changed')
    for name,digest in r['sources'].items():require(sha(repo/name)==digest,'source hash: '+name)
    for name,digest in json.loads((out/'FILES.json').read_text()).items():require(sha(out/name)==digest,'artifact hash: '+name)
    lineage(repo)
    if r['experiment']=='peer_shared':
        if r['full']:
            for name,config in json.loads((repo/'NATIVE_PROFILE.json').read_text())['selected_read_configs'].items():
                observed=dict(line.split('\t',1) for line in (out/'suite'/name).read_text().splitlines() if line.startswith('CONFIG.'))
                require(observed==config,'read IP configuration changed')
        log=out/('suite/simulation.log' if r['full'] else 'simulation.log');text=log.read_text()
        checked=dict(status='SHARED_TRACE_REVIEWED',mode='full_behavioral' if r['full'] else 'unit',
                     **audit(text,r['full']),negative_controls=negative_controls(text,r['full']))
    else:
        require(r['experiment']=='peer_commit','unexpected experiment')
        log=out/'simulation.log';text=log.read_text()
        checked=dict(status='COMMIT_TRANSACTIONS_REVIEWED',**differential(text),negative_controls=controls(text))
    checked.update(result_sha256=sha(out/'result.json'),log_sha256=sha(log),commit_profile_sha256=sha(repo/'COMMIT_PROFILE.json'),
                   native_hbm_qualified=False,physical_timing_qualified=False,board_qualified=False)
    return checked


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output.resolve()),sort_keys=True))
