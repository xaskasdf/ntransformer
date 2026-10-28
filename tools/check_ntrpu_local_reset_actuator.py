"""Independent event/identity and real edge checks for the local reset bank."""
import re
from check_ntrpu_axil_pair_drain import review as review_pair


def need(ok, message):
    if not ok:
        raise ValueError(message)


def review_local(text):
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )', text, re.M), 'simulator failure')
    names = ['unrequested_reset_and_unknown_phase', 'stopped_host_blocks_held',
             'early_release_and_missing_held_peer', 'both_local_releases_and_peer_readiness',
             'fresh_hold_clears_previous_counts', 'withdrawal_and_hold_dominance',
             'generation_is_part_of_command_identity', 'readiness_loss_requires_parent_recovery']
    cases = re.findall(r'^LOCAL_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$', text, re.M)
    need([(int(i), n) for i, n, _ in cases] == list(enumerate(names)), 'eight ordered cases')
    need(text.splitlines().count('LOCAL_SUITE_PASS cases=8 hold_cycles=64 native_endpoint=0 complete_drain=0') == 1, 'scope')
    rows = []
    for line in text.splitlines():
        if line.startswith('LOCAL_EVENT '):
            row = dict(re.findall(r'(\w+)=(\S+)', line))
            rows.append({k: v if k == 'name' else int(v, 16 if k in ['gen', 'tag'] else 10) for k, v in row.items()})
    def event(case, name):
        values = [r for r in rows if r['case'] == case and r['name'] == name]
        need(len(values) == 1, 'unique event ' + name)
        return values[0]
    for case, name, fields in [
        (0, 'NO_HOLD', dict(ack=0, hn=1, dn=1, held=0)),
        (1, 'HOST_STOP', dict(ack=0, hn=0, dn=0, held=1, hh=0, dh=1, he=0)),
        (2, 'EARLY_READY', dict(ack=0, hn=0, dn=0, held=1, hh=0, dh=1)),
        (2, 'NO_PEER_HELD', dict(ack=0, held=1, hh=1, dh=1)),
        (3, 'DEVICE_STOP_RELEASE', dict(ack=0, held=0, hn=1, dn=0)),
        (3, 'NO_PEER_READY', dict(ack=0, held=0, hn=1, dn=1)),
        (4, 'DEVICE_STOP_NEW_HOLD', dict(ack=0, held=1, hn=0, dn=0, hh=1, dh=0, de=0)),
        (5, 'WITHDRAWN', dict(ack=0, held=1, hn=0, dn=0)),
        (5, 'HOLD_DOMINATES_READY', dict(ack=0, held=1)),
        (6, 'NEW_GENERATION', dict(ack=0, held=1, hh=0, dh=1, he=0)),
        (7, 'READY_LOST', dict(ack=0, held=0, hn=1, dn=1)),
    ]:
        row = event(case, name)
        need(all(row[k] == v for k, v in fields.items()), name + ' fields')
    answers = [r for r in rows if r['name'] == 'ACK']
    need([(r['case'], r['kind']) for r in answers] == [(3,5),(4,2),(5,5),(6,2),(6,2),(7,5),(7,5)], 'response sequence')
    for r in answers:
        need(r['ack'] == 1 and r['tag'] > 0xffffffff and r['gen'] > 0xffffffff, 'full identity')
        if r['kind'] == 2:
            need(r['he'] >= 64 and r['de'] >= 64 and r['held'] == r['hh'] == r['dh'] == 1 and r['hn'] == r['dn'] == 0, 'real held edges')
        else:
            need(r['held'] == 0 and r['hn'] == r['dn'] == r['hu'] == r['du'] == 1, 'both releases')
    old, new = [r for r in answers if r['case'] == 6]
    need(old['tag'] == new['tag'] and new['gen'] == old['gen'] + 1, 'generation freshness')
    need('$finish called at time' in text, 'completion')
    return dict(cases=names, answers=answers, events=rows)


def review_integration(text, period):
    result = review_pair(text, period)
    commands = []
    for line in text.splitlines():
        if line.startswith('AUTO_RESET '):
            r = dict(re.findall(r'(\w+)=(\S+)', line))
            commands.append({k: v if k == 'event' else int(v, 16 if k == 'tag' else 10) for k, v in r.items()})
    need([r['event'] for r in commands] == ['START','HELD','READY'] * 2, 'two automatic reset cycles')
    for i in [0,3]:
        start, held, ready = commands[i:i+3]
        need(start['ps'] < held['ps'] < ready['ps'], 'phase order')
        need(start['tag'] == held['tag'] and ready['tag'] == held['tag'] + 1, 'phase identities')
        need(held['he'] >= 64 and held['de'] >= 64, 'automatic real edge proof')
    result['automatic_phases'] = commands
    result['scope'] = 'Local actuator RTL drives the real AMD AXI-Lite CDC reset. Fixture-controlled source isolation and attached peers; not the complete endpoint coordinator or DMA drain.'
    return result
