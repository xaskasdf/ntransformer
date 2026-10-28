"""Check public traces against the seven-contract composition requirements."""
import re
from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows


def review(sim, vcd):
    require(not re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:)', sim, re.M), 'simulator failure')
    rows = markers(sim, 'JOIN_CHECK')
    suite = markers(sim, 'JOIN_SUITE_PASS')
    scenarios = {}
    for case in range(10, 17): scenarios[case] = [0, 10, 11, 1, 2]
    scenarios[20] = [0, 1, 2, 20, 21, 1, 2]
    for case in range(30, 37): scenarios[case] = [0, 30, 31, 32]
    for case in range(40, 47): scenarios[case] = [0, 1, 2, 40, 41, 1, 2, 42, 43]
    for case in range(50, 57): scenarios[case] = [0, 1, 2, 50, 51]
    scenarios[60] = [0, 1, 2, 60, 61]
    scenarios[61] = [0, 1, 2, 62]
    for case in range(70, 77): scenarios[case] = [0, 1, 2, 70, 71, 72, 73, 74]
    scenarios[80] = [0, 80, 81, 82, 83]
    expected = [(case, label) for case, labels in scenarios.items() for label in labels]
    require(len(scenarios) == 39 and len(expected) == 238, 'review scenario manifest')
    require([(int(r['case']), int(r['label'])) for r in rows] == expected, 'complete scenario sequence')
    require(len(suite) == 1 and suite[0] == dict(checks='238', cases='39', native_endpoint='0', reset_driven='0'), 'scope and completion')
    require('$finish called at time' in sim, 'native simulation finish')
    for row in rows:
        answer = int(int(row['label']) in {2, 32, 83})
        require(row['expected'] == row['complete'] == str(answer), 'expected certificate for scenario')
        if answer:
            require(row['accepted'] == row['armed'] == row['request'] == '1111111', 'seven complete proofs')
            require(row['replay'] == row['revoked'] == '0' and row['faults'] == '0000000', 'healthy round')
    times = [int(r['t']) for r in rows]
    require(times == sorted(set(times)), 'unique increasing observation times')
    observations = {int(r['t']): r for r in rows}
    prefix = '/tb_ntrpu_drain_certificate_join/'
    fields = ['clk','cold','req','gen','tag','present','ready','fault','ack','ag','at',
              'request','accepted','armed','retained','pg','pt','cg','ct','complete','revoked','replay','case_id']
    found = set()
    eligible_history = set()
    lows = set()
    accepted_round = set()
    revoked_rounds = set()
    actual_identity = None
    incarnation = 0
    previous_cold = None
    complete_samples = 0
    edges = 0
    for time, before, after in vcd_rows(vcd):
        require(all(prefix+k in after for k in fields), 'public VCD signals')
        s = {k: after[prefix+k] for k in fields}
        old = {k: before[prefix+k] for k in fields}
        if all(value is None for value in s.values()):
            require(time == 0, 'only the pre-dump parser frame may be empty')
            continue
        if s['cold'] == 0:
            if previous_cold != 0: incarnation += 1
            lows.clear(); accepted_round.clear(); actual_identity = None
        previous_cold = s['cold']
        identity = (incarnation, s['pg'], s['pt'])
        if identity != actual_identity:
            lows.clear(); accepted_round.clear(); actual_identity = identity
        # Public requests establish the challenge. ACK must be low on a clock
        # after that challenge, never inferred from the expected test labels.
        rising = old['clk'] == 0 and s['clk'] == 1
        if rising:
            edges += 1
            for i in range(7):
                if old['cold'] == old['req'] == 1 and old['request'] is not None and old['ack'] is not None:
                    if old['request'] & (1<<i) and not old['ack'] & (1<<i): lows.add(i)
        qualified = set()
        for i in range(7):
            if (s['cold'] == s['req'] == 1 and s['gen'] == s['pg'] and s['tag'] == s['pt'] and
                all(s[k] is not None for k in ['ack','present','ready','ag','at']) and
                all(s[k] & (1<<i) for k in ['ack','present','ready']) and
                ((s['ag']>>(64*i)) & ((1<<64)-1)) == s['gen'] and
                ((s['at']>>(64*i)) & ((1<<64)-1)) == s['tag'] and i in lows):
                qualified.add(i)
        # Qualify only the positive assertions at settled timestamps. Revocation
        # and fresh-round handling are additionally checked at explicit markers.
        if s['complete'] == 1:
            require(len(qualified) == 7, 'complete without seven fresh matching public proofs')
            require(s['retained'] == s['fault'] == s['revoked'] == s['replay'] == 0, 'complete despite fault/revocation')
            require(s['cg'] == s['gen'] and s['ct'] == s['tag'], 'output identity')
            complete_samples += 1
        require(s['complete'] in {0, 1}, 'unknown complete output')
        if s['request'] is not None and s['request'] & 96:
            require(s['accepted'] is not None and s['accepted'] & 31 == 31, 'local proof before upstream certificates')
        if time in observations:
            row=observations[time];found.add(time)
            # A task may change its next stimulus at the same timestamp after
            # printing. Match one whole pre/post snapshot, never mix fields.
            expected_snapshot = dict(case_id=int(row['case']),complete=int(row['complete']),
                                     gen=int(row['gen'],16),tag=int(row['tag'],16))
            for name, signal in [('request','request'),('accepted','accepted'),('armed','armed'),('revoked','revoked'),('replay','replay'),('faults','retained'),('ack','ack'),('present','present'),('ready','ready')]:
                expected_snapshot[signal] = int(row[name],2) if set(row[name])<=set('01') else None
            candidates = [state for state in [old,s] if all(state[k]==v for k,v in expected_snapshot.items())]
            require(candidates, 'complete marker/VCD snapshot at '+str(time))
            observed=candidates[0]
            label=int(row['label'])
            if label in {41,43}: require(observed['revoked']==1, 'revocation retained')
            if label in {61,62}: require(observed['replay']==1, 'replay refused')
            if label==51: require(observed['retained'] & (1<<(int(row['case'])-50)), 'fault retained across request')
    require(found == set(times) and complete_samples > 39 and edges > 300, 'continuous evidence coverage')
    return dict(status='PASS_SEVEN_CONTRACT_MANAGEMENT_JOIN',cases=39,checks=len(rows),
                positive_checks=sum(r['complete']=='1' for r in rows),vcd_clock_edges=edges,
                last_observation_ps=times[-1],qualified_management_composition=True,
                qualified_external_providers=False,qualified_complete_native_drain=False,
                qualified_reset=False,qualified_physical_design=False)
