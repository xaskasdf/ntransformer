"""Check production wiring and replay against a previously reviewed public VCD.

Reuses the transaction/ownership checks without changing their assertions.
Only the testbench root name is mapped; signal values and times are unchanged.
This is bounded RTL replay, not formal equivalence or netlist qualification.
"""
from itertools import zip_longest
import re

from check_ntrpu_drain_owned_sections import review as review_sections
from check_ntrpu_drain_sections_r2 import vcd_rows


def review(sim, wave, reference_wave, period, mode):
    top = re.search(r'\$scope module (\S+) \$end', wave)[1]
    assert top.lstrip('\\').split('(')[0] == 'tb_ntrpu_owned_coordinator'
    normalized = wave.replace('tb_ntrpu_owned_coordinator',
                              'tb_ntrpu_drain_owned_sections')
    result = review_sections(sim, normalized, period, mode)
    new_root = '/' + re.search(r'\$scope module (\S+) \$end', normalized)[1] + '/'
    old_root = '/' + re.search(r'\$scope module (\S+) \$end', reference_wave)[1] + '/'
    pairs = [
        ('management_cold_rst_n', 'cold'), ('close_request', 'req'),
        ('generation', 'generation'), ('tag', 'tag'),
        ('proof_requests', 'proof_req'), ('proof_round_generation', 'pg'),
        ('proof_round_tag', 'pt'), ('complete_ack', 'complete'),
        ('complete_generation', 'cg'), ('complete_tag', 'ct'),
        ('accepted_mask', 'accepted'), ('armed_mask', 'armed'),
        ('retained_faults', 'retained_faults'), ('revoked', 'revoked'),
        ('identity_fault', 'replay'), ('proof_ack', 'owner/proof_ack'),
        ('proof_generation', 'owner/proof_generation'),
        ('proof_ack_tag', 'owner/proof_ack_tag'), ('fault_mgmt', 'owner/fault_mgmt')]
    names = None
    snapshots = 0
    for new, old in zip_longest(vcd_rows(normalized), vcd_rows(reference_wave)):
        assert new is not None and old is not None, 'public trace length differs'
        t, _, state = new
        assert t == old[0], ('public trace timestamp differs', t, old[0])
        if names is None:
            names = [name.removeprefix(old_root) for name in old[2]]
            assert len(names) > 250, 'incomplete reference scope'
        for name in names:
            assert new_root + name in state, ('missing reference signal', name)
            assert state[new_root + name] == old[2][old_root + name], (t, name)
        get = lambda name: state[new_root + name]
        core = lambda name: state[new_root + 'owner/dut/' + name]
        for output, observed in pairs:
            assert core(output) == get(observed), (t, 'coordinator binding', output)
        for name, observed in [('external_present', 'present'), ('external_ready', 'ready'),
                               ('external_ack', 'ack'), ('external_fault', 'fault')]:
            value = get(observed)
            if value is not None:
                assert core(name) == value >> 1, (t, name)
        for name, observed in [('external_generation', 'ack_gen'), ('external_tag', 'ack_tag')]:
            value = get(observed)
            if value is not None:
                assert core(name) == value >> 64, (t, name)
        if get('ready') is not None:
            assert core('owner_ready_mgmt') == get('ready') & 1
        snapshots += 1
    result.update(status='PASS_PRODUCTION_COORDINATOR_PUBLIC_REPLAY',
                  reference_public_signals=len(names), equal_timestamp_snapshots=snapshots,
                  production_bindings_checked=True, formal_equivalence=False,
                  netlist_equivalence=False)
    return result
