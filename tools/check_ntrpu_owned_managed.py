"""Reconstruct MMIO, framed payloads and ownership proofs from public VCD ports.

The pool completion source is a fixture. External drain providers are absent.
No global drain, native PCIe, DDR or physical qualification is implied.
"""
from collections import Counter, deque
import re

from check_ntrpu_drain_sections_r2 import vcd_rows
from check_ntrpu_pcie_fragment_dual_managed import payload
from check_ntrpu_xdma_stop_inventory_r4 import markers


EPOCH = 0x100000001
TAG = 0x800000010000abcd
WINDOW = 0x1000fffc0
COOKIE = 0xffffffff
NAMES = ['fast_cmd', 'fast_data', 'fast_result', 'fast_wrong',
         'slow_cmd', 'slow_data', 'slow_result', 'slow_wrong']


def expected_mmio(phase):
    job = phase + 1
    channel = job % 2
    destination = 0x1fffffffc0 if channel else 0x100000040
    descriptor = [EPOCH, TAG, WINDOW, destination]
    words = [part for word in descriptor for part in [word & 0xffffffff, word >> 32]]
    ack_words = [EPOCH & 0xffffffff, EPOCH >> 32, TAG & 0xffffffff, TAG >> 32]
    writes = [(0x14, 1, 15, 0)]
    writes += [(0x20 + 4*i, word, 15, 0) for i, word in enumerate(words)]
    writes += [(0xb0, channel, 15, 0), (0x10, 1, 15, 0), (0xb0, 1-channel, 15, 2)]
    if phase == 2:
        writes += [(0x40 + 4*i, word, 15, 0) for i, word in enumerate(ack_words)]
    writes += [(0x20, 0xdeadbeef, 15, 2)]
    writes += [(0x40 + 4*i, word, 15, 0) for i, word in enumerate(ack_words)]
    writes += [(0x10, 4, 15, 2), (0x10, 1, 15, 2), (0x20, 2, 15, 2),
               (0xb0, 1-channel, 15, 2)]
    reads = [(0, 0x4452544e, 0), (8, 0xff, 0), (0x18, 0, 0),
             (0xb0, channel, 0), (0xb4, channel, 0),
             (0x18, 1, 0), (0x1c, 0, 0), (0x50, 1, 0), (0x54, 1, 0), (0xb8, 11, 0)]
    if phase == 2:
        reads += [(0x70, 0, 0), (0x64, 1, 0), (0x6c, 0x80000001, 0)]
    return writes, reads, channel, destination


def review(sim, wave):
    assert not re.search(r'^\s*(Fatal:|ERROR:|CRITICAL WARNING:)', sim, re.M)
    assert sim.count('MANAGED_COORDINATOR_SUITE_PASS cases=8 real_ntrd=1 real_owner=1 external_contracts=0 reset_driven=0') == 1
    assert '$finish called at time' in sim
    root = '/tb_ntrpu_dual_coordinated/'
    events = markers(sim, 'MCTRL_EVENT')
    passes = markers(sim, 'MCTRL_CASE_PASS')
    assert len(passes) == 8 and {int(p['id']) for p in passes} == set(range(8))
    cases = []
    for i, name in enumerate(NAMES):
        phase = [0, 1, 2, 0][i % 4]
        bad = i % 4 == 3
        writes, reads, channel, destination = expected_mmio(phase)
        case_events = [e for e in events if int(e['id']) == i]
        wanted = ['EMPTY_OWNER_BLOCKED', 'CONTROL_ARMED', 'PAYLOAD_HELD', 'MMIO_B_HELD',
                  'SOURCE_STOPPED_BEFORE_LOSS', 'REQUEST_REPLACED_STOPPED', 'LOSS_LATCHED_SOURCE_STOPPED',
                  'WRONG_EPOCH_BLOCKED' if bad else 'OWNER_CERTIFIED', 'MMIO_B_RELEASED', 'REUSE_BLOCKED']
        if not bad:
            wanted += ['RENEW_STOPPED', 'OWNER_RENEWED']
        wanted += ['OWNER_PAYLOAD_DRAINED', 'TERMINAL']
        assert [e['name'] for e in case_events] == wanted, (i, 'event order')
        assert all(int(e['phase']) == phase and int(e['bad']) == bad and
                   int(e['period']) == (5000 if i < 4 else 37000) for e in case_events)
        assert all(int(a['ps']) < int(b['ps']) for a, b in zip(case_events, case_events[1:]))
        cases.append(dict(id=i, prefix=root+name+'/', phase=phase, bad=bad,
                          expected_writes=writes, expected_reads=reads, channel=channel,
                          destination=destination, payload=payload(phase+1, 1),
                          writes=[], reads=[], aw=deque(), w=deque(), ar=deque(),
                          frame_aw=deque(), frame_w=deque(), counts=Counter(),
                          held={}, stalls=Counter(), cold_seen=False, epoch=0,
                          ack_tags=[], last_proof=0, last_clk=None, events=case_events,
                          event_index=0, last_state=None, event_edges={},
                          resume_ps=None, proof_latency_ps=[]))

    event_map = {'epoch': 'epoch', 'proof': 'proof_ack', 'accepted': 'accepted_mask',
                 'complete': 'complete_ack', 'lost': 'session_lost', 'invalid': 'session_invalid',
                 'bv': 's_axil_bvalid', 'resp': 's_axil_bresp', 'enabled': 'source_enable',
                 'held': 'lease_held', 'bridge': 'bridge_lease_held', 'ack': 'ack_valid',
                 'gen': 'generation', 'tag': 'tag', 'obsolete': 'obsolete'}
    for t, before, after in vcd_rows(wave):
        for c in cases:
            p = c['prefix']
            get = lambda name, state=after: state[p+name]
            while c['event_index'] < len(c['events']) and int(c['events'][c['event_index']]['ps']) <= t:
                event = c['events'][c['event_index']]
                expected = {name: int(event[key], 10 if key == 'obsolete' else
                                     16 if key in {'epoch', 'resp', 'gen', 'tag'} else 2)
                            for key, name in event_map.items()}
                candidates = [before, after] if int(event['ps']) == t else [before]
                assert any(all(state[p+n] == value for n, value in expected.items()) for state in candidates), (c['id'], event['name'], 'marker/VCD')
                name = event['name']
                assert expected['generation'] == EPOCH + c['bad']
                assert expected['complete_ack'] == expected['ack_valid'] == 0
                assert expected['epoch'] == (0 if name == 'EMPTY_OWNER_BLOCKED' else EPOCH)
                assert expected['lease_held'] == expected['bridge_lease_held'] == (name != 'EMPTY_OWNER_BLOCKED')
                stopped = name in {'SOURCE_STOPPED_BEFORE_LOSS', 'REQUEST_REPLACED_STOPPED',
                                   'LOSS_LATCHED_SOURCE_STOPPED', 'RENEW_STOPPED'}
                assert expected['source_enable'] == int(not stopped)
                certified = name in {'OWNER_CERTIFIED', 'MMIO_B_RELEASED', 'REUSE_BLOCKED',
                                     'OWNER_RENEWED', 'OWNER_PAYLOAD_DRAINED', 'TERMINAL'} and not c['bad']
                assert expected['proof_ack'] == expected['accepted_mask'] == int(certified), (c['id'], name, 'proof gate')
                assert expected['session_lost'] == int(c['event_index'] >= 6)
                assert expected['session_invalid'] == int(c['event_index'] >= 4)
                held_response = 3 <= c['event_index'] <= 7
                assert expected['s_axil_bvalid'] == int(held_response)
                if held_response:
                    assert expected['s_axil_bresp'] == 2
                c['event_edges'][name] = c['counts']['source_edges']
                c['event_index'] += 1
            if get('rst_n') != 1:
                assert not c['cold_seen'], (c['id'], 'operational reset')
                continue
            c['cold_seen'] = True
            assert get('management_cold_rst_n') == 1
            assert get('complete_ack') == 0 and get('proof_requests') & ~1 == 0 and get('accepted_mask') & ~1 == 0
            assert get('external_present') == get('external_ack') == get('external_fault') == 0
            assert get('retained_faults') == get('identity_fault') == get('control_fault') == 0
            if before[p+'source_enable'] == 0 and get('source_enable') == 1:
                c['resume_ps'] = t
            if get('proof_ack'):
                assert not c['bad'] and get('session_lost') == get('session_invalid') == 1
                assert get('proof_generation') == get('epoch') == get('bridge_epoch') == EPOCH
                assert get('proof_ack_tag') == get('proof_round_tag') == get('tag')
                assert get('lease_held') == get('bridge_lease_held') == 1
            if get('proof_ack') and c['last_proof'] == 0:
                c['ack_tags'].append(get('proof_ack_tag'))
                assert c['resume_ps'] is not None and t > c['resume_ps']
                c['proof_latency_ps'].append(t - c['resume_ps'])
            c['last_proof'] = get('proof_ack')
            if get('accepted_mask') & 1:
                assert get('proof_ack') and not c['bad']
            c['last_state'] = after
            if not (before[p+'clk'] == 0 and get('clk') == 1):
                continue
            s = lambda name: before[p+name]
            assert s('source_enable') == 1
            c['counts']['source_edges'] += 1
            assert s('epoch') == s('bridge_epoch') == c['epoch']
            assert s('lease_held') == s('bridge_lease_held') == bool(c['counts']['arm'])
            assert s('next_cookie') == COOKIE + c['counts']['line']
            if s('session_lost'):
                assert s('session_invalid') and s('irq') and c['epoch'] == EPOCH
                assert s('arm_ready') == s('retire_ready') == s('line_ready') == s('cpl_ready') == 0
            assert s('ack_valid') == 0, 'NTRD must not commit a rejected retirement'
            channels = {
                'ctl_aw': ('s_axil_awvalid', 's_axil_awready', ['s_axil_awaddr']),
                'ctl_w': ('s_axil_wvalid', 's_axil_wready', ['s_axil_wdata', 's_axil_wstrb']),
                'ctl_b': ('s_axil_bvalid', 's_axil_bready', ['s_axil_bresp']),
                'ctl_ar': ('s_axil_arvalid', 's_axil_arready', ['s_axil_araddr']),
                'ctl_r': ('s_axil_rvalid', 's_axil_rready', ['s_axil_rdata', 's_axil_rresp']),
                'frame_aw': ('s_axi_awvalid', 's_axi_awready', ['s_axi_awaddr', 's_axi_awid', 's_axi_awlen', 's_axi_awsize', 's_axi_awburst', 's_axi_awlock']),
                'frame_w': ('s_axi_wvalid', 's_axi_wready', ['s_axi_wdata', 's_axi_wstrb', 's_axi_wlast']),
                'frame_b': ('s_axi_bvalid', 's_axi_bready', ['s_axi_bid', 's_axi_bresp']),
                'cmd': ('cmd_valid', 'cmd_ready', ['cmd_channel', 'cmd_addr', 'cmd_tag', 'cmd_len']),
                'data': ('data_valid', 'data_ready', ['data']),
                'result': ('result_valid', 'result_ready', ['result_epoch', 'result_tag', 'result_status'])}
            for name, (valid, ready, fields) in channels.items():
                values = tuple(s(f) for f in fields)
                if name in c['held']:
                    assert s(valid) and values == c['held'][name], (c['id'], name, 'held payload changed')
                c['held'].pop(name, None)
                if s(valid) and not s(ready):
                    c['held'][name] = values
                    c['stalls'][name] += 1
            if s('s_axil_awvalid') and s('s_axil_awready'):
                c['aw'].append(s('s_axil_awaddr'))
            if s('s_axil_wvalid') and s('s_axil_wready'):
                c['w'].append((s('s_axil_wdata'), s('s_axil_wstrb')))
            if s('s_axil_bvalid'):
                assert c['aw'] and c['w'], 'orphan MMIO B'
                row = (c['aw'][0], *c['w'][0], s('s_axil_bresp'))
                assert row == c['expected_writes'][len(c['writes'])], (c['id'], 'MMIO write', row)
                if s('s_axil_bready'):
                    c['aw'].popleft(); c['w'].popleft(); c['writes'].append(row)
            if s('s_axil_arvalid') and s('s_axil_arready'):
                c['ar'].append(s('s_axil_araddr'))
            if s('s_axil_rvalid'):
                assert c['ar'], 'orphan MMIO R'
                row = (c['ar'][0], s('s_axil_rdata'), s('s_axil_rresp'))
                assert row == c['expected_reads'][len(c['reads'])], (c['id'], 'MMIO read', row)
                if s('s_axil_rready'):
                    c['ar'].popleft(); c['reads'].append(row)
            if s('arm_valid') and s('arm_ready'):
                assert c['counts']['arm'] == 0 and len(c['writes']) >= 11
                assert (s('arm_epoch'), s('arm_tag'), s('arm_window'), s('arm_destination'), s('arm_channel')) == (EPOCH, TAG, WINDOW, c['destination'], c['channel'])
                c['epoch'] = EPOCH; c['counts']['arm'] += 1
            if s('s_axi_awvalid') and s('s_axi_awready'):
                assert tuple(s(n) for n in channels['frame_aw'][2]) == (WINDOW, c['phase']+1, 0, 6, 1, 0)
                c['frame_aw'].append(True); c['counts']['frame_aw'] += 1
            if s('s_axi_wvalid') and s('s_axi_wready'):
                half = c['counts']['frame_w']; assert half in (0, 1)
                expected = 0x313046555052544e | (EPOCH << 64) | (TAG << 128) | (half*32 << 192)
                expected |= ((c['payload'] >> (half*256)) & ((1 << 256)-1)) << 256
                assert (s('s_axi_wdata'), s('s_axi_wstrb'), s('s_axi_wlast')) == (expected, (1 << 64)-1, 1)
                c['frame_w'].append(True); c['counts']['frame_w'] += 1
            if s('s_axi_bvalid'):
                assert c['frame_aw'] and c['frame_w']
                assert (s('s_axi_bid'), s('s_axi_bresp')) == (c['phase']+1, 0)
                if s('s_axi_bready'):
                    c['frame_aw'].popleft(); c['frame_w'].popleft(); c['counts']['frame_b'] += 1
            if s('line_valid') and s('line_ready'):
                assert (s('line_epoch'), s('line_tag'), s('line_address'), s('line_data')) == (EPOCH, TAG, c['destination'], c['payload'])
                assert c['counts']['frame_w'] == 2 and c['counts']['line'] == 0
                c['counts']['line'] += 1
            if s('cmd_valid'):
                assert c['counts']['line'] == 1 and c['counts']['cmd'] == 0
                assert tuple(s(n) for n in channels['cmd'][2]) == (c['channel'], c['destination'], COOKIE, 0)
                if s('cmd_ready'): c['counts']['cmd'] += 1
            if s('data_valid'):
                assert c['counts']['cmd'] == 1 and c['counts']['data'] == 0 and s('data') == c['payload']
                if s('data_ready'): c['counts']['data'] += 1
            takes = s('cpl_valid') & s('cpl_ready')
            if takes:
                assert c['phase'] == 2 and not s('session_lost') and c['counts']['data'] == 1
                assert takes == 1 << c['channel'] and s('cpl_tag') == COOKIE << (64*c['channel']) and s('cpl_status') == 0
                c['counts']['cpl'] += 1
            if s('done_valid') and s('done_ready'):
                assert c['counts']['cpl'] == 1 and c['phase'] == 2
                assert (s('done_epoch'), s('done_tag'), s('done_status')) == (EPOCH, TAG, 0)
                c['counts']['done'] += 1
            if s('result_valid'):
                assert c['phase'] == 2 and c['counts']['done'] == 1
                assert (s('result_epoch'), s('result_tag'), s('result_status')) == (EPOCH, TAG, 0)
                assert not s('result_ready')

    results = []
    for c in cases:
        assert c['event_index'] == len(c['events'])
        assert c['writes'] == c['expected_writes'] and c['reads'] == c['expected_reads']
        assert all(not c[k] for k in ['aw', 'w', 'ar', 'frame_aw', 'frame_w'])
        expected = dict(arm=1, line=1, cmd=1, data=1, frame_aw=2, frame_w=2, frame_b=2,
                        cpl=int(c['phase'] == 2), done=int(c['phase'] == 2))
        assert all(c['counts'][k] == v for k, v in expected.items())
        assert c['ack_tags'] == ([] if c['bad'] else [2, 3])
        assert c['last_state'][c['prefix']+'obsolete'] == int(not c['bad'])
        assert c['last_state'][c['prefix']+'finished'] == 1
        edges = c['event_edges']
        assert edges['SOURCE_STOPPED_BEFORE_LOSS'] == edges['REQUEST_REPLACED_STOPPED']
        assert edges['LOSS_LATCHED_SOURCE_STOPPED'] - edges['REQUEST_REPLACED_STOPPED'] == 1
        for label, fields, actual in [('MCTRL_WRITE', ['addr', 'data', 'strb', 'resp'], c['writes']),
                                     ('MCTRL_READ', ['addr', 'data', 'resp'], c['reads'])]:
            receipts = [r for r in markers(sim, label) if int(r['id']) == c['id']]
            assert [int(r['n']) for r in receipts] == list(range(1, len(actual)+1))
            assert [tuple(int(r[k], 16) for k in fields) for r in receipts] == actual
        summary = next(p for p in passes if int(p['id']) == c['id'])
        assert int(summary['writes']) == len(c['writes']) and int(summary['reads']) == len(c['reads']) and int(summary['retired']) == 0
        results.append(dict(id=c['id'], phase=c['phase'], bad_epoch=c['bad'],
                            writes=len(c['writes']), reads=len(c['reads']),
                            events=len(c['events']), transfers=dict(c['counts']),
                            stalls=dict(c['stalls']), proof_tags=c['ack_tags'],
                            proof_latency_after_source_resume_ps=c['proof_latency_ps'],
                            stopped_request_replacement_source_edges=0,
                            loss_latching_source_edges=1,
                            final_event_ps=int(c['events'][-1]['ps'])))
    return dict(status='PASS_EIGHT_REAL_NTRD_OWNER_LIFETIMES', cases=results,
                qualified_complete_drain=False, qualified_native_endpoint=False,
                qualified_ddr=False, qualified_reset=False)
