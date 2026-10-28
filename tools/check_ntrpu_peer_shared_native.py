"""Audit unique HBM preload, physical region routing and shared ownership traces."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re
from run_ntrpu_peer_read import sha


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pattern(address):
    return (address ^ (address >> 8) ^ (address >> 16) ^ (address >> 24) ^ (address >> 32) ^ 0x5b) & 255


def memory_audit(text, native):
    need(text.count(f'SN ALL_PASS native={native}') == 1, 'native end marker')
    need(not re.search(r'(?im)^\s*(Fatal:|ERROR:)|SG_FAIL|SN_WATCHDOG', text), 'simulation failure')
    memory, writes, completed, windows = {}, {}, set(), {}
    queues = [deque(), deque()]
    bcounts = [0, 0]
    preload_done = False
    publications = []
    stats = Counter()
    boots = []
    phase = 0
    actual_cycles = {}
    for line in text.splitlines():
        p = line.split()
        if p[:1] != ['SN']:
            continue
        kind = p[1]
        if kind == 'BOOT':
            need(p[3:] == ['1', '1'], 'HBM init not complete')
            boots.append(int(p[2]))
        elif kind == 'WCMD':
            ts, lane, seq, address, beats = map(int, p[2:])
            key = lane, seq
            need(not preload_done and key not in writes and beats == 64, 'preload command')
            need(address == (seq-1)*4096 + (8589934592 if lane else 0), 'preload region/offset')
            writes[key] = dict(address=address, received=0)
        elif kind == 'WI':
            ts, lane, seq, beat = map(int, p[2:6])
            job = writes[lane, seq]
            need(beat * 64 == job['received'] and beat < 64, 'preload order')
            raw = bytes.fromhex(p[6])[::-1]
            need(len(raw) == 64, 'preload width')
            for i, value in enumerate(raw):
                address = job['address'] + beat*64 + i
                need(address not in memory and value == pattern(address), 'duplicate/corrupt preload')
                memory[address] = value
            job['received'] += 64
        elif kind == 'B':
            ts, lane, bid, resp = map(int, p[2:])
            need(bid == resp == 0 and not preload_done, 'write response')
            bcounts[lane] += 1
        elif kind == 'WCPL':
            ts, lane, seq = map(int, p[2:])
            key = lane, seq
            need(key not in completed and writes[key]['received'] == 4096, 'preload completion')
            need(bcounts[lane] == 8 * seq, 'completion without native B responses')
            completed.add(key)
        elif kind == 'PRELOAD_DONE':
            need(not preload_done and completed == {(0,i) for i in range(1,5)} | {(1,i) for i in range(1,3)}, 'preload coverage')
            need(len(memory) == 24576, 'unique initialized byte count')
            preload_done = True
        elif kind == 'LOCALITY_BEGIN':
            ts, number, a0, a1 = map(int, p[2:])
            expected = {1:(0,8589934592), 2:(8589934592,0), 3:(0,0)}
            need(preload_done and phase == 0 and number not in windows and (a0,a1) == expected[number], 'locality start')
            phase = number
            windows[number] = dict(begin_ps=ts, bases=[a0,a1], ports=[dict(ar=0,bytes=0,cycles=0,ar_stalls=0,r_stalls=0) for _ in range(2)])
        elif kind == 'LOCALITY_END':
            ts, number = map(int, p[2:])
            need(number == phase and not any(queues), 'locality end without drain')
            w = windows[number]
            need(w['ports'][0]['first_cycle'] == w['ports'][1]['first_cycle'] and w['ports'][0]['last_cycle'] == w['ports'][1]['last_cycle'], 'incomplete per-port cycle window')
            for port in w['ports']:
                need(port['ar'] == 8 and port['bytes'] == 4096, 'locality coverage')
                port['active_cycles'] = port['last_r_cycle'] - port['first_ar_cycle'] + 1
                port['bytes_per_active_cycle'] = port['bytes'] / port['active_cycles']
            w['end_ps'] = ts
            w['aggregate_active_cycles'] = max(v['last_r_cycle'] for v in w['ports']) - min(v['first_ar_cycle'] for v in w['ports']) + 1
            w['aggregate_bytes_per_active_cycle'] = 8192 / w['aggregate_active_cycles']
            phase = 0
        elif kind == 'PUBLISHED':
            need(preload_done and not any(queues) and phase == 0 and set(windows) == {1,2,3}, 'publication before ready')
            publications.append(int(p[3]))
        elif kind == 'AR':
            ts, number, direct, lane, cycle, address, beats = map(int, p[2:])
            need(preload_done and number == phase and address % 32 == 0 and 1 <= beats <= 16, 'native AR')
            need(all(address+i in memory for i in range(beats*32)), 'read uninitialized/unpublished gap')
            need(direct == bool(phase), 'read path mux')
            if direct:
                port = windows[phase]['ports'][lane]
                need(address == windows[phase]['bases'][lane] + port['ar']*512 and beats == 16, 'locality address order')
                port.setdefault('first_ar_cycle', cycle)
                port['ar'] += 1
            else:
                need(publications and address // 8589934592 == publications[-1] // 8589934592, 'shared read outside active region')
            queues[lane].append([address, beats, direct])
            stats['ar'] += 1
            stats['pc' + str(address >> 29) + '_ar'] += 1
            actual_cycles.setdefault((number,lane,cycle),Counter())['ar'] += int(bool(direct))
        elif kind == 'NR':
            ts, number, direct, lane, cycle, rid, resp, last = map(int, p[2:10])
            need(number == phase and rid == resp == 0 and queues[lane], 'unexpected/corrupt native response')
            address, beats, expected_direct = queues[lane][0]
            raw = bytes.fromhex(p[10])[::-1]
            need(direct == expected_direct and len(raw) == 32 and raw == bytes(memory[address+i] for i in range(32)) and last == int(beats == 1), 'native data/last/identity')
            stats['native_bytes'] += 32
            stats['direct_bytes' if direct else 'shared_bytes'] += 32
            if direct:
                port = windows[phase]['ports'][lane]
                port['bytes'] += 32;port['last_r_cycle'] = cycle
                actual_cycles.setdefault((number,lane,cycle),Counter())['r'] += 1
            if last:
                queues[lane].popleft()
            else:
                queues[lane][0] = [address+32, beats-1, direct]
        elif kind == 'CYCLE':
            ts, number, lane, cycle, av, ar, rv, rr = map(int, p[2:])
            need(number == phase and phase > 0, 'cycle outside window')
            port = windows[phase]['ports'][lane]
            need(cycle == port.get('last_cycle',cycle-1)+1, 'missing cycle evidence')
            port.setdefault('first_cycle',cycle)
            port['last_cycle'] = cycle;port['cycles'] += 1
            port['ar_stalls'] += int(bool(av and not ar));port['r_stalls'] += int(bool(rv and not rr))
            observed = actual_cycles.pop((number,lane,cycle),{})
            need(observed.get('ar',0) == av*ar and observed.get('r',0) == rv*rr, 'cycle/handshake conservation')
        elif kind != 'ALL_PASS':
            raise ValueError('unknown native trace: ' + kind)
    need(len(boots) == 1 and (not native or boots[0] > 40_000_000), 'single native initialization')
    need(publications == [0,8589934592] and not any(queues) and phase == 0, 'final native drain/publication')
    need(stats['native_bytes'] == 59776 and stats['direct_bytes'] == 24576 and stats['shared_bytes'] == 35200, 'native traffic coverage')
    need(set(windows) == {1,2,3} and all('end_ps' in w for w in windows.values()), 'missing locality end')
    need(not any(any(v.values()) for (phase,_,_),v in actual_cycles.items() if phase), 'unobserved direct handshakes')
    return dict(boot_ps=boots[0],unique_preload_bytes=len(memory),preload_transfers=len(completed),
                native_b_responses=sum(bcounts),publications=publications,counters=dict(stats),locality=windows)


def controls(text, native):
    trials = {}
    for token in ('SN WI ', 'SN B ', 'SN WCPL ', 'SN PRELOAD_DONE ', 'SN AR ', 'SN NR ', 'SN CYCLE ', 'SN PUBLISHED ', 'SN LOCALITY_END '):
        line = next(s for s in text.splitlines() if s.startswith(token))
        trials['drop_' + token.split()[1].lower()] = text.replace(line+'\n', '', 1)
    for token in ('SN WI ', 'SN NR '):
        line = next(s for s in text.splitlines() if s.startswith(token)); p = line.split()
        p[-1] = format(int(p[-1],16)^1, '0'+str(len(p[-1]))+'x')
        trials['corrupt_'+token.split()[1].lower()] = text.replace(line,' '.join(p),1)
    line = next(s for s in text.splitlines() if s.startswith('SN AR ') and int(s.split()[-2]) >= 8589934592)
    p = line.split();p[-2] = str(int(p[-2]) & ((1<<33)-1))
    trials['truncate_stack_bit'] = text.replace(line,' '.join(p),1)
    for name, changed in trials.items():
        try:
            memory_audit(changed,native)
        except (ValueError,KeyError,IndexError,AssertionError):
            continue
        raise ValueError('negative control accepted: '+name)
    return list(trials)


def review(out):
    r = json.loads((out/'result.json').read_text())
    need(r['experiment']=='peer_shared_native' and r['status']=='EXPORTED_PENDING_REVIEW', 'runner status')
    need(r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after'] and r['donor_unchanged'], 'runtime/source integrity')
    need(r['nice']>=10 and len(set(r['cpus']))==r['threads'], 'resource bounds')
    for n,h in json.loads((out/'FILES.json').read_text()).items():
        need((out/n).resolve().is_relative_to(out.resolve()) and sha(out/n)==h, 'artifact: '+n)
    profile = json.loads((out/'SHARED_NATIVE_PROFILE.json').read_text())
    need(r['sources']['SHARED_NATIVE_PROFILE.json']==sha(out/'SHARED_NATIVE_PROFILE.json'), 'profile binding')
    need(all(r['sources'][n]==h for n,h in profile['selected_rtl'].items()), 'qualified RTL changed')
    result = dict(result_sha256=sha(out/'result.json'),seconds=r['seconds'],threads=r['threads'],
                  selected_checkpoint_sha256=profile['selected_checkpoint_sha256'],
                  native_config_sha256=profile['native_config_sha256'],
                  integrated_timing_qualified=False,pcie_integrated=False,board_qualified=False)
    if r['stage']=='compile':
        inputs = json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
        need(len(inputs)>=10 and sum(v['bytes'] for v in inputs.values())>100_000_000,'native runtime missing')
        script = (out/'compile.sh').read_text()
        need('stub.sv' not in script and '-generic_top NATIVE=1' in script and 'ntrpu_peer_read_shared.sv' in script,'wrong elaboration')
        need(re.findall(r'--mt\s+(\d+)',script)==[str(r['threads'])],'elaboration thread count')
        for n,c in profile['selected_read_configs'].items():
            need(dict(s.split('\t',1) for s in (out/'suite'/n).read_text().splitlines() if s.startswith('CONFIG.'))==c,'IP config mismatch')
        need(sha(out/'suite/native_hbm_config.tsv')==profile['native_config_sha256'],'native config mismatch')
        need(sha(out/'suite/ntrpu_hbm_native_rw.sv')==profile['native_binding_sha256'],'native binding mismatch')
        log = (out/'console.log').read_text()
        need(not re.search(r'(?im)^\s*(ERROR:|Fatal:)',log) and 'tb_ntrpu_peer_shared_native' in log,'compile error')
        result.update(status='SHARED_NATIVE_SNAPSHOT_REVIEWED',simulation_executed=False,
                      snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),runtime_files=len(inputs),runtime_bytes=sum(v['bytes'] for v in inputs.values()))
    else:
        from check_ntrpu_shared_native_ledger import audit, negative_controls
        native = int(r['stage']=='traffic')
        text = (out/('evidence/simulate.log' if native else 'suite/simulation.log')).read_text()
        result.update(status='SHARED_NATIVE_TRAFFIC_REVIEWED' if native else 'SHARED_NATIVE_PREFLIGHT_REVIEWED',
                      simulation_executed=True,native_hbm_controller=bool(native),memory=memory_audit(text,native),
                      ownership=audit(text,True),negative_controls=controls(text,native),
                      ownership_negative_controls=negative_controls(text,True))
        need(result['ownership']['epoch_endings']==[dict(epoch=1,held=2,watermark=18,halted=True)],'final retained ownership')
        if native:
            need(r['runtime_inputs_unchanged'] and 'NTRPU_SHARED_NATIVE_WINDOW_COMPLETE' in text,'native runtime incomplete')
            prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
            need(prior['status']=='SHARED_NATIVE_SNAPSHOT_REVIEWED' and prior['result_sha256']==r['prior_result_sha256'],'snapshot provenance')
            need(prior['snapshot_manifest_sha256']==sha(out/'SNAPSHOT_INPUTS.json'),'snapshot manifest')
            rows=(out/'evidence/progress.tsv').read_text().splitlines();header=rows[0].split('\t')
            need(all(len(s.split('\t'))==len(header) for s in rows[1:]),'progress shape')
            values=[dict(zip(header,s.split('\t'))) for s in rows[1:]]
            ready=[v for v in values if v['started']=='1']
            need(ready and all(v['apb_complete_0']==v['apb_complete_1']==v['mem_rst_n']==v['apb_rst_n']=='1' and v['wfault']=='0' for v in ready),'initialization/progress')
            result['thermal_observed']=sorted({(v['DRAM_0_STAT_CATTRIP'],v['DRAM_1_STAT_CATTRIP']) for v in ready})
            result['thermal_qualified']=False
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(review(ap.parse_args().output),indent=2))
