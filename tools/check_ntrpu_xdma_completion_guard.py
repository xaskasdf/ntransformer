"""Review public-port VCD: AXI causality, register policy and bounded credits."""
import argparse
import copy
import json
import re
from pathlib import Path

from check_ntrpu_xdma_stop_inventory_r4 import markers, require, vcd_rows

TOP = '/tb_ntrpu_xdma_completion_guard/'
ADDRESSES = [0x1004, 0x1048, 0x1040, 0x1004, 0x1104, 0x1148, 0x1140, 0x1104]
PHASES = sorted(set(range(1, 57)) - {10, 12})


def samples(vcd):
    rows = []
    for t, before, after in vcd_rows(vcd):
        if before.get(TOP+'clk') == 0 and after.get(TOP+'clk') == 1 and before[TOP+'cold_rst_n'] == 1:
            row = {n[len(TOP):]: v for n, v in before.items() if n.startswith(TOP) and '/' not in n[len(TOP):]}
            row['time'] = t
            rows.append(row)
    return rows


def review_rows(rows):
    summaries = []
    for phase in PHASES:
        selected = [r for r in rows if r['phase'] == phase]
        require(selected, f'missing phase {phase}')
        published, observed, delivered = [0, 0], [0, 0], [0, 0]
        done_times = [[], []]
        pending = None
        serial = 0
        reads, poll, certificate = [], [], None
        boot_verified = False
        previous_fault = 0
        held = {}
        fault_edges = 0
        ar_stalls = r_stalls = 0
        for r in selected:
            t = r['time']
            require(all(r[n] is not None for n in ['faults','published_count','observed_count','delivered_count',
                    'descriptor_load','native_done','validated_done','admit_descriptor']), 'unknown live inventory')
            require(r['faults'] & previous_fault == previous_fault, 'fault must remain sticky')
            previous_fault = r['faults']
            for n, values in [('published_count',published),('observed_count',observed),('delivered_count',delivered)]:
                require(r[n] == values[0] | (values[1]<<32), 'inventory mismatch '+n)
            require(not r['awvalid'] and not r['wvalid'] and not r['m_req_write'] and not r['m_req_strobe'],
                    'read-only register ownership')
            for stem, valid, ready, fields in [
                ('request','m_req_valid','m_req_ready',['m_req_address','m_req_identity']),
                ('ar','arvalid','arready',['araddr','arprot']),
                ('r','rvalid','rready',['rdata','rresp'])]:
                payload = tuple(r[n] for n in fields)
                if stem in held:
                    require(r[valid] and payload == held[stem], 'offered transaction changed '+stem)
                if r[valid] and not r[ready]: held[stem] = payload
                else: held.pop(stem,None)
            ar_stalls += bool(r['arvalid'] and not r['arready'])
            r_stalls += bool(pending and pending.get('ar') and not pending.get('bus'))
            if r['m_req_valid'] and r['m_req_ready']:
                require(pending is None, 'one outstanding request')
                serial += 1
                require(r['m_req_identity'] == (phase<<64)|serial, 'unique epoch and transaction serial')
                require(r['m_req_address'] == ADDRESSES[(serial-1)%8], 'non-clearing register order')
                pending = dict(identity=r['m_req_identity'], address=r['m_req_address'], ar=None, bus=None)
            if r['arvalid'] and r['arready']:
                require(pending and pending['ar'] is None and r['araddr']==pending['address'] and r['arprot']==0,
                        'AXI AR must correspond to request')
                pending['ar'] = t
            if r['rvalid'] and r['rready']:
                require(pending and pending['ar'] is not None and pending['bus'] is None, 'R requires a real AR')
                pending['bus'] = (r['rdata'],r['rresp'],t)
            if r['m_rsp_valid'] and r['m_rsp_ready']:
                require(pending and pending['bus'], 'master must not fabricate R')
                data, code, rt = pending['bus']
                require((r['m_rsp_data'],r['m_rsp_code'])==(data,code), 'bus/response disagreement')
                identity_ok = r['m_rsp_identity']==pending['identity']
                if not identity_ok: require(phase==7, 'unexpected identity corruption')
                poll.append(dict(address=pending['address'], data=data, time=rt,
                                 ok=code==0 and identity_ok and not r['m_rsp_timeout']))
                reads.append(poll[-1])
                pending = None
                if len(poll)==8:
                    require([p['address'] for p in poll]==ADDRESSES, 'whole poll order')
                    good = all(p['ok'] for p in poll)
                    for c in range(2):
                        ctrl0,count,status,ctrl1 = poll[c*4:c*4+4]
                        good &= ctrl0['data']==ctrl1['data']==0x00f83e7f and status['data'] & ~5 == 0
                        good &= count['data']<=published[c]
                    if not boot_verified:
                        good &= poll[1]['data']==poll[5]['data']==0
                        boot_verified = bool(good)
                    certificate = copy.deepcopy(poll) if good else None
                    poll = []
            if r['admit_descriptor']:
                require(boot_verified, 'admission before initial readback')
            if r['faults']:
                fault_edges += 1
                require(r['admit_descriptor']==0 and r['validated_done']==0, 'fault must block admissions and credits')
            for c in range(2):
                load = (r['descriptor_load']>>c)&1
                raw = (r['native_done']>>c)&1
                credit = (r['validated_done']>>c)&1
                if load:
                    if phase!=45: require((r['admit_descriptor']>>c)&1, 'load without bounded admission')
                    published[c] += 1
                if raw:
                    if phase!=44: require(observed[c]<published[c], 'Done exceeds published inventory')
                    observed[c] += 1
                    done_times[c].append(t)
                if credit:
                    require(r['core_ready'] and r['exclusive_owner'] and not r['reset_request'] and not r['wait_expired']
                            and not r['external_fault'] and r['native_run']==3 and not r['m_timeout_latched'], 'revoked credit')
                    require(certificate is not None and delivered[c]<observed[c]<=published[c], 'unbacked credit')
                    count_read = certificate[c*4+1]
                    require(delivered[c]+1<=count_read['data'] and done_times[c][delivered[c]]<count_read['time']<t,
                            'credit requires count read after its Done')
                    delivered[c] += 1
                if phase!=45: require(published[c]-delivered[c]<=2, 'two-slot inventory bound')
        expected_credits = [2,2] if phase in (1,2,43) else [0,0]
        require(delivered==expected_credits, f'phase {phase} expected credits')
        if phase in (1,2): require(previous_fault==0, 'happy path fault')
        else:
            require(previous_fault and fault_edges>=6, 'negative case did not latch fault')
        if phase in (3,4,56):
            require(pending is None and selected[-1]['busy']==0, 'late real response must drain master')
        summaries.append(dict(phase=phase, credits=delivered, faults=previous_fault, reads=len(reads),
                              ar_stalls=ar_stalls, response_wait_cycles=r_stalls))
    return summaries


def review(log, vcd):
    require(not re.search(r'FATAL:|ERROR:|GUARD_ASSERT',log), 'simulator failure')
    end = markers(log,'GUARD_PASS')
    require(len(end)==1 and int(end[0]['cases'])==len(PHASES), 'complete test matrix')
    rows = samples(vcd)
    result = review_rows(rows)
    cases = markers(log,'GUARD_CASE')
    require(len(cases)==len(result), 'one case record per phase')
    for case, actual in zip(cases,result):
        require(int(case['phase'])==actual['phase'] and int(case['faults'],16)==actual['faults'] and
                [int(case['credits0']),int(case['credits1'])]==actual['credits'], 'log/VCD disagreement')
    return result, rows


def negative_controls(rows):
    mutations = [
        ('unbacked_credit', lambda r:r['phase']==1 and r['start_valid'], 'validated_done', 1),
        ('wrong_register', lambda r:r['phase']==1 and r['m_req_valid'] and r['m_req_ready'], 'm_req_address', 0x1044),
        ('stale_identity', lambda r:r['phase']==1 and r['m_rsp_valid'] and r['m_rsp_ready'], 'm_rsp_identity', 0),
        ('bus_data_changed', lambda r:r['phase']==1 and r['rvalid'] and r['rready'], 'rdata', 0),
        ('inventory_erased', lambda r:r['phase']==3 and r['faults'], 'published_count', 0),
        ('fault_cleared', lambda r:r['phase']==46 and r['faults'], 'faults', 0),
    ]
    caught = []
    for name, select, field, value in mutations:
        altered = [dict(r) for r in rows]
        positions = [i for i,r in enumerate(altered) if select(r)]
        require(positions, 'mutation target '+name)
        index = positions[1] if name=='fault_cleared' else positions[0]
        require(altered[index][field]!=value, 'mutation must change a value')
        altered[index][field] = value
        try: review_rows(altered)
        except ValueError: caught.append(name)
        else: raise ValueError('undetected evidence mutation '+name)
    return caught


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    results,rows=review((args.output/'simulation.log').read_text(),(args.output/'guard.vcd').read_text())
    print(json.dumps(dict(status='COMPLETION_GUARD_REVIEW_PASS',cases=len(results),
                         validated_credits=sum(sum(r['credits']) for r in results),
                         negative_controls=negative_controls(rows))))
