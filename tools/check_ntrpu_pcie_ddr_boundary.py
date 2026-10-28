"""Reconstruct eleven XDMA descriptors at the generic 128-GiB aperture.

Transport evidence only. Source/IP provenance and model diagnostics must also
pass before qualification; error descriptors never represent valid output.
"""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

from check_ntrpu_pcie_ddr_ffn import rows
from check_ntrpu_pcie_ddr_periodic import records, require, review_periodic
from check_ntrpu_pcie_ddr_errors import check_drain

CAPACITY = 1 << 37
DESCRIPTOR, SOURCE, DESTINATION = 0x600000100, 0x600002000, 0x700002000
CHANNELS = ('aw', 'w', 'b', 'ar', 'r')


def case_contract(case):
    require(1 <= case <= 11, 'known boundary case')
    read = case not in (1, 2, 5, 10)
    bad = case in (5, 8)
    address = 0 if case in (2, 4, 7, 10, 11) else CAPACITY - 64
    length = 128 if bad else 64
    if case in (1, 3):
        data = bytes(0x3c ^ n for n in range(64))
    elif address == 0:
        data = bytes(0x96 ^ n for n in range(64))
    else:
        data = bytes(0xc3 ^ n for n in range(64))
    if case == 5:
        data += bytes(0x69 ^ n for n in range(64))
    if case == 8:
        data += bytes(64)
    return int(read), int(bad), address, length, data


def check_boot(text):
    ui = records(text, 'NTRPU_DDR_FFN_UI', ['sample', 'ps', 'dt'])
    require(len(ui) == 2 and [u['sample'] for u in ui] == [1, 2] and
            ui[1]['ps'] - ui[0]['ps'] == 5000, 'actual calibrated UI period')
    phase = ui[0]['ps'] % 5000
    admitted = records(text, 'NTRPU_BOUNDARY_ADMITTED', ['polls', 'time_ps'])
    require(len(admitted) == 1 and 0 <= admitted[0]['polls'] <= 4000,
            'one bounded cold admission')
    when = admitted[0]['time_ps']
    state = rows(text, 'NTRPU_PCIE_DDR_STATE')
    require(state and any(s['pcie_resetn'] == s['hrst'] == '0' for s in state), 'observed PCIe reset')
    ready = lambda s: all(s[k] == '1' for k in ('pcie_resetn', 'link', 'rst', 'hrst', 'da', 'ha', 'run', 'pr', 'calib')) and s['ui_rst'] == '0'
    held = [s for s in state if s['ph'] == '1']
    live = [s for s in state if ready(s)]
    require(held and live and int(held[0]['ps']) < int(live[0]['ps']) <= when and
            ui[0]['ps'] < when, 'reset/calibration/peer admission order')
    require(all(ready(s) for s in state if int(s['ps']) >= when), 'admission stable for transport')
    require(not re.search(r'^NTRPU_PCIE_FFN_(?:JOB|DONE|IRQ|ACK|RETIRE|SERVICE_ACCEPT|MSIX) ', text, re.M),
            'no compute or interrupt workload')
    require(records(text, 'NTRPU_BOUNDARY_HOST_INIT', ['cases', 'computed']) == [dict(cases=11, computed=0)],
            'transport-only host initialization')
    begin = records(text, 'NTRPU_BOUNDARY_BEGIN', ['cases', 'time_ps'])
    ctl = records(text, 'NTRPU_BOUNDARY_DEVCTL', ['time_ps'], ['value'])
    require(len(begin) == len(ctl) == 1 and begin[0]['cases'] == 11 and
            when < ctl[0]['time_ps'] == begin[0]['time_ps'], 'suite starts after admission/configuration')
    require((ctl[0]['value'] >> 5) & 7 == 1 and (ctl[0]['value'] >> 12) & 7 == 1,
            'actual MPS and MRRS 256')
    return dict(ui_phase_ps=phase, admitted_ps=when, begin_ps=begin[0]['time_ps'])


def check_axi(axi, case, direction, bad, address, data, start, end, phase, summaries):
    buses = []
    for scope in range(3):
        observed = {}
        for channel in CHANNELS:
            observed[channel] = [r for r in axi[channel] if r['case'] == case and r['scope'] == scope]
        count = 2 if scope < 2 and bad else 1
        want = {ch: count if (ch in ('ar','r') if direction else ch in ('aw','w','b')) else 0 for ch in CHANNELS}
        require({ch: len(v) for ch,v in observed.items()} == want, 'exact AXI split/no invalid MIG forwarding')
        summary = [r for r in summaries if r['case'] == case and r['scope'] == scope]
        require(len(summary) == 1 and {ch: summary[0][ch] for ch in CHANNELS} == want and
                start < summary[0]['ps'] <= end, 'AXI summary from accepted handshakes')
        stop = summary[0]['ps']
        for channel, events in observed.items():
            require(all(a['ps'] < b['ps'] for a,b in zip(events, events[1:])), 'channel handshake order')
            for n, r in enumerate(events):
                require(start < r['ps'] < stop and r['ps'] % (4000 if scope == 0 else 5000) ==
                        (0 if scope == 0 else phase), 'handshake on actual admitted clock edge')
                response = 3 if scope < 2 and bad and n == 1 else 0
                if channel in ('aw','ar'):
                    require((r['addr'],r['len'],r['size'],r['burst']) == (address+64*n,0,6,1) and
                            0 <= r['id'] < 16, 'full-width aligned split address/ID')
                    require(scope != 2 or r['addr']+64 <= CAPACITY, 'MIG valid range')
                elif channel == 'w':
                    require(r['data'].to_bytes(64,'little') == data[64*n:64*(n+1)] and
                            r['strb'] == (1 << 64)-1 and r['last'] == 1, 'actual full write bytes/strobes/LAST')
                elif channel == 'b':
                    aw, w = observed['aw'][n], observed['w'][n]
                    require(r['id'] == aw['id'] and r['resp'] == response and
                            r['ps'] > max(aw['ps'],w['ps']), 'B follows its accepted AW and W')
                else:
                    ar = observed['ar'][n]
                    require(r['id'] == ar['id'] and r['resp'] == response and r['last'] == 1 and
                            r['ps'] > ar['ps'] and r['data'].to_bytes(64,'little') == data[64*n:64*(n+1)],
                            'actual read bytes, ID, response and LAST')
        buses.append(observed)
    for ch in CHANNELS:
        for n, mig in enumerate(buses[2][ch]):
            host, pre = buses[0][ch][n], buses[1][ch][n]
            require({k:v for k,v in host.items() if k not in ('scope','ps')} ==
                    {k:v for k,v in pre.items() if k not in ('scope','ps')} ==
                    {k:v for k,v in mig.items() if k not in ('scope','ps')}, 'forwarded valid transaction identity')
            require(host['ps'] < pre['ps'] <= mig['ps'] if ch in ('aw','w','ar') else
                    mig['ps'] <= pre['ps'] < host['ps'], 'CDC and guard causal ordering')
        if bad and buses[0][ch]:
            host, pre = buses[0][ch][1], buses[1][ch][1]
            require({k:v for k,v in host.items() if k not in ('scope','ps')} ==
                    {k:v for k,v in pre.items() if k not in ('scope','ps')}, 'rejected upper-address identity')
            require(host['ps'] < pre['ps'] if ch in ('aw','w','ar') else pre['ps'] < host['ps'],
                    'rejected response follows CDC')
    return buses


def check_host(text, direction, bad, length, data, start, end, buses, summary):
    result = bytearray([0xa5]*128)
    seen = set(); packet = None; desc_bytes = source_bytes = 0; requests = []
    for line in text.splitlines():
        if not line.startswith('NTRPU_PCIE_FFN_CQ_'): continue
        r = dict(re.findall(r'(\w+)=([^ ]+)', line))
        ps = int(r['time_ps'])
        if not start <= ps <= end: continue
        if line.startswith('NTRPU_PCIE_FFN_CQ_REQUEST '):
            require(packet is None, 'no overlapping host packet')
            address, count, typ = int(r['addr'],16), int(r['bytes']), int(r['type'])
            require(int(r['tc']) == 0 and int(r['attr']) == (2 if typ == 0 else 0) and 0 < count <= 256 and count % 4 == 0 and
                    address % 4 == 0 and (address & 4095)+count <= 4096, 'host packet attributes/page/size')
            requests.append(r)
            if typ == 0:
                if address == DESCRIPTOR+desc_bytes and desc_bytes+count <= 32:
                    desc_bytes += count
                else:
                    require(not direction and address == SOURCE+source_bytes and source_bytes+count <= length,
                            'actual host read maps descriptor or H2C source')
                    source_bytes += count
            else:
                require(typ == 1 and direction and DESTINATION <= address and address+count <= DESTINATION+length,
                        'posted C2H stays inside separate result buffer')
                packet = dict(address=address, count=count, used=0, ps=ps)
        else:
            require(line.startswith('NTRPU_PCIE_FFN_CQ_DATA ') and packet is not None, 'data follows packet')
            first, count, left = int(r['first']), int(r['bytes']), int(r['left'])
            require(int(r['addr'],16) == packet['address'] and left == packet['count']-packet['used'] and
                    first == (0 if packet['used'] else 16) and count == min(left,64-first) and ps >= packet['ps'],
                    'actual CQ payload progression')
            payload = int(r['data'],16).to_bytes(64,'little')[first:first+count]
            for lane, value in enumerate(payload):
                offset = packet['address']-DESTINATION+packet['used']+lane
                require(offset not in seen and value == data[offset], 'exact published bytes without overlap')
                # Published error bytes are measured, never reported as valid output.
                require(buses[0]['r'][offset//64]['ps'] < ps, 'published byte follows actual XDMA AXI R')
                seen.add(offset); result[offset] = value
            packet['used'] += count; packet['ps'] = ps
            if packet['used'] == packet['count']: packet = None
    require(packet is None and desc_bytes == 32 and source_bytes == (0 if direction else length),
            'complete descriptor and requested H2C payload')
    require((direction and (bad or seen == set(range(length)))) or (not direction and not seen),
            'all valid C2H bytes, observed partial error C2H, no H2C publication')
    coverage = sum(1 << i for i in seen)
    require((summary['descriptors'],summary['source'],summary['received'],summary['coverage'],summary['contents']) ==
            (desc_bytes,source_bytes,len(seen),coverage,int.from_bytes(result,'little')) and
            start < summary['time_ps'] <= end, 'host summary including untouched bytes reconstructed from CQ')
    logged = [r for r in rows(text,'NTRPU_PCIE_RX_LOG_PACKET') if r['channel'] == '0' and start <= int(r['time_ps']) <= end]
    require(len(logged) == len(requests), 'independent packet log count')
    for log, request in zip(logged,requests):
        write = int(request['type']) == 1; count = int(request['bytes'])
        require(int(log['head'],16) == ((0x60000000 if write else 0x20002000) | count//4) and
                int(log['bytes']) == 16+(count if write else 0) and int(log['time_ps']) >= int(request['time_ps']),
                'actual 64-bit PCIe header and length')
    return dict(c2h_observed_bytes=len(seen), coverage_hex=f'{coverage:032x}', contents_hex=result.hex())


def review_transfers(text):
    require(not re.search(r'(?mi)\b(?:ERROR|FATAL)\s*:|^\s*FAIL\b|^\s*CRITICAL WARNING:',text), 'no simulator/model error')
    boot = check_boot(text)
    descriptions = records(text,'NTRPU_BOUNDARY_DESCRIPTOR',['case','direction','bad','bytes','time_ps'],['address','source','destination','raw'])
    passes = records(text,'NTRPU_BOUNDARY_CASE_PASS',['case','direction','bad','completed','polls','time_ps'],['status'])
    starts = records(text,'NTRPU_BOUNDARY_AXI_BEGIN',['case','direction','bad','ps'])
    controls = records(text,'NTRPU_BOUNDARY_CONTROL',['case','direction','time_ps'],['value'])
    runs = records(text,'NTRPU_BOUNDARY_RUN',['case','direction','time_ps'],['value'])
    hosts = records(text,'NTRPU_BOUNDARY_HOST_PASS',['case','descriptors','source','received','time_ps'],['coverage','contents'])
    summaries = records(text,'NTRPU_BOUNDARY_AXI_PASS',['case','scope','aw','w','b','ar','r','ps'])
    polls_all = records(text,'NTRPU_BOUNDARY_STATUS',['case','direction','time_ps'],['value'])
    require(all(len(a) == 11 for a in (descriptions,passes,starts,controls,runs,hosts)) and len(summaries) == 33,
            'exact eleven-case evidence coverage')
    require(all(1 <= r['case'] <= 11 for r in polls_all), 'no unscoped status')
    axi = {}
    for channel in CHANNELS:
        decimal = {'aw': ['len','size','burst','id'], 'ar': ['len','size','burst','id'],
                   'w': ['last'], 'b': ['id','resp'], 'r': ['id','resp','last']}[channel]
        hexadecimal = {'aw': ['addr'], 'ar': ['addr'], 'w': ['data','strb'], 'b': [], 'r': ['data']}[channel]
        axi[channel] = records(text,'NTRPU_BOUNDARY_AXI_'+channel.upper(),['case','scope','ps']+decimal,hexadecimal)
        require(all(1 <= r['case'] <= 11 and r['scope'] in range(3) for r in axi[channel]), 'no unscoped bus event')
    csr = [(int(ps),int(v,16),int(a,16)) for ps,v,a in re.findall(
        r'^\[\s*(\d+)\] : Data read ([0-9a-fA-F]+) from Address ([0-9a-fA-F]+)$',text,re.M)]
    cases = []; totals = [dict.fromkeys(CHANNELS,0) for _ in range(3)]
    for i in range(11):
        case = i+1
        direction,bad,address,length,data = case_contract(case)
        d,p,start,c,h = descriptions[i],passes[i],starts[i],controls[i],hosts[i]
        require(all((r['case'],r['direction'],r['bad']) == (case,direction,bad) for r in (d,p,start)) and
                c['case'] == h['case'] == case and c['direction'] == direction, 'case order and identity')
        source,destination = (address,DESTINATION) if direction else (SOURCE,address)
        descriptor = 0xad4b0013 | (length << 32) | (source << 64) | (destination << 128)
        require((d['address'],d['source'],d['destination'],d['bytes'],d['raw']) ==
                (DESCRIPTOR,source,destination,length,descriptor), 'exact full-width descriptor')
        require((passes[i-1]['time_ps'] if i else boot['begin_ps']) < d['time_ps'] <= start['ps'] <
                c['time_ps'] < p['time_ps'], 'isolated sequential descriptor lifetime')
        require(c['value'] == 0xfffe7e, 'error logging, incrementing and stopped control')
        csr_base = 0x1000 if direction else 0
        require((c['time_ps'],c['value'],csr_base+4) in csr, 'actual control read completion')
        run = runs[i]
        require(run['case'] == case and run['direction'] == direction and run['value'] == 0xfffe7f and
                c['time_ps'] < run['time_ps'] < p['time_ps'] and
                (run['time_ps'],run['value'],csr_base+4) in csr, 'Run rising edge read back from the actual channel')
        mask = (0x200 if direction else 0x4000) if bad else 0
        state_mask = 0x47 if bad else 7
        require(p['status'] & ~state_mask == mask and not p['status'] & 1 and 0 < p['polls'] <= 100 and
                (p['completed'] in (0,1) if bad else p['status'] == 6 and p['completed'] == 1), 'terminal DMA status/count')
        polls = [r for r in polls_all if r['case'] == case]
        require(len(polls) == p['polls'] and polls[-1]['value'] == p['status'] and
                all(r['direction'] == direction and run['time_ps'] < r['time_ps'] < p['time_ps'] and
                    r['value'] & ~state_mask & ~mask == 0 and (r['time_ps'],r['value'],csr_base+0x40) in csr for r in polls),
                'status from actual root completions')
        require(all(a['time_ps'] < z['time_ps'] for a,z in zip(polls,polls[1:])), 'status poll chronology')
        counts = [(ps,value) for ps,value,addr in csr if addr == csr_base+0x48 and polls[-1]['time_ps'] < ps < p['time_ps']]
        require(len(counts) == 1 and counts[0][1] == p['completed'], 'actual descriptor completion count')
        buses = check_axi(axi,case,direction,bad,address,data,start['ps'],p['time_ps'],boot['ui_phase_ps'],summaries)
        host = check_host(text,direction,bad,length,data,start['ps'],p['time_ps'],buses,h)
        for scope,bus in enumerate(buses):
            for channel in CHANNELS: totals[scope][channel] += len(bus[channel])
        cases.append(dict(case=case,direction=direction,rejected=bad,status=p['status'],completed=p['completed'],
                          elapsed_ps=p['time_ps']-start['ps'],**host))
    require(totals == [dict(aw=5,w=5,b=5,ar=8,r=8),dict(aw=5,w=5,b=5,ar=8,r=8),dict(aw=4,w=4,b=4,ar=7,r=7)],
            'exact bank totals')
    intervals = [(s['ps'],p['time_ps']) for s,p in zip(starts,passes)]
    require(all(any(a <= int(r['time_ps']) <= z for a,z in intervals) for marker in
                ('NTRPU_PCIE_FFN_CQ_REQUEST','NTRPU_PCIE_FFN_CQ_DATA') for r in rows(text,marker)),
            'no host DMA traffic outside cases')
    final = records(text,'NTRPU_BOUNDARY_PASS',['cases','rejected','computed','time_ps'])
    require(len(final) == 1 and (final[0]['cases'],final[0]['rejected'],final[0]['computed']) == (11,2,0) and
            passes[-1]['time_ps'] < final[0]['time_ps'], 'complete transport-only suite')
    return dict(status='PASS_BOUNDARY_TRANSFERS_ONLY',**boot,cases=cases,axi_counts=totals,
                h2c_requested_bytes=320,mig_written_bytes=256,valid_c2h_bytes=384,
                rejected_c2h_observed_bytes=cases[7]['c2h_observed_bytes'],final_ps=final[0]['time_ps'])


def review(text):
    result = review_transfers(text)
    periodic = review_periodic(text)
    drain = check_drain(text,periodic,'NTRPU_DDR_PERIODIC_DRAIN')
    finishes = re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    require(finishes and {int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in finishes} ==
            {drain['end_ps']} == {result['final_ps']}, 'actual finish coincides with complete drain')
    result.update(status='PASS_COLD_PCIE_DDR_BOUNDARY',periodic=periodic,periodic_drain=drain,
        scope='Eleven transport-only descriptors at zero and the generic 128-GiB aperture. '
              'Valid prefix may commit before a later AXI burst is rejected. Error C2H coverage is observed, '
              'never valid output. No compute, exact DIMM, warm reset, ownership or physical throughput qualification.')
    return result


def mutations(text):
    variants = {}
    def change(name,marker,key,base=10,case=None,scope=None):
        row = next(r for r in rows(text,marker) if (case is None or int(r.get('case',-1)) == case) and
                   (scope is None or int(r.get('scope',-1)) == scope))
        line = next(l for l in text.splitlines() if l.startswith(marker+' ') and dict(re.findall(r'(\w+)=([^ ]+)',l)) == row)
        value = int(row[key],base)^1
        changed = re.sub(r'\b'+key+'='+re.escape(row[key])+r'(?= |$)',key+'='+(format(value,'x') if base == 16 else str(value)),line,count=1)
        variants[name] = text.replace(line,changed,1)
    for case in range(1,12):
        change(f'descriptor_{case}','NTRPU_BOUNDARY_DESCRIPTOR','raw',16,case)
        change(f'status_{case}','NTRPU_BOUNDARY_CASE_PASS','status',16,case)
        change(f'host_coverage_{case}','NTRPU_BOUNDARY_HOST_PASS','coverage',16,case)
        change(f'host_contents_{case}','NTRPU_BOUNDARY_HOST_PASS','contents',16,case)
    for ch,key,base,case in [('AW','addr',16,5),('AW','len',10,5),('AW','size',10,5),('AW','burst',10,5),
                            ('W','data',16,5),('W','strb',16,5),('W','last',10,5),('B','resp',10,5),('B','id',10,5),
                            ('AR','addr',16,8),('R','data',16,8),('R','resp',10,8),('R','last',10,8),('R','id',10,8),
                            ('R','data',16,7),('R','data',16,9),('R','data',16,11)]:
        for scope in range(3): change(f'bus_{case}_{scope}_{ch}_{key}','NTRPU_BOUNDARY_AXI_'+ch,key,base,case,scope)
    for marker in ('ADMITTED','BEGIN','DEVCTL','DESCRIPTOR','CONTROL','RUN','STATUS','AXI_BEGIN','AXI_PASS','HOST_PASS','CASE_PASS','PASS'):
        full = 'NTRPU_BOUNDARY_'+marker
        line = next(l for l in text.splitlines() if l.startswith(full+' '))
        variants['missing_'+marker] = text.replace(line+'\n','',1)
        variants['duplicate_'+marker] = text.replace(line,line+'\n'+line,1)
    for ch in ('AW','W','B','AR','R'):
        case = 5 if ch in ('AW','W','B') else 8
        lines = [l for l in text.splitlines() if l.startswith(f'NTRPU_BOUNDARY_AXI_{ch} case={case} scope=1 ')]
        require(len(lines) == 2, 'negative test uses real split evidence')
        line = lines[1]
        variants['invalid_mig_'+ch] = text.replace(line,line+'\n'+line.replace('scope=1','scope=2',1),1)
        variants['missing_rejected_'+ch] = text.replace(line+'\n','',1)
    line = next(l for l in text.splitlines() if l.startswith('NTRPU_PCIE_FFN_CQ_DATA '))
    row = dict(re.findall(r'(\w+)=([^ ]+)',line))
    value = int(row['data'],16) ^ (1 << (8*int(row['first'])))
    variants['published_payload'] = text.replace(line,line.replace('data='+row['data'],'data='+format(value,'x'),1),1)
    for name,marker,old,new in [
        ('read_relaxed_ordering','NTRPU_PCIE_FFN_CQ_REQUEST type=0 ','attr=2','attr=0'),
        ('read_no_snoop','NTRPU_PCIE_FFN_CQ_REQUEST type=0 ','attr=2','attr=3'),
        ('write_relaxed_ordering','NTRPU_PCIE_FFN_CQ_REQUEST type=1 ','attr=0','attr=2'),
        ('read_serialized_attribute','NTRPU_PCIE_RX_LOG_PACKET channel=0 ','head=20002008','head=20000008')]:
        line=next(l for l in text.splitlines() if l.startswith(marker) and old in l)
        variants[name]=text.replace(line,line.replace(old,new,1),1)
    variants['prefixed_error'] = text+'\nModel:ERROR: deliberate evidence corruption\n'
    variants['unexpected_compute'] = text+'\nNTRPU_PCIE_FFN_JOB sequence=00000001 time_ps=0\n'
    return variants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'))
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
