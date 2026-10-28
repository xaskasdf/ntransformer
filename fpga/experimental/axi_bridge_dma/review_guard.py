"""Review local guard evidence from public VCD ports and structured console events."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HEALTHY = {0, 1, 2, 11}
DATA_FAULT = {9, 10, 15}
CODES = {3:2, 4:2, 5:1, 6:1, 7:1, 8:1, 9:2, 10:1, 14:3, 15:2, 16:2}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def vcd_rows(text):
    header, body = text.split('$enddefinitions $end', 1)
    require(re.search(r'\$timescale\s+1ps\s+\$end', header), 'VCD timescale')
    names = {}
    for line in header.splitlines():
        fields = line.split()
        if fields and fields[0] == '$var':
            names.setdefault(fields[3], []).append(fields[4])
    current, changes, now = {}, {}, 0
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('$'):
            continue
        if line.startswith('#'):
            after = {**current, **changes}
            yield now, current, after
            current, changes = after, {}
            value = int(line[1:]); require(value >= now, 'VCD timestamp order'); now = value
        else:
            raw, code = line[1:].split() if line.startswith('b') else (line[0], line[1:])
            value = None if re.search('[xzXZ]', raw) else int(raw, 2)
            require(code in names, 'VCD identifier')
            for name in names[code]:
                changes[name] = value
    yield now, current, {**current, **changes}


def review(log, vcd, case, clocks):
    require('FATAL' not in log and 'ERROR' not in log, 'Simulator diagnostics')
    require(f'BRIDGE_GUARD_LOCAL_PASS case={case} axi_half={clocks[0]} mgmt_half={clocks[1]} phase={clocks[2]}' in log,
            'Complete test identity')
    summaries = re.findall(r'^BG CASE_PASS (.*)$', log, re.M)
    require(len(summaries) == 1, 'Single verdict')
    result = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', summaries[0])}
    require(result['case'] == case, 'Case identity')
    expected_code = CODES.get(case, 0)
    require(result['code'] == expected_code and result['fault'] == int(case in CODES), 'Fault classification')
    expected_owned = int(case in CODES and case != 16)
    require(result['owned'] == expected_owned, 'Final ownership')
    expected_counts = dict(aw=0, w=0, b=0, ack=0, results=0)
    if case in HEALTHY | DATA_FAULT:
        expected_counts.update(aw=1, w=2, b=1, ack=1, results=int(case in HEALTHY))
    require(all(result[k] == v for k,v in expected_counts.items()), 'Expected public transaction totals')
    counts = dict(aw=0, w=0, b=0, ack=0, results=0)
    mgmt_edges=0; event_cycle=None; fault_cycle=None; cold_after_fault=False
    saw_fault=False; saw_closed_without_clock=False; last_mgmt=None; last_axi=None
    aw_stall=None; w_stall=None; final={}; edge_count=0
    for now, before, after in vcd_rows(vcd):
        final=after
        if before.get('mgmt_clk') == 0 and after.get('mgmt_clk') == 1:
            mgmt_edges+=1
            if last_mgmt is not None:
                require(now-last_mgmt == clocks[1]*2000, 'Independent management clock period')
            last_mgmt=now
            if before.get('guard_fault') == 1 and fault_cycle is None:
                fault_cycle=mgmt_edges
        if before.get('armed') == 1 and event_cycle is None:
            if any(before.get(k)==1 and after.get(k)==0 for k in
                   ('clock_enable','native_link_up','native_axi_reset_n','native_ctl_reset_n','perst_n')):
                event_cycle=mgmt_edges
        if saw_fault and after.get('cold_rst_n') == 0:
            cold_after_fault=True
        if after.get('guard_fault') == 1:
            saw_fault=True
            require(after.get('client_stop') == 1 and after.get('client_link_up') == 0, 'VCD fault closes admission asynchronously')
            if not cold_after_fault:
                require(after.get('client_cold_rst_n') == 1, 'VCD operational fault preserves cold reset')
                require(after.get('owned') == int(case != 16), 'VCD retained ownership')
                require(after.get('result_valid') == 0, 'VCD no fabricated success')
                if case != 16:
                    require((after.get('src_req_tag'),after.get('src_req_address'),after.get('src_req_bytes')) ==
                            ((9<<64)|1,0x1000,128), 'VCD retained source identity')
            if after.get('clock_enable') == 0:
                saw_closed_without_clock=True
        if before.get('axi_clk') == 0 and after.get('axi_clk') == 1:
            edge_count+=1
            if last_axi is not None and before.get('clock_enable') == 1:
                require(now-last_axi >= clocks[0]*2000, 'AXI clock edge spacing')
            last_axi=now
            if before.get('client_cold_rst_n') != 1:
                aw_stall=w_stall=None
                continue
            if aw_stall is not None:
                require(before.get('m_awvalid') == 1 and before.get('m_awaddr') == aw_stall, 'VCD AW stable across fault')
            if w_stall is not None:
                require(before.get('m_wvalid') == 1 and (before.get('m_wdata'),before.get('m_wlast')) == w_stall,
                        'VCD W stable across fault')
            aw_stall=before.get('m_awaddr') if before.get('m_awvalid')==1 and before.get('m_awready')==0 else None
            w_stall=(before.get('m_wdata'),before.get('m_wlast')) if before.get('m_wvalid')==1 and before.get('m_wready')==0 else None
            for kind,valid,ready in [('aw','m_awvalid','m_awready'),('w','m_wvalid','m_wready'),
                                     ('b','m_bvalid','m_bready'),('ack','ack_valid','ack_ready'),
                                     ('results','result_valid','result_ready')]:
                if before.get(valid)==1 and before.get(ready)==1:
                    if kind=='aw':
                        require(tuple(before.get(k) for k in ('m_awaddr','m_awlen','m_awsize','m_awburst','m_awid')) ==
                                (0x10000000,1,6,1,0), 'VCD AW identity')
                    elif kind=='w':
                        require(before.get('m_wdata') == int.from_bytes(bytes(range(counts['w']*64,(counts['w']+1)*64)),'little') and
                                before.get('m_wstrb')==(1<<64)-1 and before.get('m_wlast')==int(counts['w']==1), 'VCD payload/framing')
                    elif kind=='b':
                        require(counts['aw']==1 and counts['w']==2, 'VCD B follows complete write')
                    elif kind=='ack':
                        require(counts['b']==1, 'VCD attributed fixture ACK follows B')
                    elif kind=='results':
                        require(counts['ack']==1 and before.get('result_tag')==(9<<64)|1 and before.get('result_bytes')==128, 'VCD result causality')
                    counts[kind]+=1
    require(counts==expected_counts, 'Log/VCD transaction totals')
    require((final.get('guard_fault'),final.get('guard_code'),final.get('owned')) ==
            (result['fault'],result['code'],result['owned']), 'Log/VCD final state')
    if case not in HEALTHY | {12}:
        require(saw_fault and event_cycle is not None and fault_cycle is not None, 'VCD fault coverage')
        require(0 < fault_cycle-event_cycle <= 44 and result['latency']==fault_cycle-event_cycle, 'VCD independent detection bound')
        if case in {3,4,9,13,14,15,16}:
            require(saw_closed_without_clock, 'VCD closure with AXI clock stopped')
        if case==13:
            require(cold_after_fault, 'Explicit new configuration exercised')
    else:
        require(not saw_fault, 'VCD healthy/startup path')
    return dict(case=case,clocks=clocks,public_counts=counts,axi_edges=edge_count,
                detection_mgmt_cycles=result['latency'],fault_code=expected_code,
                closed_without_axi_clock=saw_closed_without_clock,status='GUARD_LOG_VCD_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('vcd',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--case',type=int,required=True)
    parser.add_argument('--clocks',type=int,nargs=3,required=True)
    args=parser.parse_args()
    result=review(args.log.read_text(),args.vcd.read_text(),args.case,args.clocks)
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))
