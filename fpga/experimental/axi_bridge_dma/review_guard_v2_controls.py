"""Review actual public traces for short native pulses and admission retention."""
import re
from review_guard import require, vcd_rows


def review_controls(log, vcd, case):
    input_id=case//3 if case<12 else case-12
    stop=case%3 if case<12 else 0
    code=3 if stop else 1
    require(not re.search(r'(?im)^\s*(?:fatal|error):',log),'Simulator diagnostics')
    require(log.count(f'V2_CONTROL_PASS case={case} input={input_id} stop={stop} code={code}')==1,'Unique control case verdict')
    match=re.findall(rf'^V2 CONTROL_DROP case={case} input={input_id} stop={stop} t=(\d+)$',log,re.M)
    require(len(match)==1,'Control event identity');start=int(match[0])
    drop_seen=False;fault_at=None;last_mgmt=None;held_edges=0;recovery_edges=0
    lows={};widths={};last={}
    for now,before,after in vcd_rows(vcd):
        last=after
        if before.get('mgmt_clk')==0 and after.get('mgmt_clk')==1:
            if last_mgmt is not None:require(now-last_mgmt==14000,'Continuous management clock')
            last_mgmt=now
        if now<start:continue
        if now==start:
            require(before.get('native_status')==15 and after.get('native_status')==15-(1<<input_id),'Actual native input falls')
            require(before.get('client_link_up')==before.get('armed')==1,'Admitted before input loss')
            drop_seen=True
        require(after.get('client_link_up')==0,'No reopening after the native pulse')
        require(after.get('client_cold_rst_n')==after.get('cold_rst_n')==1,'Operational loss does not cold-reset')
        for bit in range(4):
            old=(before.get('native_status',15)>>bit)&1
            new=(after.get('native_status',15)>>bit)&1
            if old==1 and new==0:lows[bit]=now
            if old==0 and new==1:
                require(bit in lows and bit not in widths,'Attributed pulse return')
                widths[bit]=now-lows[bit]
        if after.get('fault')==1:
            if fault_at is None:fault_at=now
            require(after.get('client_stop')==1,'Fault closes stop asynchronously')
        if fault_at is not None:require(after.get('fault')==1,'Fault remains sticky')
        if stop and after.get('clock_enable')==0:
            require(after.get('axi_clk')==stop-1,'Stopped AXI level')
            require(before.get('axi_clk')==after.get('axi_clk'),'No AXI transitions during loss')
            if before.get('mgmt_clk')==0 and after.get('mgmt_clk')==1:held_edges+=1
        if stop and after.get('clock_enable')==1 and before.get('axi_clk')==0 and after.get('axi_clk')==1:
            recovery_edges+=1
    require(drop_seen and fault_at is not None and 0<fault_at-start<=28000,'Bounded management detection of native pulse')
    expected={input_id:100}
    if case>=12:expected[(input_id+1)%4]=100
    require(widths==expected,'Measured 100 ps native pulses, including overlap')
    require(last.get('fault_code')==code and last.get('clock_alive')==1,'Final attributed fault after recovery')
    if stop:require(held_edges==48 and recovery_edges>=32,'Measured stopped-clock and recovery coverage')
    return dict(status='GUARD_V2_CONTROL_TRACE_PASS',case=case,input_id=input_id,stop_kind=stop,
                native_detection_ps=fault_at-start,pulse_widths_ps=widths,held_management_edges=held_edges,
                recovery_axi_edges=recovery_edges,admission_never_reopened=True,physical_pulse_width_validated=False)
