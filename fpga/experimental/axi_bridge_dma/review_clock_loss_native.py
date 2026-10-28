"""Check a public AXI-clock injection trace; never infer physical GT failure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from review_native import require, vcd_rows

P='/board/probe/'
EP='/board/AXI_PCIE_EP/'
CLOCKS=[EP+'axi_aclk_out',P+'clk',P+'guard/axi_clk',P+'dut/clk']
NATIVE=[EP+'user_lnk_up',EP+'axi_aresetn',EP+'axi_ctl_aresetn','/board/sys_rst_n']


def review(log,vcd):
    require(not re.search(r'(?im)^\s*(?:fatal:|fatal\b|error:|error\b|BN CHECK|BN TIMEOUT)',log),'Simulator diagnostics')
    verdict=re.findall(r'^BRIDGE_CLOCK_LOSS_PASS level=([01]) timeout=64 hold_cycles=128 recovery_cycles=64$',log,re.M)
    require(len(verdict)==1,'Unique clock-loss verdict');level=int(verdict[0])
    events={}
    for kind in ('PAUSE','HELD','RELEASE','RECOVERED'):
        rows=re.findall(r'^CL '+kind+r' level=([01]) t=(\d+)(?: code=(\d+))?$',log,re.M)
        require(len(rows)==1 and int(rows[0][0])==level,'Complete clock-loss event identity: '+kind)
        events[kind]=int(rows[0][1])
        if kind=='HELD':require(rows[0][2]=='2','Pure clock-loss cause in log')
    start,held,end,recovered=[events[k] for k in ('PAUSE','HELD','RELEASE','RECOVERED')]
    require(0<start<held<end<recovered,'Clock-loss event order')
    require('BN DATA_PASS bytes=192 maps=2 clock_period_ps=4000' in log,'Native warmup marker')
    # The clock test reviews command admission and retained context. It does not
    # independently reconstruct the warmup payload or certify host isolation.
    seen=set();samples={};admissions=[];fault_time=None
    mgmt_last=None;falling_held=0;falling_recovery=0;recovery_edges=[];pre_edges=[]
    stopped_changes=0
    for now,before,after in vcd_rows(vcd):
        if now>recovered:break
        seen.update(after)
        axi_rise=before.get(CLOCKS[0])==0 and after.get(CLOCKS[0])==1
        if after.get(P+'phase') in (1,2,3):
            require(all(after.get(k)==after.get(CLOCKS[0]) for k in CLOCKS),'Client and guard observe the same public AXI clock')
        if axi_rise and before.get(P+'phase')==3 and before.get(P+'cmd_valid')==before.get(P+'cmd_ready')==1:
            admissions.append((now,tuple(before.get(P+k) for k in ('cmd_tag','cmd_generation','cmd_source','cmd_bytes'))))
        if before.get(P+'mgmt_clk')==0 and after.get(P+'mgmt_clk')==1:
            if mgmt_last is not None:require(now-mgmt_last==10000,'Continuous independent 100 MHz management clock')
            mgmt_last=now
        if start<now<=held and before.get(P+'mgmt_clk')==1 and after.get(P+'mgmt_clk')==0:falling_held+=1
        if end<now<=recovered and before.get(P+'mgmt_clk')==1 and after.get(P+'mgmt_clk')==0:falling_recovery+=1
        if now<start and axi_rise and before.get(P+'phase')==3:pre_edges.append(now)
        if end<=now<=recovered and axi_rise:recovery_edges.append(now)
        if now in events.values():samples[now]=after.copy()
        if start<=now<=recovered:
            require(after.get(P+'phase')==3,'Dedicated clock-loss phase')
            require(all(after.get(k)==1 for k in NATIVE),'No native reset/link loss mixed with clock injection')
            require(all(after.get(P+k)==1 for k in ('configuration_rst_n','cold_rst_n','owned','src_req_valid','guard_armed')),
                    'Retained cold reset, ownership and source offer')
            require(tuple(after.get(P+k) for k in ('src_req_tag','src_req_address','src_req_bytes'))==((9<<64)|3,4096,64),
                    'Retained source identity')
            require(all(after.get(P+k)==0 for k in ('src_req_ready','src_valid','src_cpl_valid','ack_valid','result_valid','cmd_ready','cfg_ready')),
                    'No new admission or synthetic completion')
            if start<=now<end:
                require(after.get(P+'pause_active')==1 and after.get(P+'pause_complete')==0,'Pause markers match events')
                require(all(after.get(k)==level for k in CLOCKS),'Actual public clocks held at requested level')
                if now>start:stopped_changes+=sum(before.get(k)!=after.get(k) for k in CLOCKS)
                require(after.get(P+'fault')==after.get(P+'mapper_fault')==0,'AXI fault registers frozen until clock returns')
            else:require(after.get(P+'pause_active')==0 and after.get(P+'pause_complete')==1,'Released clock marker')
            if after.get(P+'guard_fault')==1:
                if fault_time is None:
                    require(before.get(P+'mgmt_clk')==0 and after.get(P+'mgmt_clk')==1,'Watchdog fault recorded on management clock')
                    fault_time=now
                require(after.get(P+'guard_code')==2 and after.get(P+'stop')==1 and after.get(P+'link_up')==0,
                        'Asynchronous watchdog closure with clock-only cause')
            if fault_time is not None:require(after.get(P+'guard_fault')==1,'Fault sticky after clock return')
    required=set(CLOCKS+NATIVE+[P+x for x in ('mgmt_clk','guard_alive','clock_case_done','stop_level')])
    require(required.issubset(seen),'Required public trace signals')
    require(set(samples)==set(events.values()),'Log events have exact VCD timestamps')
    require(len(admissions)==1 and admissions[0][0]<start and admissions[0][1]==((9<<64)|3,2,4096,64),
            'Exactly one attributed third command before clock loss')
    require(len(pre_edges)>=3 and all(b-a==4000 for a,b in zip(pre_edges,pre_edges[1:])),'250 MHz AXI before injection')
    require(all(samples[start].get(P+k)==v for k,v in [('guard_fault',0),('guard_alive',1),('link_up',1),('stop',0),('stop_level',level)]),
            'Healthy start at requested level')
    require(fault_time is not None and 0<fault_time-start<=760000,'Watchdog detection within 76 management periods')
    require(stopped_changes==0 and falling_held==128 and falling_recovery==64,'Measured clock stop and recovery coverage')
    require(samples[held].get(P+'guard_fault')==1 and samples[held].get(P+'guard_alive')==0,'Watchdog expired before release')
    require(len(recovery_edges)>=32 and all(b-a==4000 for a,b in zip(recovery_edges[1:],recovery_edges[2:])),
            'AXI resumes at 250 MHz after the release boundary')
    final=samples[recovered]
    require(all(final.get(P+k)==1 for k in ('guard_alive','guard_fault','fault','mapper_fault','clock_case_done')),
            'Recovered clock with sticky local faults and completed scenario')
    return dict(status='CLOCK_LOSS_TRACE_CONTRACT_PASS',stop_level=level,events_ps=events,
                detection_ps=fault_time-start,management_falling_edges_held=falling_held,
                management_falling_edges_recovery=falling_recovery,axi_transitions_during_hold=stopped_changes,
                recovered_axi_rising_edges=len(recovery_edges),context_retained=True,
                injection='Forced public AXI export; not physical GT/refclk loss',
                warmup_payload_independently_reconstructed=False,physical_cdc_validated=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('log',type=Path);ap.add_argument('vcd',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=review(a.log.read_text(errors='replace'),a.vcd.read_text())
    result.update(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),vcd_sha256=hashlib.sha256(a.vcd.read_bytes()).hexdigest())
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
