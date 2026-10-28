"""Check the V2 OOC netlist and scoped timing evidence, not physical signoff."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from review_guard_cdc import load, require, rows

CHAINS={
    'cold_m':('FDCE','mgmt_clk','VCC/P','cold_rst_n'),
    'cold_a':('FDCE','axi_clk','VCC/P','cold_rst_n'),
    'armed_a':('FDCE','axi_clk','armed_reg/Q','cold_a_reg[2]/Q'),
    'stop_a':('FDPE','axi_clk','GND/G','fault_reg/Q'),
    'echo_a':('FDRE','axi_clk','request_reg/Q',None),
    'echo_m':('FDRE','mgmt_clk','echo_a_reg[2]/Q',None),
}
for i,port in enumerate(('native_link_up','native_axi_reset_n','native_ctl_reset_n','perst_n')):
    CHAINS[f'availability[{i}].release_m']=('FDCE','mgmt_clk','VCC/P',port)
    CHAINS[f'availability[{i}].release_a']=('FDCE','axi_clk','VCC/P',f'availability[{i}].release_m_reg[2]/Q')


def structure(report):
    cells={r['cell']:r for r in rows(report['cells.tsv'])}
    pins={r['pin']:r for r in rows(report['pins.tsv'])}
    expected={f'{name}_reg[{i}]' for name in CHAINS for i in range(3)}
    require({k for k,v in cells.items() if v['async_reg'].lower() in ('1','true')}==expected,'Exactly 42 ASYNC_REG FF')
    require(sum(c['ref'].startswith('FD') for c in cells.values())==57,'Exactly 57 synthesized FF')

    def source(pin,active=frozenset()):
        require(pin not in active,'Acyclic asynchronous cone')
        row=pins[pin]
        if row['direction']=='OUT':
            cell=pin.rsplit('/',1)[0];ref=cells[cell]['ref']
            if ref.startswith('FD') or ref in ('GND','VCC'):return {pin}
            require(ref=='LUT1' and cells[cell]['init']=="2'h1",'Only single-input inverters in CLR/PRE cones')
            inputs=[p for p,r in pins.items() if p.rsplit('/',1)[0]==cell and r['direction']=='IN']
            require(len(inputs)==1,'One inverter input')
            return source(inputs[0],active|{pin})
        result=set(filter(None,row['input_ports'].split(',')))
        for driver in filter(None,row['drivers'].split(',')):result|=source(driver,active|{pin})
        return result

    endpoints=set();control_pins=set()
    for name,(ref,clock,first,control) in CHAINS.items():
        nets=set()
        for i in range(3):
            cell=f'{name}_reg[{i}]'
            require(cells[cell]['ref']==ref and cells[cell]['init']=="1'b0",'Chain type and INIT: '+cell)
            require(pins[cell+'/C']['clocks']==clock and pins[cell+'/CE']['drivers']=='VCC/P','Chain clock and enable: '+cell)
            driver=first if i==0 else f'{name}_reg[{i-1}]/Q'
            require(pins[cell+'/D']['drivers']==driver,'Chain D/Q continuity: '+cell)
            if i:
                consumers={p for p,r in pins.items() if r['direction']=='IN' and r['net']==pins[driver]['net']}
                require(consumers=={cell+'/D'},'Exclusive interstage fanout: '+cell)
                endpoints.add(cell+'/D')
            if ref=='FDRE':require(pins[cell+'/R']['drivers']=='GND/G','No data-chain reset')
            else:
                pin=cell+('/CLR' if ref=='FDCE' else '/PRE');control_pins.add(pin)
                require(source(pin)=={control},'Exact single asynchronous source: '+pin)
                nets.add(pins[pin]['net'])
        if control:require(len(nets)==1,'Common control across chain: '+name)
    actual={p for p in pins if p.rsplit('/',1)[-1] in ('CLR','PRE')}
    require(actual==control_pins,'Every asynchronous FF control was inspected')
    return cells,pins,endpoints


def review(baseline,scoped):
    bcells,bpins,endpoints=structure(baseline)
    require((bcells,bpins,endpoints)==structure(scoped),'Constraints leave the netlist unchanged')
    data={('request_reg/C','echo_a_reg[0]/D'):4.0,('echo_a_reg[2]/C','echo_m_reg[0]/D'):10.0,
          ('armed_reg/C','armed_a_reg[0]/D'):4.0}
    controls={(f'availability[{n}].release_m_reg[2]/C',f'availability[{n}].release_a_reg[{i}]/CLR') for n in range(4) for i in range(3)}
    controls|={('fault_reg/C',f'stop_a_reg[{i}]/PRE') for i in range(3)}
    for report in (baseline,scoped):
        require('No waivers found.' in report['waivers.rpt'],'No waivers')
        counts={m[0]:(m[1],int(m[2])) for m in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',report['cdc.rpt'],re.M)}
        require(counts=={'CDC-3':('Info',3),'CDC-9':('Info',5)},'All eight internal CDC findings are informational')
        details=set()
        for line in report['cdc.rpt'].splitlines():
            if re.match(r'^\s+\d+\s+CDC-',line):
                fields=line.split();details.add((fields[-2],fields[-1]))
        expected=set(data)|{pair for pair in controls if pair[1].endswith('[0]/CLR') or pair[1].endswith('[0]/PRE')}
        require(details==expected,'CDC endpoint identities match the actual chains')
        require(not re.search(r'^LUTAR-\d+#',report['methodology.rpt'],re.M),'No combinational asynchronous-reset findings')
        for key,count in [('no_clock',0),('unconstrained_internal_endpoints',0),('no_input_delay',5),('no_output_delay',8)]:
            require(f'checking {key} ({count})' in report['check_timing.rpt'],'Timing coverage explicitly retained: '+key)
    paths=[rows(r['all_paths.tsv']) for r in (baseline,scoped)]
    for items in paths:
        require(max(Counter((r['source_clock'],r['destination_clock'],r['delay']) for r in items).values())<10000,'Untruncated path export')
    intra=lambda items:Counter(tuple(sorted(r.items())) for r in items if r['source_clock']==r['destination_clock'])
    require(intra(paths[0])==intra(paths[1]),'All same-clock paths preserved')
    require(sum(intra(paths[0]).values())==1804,'Complete same-clock path coverage')
    # In particular, keep recovery/removal for cold_a -> armed_a in its own domain.
    for i in range(3):
        for delay in ('max','min'):
            found=[r for r in paths[1] if r['startpoint']=='cold_a_reg[2]/C' and r['endpoint']==f'armed_a_reg[{i}]/CLR' and r['delay']==delay]
            require(found and all(r['slack'] and float(r['slack'])>0 for r in found),'Cold-reset recovery/removal remains timed')
    stages=rows(scoped['stage_paths.tsv'])
    require(len(stages)==56 and {(r['endpoint'],r['delay']) for r in stages}=={(e,d) for e in endpoints for d in ('max','min')},'All 28 interstage connections checked twice')
    for r in stages:
        require(r['before_slack']==r['after_slack'] and float(r['after_slack'])>0,'Interstage margins unchanged and positive')
        actual=[float(p['slack']) for p in paths[0] if p['endpoint']==r['endpoint'] and p['delay']==r['delay']]
        require(actual and min(actual)==float(r['before_slack']),'Margins match independent baseline paths')
    commands=lambda r:[s.strip() for s in r['effective.xdc'].splitlines() if s.strip() and not s.lstrip().startswith('#')]
    before,after=commands(baseline),commands(scoped)
    require(len(before)==3 and set(before)=={
        'create_clock -period 10.000 -name mgmt_clk [get_ports mgmt_clk]',
        'create_clock -period 4.000 -name axi_clk [get_ports axi_clk]',
        'current_instance -quiet'},'Only the reviewed 100/250 MHz clocks in baseline')
    require(set(before).issubset(after),'Original clock constraints retained')
    additions=[s for s in after if s not in before]
    require(len(additions)==18,'Exactly 18 scoped additions')
    select=r'\[get_pins -quiet -filter \{NAME == "([^"]+)"\}\]'
    pairs=set();maxima={}
    for line in additions:
        f=re.fullmatch(r'set_false_path -from '+select+' -to '+select,line)
        m=re.fullmatch(r'set_max_delay -datapath_only -from '+select+' -to '+select+r' ([0-9.]+)',line)
        require(f or m,'Exact pin constraints only; no blanket clock cuts')
        if f:pairs.add(f.groups())
        else:maxima[m.groups()[:2]]=float(m[3])
    require(pairs==controls and maxima==data,'Only the 15 control pairs and 3 first data stages')
    for r in paths[1]:
        if r['source_clock']!=r['destination_clock'] and r['slack']:
            require(r['delay']=='max' and (r['startpoint'],r['endpoint']) in data and float(r['slack'])>0,'Only intended interclock max paths remain timed')
    require('No ignored timing exceptions found.' in scoped['exceptions_ignored.rpt'],'No ignored exceptions')
    require(not re.search(r'^TIMING-[67]#',scoped['methodology.rpt'],re.M),'No assumed interclock phase relationship')
    require(len(re.findall(r'^TIMING-18#\d+ Warning',scoped['methodology.rpt'],re.M))==13,'External timing omissions remain visible')
    logic=re.search(r'\|\s+LUT as Logic\s+\|\s+(\d+)\s+\|',scoped['utilization.rpt'])
    require(logic is not None,'Utilization LUT estimate present')
    return dict(status='GUARD_V2_INTERNAL_CDC_OOC_PASS',synchronizer_chains=14,async_reg_cells=42,flip_flops=57,
                lut_primitives=sum(c['ref'].startswith('LUT') for c in bcells.values()),estimated_logic_luts=int(logic[1]),same_clock_paths_preserved=1804,
                interstage_connections=28,interstage_setup_hold_checks=56,
                interstage_min_setup_ns=min(float(r['after_slack']) for r in stages if r['delay']=='max'),
                interstage_min_hold_ns=min(float(r['after_slack']) for r in stages if r['delay']=='min'),
                scoped_false_path_pairs=15,bounded_data_paths=3,cdc_critical=0,cdc_warning=0,lutar_warnings=0,
                ports_without_input_delay=5,ports_without_output_delay=8,physical_timing_closed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('reports',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=review(load(a.reports/'baseline'),load(a.reports/'scoped'))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
