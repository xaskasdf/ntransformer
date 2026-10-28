"""Check BAR simulations, native binding, and applied/reopened CDC evidence."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_control_plane import parse_trace, require, sha

BAR_GROUPS=['all_lanes_ids','strobes_errors_duplex','decode_rejection','burst_drain_framing','reset_boundaries','last_word_recovery']
CSR_WIDTHS=dict(awaddr=12,araddr=12,awprot=3,arprot=3,awvalid=1,wvalid=1,bready=1,arvalid=1,rready=1,
               wdata=32,wstrb=4,awready=1,wready=1,arready=1,bvalid=1,rvalid=1,bresp=2,rresp=2,rdata=32)
BRIDGE_WIDTHS=dict(awaddr=64,araddr=64,awid=4,arid=4,awlen=8,arlen=8,awsize=3,arsize=3,awprot=3,arprot=3,
    awburst=2,arburst=2,awvalid=1,wvalid=1,wlast=1,bready=1,arvalid=1,rready=1,
    wdata=512,wstrb=64,awready=1,wready=1,arready=1,bid=4,rid=4,bresp=2,rresp=2,bvalid=1,rvalid=1,rlast=1,rdata=512)


def exact_connections(table,widths,first,second,prefix_a,prefix_b):
    expected={(prefix_a+name+(f'[{bit}]' if width>1 else ''),prefix_b+name+(f'[{bit}]' if width>1 else ''))
              for name,width in widths.items() for bit in range(width)}
    require(len(table)==len(expected) and {(r[first],r[second]) for r in table}==expected,'Connection endpoint coverage')


def snapshot_bounds(table):
    require({r['group'] for r in table}=={'request_1','request_2','request_3','ack_1','ack_2','ack_3','data_1','data_2','data_3'},'Snapshot group coverage')
    require(len({r['to'] for r in table})==len(table),'Duplicate snapshot endpoint')
    for row in table:
        require(abs(float(row['requirement_ns'])-(2.222 if row['group']=='request_3' else 4.0))<0.001,'Wrong snapshot bound')


def lock_constants(table):
    require(len(table)==2 and {r['bridge'] for r in table}=={'ntrpu_bridge_i/m_axib_awlock','ntrpu_bridge_i/m_axib_arlock'},'LOCK constant coverage')
    for row in table:
        require(row['adapter']=='bar_control/s_'+row['bridge'].split('m_axib_')[1] and
                row['bridge_leaf_drivers']=='0' and row['adapter_ref']=='GND' and
                row['adapter_driver']=='GND/G','Unexpected LOCK constant')


def rows(root, name):
    return list(csv.DictReader((root/'output'/name).read_text().splitlines(),delimiter='\t'))


def closed(root):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Closed file set')
    for name,digest in files.items():require(sha(root/name)==digest,'Hash mismatch '+name)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status'].endswith('_EXPORTED') and result['sources_unchanged'],'Failed/changed run')
    for name,digest in result['sources'].items():require(sha(root/'source'/name)==digest,'Source hash '+name)
    if 'returncode' in result:require(result['returncode']==0 and not result['timed_out'],'Failed native stage')
    else:require(all(s['returncode']==0 and not s['timed_out'] for s in result['steps']),'Failed simulation stage')
    return result


def bar_trace(text,seed):
    require(not re.search(r'(?im)^\s*(ERROR|Fatal):|BAR CHECK|BAR WATCHDOG',text),'BAR trace failure')
    matches=re.findall(r'^BAR PASS seed=(\d+) groups=(\d+) checks=(\d+) downstream_w=(\d+) downstream_r=(\d+)$',text,re.M)
    require(len(matches)==1,'Exactly one BAR PASS required')
    value=tuple(map(int,matches[0]))
    require(value[:2]==(seed,6) and value[2]>500 and value[3:]==(22,23),'BAR counts/identity')
    groups=re.findall(r'^BAR GROUP seed=(\d+) name=(\w+)$',text,re.M)
    require(groups==[(str(seed),name) for name in BAR_GROUPS],'BAR group coverage/order')
    require(re.findall(r'^BAR DRAIN beats=(\d+) no_side_effects=(\d)$',text,re.M)==[('2','1'),('256','1')],'Burst drain coverage')
    return dict(writer=0,seed=seed,groups=6,checks=value[2])


def compare_table(a,b,name,keys):
    left=rows(a,name);right=rows(b,name)
    require([{k:r[k] for k in keys} for r in left]==[{k:r[k] for k in keys} for r in right], 'Reopen changed '+name)
    return left


def review(sim,synth,apply,reopen):
    results=[closed(root) for root in (sim,synth,apply,reopen)]
    require(results[0]['seeds']==[0,1,2,3],'Full BAR bank required')
    records=[];text0='';writer0=''
    for writer in range(2):
        for seed in range(4):
            text=(sim/f'writer-{writer}-seed-{seed}.log').read_text()
            if writer:record=parse_trace(text,1,seed)
            else:record=bar_trace(text,seed)
            records.append(record)
            if seed==0:
                if writer:writer0=text
                else:text0=text
    require(results[0]['completed']==[{k:r[k] for k in ('writer','seed','groups','checks')} for r in records],'Bank summary disagrees')
    for name in ('ntrpu_bar32_adapter.sv','ntrpu_hbm_control_plane.sv','ntrpu_control_snapshot.sv','ntrpu_axi_bridge_writer.sv'):
        require(results[0]['sources'][name]==results[1]['sources'][name],'Native/simulation RTL differs '+name)
    require(results[2]['checkpoint_sha256']==sha(synth/'output/synth.dcp'),'Wrong apply donor')
    require(results[3]['checkpoint_sha256']==sha(apply/'output/bar_bounded.dcp'),'Wrong reopen donor')
    binding=json.loads((synth/'source/BAR_BINDING.json').read_text())
    require(binding['top_sha256']==sha(synth/'source/ntrpu_axi_bridge_hbm_controlled.sv'),'Top binding differs')
    require(binding['removed_bits']==1213 and not binding['native_ip_configuration_changed'],'Unexpected native boundary/config change')
    source=(synth/'source/ntrpu_axi_bridge_hbm_controlled.sv').read_text()
    require('xilinx_axi_pcie_ep_app_i (' not in source,'Example BRAM still attached')
    telemetry=binding['telemetry']; require(sum(item['width'] for item in telemetry)==133,'Telemetry bits missing')
    allocated=set()
    for item in telemetry:
        require(item['bit']>=32 and item['bit']+item['width']<=128,'Invalid telemetry placement')
        for bit in range(item['bit'],item['bit']+item['width']):
            key=(item['domain'],bit);require(key not in allocated,'Telemetry overlap');allocated.add(key)
    ports=compare_table(apply,reopen,'physical_ports.tsv',['port','direction'])
    require(len(ports)==74,'Unexpected external port bits')
    for row in ports:require(row['port'].split('[')[0] in {'pci_exp_txp','pci_exp_txn','pci_exp_rxp','pci_exp_rxn',
        'sys_clk_p','sys_clk_n','sys_rst_n','mgmt_clk_in','fabric_ref_clk_in','fabric_clock_reset',
        'hbm_ref0_clk_in','hbm_ref1_clk_in','fabric_clock_locked','configuration_rst_n'},'Logical test port escaped')
    csr=compare_table(apply,reopen,'csr_connections.tsv',['adapter','control','driver']);require(len(csr)==112,'CSR wiring incomplete')
    bridge=compare_table(apply,reopen,'bridge_connections.tsv',['bridge','adapter','driver']);require(len(bridge)==1280,'Bridge wiring incomplete')
    constants=compare_table(apply,reopen,'bridge_constants.tsv',['bridge','adapter','bridge_leaf_drivers','adapter_driver','adapter_ref'])
    lock_constants(constants)
    for name in ('awlock','arlock'):
        require(len(re.findall(r'\.m_axib_'+name+r'\s*\(m_axi_'+name+r'\s*\)',source))==1 and
                len(re.findall(r'\.s_'+name+r'\s*\(m_axi_'+name+r'\s*\)',source))==1,
                'Native source must bind LOCK to the IP output before synthesis')
    exact_connections(csr,CSR_WIDTHS,'adapter','control','bar_control/m_','control/s_')
    exact_connections(bridge,BRIDGE_WIDTHS,'bridge','adapter','ntrpu_bridge_i/m_axib_','bar_control/s_')
    cold=compare_table(apply,reopen,'cold_reset_bindings.tsv',['source','consumer','driver']);require(len(cold)==10,'Cold reset bindings incomplete')
    old=compare_table(apply,reopen,'paths.tsv',['kind','group','from','to','start_clock','end_clock','requirement_ns'])
    require(len(old)==1503,'Old bounded endpoint count changed')
    inter=compare_table(apply,reopen,'interstage.tsv',['from','to','type','start_clock','end_clock','requirement_ns'])
    require(len(inter)==180,'Old interstage analysis changed')
    new=compare_table(apply,reopen,'snapshot_bounds.tsv',['kind','group','from','to','requirement_ns'])
    snapshot_bounds(new)
    tokens=[row for row in new if row['kind']=='token'];data=[row for row in new if row['kind']=='held_data']
    require(len(tokens)==6 and {r['group'] for r in data}=={'data_1','data_2','data_3'},'New snapshot bounds incomplete')
    expected_samples={f'control/snapshot/taps[{domain}].tap/sample_reg[{bit}]/D' for domain in (1,2,3) for bit in range(128)}
    require(len(data)==384 and {r['to'] for r in data}==expected_samples,'Snapshot data endpoint coverage')
    new_inter=compare_table(apply,reopen,'snapshot_interstage.tsv',['from','to','type','requirement_ns'])
    require(len(new_inter)==32,'Snapshot synchronizer stages incomplete')
    sampled=compare_table(apply,reopen,'telemetry_capture_sources.tsv',['domain','capture','source_clock','source_pin'])
    require({r['domain'] for r in sampled}=={'0','1','2','3'},'Missing telemetry clock domain')
    for root in (synth,apply,reopen):
        text=(root/'tool.log').read_text(errors='replace')
        critical=re.findall(r'^CRITICAL WARNING:.*$',text,re.M)
        require(all('[Constraints 18-515]' in line and 'clock_generator/inst/mmcme4_adv_inst/LOCKED' in line for line in critical),'Unreviewed critical warning')
        if root==apply:require(len(critical)==1,'Expected visible LOCKED segmentation warning')
        require(not re.search(r'(?im)^\s*(ERROR|Fatal):',text),'Tool error')
    for root in (apply,reopen):
        lock=rows(root,'lock_path.tsv');require(len(lock)==2 and float(lock[-1]['requirement_ns'])==10,'LOCKED bound missing')
        props=rows(root,'lock_arc_properties.tsv')
        for key in ('IS_DISABLED','IS_USER_DISABLED'):
            found=[r for r in props if r['property']==key]
            require(len(found)==4 and all(r['value']==('1' if key=='IS_DISABLED' else '0') for r in found),'LOCKED arc behavior changed')
    drc=(synth/'output/drc.rpt').read_text()
    findings=re.findall(r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Error)\s*\| [^|]+\| (\d+)\s*\|$',drc,re.M)
    require(findings==[('AVAL-326','Critical Warning','1'),('NSTD-1','Critical Warning','1'),
                      ('UCIO-1','Critical Warning','1')],'Unreviewed native DRC findings')
    require('cell refclk_ibuf is missing a valid LOC' in drc and
            'Problem ports: fabric_ref_clk_in, sys_rst_n.' in drc and
            'Problem ports: fabric_ref_clk_in, sys_clk_n, sys_clk_p, sys_rst_n.' in drc,'Changed physical DRC scope')
    timing={}
    for label,root in [('synth',synth),('apply',apply),('reopen',reopen)]:
        coverage=(root/'output/check_timing.rpt').read_text()
        timing[label]={name:int(count) for name,count in re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',coverage,re.M)}
        require(all(timing[label][key]==0 for key in ('no_clock','multiple_clock','loops','latch_loops')),'Clock topology problem')
        anchor='4. checking unconstrained_internal_endpoints'
        start=coverage.index(anchor,coverage.index(anchor)+1)
        section=coverage[start:coverage.index('5. checking no_input_delay',start)]
        pins=[line.strip() for line in section.splitlines() if '/' in line]
        require(len(pins)==1154 and all(pin.startswith('hbm_binding/memory/') and 'xsdb' in pin.lower() for pin in pins),
                'Unexpected unconstrained internal endpoint')
    require(timing['apply']==timing['reopen'],'Timing coverage changed on reopen')
    cdc={}
    for label,root in [('apply',apply),('reopen',reopen)]:
        cdc[label]=[dict(rule=rule,severity=severity,count=int(count)) for rule,severity,count in re.findall(
            r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+', (root/'output/cdc.rpt').read_text(),re.M)]
        require([(r['rule'],r['severity'],r['count']) for r in cdc[label]]==[
            ('CDC-3','Info',196),('CDC-6','Warning',9),('CDC-9','Info',36),('CDC-15','Warning',2130)],'Changed CDC findings')
    negative=[]
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError):negative.append(name)
        else:raise ValueError('Negative control escaped '+name)
    rejects('missing_bar_pass',lambda:bar_trace(text0.replace('BAR PASS','REMOVED PASS'),0))
    rejects('wrong_seed',lambda:bar_trace(text0,1))
    rejects('bad_drain_side_effect',lambda:bar_trace(text0.replace('no_side_effects=1','no_side_effects=0'),0))
    rejects('missing_long_burst',lambda:bar_trace(text0.replace('beats=256','beats=255'),0))
    rejects('bad_csr_count',lambda:bar_trace(text0.replace('downstream_w=22','downstream_w=23'),0))
    rejects('missing_reset_test',lambda:bar_trace(text0.replace('name=reset_boundaries','name=missing'),0))
    rejects('late_failure',lambda:bar_trace(text0+'\nFatal: injected\n',0))
    rejects('missing_writer_retention',lambda:parse_trace(writer0.replace('CP B_ONLY','REMOVED'),1,0))
    rejects('bad_ack_released',lambda:parse_trace(writer0.replace('CP BAD_ACK owned=1','CP BAD_ACK owned=0'),1,0))
    rejects('missing_csr_wire',lambda:exact_connections(csr[:-1],CSR_WIDTHS,'adapter','control','bar_control/m_','control/s_'))
    rejects('duplicate_csr_wire',lambda:exact_connections(csr[:-1]+[csr[0]],CSR_WIDTHS,'adapter','control','bar_control/m_','control/s_'))
    rejects('missing_native_wire',lambda:exact_connections(bridge[:-1],BRIDGE_WIDTHS,'bridge','adapter','ntrpu_bridge_i/m_axib_','bar_control/s_'))
    rejects('missing_lock_constant',lambda:lock_constants(constants[:-1]))
    bad_constants=[dict(r) for r in constants];bad_constants[0]['adapter_ref']='VCC'
    rejects('wrong_lock_constant',lambda:lock_constants(bad_constants))
    changed=[dict(r) for r in new];changed[0]['requirement_ns']='100'
    rejects('loosened_snapshot_bound',lambda:snapshot_bounds(changed))
    rejects('duplicate_snapshot_endpoint',lambda:snapshot_bounds(new+[new[0]]))
    return dict(status='BAR_CONTROL_NATIVE_REVIEWED',simulations=records,total_checks=sum(r['checks'] for r in records),
        total_groups=sum(r['groups'] for r in records),external_port_bits=74,removed_internal_port_bits=1213,
        csr_connection_bits=len(csr),bridge_connection_bits=len(bridge),bridge_constant_bits=len(constants),telemetry_bits=133,
        previous_bounded_endpoints=len(old),snapshot_bounded_endpoints=len(new),snapshot_held_data_endpoints=len(data),
        snapshot_min_estimated_slack_ns=min(float(r['slack_ns']) for r in new),
        checkpoint_sha256=sha(apply/'output/bar_bounded.dcp'),negative_controls=negative,
        physical_drc_findings=[dict(rule=rule,severity=severity,count=int(count)) for rule,severity,count in findings],
        timing_coverage=timing,cdc_findings=cdc,
        native_tlp_traffic_tested=False,routed_timing_qualified=False,board_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('simulation','synthesis','apply','reopen'):p.add_argument(name,type=Path)
    p.add_argument('--output',type=Path);args=p.parse_args()
    text=json.dumps(review(args.simulation,args.synthesis,args.apply,args.reopen),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')
