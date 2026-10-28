"""Review read-only timing diagnosis and one isolated post-route physical experiment.

Implementation contracts are frozen from the preceding reviewed milestone.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
from review_hbm_bar_control import closed
from review_hbm_loader_readiness import expand_xdc,loader_paths,loader_sync,hbm_connections,snapshot_bounds
from review_hbm_control_plane import require,sha
from review_hbm_board_stage import report_timing

PINS={
    'sys_clk_p':('AR40','NONE'),'sys_clk_n':('AR41','NONE'),
    'sys_rst_n':('BF41','LVCMOS18'),'fabric_ref_clk_in':('BK31','LVCMOS18'),
    'mgmt_clk_p':('BJ33','LVDS'),'mgmt_clk_n':('BJ34','LVDS'),
    'hbm_ref0_clk_p':('BK33','LVDS'),'hbm_ref0_clk_n':('BL33','LVDS'),
    'hbm_ref1_clk_p':('BH32','LVDS'),'hbm_ref1_clk_n':('BJ32','LVDS'),
    'supervisor_rst_n_pad':('BG33','LVCMOS18')}


def table(path):return list(csv.DictReader(path.read_text().splitlines(),delimiter='\t'))


def stable(rows):
    return Counter(tuple(sorted((k,v) for k,v in row.items() if k not in ('slack_ns','datapath_ns'))) for row in rows)


def path_slack(rows):
    timed=[row for row in rows if row.get('slack_ns')]
    require(timed,'Empty timing sample')
    worst=min(timed,key=lambda row:float(row['slack_ns']))
    return dict(timed=len(timed),failing=sum(float(row['slack_ns'])<0 for row in timed),
                minimum_slack_ns=float(worst['slack_ns']),worst=worst)


def pin_constraints(before,after):
    expected=Counter('set_property PACKAGE_PIN '+ball+' [get_ports '+p+']' for p,(ball,_) in PINS.items())
    expected.update('set_property IOSTANDARD '+standard+' [get_ports '+p+']' for p,(_,standard) in PINS.items() if standard!='NONE')
    expected['set_property LOC GTYE4_COMMON_X0Y1 [get_cells refclk_ibuf]']=1
    a,b=expand_xdc(before),expand_xdc(after)
    require(not a-b and b-a==expected,'Pin stage changed more than the 21 assignments')


def findings(path):
    result={r:dict(severity=s,count=int(n)) for r,s,n in re.findall(
        r'^\|\s*([A-Z]+-\d+)\s*\|\s*(Critical Warning|Warning|Error|Info|Advisory)\s*\|[^|]+\|\s*(\d+)\s*\|',path.read_text(),re.M)}
    total=re.search(r'Checks found:\s*(\d+)',path.read_text())
    require(total and sum(x['count'] for x in result.values())==int(total[1]),'Incomplete DRC/methodology summary: '+path.name)
    return result


def route_status(path,routed):
    text=path.read_text()
    require('Design Route Status' in text,'Missing route status report')
    counts={k.strip():int(n) for k,n in re.findall(r'^\s*# of ([^.]+)\.+\s*:\s*(\d+)\s*:',text,re.M)}
    require(counts.get('logical nets',0)>0 and counts.get('routable nets',0)>0,'Empty route inventory')
    complete=(counts.get('fully routed nets',0)==counts['routable nets'] and
              all(counts.get(k,0)==0 for k in ('unrouted nets','nets with routing errors','nets with no placed pins',
                                            'nets with explicit gaps','nets with some unplaced pins','nets with some unrouted pins')))
    if routed:require(complete,'Routing incomplete or erroneous')
    return dict(counts=counts,complete=complete)


def review_probe(root,donor):
    result=closed(root);out=root/'output'
    require(result['checkpoint_sha256']==sha(donor),'Probe checkpoint mismatch')
    require((out/'STATUS.txt').read_text().strip()=='BOARD_PHYSICAL_RESET_PROBED','Incomplete probe')
    require(expand_xdc((out/'before.xdc').read_text())==expand_xdc((out/'after.xdc').read_text()),'Probe changed constraints')
    properties={}
    for row in table(out/'reset_properties.tsv'):
        key=(row['object'],row['property'])
        require(key not in properties or properties[key]==row['value'],'Conflicting probe properties')
        properties[key]=row['value']
    ready='loader_readiness/ready_reg';canonical='core/endpoint/guard/cold_m_reg[2]'
    expected={(ready,'REF_NAME'):'FDCE',(ready,'INIT'):"1'b0",(ready,'IS_CLR_INVERTED'):"1'b1",
              (canonical,'REF_NAME'):'FDCE',(canonical,'INIT'):"1'b0",
              (canonical,'IS_CLR_INVERTED'):"1'b1",(canonical+'/Q','IS_INVERTED'):'0',
              (canonical+'/CLR','IS_INVERTED'):'1'}
    for pin in (ready+'/CLR','core/management_cold_rst_n','loader_readiness/cold_rst_n'):
        expected[pin,'LEAF_DRIVERS']=canonical+'/Q'
        expected[pin,'FANIN_STARTS']=canonical+'/C'
    require(all(properties.get(k)==v for k,v in expected.items()),'Optimized readiness reset proof differs')
    return dict(status='BOARD_IMPLEMENTATION_RESET_PROBE_REVIEWED',donor_sha256=result['checkpoint_sha256'],
                properties_checked=len(expected),constraints_unchanged=True,ready_initial_value=0,
                canonical_reset=canonical+'/Q',reset_encoding='PRIMITIVE_INVERTED_CLR')


def review_origin(root,baseline):
    # This run intentionally remains FAILED; later independent evidence qualifies its DCP.
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Origin file inventory')
    for name,digest in files.items():require(sha(root/name)==digest,'Origin hash mismatch: '+name)
    r=json.loads((root/'RESULT.json').read_text());out=root/'output'
    require(r['status']=='FAILED' and r['returncode']==1 and not r['timed_out'] and r['sources_unchanged'],'Unexpected optimization-origin outcome')
    for name,digest in r['sources'].items():require(sha(root/'source'/name)==digest,'Origin source changed')
    require(r['checkpoint_sha256']==sha(baseline/'output/board_cdc_bounded.dcp'),'Wrong CDC donor')
    require('\nReadiness reset polarity changed\n' in (root/'tool.log').read_text(),'Unexpected audit failure')
    pin_constraints((out/'before.xdc').read_text(),(out/'pinned.xdc').read_text())
    require(expand_xdc((out/'before.xdc').read_text())==expand_xdc((baseline/'output/effective.xdc').read_text()),'Origin pre-pin constraints differ')
    require(not findings(out/'pinned_drc.rpt') and not findings(out/'reports/drc.rpt'),'Origin DRC findings')
    require((out/'pinned.dcp').stat().st_size>0 and (out/'optimized.dcp').stat().st_size>0,'Origin checkpoint missing')
    return dict(status='BOARD_IMPLEMENTATION_FAILED_ORIGIN_PRESERVED',original_status=r['status'],
                donor_sha256=r['checkpoint_sha256'],checkpoint_sha256=sha(out/'optimized.dcp'),
                pinned_sha256=sha(out/'pinned.dcp'),trial_assignments_verified=21,
                post_optimization_audit_required=True)


def physical_readiness(contract,clocks):
    records=table(contract/'readiness_source.tsv');resets=table(contract/'readiness_reset.tsv')
    require(len(records)==len(resets)==1,'Readiness source/reset records missing')
    r,q=records[0],resets[0];source='loader_readiness/ready_reg'
    require(r['cell']==q['cell']==source and r['reference']=='FDCE' and r['clock']==clocks['mgmt_clk'] and r['async_reg'].upper() not in ('1','TRUE'),'Readiness register identity')
    require(r['ce_reference']=='VCC' and r['ce_driver'].endswith('VCC/P') and r['fanout']=='weight_loader/mem_ready_sync_reg[0]/D','Readiness CE/fanout')
    bindings=table(contract/'readiness_bindings.tsv')
    expected={('core/mgmt_clk','loader_readiness/clk'),('core/management_cold_rst_n','loader_readiness/cold_rst_n'),
              ('availability/fault','loader_readiness/fault_mgmt'),('availability/ready_mgmt[0]','loader_readiness/ready_mgmt[0]'),
              ('availability/ready_mgmt[1]','loader_readiness/ready_mgmt[1]'),('loader_readiness/ready','weight_loader/memory_ready_async')}
    require(len(bindings)==6 and {(x['source'],x['consumer']) for x in bindings}==expected,'Readiness wiring')
    drivers={x['source']:x['driver'].removeprefix('{').removesuffix('}') for x in bindings}
    require(drivers['loader_readiness/ready']==source+'/Q' and r['reset_source']==q['canonical_driver']==drivers['core/management_cold_rst_n'],'Canonical readiness reset binding')
    for port,reg in [('fault','fault_reg'),('ready_mgmt[0]','ready_mgmt_reg[0]'),('ready_mgmt[1]','ready_mgmt_reg[1]')]:
        require(drivers['availability/'+port]=='availability/'+reg+'/Q','Readiness input driver')
    require((r['clear_driver'],r['clear_reference'],r['clear_init'])==(q['clear_driver'],q['driver_reference'],q['driver_init']),'Reset records disagree')
    if q['encoding']=='PRIMITIVE_INVERTED_CLR':
        require(q['clear_driver']==q['canonical_driver']=='core/endpoint/guard/cold_m_reg[2]/Q' and q['driver_reference']=='FDCE' and q['driver_init']=="1'b0" and q['clear_inverted'] in ('1',"1'b1",'TRUE','true') and q['pin_inverted']=='1','Active-low CLR not preserved')
    else:
        require(q['encoding']=='EXPLICIT_INVERTER' and q['driver_reference']=='LUT1' and q['driver_init']=="2'h1" and q['clear_inverted'] in ('0',"1'b0",'FALSE','false') and q['pin_inverted']=='0','Explicit reset inverter differs')
    fanin=table(contract/'loader_readiness_fanin.tsv')
    require(len(fanin)==3 and {x['source'] for x in fanin}=={'availability/fault_reg/C','availability/ready_mgmt_reg[0]/C','availability/ready_mgmt_reg[1]/C'} and {x['to'] for x in fanin}=={source+'/D'} and len({x['direct_driver'] for x in fanin})==1,'Readiness input cone changed')
    return q['encoding']


def review_implementation(root,baseline,donor,mode):
    r=closed(root)
    require(r['checkpoint_sha256']==sha(donor),'Checkpoint chain mismatch')
    out=root/'output';reports=out/'reports';contract=out/'contracts'
    require((out/'STATUS.txt').read_text().strip()==f'BOARD_IMPLEMENTATION_{mode}_EXPORTED_REVIEW_REQUIRED','Missing stage completion')
    require((contract/'AUDIT_STATUS.txt').read_text().strip()=='BOARD_IMPLEMENTATION_CONTRACTS_EXPORTED_REVIEW_REQUIRED','Missing contract audit')
    if mode=='optimized':
        pin_constraints((out/'before.xdc').read_text(),(out/'pinned.xdc').read_text())
        require(expand_xdc((out/'before.xdc').read_text())==expand_xdc((baseline/'output/effective.xdc').read_text()),'Wrong pre-pin constraints')
        before=expand_xdc((out/'pinned.xdc').read_text())
    else:before=expand_xdc((out/'before.xdc').read_text())
    after=expand_xdc((reports/'effective.xdc').read_text())
    changed_timing={s:n for s,n in (after-before).items() if s.startswith(('set_max_delay ','set_min_delay ','set_bus_skew ','set_false_path ','set_clock_groups ','set_multicycle_path ','set_disable_timing ','create_waiver ','create_clock ','create_generated_clock '))}
    require(not changed_timing,'Implementation introduced timing exceptions or clocks')
    pins=table(out/'trial_pins.tsv')
    require(len(pins)==11 and {x['port']:(x['ball'],x['standard']) for x in pins}==PINS,'Pin inventory differs')
    require(all(x['status']=='EXPERIMENT_ONLY' for x in pins),'Unqualified pin release')
    require(all(x['slr']=='SLR0' for x in pins if x['bank']=='64'),'Reference/control bank left SLR0')
    domains=table(contract/'domain_clocks.tsv');clocks={x['logical']:x['clock'] for x in domains}
    require(len(domains)==4,'Native clock domains missing')
    counts={}
    for name,count in [('paths.tsv',1503),('snapshot_bounds.tsv',390),('loader_bounds.tsv',302),
                       ('snapshot_interstage.tsv',32),('loader_interstage.tsv',16),('readiness_source_paths.tsv',6)]:
        current=table(contract/name);old=table(baseline/'output'/name)
        require(len(current)==count and stable(current)==stable(old),'Preserved path contract differs: '+name)
        counts[name]=len(current)
    loader_paths(table(contract/'loader_bounds.tsv'),clocks)
    loader_sync(table(contract/'loader_synchronizers.tsv'),table(contract/'loader_interstage.tsv'),clocks)
    hbm_connections(table(contract/'loader_hbm_connections.tsv'))
    snapshot_bounds(table(contract/'snapshot_bounds.tsv'))
    reset_encoding=physical_readiness(contract,clocks)
    synchronous=table(contract/'synchronous_paths.tsv');old=table(baseline/'output/board_synchronous_paths.tsv')
    require(len(synchronous)==88 and stable([{k:v for k,v in row.items() if k!='status'} for row in synchronous])==stable(old),'Synchronous contract changed')
    missing=[x for x in synchronous if x['status']!='TIMED']
    require(all(x['status']=='PRUNED_UNUSED_LOCK_FAULT' and x['to'] in ('board_startup/lock_fault_reg/D','board_startup/lock_fault_reg/CLR') for x in missing),'Required synchronous timing pruned')
    require(len(missing) in (0,4),'Partial lock-fault pruning')
    tree=table(contract/'reference_tree.tsv')
    require(stable(tree)==stable(table(baseline/'output/reference_tree.tsv')),'HBM reference topology changed')
    locks=table(contract/'board_lock_paths.tsv');arcs=table(contract/'board_lock_arcs.tsv')
    require(len(locks)==4 and all(float(x['requirement_ns'])==10 for x in locks),'LOCKED transport bound missing')
    require(len(arcs)==4 and all(x['disabled']=='1' and x['user_disabled']=='0' for x in arcs),'LOCKED timing model altered')
    cdc={k:dict(severity=s,count=int(n)) for k,s,n in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+', (reports/'cdc.rpt').read_text(),re.M)}
    require(cdc and not any(x['severity']=='Critical' for x in cdc.values()),'Critical CDC remains')
    drc=findings(reports/'drc.rpt');method=findings(reports/'methodology.rpt')
    routed=mode in ('routed','reopened','postroute_explore')
    require(not any(x['severity']=='Error' or (x['severity']=='Critical Warning' and (k!='RTSTAT-13' or routed)) for k,x in drc.items()),'Blocking physical DRC')
    require(not any(method.get(k,{}).get('count',0) for k in ('TIMING-2','TIMING-4','TIMING-6','TIMING-7','CLKC-5','AVAL-324')),'Clock methodology regression')
    require('No waivers found.' in (reports/'waivers.rpt').read_text(),'Waivers present')
    timing=report_timing(reports/'timing.rpt')
    coverage={k:int(n) for k,n in re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',(reports/'check_timing.rpt').read_text(),re.M)}
    require(all(coverage.get(k)==0 for k in ('no_clock','multiple_clock','loops','latch_loops')),'Clock topology failure')
    skew_text=(reports/'bus_skew.rpt').read_text().split('2. Bus Skew Report Per Constraint')[1]
    skews=re.findall(r'^\s*(Slow|Fast)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$',skew_text,re.M)
    require(len(skews)==18,'Skew report coverage changed')
    skew=dict(groups=len(skews),minimum_slack_ns=min(float(x[3]) for x in skews),failing=sum(float(x[3])<0 for x in skews))
    route=route_status(reports/'route_status.rpt',routed)
    scoped_timing={name:path_slack(table(contract/name)) for name in
                  ('paths.tsv','snapshot_bounds.tsv','loader_bounds.tsv','snapshot_interstage.tsv',
                   'loader_interstage.tsv','readiness_source_paths.tsv','synchronous_paths.tsv','board_lock_paths.tsv')}
    result=dict(status='BOARD_IMPLEMENTATION_STAGE_REVIEWED',stage=mode,seconds=r['seconds'],
        donor_sha256=r['checkpoint_sha256'],checkpoint_sha256=sha(out/(mode+'.dcp')) if mode not in ('reopened','audited') else None,
        physical_ports=75,trial_assignments=21,preserved_bounded_paths=2195,
        synchronous_timed=88-len(missing),unused_lock_fault_checks_pruned=len(missing),
        path_counts=counts,readiness_reset_encoding=reset_encoding,removed_xdc_commands=dict(before-after),cdc=cdc,drc=drc,methodology=method,
        timing=timing,timing_coverage=coverage,scoped_timing=scoped_timing,bus_skew=skew,route_status=route,physical_route_executed=routed,
        timing_qualified=False,board_qualified=False,pinout_released=False,native_traffic_qualified=False)
    return result



TIMING_PREFIXES=('set_max_delay ','set_min_delay ','set_bus_skew ','set_false_path ',
                 'set_clock_groups ','set_multicycle_path ','set_disable_timing ',
                 'create_waiver ','create_clock ','create_generated_clock ',
                 'set_clock_uncertainty ','set_input_delay ','set_output_delay ')


def timing_commands(path):
    return Counter({s:n for s,n in expand_xdc(path.read_text()).items() if s.startswith(TIMING_PREFIXES)})


def complete_timing(path):
    result=report_timing(path)
    rows=re.findall(r'^\s*((?:-?\d+(?:\.\d+)?\s+){11}-?\d+(?:\.\d+)?)\s*$',path.read_text(),re.M)
    require(rows,'Missing complete timing summary')
    values=rows[0].split()
    require(float(values[0])==result['wns_ns'] and int(values[2])==result['setup_failing'],
            'Timing summary parsers disagree')
    result.update(tns_ns=float(values[1]),setup_total=int(values[3]),ths_ns=float(values[5]),
                  hold_total=int(values[7]),tpws_ns=float(values[9]),pulse_total=int(values[11]))
    return result


def methodology_paths(path):
    result={}
    pattern=r'(TIMING-16#\d+) Warning\s+Large setup violation\s+There is a large setup violation of ([-\d.]+) ns between\s+(.*?) \(clocked by (.*?)\) and (.*?) \(clocked by (.*?)\)'
    for match in re.finditer(pattern,path.read_text(),re.S):
        name,slack,src,clock,dst,end_clock=match.groups()
        key=(''.join(src.split()),''.join(dst.split()))
        require(key not in result,'Duplicate methodology endpoint')
        result[key]=dict(finding=name,slack_ns=float(slack))
    require(len(result)==findings(path).get('TIMING-16',{}).get('count',0),'Incomplete methodology path extraction')
    return result


def review(root,baseline,donor,mode,reference):
    r=closed(root);out=root/'output'
    require(r['checkpoint_sha256']==sha(donor),'Wrong checkpoint donor')
    binding=json.loads((root/'source/TIMING_BINDING.json').read_text())
    require(binding['checkpoint_sha256']==sha(donor),'Prepared checkpoint binding differs')
    require(binding['cpu_count']==2 and binding['nice']==10,'Resource isolation differs')
    for name,digest in binding['frozen_audits'].items():
        require(sha(root/'source'/name)==digest,'Frozen audit changed: '+name)
    before=expand_xdc((out/'before.xdc').read_text())
    require(before==expand_xdc((reference/'output/reports/effective.xdc').read_text()),'Donor constraints changed before run')
    if mode=='diagnosed':
        require(binding['mode']=='diagnose','Wrong diagnosis mode')
        require((out/'STATUS.txt').read_text().strip()=='HBM_ROUTED_TIMING_DIAGNOSTICS_EXPORTED','Diagnosis incomplete')
        require(before==expand_xdc((out/'after.xdc').read_text()),'Read-only diagnosis changed constraints')
        timing=complete_timing(out/'timing_before_methodology.rpt')
        require(timing==complete_timing(out/'timing_after_methodology.rpt'),'Methodology changed reported timing')
        paths={k:table(out/(k+'.tsv')) for k in ('worst_setup','worst_hold','hbm_response','hbm_recovery',
               'transport_return','pcie_msi23','pcie_msi10','pcie_slave','pcie_msi23_after',
               'pcie_msi23_ignored','pcie_msi10_ignored','pcie_slave_ignored')}
        require(len(paths['worst_setup'])==128 and len(paths['worst_hold'])==8,'Incomplete worst path sample')
        require(all(paths[k] for k in ('hbm_response','hbm_recovery','transport_return','pcie_msi23','pcie_msi10','pcie_slave')),'Missing targeted timing')
        require(paths['pcie_msi23']==paths['pcie_msi23_after'],'Methodology changed targeted PCIe timing')
        return dict(status='HBM_ROUTED_TIMING_DIAGNOSIS_REVIEWED',seconds=r['seconds'],
                    donor_sha256=r['checkpoint_sha256'],constraints_unchanged=True,timing=timing,
                    methodology=findings(out/'methodology_fresh.rpt'),paths=paths,
                    route_status=route_status(out/'route_status.rpt',True),
                    timing_qualified=False,board_qualified=False)
    require(mode in ('postroute_explore','reopened'),'Unsupported physical review stage')
    result=review_implementation(root,baseline,donor,mode)
    require(timing_commands(out/'before.xdc')==timing_commands(out/'reports/effective.xdc'),
            'Physical experiment added or removed timing commands')
    require(result['bus_skew']['failing']==0,'Bus skew regression')
    require(all(x['failing']==0 for x in result['scoped_timing'].values()),'Scoped contract timing failure')
    require(route_status(out/'route_before.rpt',True)['complete'],'Donor was not fully routed')
    experiment=table(out/'experiment.tsv')
    expected_mode='explore' if mode=='postroute_explore' else 'reopen'
    command='phys_opt_design -directive Explore' if mode=='postroute_explore' else 'open_checkpoint'
    require(binding['mode']==expected_mode and len(experiment)==1,'Wrong experiment binding')
    require(experiment[0]==dict(mode=expected_mode,command=command,cpu_limit='2',clock_or_exception_edits='0',qualified='0'),'Unexpected experiment command')
    result.update(status='HBM_POSTROUTE_TIMING_EXPERIMENT_REVIEWED',experiment=experiment[0],
                  timing_commands_preserved=True,before_timing=complete_timing(out/'timing_before.rpt'),
                  timing=complete_timing(out/'reports/timing.rpt'))
    if mode=='reopened':
        expected=table(root/'source/pcie_methodology_targets.tsv')
        actual=table(out/'pcie_reconciliation/pcie_paths.tsv')
        require(len(expected)==len(actual)==18,'Incomplete PCIe reconciliation')
        current=methodology_paths(out/'reports/methodology.rpt')
        compared=[]
        for before,row in zip(expected,actual):
            require(all(before[k]==row[k] for k in ('finding','from','to')) and
                    before['warning_ns']==row['baseline_warning_ns'],'Changed PCIe target')
            require(float(row['requirement_ns'])==4 and float(row['period_ns'])==2 and
                    row['exception']=='MultiCycle Path   Setup -start 2','PCIe timing exception changed')
            warning=current.get((row['from'],row['to']))
            compared.append(dict(**row,current_warning=warning,
                warning_to_timed_delta_ns=round(float(row['slack_ns'])-warning['slack_ns'],3) if warning else None))
        result['pcie_reconciliation']=dict(paths=compared,queried=18,
            minimum_slack_ns=min(float(x['slack_ns']) for x in actual),
            warning_deltas_ns=sorted({x['warning_to_timed_delta_ns'] for x in compared if x['current_warning']}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','baseline','donor','reference'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=('diagnosed','postroute_explore','reopened'),required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=review(a.root,a.baseline,a.donor,a.mode,a.reference)
    value=json.dumps(result,indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(value)
    else:print(value,end='')
