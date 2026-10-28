"""Review full-RTL capture integration without claiming board timing closure."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_reset_integration import (
    closed, sha, require, table, stable, mapped_paths, path_slack, complete_timing,
    semantic_timing, findings, route_status, PINS, incremental_reuse,
    loader_paths, loader_sync, hbm_connections, snapshot_bounds, physical_readiness)


def capture_semantic_timing(path, groups):
    result=Counter()
    for command,count in semantic_timing(path,groups).items():
        # Resynthesis regenerates timestamps on identical internal IP waivers.
        # Compare their complete commands and multiplicities, excluding only
        # this nonfunctional trailing metadata; never discard waiver commands.
        if command.startswith('create_waiver ') and ' -scope -internal ' in command:
            command=re.sub(r' -timestamp "[^"\r\n]+"$', '', command)
        result[command]+=count
    return result


def queue_proof(out, timed):
    rows=table(out/'response_capture/enables.tsv')
    expected={f'core/reader/channels[{c}].adapter/tracker/response_cut/slot{s}_reg[{b}]'
              for c in range(2) for s in range(2) for b in range(265)}
    require(len(rows)==1060 and {r['cell'] for r in rows}==expected, 'Full-board response inventory')
    for row in rows:
        prefix=f"core/reader/channels[{row['channel']}].adapter/tracker/response_cut"
        require(row['cell'].startswith(prefix+'/') and row['primitive']=='FDRE'
                and row['clock']=='clk_out2_fabric_clocks' and row['driver'], 'Response primitive/clock')
        starts=set(row['startpoints'].split(','))
        allowed={prefix+'/count_reg[0]/C',prefix+'/count_reg[1]/C',prefix+'/write_slot_reg/C',
                 'core/reader/startup/memory_distribution_reg/C'}
        require(starts and '' not in starts and starts<=allowed, 'External signal in payload enable')
    fanout=table(out/'response_capture/rvalid_endpoints.tsv')
    for channel in range(2):
        prefix=f'core/reader/channels[{channel}].adapter/tracker/response_cut/'
        ends=[r['endpoint'] for r in fanout if r['channel']==str(channel)]
        require(ends and len(ends)==len(set(ends)), 'RVALID endpoint export')
        require(not any(e.startswith(prefix+'slot') and e.endswith('/CE') for e in ends), 'RVALID reaches payload enable')
        require(any(e.startswith(prefix+'count_reg') or e.startswith(prefix+'write_slot_reg') for e in ends), 'Ownership lost RVALID')
    require((out/'response_capture/STATUS.txt').read_text().strip()=='CAPTURE_BOTH_CHANNELS_EXPORTED', 'Incomplete queue probe')
    result=dict(channels=2,payload_registers=1060,enables_from_rvalid=0,ownership_depends_on_rvalid=True)
    if timed:
        for kind in ('max','min'):
            paths=table(out/f'response_capture/enable_{kind}.tsv')
            require(len(paths)==1060 and {r['to'] for r in paths}=={e+'/CE' for e in expected}, 'Enable timing coverage')
            require(all(r['clock']==r['end_clock']=='clk_out2_fabric_clocks' for r in paths), 'Enable timing clock')
            result[kind]=path_slack(paths)
    return result


def synthesis_review(root,reference):
    native=closed(root);out=root/'output'
    require(native['stage']=='synth' and native['sources_unchanged'],'Synthesis source/status')
    require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Synthesis incomplete')
    binding=json.loads((root/'source/CAPTURE_PHYSICAL_BINDING.json').read_text())
    for name,digest in binding['donor_files'].items():require(sha(reference/'rtl-donor'/name)==digest,'Frozen RTL donor changed')
    for name,digest in binding['prepared_files'].items():require(sha(root/'source'/name)==digest,'Prepared synthesis source changed')
    require(binding['changed_original_files']==['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv','run_hbm_native_join.py'], 'Unexpected original source change')
    old=(reference/'rtl-donor/ntrpu_hbm_read_adapter.sv').read_text()
    candidate=(reference/'unit/ntrpu_hbm_response_capture_queue.sv').read_text()
    anchor='ntrpu_hbm_read_response_queue #(.WIDTH(265)) response_cut ('
    replacement='ntrpu_hbm_response_capture_queue #(.WIDTH(265)) response_cut ('
    require(old.count(anchor)==1 and old.replace(anchor,replacement,1)+'\n'+candidate==
            (root/'source/ntrpu_hbm_read_adapter.sv').read_text(),'Full RTL delta exceeds response binding')
    require(sha(reference/'unit/ntrpu_hbm_response_capture_queue.sv')==binding['tested_candidate_sha256'],'Untested queue candidate')
    ports=table(out/'ports.tsv');require(len(ports)==75 and len({p['port'] for p in ports})==75,'Physical port inventory')
    instances=table(out/'native_instances.tsv')
    require({r['cell'] for r in instances}=={'ntrpu_bridge_i','hbm_binding/memory','clock_generator','core'},'Native IP composition')
    require('No waivers found.' in (out/'reference_waivers.rpt').read_text(),'Synthesis waivers')
    return dict(status='CAPTURE_FULL_SYNTHESIS_REVIEWED',seconds=native['seconds'],
                queue=queue_proof(out,False),physical_ports=75,
                checkpoint_sha256=sha(out/'synth.dcp'),full_rtl_resynthesis=True,
                physical_timing_qualified=False,native_pcie_hbm_traffic=False,board_qualified=False)


def reset_probe_review(root,donor):
    native=closed(root)
    require(native['checkpoint_sha256']==sha(donor),'Reset probe donor changed')
    require((root/'output/STATUS.txt').read_text().strip()=='CAPTURE_RESET_PROBE_EXPORTED','Reset probe incomplete')
    props={}
    for row in table(root/'output/reset_probe.tsv'):
        key=row['object'],row['property']
        require(key not in props or props[key]==row['value'],'Contradictory reset probe')
        props[key]=row['value']
    canonical='core/endpoint/guard/cold_m_reg[2]/Q'
    require(props['core/management_cold_rst_n','DRIVERS']==canonical,'Canonical reset differs')
    driver=props['loader_readiness/ready_reg/CLR','DRIVERS'];inv=driver.removesuffix('/O')
    require(props['loader_readiness/ready_reg/CLR','IS_INVERTED']=='0'
            and props['loader_readiness/ready_reg','IS_CLR_INVERTED']==''
            and props['loader_readiness/ready_reg','REF_NAME']=='FDCE'
            and props[inv,'REF_NAME']=='LUT1' and props[inv,'INIT']=="2'h1"
            and props[inv+'/I0','DRIVERS']==canonical,'Unproven explicit reset inverter')
    return dict(status='CAPTURE_SYNTH_RESET_ENCODING_REVIEWED',seconds=native['seconds'],
                cell_inversion_raw='',pin_inversion='0',inverter=inv,init="2'h1",
                canonical_driver=canonical,netlist_changed=False)


def reset_proof(out,timed):
    rows=table(out/'reset/reset.tsv')
    props={(r['object'],r['property']):r['value'].strip('{}') for r in rows}
    require(len(props)==len(rows),'Duplicate reset property')
    sync='core/reader/startup/memory_release_reg[2]'
    dist='core/reader/startup/memory_distribution_reg'
    for cell in (sync,dist):
        require(props[cell,'REF_NAME']=='FDCE' and props[cell,'INIT']=="1'b0",'Reset primitive')
        require(props[cell+'/C','CLOCK']=='clk_out2_fabric_clocks','Reset clock')
    require(props[sync,'ASYNC_REG']=='1' and props[dist,'ASYNC_REG'] not in ('1','true','TRUE')
            and props[dist,'DONT_TOUCH'] not in ('1','true','TRUE'), 'Reset register attributes')
    driver=props[dist+'/CLR','DRIVER']
    encoding=props[dist,'RESET_ENCODING']
    if encoding=='PRIMITIVE_INVERTED_CLR':
        require(driver==sync+'/Q' and props[dist+'/CLR','IS_INVERTED']=='1'
                and props[sync+'/Q','LOADS']==dist+'/CLR','Primitive reset polarity/fanout')
    else:
        inv=driver.removesuffix('/O')
        require(encoding=='EXPLICIT_INVERTER' and props[inv,'REF_NAME']=='LUT1'
                and props[inv,'INIT']=="2'h1" and props[inv+'/I0','DRIVER']==sync+'/Q'
                and props[inv+'/O','LOADS']==dist+'/CLR' and props[dist+'/CLR','IS_INVERTED']=='0'
                and props[sync+'/Q','LOADS']==inv+'/I0','Explicit reset polarity/fanout')
    require(props['core/reader/startup/ready_meta_reg[2]/D','DRIVER']==dist+'/Q','Readiness bypass')
    loads=(out/'reset/distribution_loads.txt').read_text().splitlines()
    require(len(loads)==len(set(loads)) and len(loads)>1000,'Distribution fanout coverage')
    result=dict(encoding=encoding,loads=len(loads),canonical_release_loads=1)
    if timed:
        result['timing']={}
        for kind in ('max','min'):
            paths=table(out/f'reset/recovery_{kind}.tsv')
            require(len(paths)==1 and paths[0]['STARTPOINT_PIN']==sync+'/C'
                    and paths[0]['ENDPOINT_PIN']==dist+'/CLR', 'Recovery/removal coverage')
            require(paths[0]['STARTPOINT_CLOCK']==paths[0]['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks'
                    and not paths[0]['EXCEPTION'], 'Recovery/removal must remain synchronous and unexcepted')
            result['timing'][kind]=float(paths[0]['SLACK'])
    return result


def contract_proof(out,reference,timed):
    c=out/'contracts';r=out/'reports';baseline=reference/'cdc/output'
    require((c/'AUDIT_STATUS.txt').read_text().strip()=='BOARD_IMPLEMENTATION_CONTRACTS_EXPORTED_REVIEW_REQUIRED','Missing board audit')
    clocks={row['logical']:row['clock'] for row in table(c/'domain_clocks.tsv')}
    require(len(clocks)==4,'Clock inventory')
    for name,count in [('paths.tsv',1503),('snapshot_bounds.tsv',390),('loader_bounds.tsv',302),
                       ('snapshot_interstage.tsv',32),('loader_interstage.tsv',16),('readiness_source_paths.tsv',6)]:
        rows=table(c/name)
        require(len(rows)==count and stable(rows)==mapped_paths(table(baseline/name),name),'Path contract changed: '+name)
    loader_paths(table(c/'loader_bounds.tsv'),clocks)
    loader_sync(table(c/'loader_synchronizers.tsv'),table(c/'loader_interstage.tsv'),clocks)
    hbm_connections(table(c/'loader_hbm_connections.tsv'))
    snapshot_bounds(table(c/'snapshot_bounds.tsv'))
    physical_readiness(c,clocks)
    for row in table(c/'readiness_reset.tsv'):
        if 'clear_inverted_raw' in row:
            require(row['clear_inverted_raw']=='' and row['clear_inverted']==row['pin_inverted']=='0'
                    and row['driver_reference']=='LUT1' and row['driver_init']=="2'h1"
                    and row['inversion_basis']=='PIN_IS_INVERTED_WITH_VERIFIED_LUT',
                    'Unproven unset inversion normalization')
    sync=table(c/'synchronous_paths.tsv')
    require(len(sync)==88 and stable([{k:v for k,v in row.items() if k!='status'} for row in sync])==
            stable(table(baseline/'board_synchronous_paths.tsv')), 'Board synchronous contract')
    missing=[row for row in sync if row['status']!='TIMED']
    require(len(missing) in (0,4) and all(row['status']=='PRUNED_UNUSED_LOCK_FAULT' and row['to'] in
            ('board_startup/lock_fault_reg/D','board_startup/lock_fault_reg/CLR') for row in missing), 'Unexpected synchronous path pruning')
    require(stable(table(c/'reference_tree.tsv'))==stable(table(baseline/'reference_tree.tsv')), 'HBM reference tree')
    locks=table(c/'board_lock_paths.tsv');arcs=table(c/'board_lock_arcs.tsv')
    require(len(locks)==4 and all(float(x['requirement_ns'])==10.0 for x in locks),'LOCKED bound changed')
    require(len(arcs)==4 and all(x['disabled']=='1' and x['user_disabled']=='0' for x in arcs),'LOCKED model changed')
    pins=table(out/'trial_pins.tsv')
    require(len(pins)==11 and {row['port']:(row['ball'],row['standard']) for row in pins}==PINS,'Trial pin assignment')
    require(all(row['status']=='EXPERIMENT_ONLY' for row in pins),'Pin release not authorized')
    selectors=table(out/'calibration_selectors.tsv')
    expected_selectors=table(reference/'calibration_selectors.tsv')
    require(selectors==expected_selectors,'PCIe calibration selector members changed')
    groups={}
    for row in selectors:groups.setdefault((int(row['lane']),row['family']),set()).add(row['cell'])
    require(len(groups)==32 and sum(map(len,groups.values()))==624,'Calibration selector coverage')
    current=capture_semantic_timing(r/'effective.xdc',groups)
    original=capture_semantic_timing(reference/'reset_donor_effective.xdc',groups)
    require(current==original,'Full-board timing clocks/exceptions differ from reset donor')
    if timed: require(capture_semantic_timing(out/'before.xdc',groups)==current,'Physical flow changed timing constraints')
    require('No waivers found.' in (r/'waivers.rpt').read_text(),'Timing/CDC waivers present')
    coverage={k:int(n) for k,n in re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',(r/'check_timing.rpt').read_text(),re.M)}
    require(all(coverage.get(k)==0 for k in ('no_clock','multiple_clock','loops','latch_loops')), 'Clock coverage failure')
    cdc={k:dict(severity=s,count=int(n)) for k,s,n in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+', (r/'cdc.rpt').read_text(),re.M)}
    require(cdc and not any(x['severity']=='Critical' for x in cdc.values()),'Critical CDC')
    drc=findings(r/'drc.rpt');method=findings(r/'methodology.rpt')
    require(not any(x['severity']=='Error' or (x['severity']=='Critical Warning' and (k!='RTSTAT-13' or timed))
                    for k,x in drc.items()),'Blocking DRC')
    require(not any(method.get(k,{}).get('count',0) for k in ('TIMING-2','TIMING-4','TIMING-6','TIMING-7','CLKC-5','AVAL-324')),'Clock methodology regression')
    skew_text=(r/'bus_skew.rpt').read_text().split('2. Bus Skew Report Per Constraint')[1]
    skews=re.findall(r'^\s*(Slow|Fast)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$',skew_text,re.M)
    require(len(skews)==18,'Bus skew coverage')
    scoped={name:path_slack(table(c/name)) for name in ('paths.tsv','snapshot_bounds.tsv','loader_bounds.tsv',
            'snapshot_interstage.tsv','loader_interstage.tsv','readiness_source_paths.tsv','synchronous_paths.tsv','board_lock_paths.tsv')}
    return dict(preserved_bounded_paths=2195,synchronous_timed=88-len(missing),timing=complete_timing(r/'timing.rpt'),
                scoped=scoped,cdc=cdc,drc=drc,methodology=method,timing_coverage=coverage,
                bus_skew=dict(groups=18,minimum_slack_ns=min(float(x[3]) for x in skews),failing=sum(float(x[3])<0 for x in skews)),
                route=route_status(r/'route_status.rpt',timed))


def review(root,reference,donor,mode):
    native=closed(root);out=root/'output'
    require(native['sources_unchanged'] and native['checkpoint_sha256']==sha(donor),'Input checkpoint chain')
    require((out/'STATUS.txt').read_text().strip()==f'CAPTURE_BOARD_{mode}_EXPORTED','Native stage incomplete')
    binding=json.loads((root/'source/CAPTURE_STAGE_BINDING.json').read_text())
    require(binding['mode']==mode and binding['checkpoint_sha256']==sha(donor),'Stage binding differs')
    for name,digest in binding['prepared_files'].items():require(sha(root/'source'/name)==digest,'Prepared source modified')
    for name,digest in binding['frozen_audits'].items():require(sha(root/'source'/name)==digest,'Frozen audit modified')
    timed=mode!='constrain'
    result=dict(status='CAPTURE_FULL_BOARD_REVIEWED',mode=mode,seconds=native['seconds'],
                queue=queue_proof(out,timed),reset=reset_proof(out,timed),contracts=contract_proof(out,reference,timed),
                full_rtl_resynthesis=True,physical_timing_qualified=False,native_pcie_hbm_traffic=False,board_qualified=False)
    if mode=='route':
        require((out/'incremental_sha256.txt').read_text().strip()==binding['incremental_sha256'],'Incremental donor changed')
        result['reuse']={stage:incremental_reuse(out/f'reuse_{stage}.rpt') for stage in ('before','placed','routed')}
        result['checkpoint_sha256']=sha(out/'routed.dcp')
    elif mode=='constrain':result['checkpoint_sha256']=sha(out/'optimized.dcp')
    else:result['checkpoint_sha256']=sha(donor)
    if timed:
        for name in ('rready_max','rready_min','worst_setup'):
            rows=table(out/(name+'.tsv'));require(rows,'Empty targeted timing sample')
            result[name]=dict(paths=len(rows),minimum_slack_ns=min(float(row['SLACK']) for row in rows),worst=rows[0])
        result['preserved_bounds_pass']=all(x['failing']==0 for x in result['contracts']['scoped'].values())
        result['reset_timing_pass']=min(result['reset']['timing'].values())>=0
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','donor'):p.add_argument(name,type=Path)
    p.add_argument('--mode',choices=['constrain','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(review(a.root,a.reference,a.donor,a.mode),indent=2))
