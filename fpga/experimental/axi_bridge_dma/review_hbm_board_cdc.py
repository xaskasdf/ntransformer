"""Review canonical reset RTL and the applied/reopened board CDC constraints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_loader_readiness import (closed, rows, require, sha, expand_xdc,
    loader_paths, loader_sync, hbm_connections, readiness_source, snapshot_bounds)


def simulation(root):
    result=closed(root)
    require(result['status']=='HBM_COLD_RELEASE_SIM_EXPORTED', 'Wrong simulation')
    positive=(root/'positive.log').read_text()
    cases=re.findall(r'RC CASE PASS case=(\d+) checks=(\d+) groups=(\d+)',positive)
    require(sorted(tuple(map(int,c)) for c in cases)==[(i,37,12) for i in range(4)] and
            'RC PASS cases=4' in positive, 'Missing cold-release scenarios')
    negative=(root/'mutation-run.log').read_text()
    require(re.search(r'^Fatal: RC FAIL case=\d+ status held until canonical reader release$',negative,re.M),
            'Raw-reset mutation did not fail the intended invariant')
    candidate=(root/'source/ntrpu_hbm_startup.sv').read_text()
    anchor='always @(posedge clk or negedge src_rst_n)begin\n        if(!src_rst_n)begin'
    require(candidate.count(anchor)==1 and (root/'mutation/ntrpu_hbm_startup.sv').read_text()==
            candidate.replace(anchor,'always @(posedge clk or negedge cold_reset_n)begin\n        if(!cold_reset_n)begin',1),
            'Wrong actual RTL mutation')
    return dict(status='BOARD_COLD_RELEASE_SIM_REVIEWED',cases=4,checks=148,scenario_groups=48,
                original_reset_mutation_rejected=True,seconds=result['seconds'],
                candidate_sha256=sha(root/'source/ntrpu_hbm_startup.sv'),native_traffic_qualified=False)


def same_table(applied,reopened,name,ignore=('slack_ns','datapath_ns')):
    a,b=rows(applied,name),rows(reopened,name)
    clean=lambda data:[{k:v for k,v in row.items() if k not in ignore} for row in data]
    require(clean(a)==clean(b),'Reopened table differs: '+name)
    return a


def integration(sim,synth,applied,reopened,baseline):
    sim_review=simulation(sim)
    sr,ar,rr=[closed(p) for p in (synth,applied,reopened)]
    require(sr['stage']=='synth' and ar['checkpoint_sha256']==sha(synth/'output/synth.dcp') and
            rr['checkpoint_sha256']==sha(applied/'output/board_cdc_bounded.dcp'),'Checkpoint chain differs')
    require(sha(synth/'source/ntrpu_hbm_startup.sv')==sim_review['candidate_sha256'],'Untested RTL synthesized')
    binding=json.loads((synth/'source/BOARD_CDC_BINDING.json').read_text())
    require(binding['changed_rtl']=={'ntrpu_hbm_startup.sv':sim_review['candidate_sha256']},'Unexpected candidate scope')
    for name,digest in binding['donor_files'].items():
        if name!='ntrpu_hbm_startup.sv':require(sha(synth/'source'/name)==digest,'Unrelated donor changed: '+name)
    for root,mode in [(applied,1),(reopened,0)]:
        require((root/'output/STATUS.txt').read_text().strip()==f'BOARD_CDC_{mode}_EXPORTED_REVIEW_REQUIRED','Missing audit completion')
    table=lambda name:same_table(applied,reopened,name)
    domains=table('domain_clocks.tsv');clocks={r['logical']:r['clock'] for r in domains}
    require(len(domains)==4,'Missing native domains')
    for name,count in [('paths.tsv',1503),('snapshot_bounds.tsv',390),('loader_bounds.tsv',302),
                       ('interstage.tsv',180),('snapshot_interstage.tsv',32),('loader_interstage.tsv',16),
                       ('readiness_source_paths.tsv',6),('reset_arcs.tsv',18)]:
        current=table(name)
        require(len(current)==count,'Coverage count differs: '+name)
        previous=rows(baseline,name)
        clean=lambda data:Counter(tuple(sorted((k,v) for k,v in r.items() if k not in ('slack_ns','datapath_ns'))) for r in data)
        require(clean(current)==clean(previous),'Frozen timing contract differs: '+name)
    loader_paths(table('loader_bounds.tsv'),clocks)
    loader_sync(table('loader_synchronizers.tsv'),table('loader_interstage.tsv'),clocks)
    hbm_connections(table('loader_hbm_connections.tsv'))
    snapshot_bounds(table('snapshot_bounds.tsv'))
    readiness_source(table('readiness_source_paths.tsv'),table('readiness_source.tsv'),
                     table('readiness_bindings.tsv'),table('loader_readiness_fanin.tsv'),clocks)
    for name in ('csr_connections.tsv','bridge_connections.tsv','bridge_constants.tsv','cold_reset_bindings.tsv',
                 'loader_clock_reset.tsv','telemetry_capture_sources.tsv','capture_topology.tsv','synchronizers.tsv'):
        require(table(name)==rows(baseline,name),'Frozen connectivity differs: '+name)
    require(len(table('physical_ports.tsv'))==75,'Board interface changed')
    sync=table('board_synchronizers.tsv')
    require(len(sync)==6 and {r['cell'] for r in sync}==
            {f'board_startup/{chain}_reg[{bit}]' for chain in ('release_sync','locked_sync') for bit in range(3)},
            'Board synchronizer coverage')
    require(all(r['clock']==clocks['mgmt_clk'] and r['async_reg'].upper() in ('1','TRUE') for r in sync),'Board ASYNC_REG/clock')
    timed=table('board_synchronous_paths.tsv')
    require(Counter(r['kind'] for r in timed)==dict(interstage=8,board_recovery=30,configuration_recovery=6,configuration_control=14,reader_status_recovery=30),
            'Recovery/interstage coverage')
    board='board_startup/'
    local_controls={board+'configuration_rst_n_reg/D',board+'lock_fault_reg/D'}|{board+f'lock_count_reg[{i}]/CE' for i in range(5)}
    recovered={board+n+'_reg/CLR' for n in ('configuration_rst_n','lock_fault')}
    recovered|={board+'fabric_clock_reset_reg/PRE'}
    recovered|={board+f'{chain}_reg[{i}]/CLR' for chain,width in [('hold_count',4),('lock_count',5),('locked_sync',3)] for i in range(width)}
    expected_timed={('interstage',board+f'{chain}_reg[{i}]/C',board+f'{chain}_reg[{i+1}]/D')
                    for chain in ('release_sync','locked_sync') for i in (0,1)}
    expected_timed|={('board_recovery',board+'release_sync_reg[2]/C',to) for to in recovered}
    expected_timed|={('configuration_control',board+'configuration_rst_n_reg/C',to) for to in local_controls}
    expected_timed|={('configuration_recovery',board+'configuration_rst_n_reg/C',f'core/endpoint/guard/cold_m_reg[{i}]/CLR') for i in range(3)}
    expected_timed|={('reader_status_recovery','core/reader/startup/source_release_reg[2]/C',f'core/reader/startup/{chain}_reg[{i}]/CLR')
                    for chain,width in [('ready_meta',3),('ready_sync',3),('ready_final',3),('fault_meta',2),('fault_sync',2),('fault_final',2)] for i in range(width)}
    require(Counter((r['kind'],r['from'],r['to'],r['type']) for r in timed)==
            Counter((*path,typ) for path in expected_timed for typ in ('max','min')),'Synchronous endpoint coverage differs')
    for r in timed:
        clock=clocks['reader_clk'] if r['kind']=='reader_status_recovery' else clocks['mgmt_clk']
        period=5. if r['kind']=='reader_status_recovery' else 10.
        require(r['clock']==clock and float(r['requirement_ns'])==(period if r['type']=='max' else 0.),'Synchronous timing cut')
    expected_new_resets={('supervisor_rst_n_pad',f'board_startup/release_sync_reg[{i}]/CLR') for i in range(3)}
    expected_new_resets|={('board_startup/configuration_rst_n_reg/C',f'{chain}_reg[{i}]/CLR')
        for chain in ('core/endpoint/guard/cold_a','core/reader/startup/source_release','core/reader/startup/memory_release') for i in range(3)}
    resets=table('board_reset_exceptions.tsv')
    require(len(resets)==12 and {(r['from'],r['to']) for r in resets}==expected_new_resets,'Overbroad reset cut')
    registers=table('board_registers.tsv');fanout=table('board_reset_fanout.tsv')
    expected_cells={pin.rsplit('/',1)[0] for pin in recovered}|{board+f'release_sync_reg[{i}]' for i in range(3)}
    require(len(registers)==18 and {r['cell'] for r in registers}==expected_cells,'Board register inventory differs')
    require(all(r['clock']==clocks['mgmt_clk'] and r['reference']==('FDPE' if r['cell']==board+'fabric_clock_reset_reg' else 'FDCE') for r in registers),'Board register mapping differs')
    expected_fanout={board+'release_sync_reg[2]/Q':recovered,
                     board+'configuration_rst_n_reg/Q':local_controls|{f'{chain}_reg[{i}]/CLR'
                        for chain in ('core/endpoint/guard/cold_m','core/endpoint/guard/cold_a','core/reader/startup/source_release','core/reader/startup/memory_release') for i in range(3)},
                     'supervisor_rst_n_pad':{board+f'release_sync_reg[{i}]/CLR' for i in range(3)}}
    for source,ends in expected_fanout.items():
        require(Counter(r['endpoint'] for r in fanout if r['source']==source)==Counter(ends),'Reset fanout differs: '+source)
    for root in (applied,reopened):
        lock=rows(root,'board_lock_paths.tsv');arcs=rows(root,'board_lock_arcs.tsv')
        require(len(lock)==4 and {(r['phase'],r['to']) for r in lock}==
                {(phase,dest) for phase in ('before','after') for dest in ('availability/lock_sync_reg[0]/D','board_startup/locked_sync_reg[0]/D')},'LOCKED endpoint coverage')
        require(all(r['end_clock']==clocks['mgmt_clk'] and (r['phase']=='before' and root==applied or float(r['requirement_ns'])==10.) for r in lock),'LOCKED bound differs')
        require(len(arcs)==4 and all(r['disabled']=='1' and r['user_disabled']=='0' for r in arcs),'LOCKED model arcs altered')
    before=expand_xdc((applied/'output/before.xdc').read_text())
    after=expand_xdc((applied/'output/effective.xdc').read_text())
    require(not before-after,'Vendor constraints removed')
    added=after-before
    counts={kind:sum(n for s,n in added.items() if s.startswith(kind+' '))
            for kind in ('set_max_delay','set_bus_skew','set_false_path')}
    require(counts==dict(set_max_delay=54,set_bus_skew=10,set_false_path=30),'Scoped constraint counts')
    require(not any(s.startswith(('set_clock_groups ','set_multicycle_path ','set_disable_timing ','create_waiver ')) for s in added),'Broad exception/waiver added')
    require(all('-datapath_only' in s for s in added if s.startswith('set_max_delay ')),'Non-datapath bound added')
    def pin(name,quiet=True):return '[get_pins '+('-quiet ' if quiet else '')+('{'+name+'}' if '[' in name else name)+']'
    expected=Counter('set_false_path -from '+pin(r['from'])+' -to '+pin(r['to']) for r in rows(applied,'reset_arcs.tsv'))
    for r in resets:
        source='[get_ports supervisor_rst_n_pad]' if r['kind']=='supervisor' else pin(r['from'],False)
        expected['set_false_path -from '+source+' -to '+pin(r['to'],False)]+=1
    require(Counter({s:n for s,n in added.items() if s.startswith('set_false_path ')})==expected,'Serialized reset exception scope')
    require(after==expand_xdc((reopened/'output/before.xdc').read_text())==expand_xdc((reopened/'output/effective.xdc').read_text()),'DCP constraints changed')
    cdc={};coverage={};method={}
    for label,root in [('synthesis',synth),('apply',applied),('reopen',reopened)]:
        text=(root/'output/cdc.rpt').read_text()
        cdc[label]={code:dict(severity=sev,count=int(n)) for code,sev,n in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',text,re.M)}
        require(cdc[label] and not any(v['severity']=='Critical' for v in cdc[label].values()),'Critical CDC remains')
        report=(root/'output/check_timing.rpt').read_text()
        coverage[label]={k:int(v) for k,v in re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',report,re.M)}
        require(all(coverage[label][k]==0 for k in ('no_clock','multiple_clock','loops','latch_loops')),'Clock/timing topology failure')
        require(coverage[label]['unconstrained_internal_endpoints']==1154 and coverage[label]['no_input_delay']==2,'Unexpected unconstrained boundary')
        text=(root/'output/reference_methodology.rpt').read_text()
        method[label]={k:int(v) for k,v in re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|[^|]+\|[^|]+\|\s*(\d+)\s*\|',text,re.M)}
        require(method[label] and not any(method[label].get(k,0) for k in ('TIMING-2','TIMING-4','CLKC-5')),'Reference-clock regression')
        if label!='synthesis':
            require(not any(method[label].get(k,0) for k in ('TIMING-6','TIMING-7')),'Unbounded clock relation remains')
        require('No waivers found.' in (root/'output/reference_waivers.rpt').read_text(),'Unexpected waiver')
    require(cdc['apply']==cdc['reopen'] and coverage['apply']==coverage['reopen'],'Reopen coverage differs')
    return dict(status='HBM_BOARD_CDC_REBOUND_AND_REVIEWED',simulation=sim_review,
                checkpoint_sha256=sha(applied/'output/board_cdc_bounded.dcp'),preserved_data_paths=2195,
                added_reset_exceptions=12,preserved_reset_exceptions=18,new_synchronous_checks=88,
                scoped_constraints=counts,cdc=cdc,timing_coverage=coverage,methodology=method,
                board_qualified=False,native_traffic_qualified=False,routed_timing_qualified=False,
                placement_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('simulation',type=Path)
    for name in ('synthesis','apply','reopen','baseline'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.synthesis:
        if not all((a.apply,a.reopen,a.baseline)):p.error('Integration review requires all four evidence roots')
        result=integration(a.simulation,a.synthesis,a.apply,a.reopen,a.baseline)
    else:result=simulation(a.simulation)
    print(json.dumps(result,indent=2))
