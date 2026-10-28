"""Review board reset integration with one named CDC source substitution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from review_hbm_routed_timing import *

SYNC='core/reader/startup/memory_release_reg[2]'
DIST='core/reader/startup/memory_distribution_reg'
READY='core/reader/startup/ready_meta_reg[2]/D'
OLD='set_max_delay -datapath_only -from [get_cells {'+SYNC+'}] -to [get_pins -quiet {'+READY+'}] 4.000'
NEW='set_max_delay -datapath_only -from [get_cells '+DIST+'] -to [get_pins -quiet {'+READY+'}] 4.000'

def list_encoding(text):
    # Vivado reserializes exactly these two four-element Tcl lists on read_xdc.
    # Normalize only the observed forms; never drop/reorder a timing endpoint.
    for command,prefix,suffix in [('get_cells','weight_loader/memory_result_reg',''),
                                  ('get_pins','weight_loader/completion_reg','/D')]:
        items=' '.join('{'+prefix+f'[{i}]'+suffix+'}' for i in range(4))
        text=text.replace('['+command+' {'+items+'}]','['+command+' [list '+items+']]')
    return text


def semantic_xdc(text):
    # Retain scope for every command. Re-reading XDC can repeat idempotent
    # physical property assignments; timing constraints and waivers keep counts.
    logical=re.sub(r'\\\r?\n\s*',' ',text)
    aliases={};scope='';result=Counter()
    for line in logical.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        match=re.fullmatch(r'set (_xlnx_shared_i\d+) (.+)',line)
        if match:
            require(match[1] not in aliases,'Duplicate alias definition')
            aliases[match[1]]=match[2];continue
        if line.startswith('current_instance '):
            scope=line.removeprefix('current_instance ').removeprefix('{').removesuffix('}')
            if scope=='-quiet':scope=''
            continue
        for _ in range(len(aliases)+1):
            new=re.sub(r'\$(_xlnx_shared_i\d+)\b',lambda m:aliases[m[1]],line)
            if new==line:break
            line=new
        require(not re.search(r'\$_xlnx_shared_i',line),'Unresolved scoped alias')
        line=list_encoding(line);key=scope+' :: '+line
        if line.startswith('set_property '):result[key]=1
        else:result[key]+=1
    return result


def normalized_timing(path):
    result=Counter()
    for command,count in timing_commands(path).items():result[list_encoding(command)]+=count
    return result

def mapped_paths(rows,name):
    mapped=[dict(row) for row in rows]
    if name=='paths.tsv':
        changed=0
        for row in mapped:
            if row['from']==SYNC+'/C' and row['to']==READY:
                require(float(row['requirement_ns'])==4.,'Original readiness bound differs')
                row['from']=DIST+'/C';changed+=1
        require(changed==1,'Expected one readiness path mapping')
    return stable(mapped)

def lines(path):return path.read_text().splitlines()


def reset_topology(out,expected_consumers):
    rows=table(out/'reset_topology.tsv')
    props={(r['object'],r['property']):r['value'].removeprefix('{').removesuffix('}') for r in rows}
    require(len(props)==len(rows),'Duplicate reset property')
    for cell in (SYNC,DIST):
        require(props[cell,'REF_NAME']=='FDCE' and props[cell,'INIT']=="1'b0" and
                props[cell,'IS_CLR_INVERTED']=="1'b1",'Reset primitive differs')
    require(props[SYNC,'ASYNC_REG']=='1' and props[DIST,'ASYNC_REG'] not in ('1','TRUE') and
            props[DIST,'DONT_TOUCH'] not in ('1','TRUE'),'Synchronization/distribution attributes differ')
    require(props[DIST+'/CLR','DRIVERS']==SYNC+'/Q' and props[SYNC+'/Q','LOADS']==DIST+'/CLR',
            'Canonical reset no longer drives only distribution CLR')
    require(props[DIST+'/C','CLOCK']=='clk_out2_fabric_clocks' and
            props[DIST+'/C','DRIVERS']==props[SYNC+'/C','DRIVERS']=='clock_generator/inst/clkout2_buf/O',
            'Reset clock differs')
    require(props[DIST+'/D','DRIVERS']==props[DIST+'/CE','DRIVERS']==props[SYNC+'/CE','DRIVERS']==
            'core/reader/startup/VCC/P','Distribution constant differs')
    require(props[READY,'DRIVERS']==DIST+'/Q','Ready status bypasses distribution')
    consumers=lines(out/'reset_consumers.txt')
    require(consumers==sorted(set(consumers))==expected_consumers and len(consumers)==1091,'Reset consumer set changed')
    return dict(consumer_pins=len(consumers),canonical_leaf_loads=1,clock=props[DIST+'/C','CLOCK'],
                distribution_location=props[DIST,'LOC'],distribution_bel=props[DIST,'BEL'])


def binding(root,reference,donor,mode):
    result=closed(root)
    require(result['checkpoint_sha256']==sha(donor),'Wrong native input checkpoint')
    record=json.loads((root/'source/RESET_INTEGRATION_BINDING.json').read_text())
    require(record['mode']==mode and record['checkpoint_sha256']==sha(donor),'Preparation/input mismatch')
    require(record['cpu_count']==2 and record['nice']==10 and
            record['timeout_seconds']==(1800 if mode=='route' else 900),'Resource boundary changed')
    require(sha(root/'source/integrate_hbm_reset_distribution.tcl')==record['script_sha256'],'Integration script differs')
    for name,digest in record['frozen_audit_origins'].items():
        origin=reference/'prepared-audits'/name
        require(sha(origin)==digest and sha(root/'source'/name)==record['frozen_audits'][name],'Frozen audit hash differs')
        if name==record['changed_audit']:
            text=origin.read_text();old=record['old_audit_row'];new=record['new_audit_row']
            require(text.count(old)==1 and text.replace(old,new,1)==(root/'source'/name).read_text(),
                    'Audit edits exceed readiness source mapping')
        else:require(record['frozen_audits'][name]==digest,'Unrelated audit modified')
    return result


def incremental_reuse(path):
    result={}
    for line in path.read_text().splitlines():
        if not re.match(r'^\|\s*(Cells|Nets|Pins|Ports)\s*\|',line):continue
        columns=[x.strip() for x in line.strip().strip('|').split('|')]
        require(len(columns) in (5,6),'Unexpected reuse columns')
        # Placement adds an Initial Reuse column; Current Reuse remains third
        # from the end. Preserve that distinction in the archived review.
        name,matched=columns[:2];reused,fixed,total=columns[-3:]
        require(name not in result,'Duplicate reuse summary')
        result[name]=dict(matched_percent=None if matched=='-' else float(matched),
                          initial_reused_percent=float(columns[2]) if len(columns)==6 else None,
                          reused_percent=float(reused),fixed_percent=None if fixed=='-' else float(fixed),
                          total=int(total))
    require(set(result)=={'Cells','Nets','Pins','Ports'},'Incomplete reuse summary')
    return result


def calibration_constraint(lane):
    def selector(family):
        return '[get_cells -hierarchical -filter {NAME =~ *gen_cpll_cal_inst['+str(lane)+'].*U_TXOUTCLK_FREQ_COUNTER/'+family+'*}]'
    return 'set_false_path -from '+selector('testclk_cnt_reg')+' -to '+selector('freq_cnt_o_reg')+' -quiet'


def semantic_timing(path,groups):
    result=Counter()
    for command,count in normalized_timing(path).items():
        match=re.fullmatch(r'set_false_path -from \[get_cells \[list (.*?)\]\] -to \[get_cells \[list (.*?)\]\] -quiet',command)
        if match:
            sets=[]
            for items in match.groups():
                names=re.findall(r'\{([^{}]+)\}',items)
                if not names or re.sub(r'\{[^{}]+\}','',items).strip():break
                require(len(names)==len(set(names)),'Duplicated explicit timing endpoint')
                sets.append(set(names))
            if len(sets)==2:
                matches=[lane for lane in range(16) if sets==[groups[lane,'testclk_cnt_reg'],groups[lane,'freq_cnt_o_reg']]]
                require(len(matches)<=1,'Ambiguous calibration selector')
                if matches:command=calibration_constraint(matches[0])
        result[command]+=count
    require(all(result[calibration_constraint(lane)]==1 for lane in range(16)),
            'Calibration false-path endpoint set changed')
    return result


def semantic_probe(probe,reference,candidate,edited):
    require(probe is not None,'Closed alias/selector probe required')
    binding(probe,reference,candidate,'reopen')
    require(not re.search(r'^ERROR:',(probe/'tool.log').read_text(),re.M),'Native probe error')
    b=json.loads((probe/'source/RESET_INTEGRATION_BINDING.json').read_text());out=probe/'output'
    require(b['alias_reference_sha256']==sha(edited/'output/edited.dcp')==
            (out/'alias_reference_sha256.txt').read_text().strip(),'Wrong alias reference checkpoint')
    expected={prefix for stack in range(2) for signal in ('PREADY','PSLVERR') for prefix in (
        f'hbm_binding/memory/inst/APB_{stack}_{signal}',
        f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/APB_{stack}_{signal}',
        f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_apb_arbiter_{stack}/extrnl_apb_{signal.lower()}')}
    before=table(out/'reference_aliases.tsv');after=table(out/'current_aliases.tsv')
    require(len(before)==len(after)==12 and {r['net'] for r in before}=={r['net'] for r in after}==expected,
            'Alias probe inventory differs')
    require(all(r['exists']=='1' and not r['leaf_inputs'] and not r['top_ports'] for r in before),
            'Removed APB net has consumers or a top port')
    require(all(r['exists']=='0' and not any(r[k] for k in ('leaf_inputs','leaf_outputs','top_ports')) for r in after),
            'APB alias pruning not reproduced')
    rows=table(out/'reference_calibration_selectors.tsv')
    require(rows==table(out/'current_calibration_selectors.tsv'),'Calibration selector membership changed')
    groups={}
    for r in rows:
        key=int(r['lane']),r['family'];members=groups.setdefault(key,set())
        require(r['cell'] not in members,'Duplicate calibration selector member');members.add(r['cell'])
    require(set(groups)=={(i,f) for i in range(16) for f in ('testclk_cnt_reg','freq_cnt_o_reg')},
            'Incomplete calibration selector probe')
    return expected,groups


def physical_review(root,reference,donor,edited,mode,probe=None):
    require(mode in ('route','reopen'),'Unsupported physical mode')
    binding(root,reference,donor,mode)
    out=root/'output';edit=edited/'output'
    removed_aliases,groups=semantic_probe(probe,reference,out/'routed.dcp' if mode=='route' else donor,edited)
    log=(root/'tool.log').read_text()
    require(not re.search(r'^ERROR:',log,re.M),'Native error in physical run')
    reuse={}
    if mode=='route':
        source=(root/'source/integrate_hbm_reset_distribution.tcl').read_text()
        require('place_design -eco' not in source and 'route_design -eco' not in source,
                'Versal-only flow on UltraScale+')
        require('read_checkpoint -incremental $checkpoint' in source and
                'place_design -no_psip' in source and 'route_design -no_psir' in source,
                'Unsupported integration flow')
        require('Starting Incremental Placer Task' in log,'Native incremental flow not entered')
        for stage in ('before','placed','routed'):
            reuse[stage]=incremental_reuse(out/('reuse_'+stage+'.rpt'))
            require(reuse[stage]['Cells']['reused_percent']>=75,'Unexpected placement reuse loss')
    require(semantic_timing(out/'before.xdc',groups)==semantic_timing(edit/'remap_effective.xdc',groups),
            'Physical donor timing differs from mapped edit')
    result=integration_contracts(root,reference/'cdc',donor,'routed' if mode=='route' else 'reopened',groups)
    require(semantic_timing(out/'before.xdc',groups)==semantic_timing(out/'reports/effective.xdc',groups),
            'Physical run changed timing constraints')
    require(lines(out/'cells.txt')==lines(edit/'cells.txt'),'Physical run changed edited cell inventory')
    old_nets=set(lines(edit/'nets.txt'));new_nets=set(lines(out/'nets.txt'))
    require(old_nets-new_nets==removed_aliases and not new_nets-old_nets,'Unproven net inventory change')
    topology=reset_topology(out,lines(edit/'reset_consumers.txt'))
    require(topology['distribution_location'] and topology['distribution_bel'],'Distribution register unplaced')
    paths=table(out/'distribution_paths.tsv')
    require(len(paths)==2 and {p['type'] for p in paths}=={'max','min'},'Missing distribution recovery/removal checks')
    for row in paths:
        require(row['from']==SYNC+'/C' and row['to']==DIST+'/CLR' and
                row['clock']==row['end_clock']=='clk_out2_fabric_clocks','Distribution timing endpoints differ')
        require(abs(float(row['requirement_ns'])-(2.222 if row['type']=='max' else 0.0))<0.001,
                'Distribution recovery/removal requirement differs')
    preserved_pass=result['bus_skew']['failing']==0 and all(x['failing']==0 for x in result['scoped_timing'].values())
    distribution_pass=all(float(p['slack_ns'])>=0 for p in paths)
    result.update(status='HBM_RESET_BOARD_PHYSICAL_REVIEWED',mode=mode,
                  incremental_reuse=reuse,
                  semantic_reconciliation=dict(pruned_unloaded_apb_aliases=sorted(removed_aliases),
                      calibration_constraints=16,calibration_selector_members=sum(map(len,groups.values())),
                      native_probe_job=probe.parent.name,reference_sha256=sha(edit/'edited.dcp')),
                  topology=topology,distribution_paths=paths,distribution_recovery_removal_pass=distribution_pass,
                  preserved_timing_pass=preserved_pass,timing=complete_timing(out/'reports/timing.rpt'),
                  before_timing=complete_timing(out/'timing_before.rpt'),
                  original_routed_timing=complete_timing(reference/'routed-reopened/reports/timing.rpt'),
                  constraint_source_remapped=True,netlist_eco_integrated=True,
                  full_rtl_resynthesis_executed=False,native_traffic_qualified=False,timing_qualified=False)
    return result




def edit_review(root,reference,donor):
    result=binding(root,reference,donor,'edit');out=root/'output'
    require((out/'STATUS.txt').read_text().strip()=='HBM_RESET_BOARD_EDIT_EXPORTED','Incomplete edit')
    before=expand_xdc((out/'before.xdc').read_text())
    require(before==expand_xdc((reference/'routed-reopened/reports/effective.xdc').read_text()),'Donor constraints changed')
    require(before[OLD]==1 and not before[NEW],'Unexpected original readiness constraint')
    expected=before.copy();expected.subtract({OLD:1});expected+=Counter({NEW:1})
    remapped=expand_xdc((out/'remapped.xdc').read_text())
    require(expected==remapped,'More than one exported constraint changed')
    scoped_expected=semantic_xdc((out/'remapped.xdc').read_text())
    after=semantic_xdc((out/'remap_effective.xdc').read_text())
    require(not scoped_expected-after,'Constraint reload removed scoped commands')
    extra=after-scoped_expected
    allowed=Counter({" :: set_property INIT 1'b0 [get_cells "+DIST+"]":1,
                     " :: set_property IS_CLR_INVERTED 1'b1 [get_cells "+DIST+"]":1})
    require(extra==allowed,'Constraint reload added unrelated scoped commands')
    require(normalized_timing(out/'remapped.xdc')==normalized_timing(out/'remap_effective.xdc'),
            'Constraint reload changed timing beyond mapping')
    consumers=lines(out/'original_consumers.txt')
    baseline=sorted(r['pin'] for r in table(reference/'original_reset_consumers.tsv'))
    require(consumers==baseline,'Original reset probe inventory differs')
    topology=reset_topology(out,consumers)
    old_cells=set(lines(out/'original_cells.txt'));new_cells=set(lines(out/'cells.txt'))
    old_nets=set(lines(out/'original_nets.txt'));new_nets=set(lines(out/'nets.txt'))
    require(new_cells-old_cells=={DIST} and not old_cells-new_cells,'Cell edit exceeds one register')
    require(new_nets-old_nets=={'core/reader/startup/memory_sync_release'} and not old_nets-new_nets,
            'Net edit exceeds one reset net')
    require((out/'edited.dcp').stat().st_size>0,'Missing mapped edit checkpoint')
    return dict(status='HBM_RESET_BOARD_EDIT_REVIEWED',checkpoint_sha256=sha(out/'edited.dcp'),
                donor_sha256=result['checkpoint_sha256'],topology=topology,cell_count=len(new_cells),net_count=len(new_nets),
                explicit_constraint_mapping=dict(before=OLD,after=NEW),added_properties=dict(extra),
                timing_qualified=False,native_traffic_qualified=False,board_qualified=False)

def integration_contracts(root,baseline,donor,mode,groups):
    r=closed(root)
    require(r['checkpoint_sha256']==sha(donor),'Checkpoint chain mismatch')
    out=root/'output';reports=out/'reports';contract=out/'contracts'
    require((out/'STATUS.txt').read_text().strip()=='HBM_RESET_BOARD_'+{'routed':'route','reopened':'reopen'}[mode]+'_EXPORTED','Missing stage completion')
    require((contract/'AUDIT_STATUS.txt').read_text().strip()=='BOARD_IMPLEMENTATION_CONTRACTS_EXPORTED_REVIEW_REQUIRED','Missing contract audit')
    if mode=='optimized':
        pin_constraints((out/'before.xdc').read_text(),(out/'pinned.xdc').read_text())
        require(expand_xdc((out/'before.xdc').read_text())==expand_xdc((baseline/'output/effective.xdc').read_text()),'Wrong pre-pin constraints')
        before=expand_xdc((out/'pinned.xdc').read_text())
    else:before=expand_xdc((out/'before.xdc').read_text())
    after=expand_xdc((reports/'effective.xdc').read_text())
    require(semantic_timing(out/'before.xdc',groups)==semantic_timing(reports/'effective.xdc',groups),
            'Implementation introduced timing exceptions or clocks')
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
        require(len(current)==count and stable(current)==mapped_paths(old,name),'Preserved path contract differs: '+name)
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


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['edit','route','reopen'])
    for name in ('root','reference','donor'):p.add_argument(name,type=Path)
    p.add_argument('--edited',type=Path)
    p.add_argument('--probe',type=Path)
    a=p.parse_args()
    result=edit_review(a.root,a.reference,a.donor) if a.mode=='edit' else physical_review(a.root,a.reference,a.donor,a.edited,a.mode,a.probe)
    print(json.dumps(result,indent=2))
