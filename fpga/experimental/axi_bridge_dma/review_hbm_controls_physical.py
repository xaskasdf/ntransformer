"""Review the complete-board control experiment and inherited timing contracts."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import review_hbm_ready_physical as ready
from review_hbm_capture_physical import closed,sha,require,table,queue_proof
from prepare_hbm_controls_physical import board_adapter


def control_proof(out,timed,candidate):
    root=out/'control_predictors'
    require((root/'STATUS.txt').read_text().strip()==('CONTROLS_CANDIDATE_EXPORTED' if candidate else 'CONTROLS_BASELINE_EXPORTED'),'Control audit incomplete')
    rows=table(root/'registers.tsv');props={(int(r['channel']),r['object'],r['property']):r['value'] for r in rows}
    require(len(rows)==len(props) and {k[0] for k in props}=={0,1},'Control property inventory')
    payload=table(root/'payload.tsv');pruned=table(root/'pruned.tsv');interfaces=table(root/'pruned_interfaces.tsv')
    require(len(interfaces)==3*len(pruned),'Pruned ID interface inventory')
    ends=table(root/'endpoints.tsv');owners=table(root/'ownership.tsv');valid=table(root/'input_valid_endpoints.tsv')
    require(len(owners)==18,'Return ownership count')
    dist='core/reader/startup/memory_distribution_reg'
    result=[]
    for ch in range(2):
        adapter=f'core/reader/channels[{ch}].adapter';queue=adapter+'/return_cut'
        crossing=f'core/reader/channels[{ch}].crossing';name=queue+'/valid_reg_reg'
        p=lambda obj,key:props[ch,obj,key]
        cells=[r for r in payload if r['channel']==str(ch)]
        missing=[r for r in pruned if r['channel']==str(ch)]
        expected={f'{queue}/slot{s}_reg[{b}]' for s in range(2) for b in range(519)}
        missing_names={f"{queue}/slot{r['slot']}_reg[{r['bit']}]" for r in missing}
        actual={r['cell'] for r in cells}
        require(len(cells)==len(actual) and len(missing_names)==len(missing) and not actual&missing_names and actual|missing_names==expected,'Return payload/pruning inventory')
        for r in missing:
            require(515<=int(r['bit'])<519 and r['primitive'] in ('GND','VCC','UNCONNECTED') and
                    r['port']==f"{queue}/m_data[{r['bit']}]",'Unproven metadata pruning')
            boundary=[v for v in interfaces if v['channel']==r['channel'] and v['slot']==r['slot'] and v['bit']==r['bit']]
            bit=int(r['bit'])-515
            require(len(boundary)==3 and {v['port'] for v in boundary}=={r['port'],f'{adapter}/s_rid[{bit}]',f'{crossing}/m_axi_rid[{bit}]'},'Pruned ID boundaries')
            for v in boundary:
                if v['primitive']=='UNCONNECTED':require(v['nets']=='0' and not v['driver'],'Pruned ID still connected')
                else:require(v['primitive'] in ('GND','VCC') and int(v['nets'])>0 and v['driver'],'Pruned ID not constant')
                if v['port']==r['port']:require((v['driver'],v['primitive'])==(r['driver'],r['primitive']),'Pruned ID proof mismatch')
        allowed={dist+'/C',queue+'/count_reg[0]/C',queue+'/count_reg[1]/C',queue+'/write_slot_reg/C'}
        for r in cells:
            starts=set(r['startpoints'].split(','))
            require(r['primitive']=='FDRE' and r['clock']=='clk_out2_fabric_clocks' and starts and '' not in starts and starts<=allowed,'Return payload enable/clock')
        owner_rows=[r for r in owners if r['channel']==str(ch)]
        owner_set={queue+'/'+tail for tail in ('count_reg[0]','count_reg[1]','write_slot_reg')}
        require(len(owner_rows)==9 and {(r['cell'],r['pin']) for r in owner_rows}=={(c,pin) for c in owner_set for pin in ('CE','D','R')},'Return ownership pins')
        valid_ends=[r['endpoint'] for r in valid if r['channel']==str(ch)]
        require(valid_ends and len(valid_ends)==len(set(valid_ends)),'Return input VALID fanout')
        require(not any(e.startswith(queue+'/slot') and e.endswith('/CE') for e in valid_ends),'Return input VALID reaches payload enable')
        for r in owner_rows:require(r['valid_dependency']==str(int(r['cell']+'/'+r['pin'] in valid_ends)),'Return ownership fanout mismatch')
        for c in owner_set:require(any(r['cell']==c and r['valid_dependency']=='1' for r in owner_rows),'Return ownership lost VALID')
        driver=p(queue+'/m_valid','DRIVER')
        require(driver==p(adapter+'/s_rvalid','DRIVER')==p(crossing+'/m_axi_rvalid','DRIVER'),'Return/CDC VALID binding')
        d_cells=[adapter+'/tracker/rready_reg_reg']
        if candidate:
            require(driver==name+'/Q' and p(name,'REF_NAME')=='FDCE' and p(name,'INIT')=="1'b0",'Return VALID register')
            require(p(name,'ASYNC_REG') not in ('1','true','TRUE') and p(name,'DONT_TOUCH') not in ('1','true','TRUE'),'Return register attributes')
            require(p(name+'/C','CLOCK')=='clk_out2_fabric_clocks' and p(name+'/CE','DRIVER_PRIMITIVE')=='VCC','Return register clock/enable')
            require(p(name+'/CLR','STARTPOINTS')==dist+'/C','Return reset source')
            clr=p(name+'/CLR','DRIVER');encoding=p(name,'RESET_ENCODING');inverted=p(name+'/CLR','IS_INVERTED')
            if encoding=='PRIMITIVE_INVERTED_CLR':require(clr==dist+'/Q' and inverted=='1','Return direct reset polarity')
            else:
                inv=clr.removesuffix('/O')
                require(encoding=='EXPLICIT_INVERTER' and inverted=='0' and p(inv,'REF_NAME')=='LUT1' and
                        p(inv,'INIT')=="2'h1" and p(inv+'/I0','DRIVER')==dist+'/Q','Return explicit reset polarity')
            d_cells.append(name)
        else:require(p(driver.removesuffix('/O'),'REF_NAME').startswith('LUT'),'Return baseline gate')
        dependencies={}
        for cell in d_cells:
            starts=p(cell+'/D','STARTPOINTS').split(',')
            require(starts and '' not in starts and len(starts)==len(set(starts)),'Control D fan-in inventory')
            observed=dist+'/C' in starts
            require(p(cell+'/D','DISTRIBUTION_DEPENDENCY')==str(int(observed)),'D reset dependency differs')
            dependencies[cell]=observed
        endpoints=[r for r in ends if r['channel']==str(ch)]
        require(endpoints and len(endpoints)==len({r['endpoint'] for r in endpoints}),'Return output fanout inventory')
        we=set()
        for r in endpoints:
            is_we=r['endpoint'].startswith(crossing+'/') and r['primitive'] in ('RAMD32','RAMS32') and r['pin']=='WE'
            require(r['write_enable']==str(int(is_we)),'CDC WE classification')
            if is_we:we.add(r['endpoint'])
        require(we,'CDC write enables missing')
        item=dict(channel=ch,driver=driver,payload_registers=len(cells),pruned_id_registers=len(missing),
                  d_reset_dependencies=dependencies,cdc_write_enables=len(we))
        if timed:
            timing={}
            for kind in ('max','min'):
                paths=table(root/f'ch{ch}_cdc_we_{kind}.tsv')
                require(len(paths)==len(we) and {r['ENDPOINT_PIN'] for r in paths}==we,'CDC WE timing inventory')
                categories=['return_output']+(['return_input','return_reset'] if candidate else [])
                samples={}
                for category in categories:
                    rows=table(root/f'ch{ch}_{category}_{kind}.tsv');require(len(rows)==1,'Return control timing inventory')
                    r=rows[0];samples[category]=r
                    if category=='return_output':
                        require(r['ENDPOINT_PIN'] in we,'Return output endpoint')
                        if candidate:require(r['STARTPOINT_PIN']==name+'/C','Return output register source')
                    elif category=='return_input':require(r['ENDPOINT_PIN']==name+'/D','Return D endpoint')
                    else:require(r['ENDPOINT_PIN']==name+'/CLR' and r['STARTPOINT_PIN']==dist+'/C','Return clear timing endpoints')
                for r in paths+list(samples.values()):
                    require(r['DELAY_TYPE']==kind and r['STARTPOINT_CLOCK']==r['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks' and
                            not r['EXCEPTION'] and abs(float(r['REQUIREMENT'])-(2.222 if kind=='max' else 0))<0.002,'Return timing clock/exception/requirement')
                timing[kind]=dict(write_enables=dict(paths=len(paths),minimum_slack_ns=min(float(r['SLACK']) for r in paths),
                                 failing=sum(float(r['SLACK'])<0 for r in paths),worst=paths[0]),samples=samples)
            item['timing']=timing
        result.append(item)
    return dict(channels=result,candidate=candidate,
                all_predictor_d_reset_free=all(not v for ch in result for v in ch['d_reset_dependencies'].values()),
                physical_timing_qualified=False)


def synthesis_review(root,reference,original=None):
    native=closed(root);out=root/'output';source=root/'source'
    require(native['sources_unchanged'],'Synthesis source changed')
    if original is None:
        require(native['stage']=='synth','Synthesis status')
        require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Incomplete synthesis')
        checkpoint=out/'synth.dcp'
    else:
        checkpoint=original/'output/synth.dcp';source=original/'source'
        require(native['stage']=='inspect' and native['checkpoint_sha256']==sha(checkpoint),'Recovery input checkpoint differs')
        require((out/'STATUS.txt').read_text().strip()=='CONTROLS_SYNTHESIS_RECOVERY_EXPORTED','Incomplete recovery')
        record=json.loads((root/'source/CONTROLS_SYNTH_RECOVERY.json').read_text())
        require(record['read_only'] and record['checkpoint_sha256']==sha(checkpoint),'Recovery binding differs')
        for n,h in record['prepared_files'].items():require(sha(root/'source'/n)==h,'Recovery source changed')
        for n,h in record['original_source_files'].items():require(sha(source/n)==h,'Original synthesis source changed')
        old=json.loads((original/'RESULT.json').read_text())
        require(old['stage']=='synth' and old['sources_unchanged'] and not old['timed_out'],'Unexpected original synthesis state')
        if old['status']=='FAILED':
            require(old['returncode']==1 and "rp_driver $port" in (original/'tool.log').read_text() and
                    "Expected one driver, got 0" in (original/'tool.log').read_text(),'Original failure differs')
        else:require(old['returncode']==0 and (original/'output/STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Original synthesis incomplete')
    binding=json.loads((source/'CONTROLS_PHYSICAL_BINDING.json').read_text())
    for n,h in binding['donor_files'].items():require(sha(reference/'rtl-donor'/n)==h,'Frozen donor changed')
    for n,h in binding['prepared_files'].items():require(sha(source/n)==h,'Prepared source changed')
    require(binding['changed_original_files']==['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv'],'Collateral source change')
    require(binding['tested_unit_binding_sha256']==sha(reference/'unit/PREPARED.json'),'Unit binding differs')
    require((source/'ntrpu_hbm_read_adapter.sv').read_text()==board_adapter(reference/'rtl-donor/ntrpu_hbm_read_adapter.sv',reference/'unit'),'Untested full-board RTL')
    ports=table(out/'ports.tsv');require(len(ports)==len({r['port'] for r in ports})==75,'Physical port inventory')
    require({r['cell'] for r in table(out/'native_instances.tsv')}=={'ntrpu_bridge_i','hbm_binding/memory','clock_generator','core'},'Native composition')
    require('No waivers found.' in (out/'reference_waivers.rpt').read_text(),'Synthesis waivers')
    return dict(status='CONTROLS_FULL_SYNTHESIS_REVIEWED',seconds=native['seconds'],checkpoint_sha256=sha(checkpoint),recovered_read_only=original is not None,
                queue=queue_proof(out,False),ready=ready.ready_proof(out,False),controls=control_proof(out,False,True),
                full_board_integrated=True,physical_timing_qualified=False)


def review(root,reference,donor,mode):
    result=ready.review(root,reference,donor,mode)
    binding=json.loads((root/'source/CAPTURE_STAGE_BINDING.json').read_text())
    candidate=binding['control_predictors']
    require(candidate!=binding['baseline_read_only'] and (candidate or mode=='reopen'),'Wrong control stage binding')
    result['status']='CONTROLS_FULL_BOARD_REVIEWED';result['baseline_read_only']=not candidate
    result['full_rtl_resynthesis']=candidate;result['controls']=control_proof(root/'output',mode!='constrain',candidate)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reference'):p.add_argument(key,type=Path)
    p.add_argument('--donor',type=Path);p.add_argument('--mode',choices=['synth','constrain','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(synthesis_review(a.root,a.reference) if a.mode=='synth' else review(a.root,a.reference,a.donor,a.mode),indent=2))
