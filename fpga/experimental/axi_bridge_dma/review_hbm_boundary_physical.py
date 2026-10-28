"""Review the tested boundary in a complete board, retaining inherited contracts."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import review_hbm_ready_physical as ready
from review_hbm_controls_physical import control_proof
from review_hbm_capture_physical import closed,sha,require,table,queue_proof
from prepare_hbm_boundary_physical import board_adapter


def boundary_proof(out,timed,candidate):
    root=out/'control_boundary'
    require((root/'STATUS.txt').read_text().strip()==('BOUNDARY_CANDIDATE_EXPORTED' if candidate else 'BOUNDARY_BASELINE_EXPORTED'),'Boundary audit incomplete')
    rows=table(root/'registers.tsv');props={(int(r['channel']),r['object'],r['property']):r['value'] for r in rows}
    require(len(rows)==len(props) and {k[0] for k in props}=={0,1},'Boundary property inventory')
    payload=table(root/'payload.tsv');pruned=table(root/'pruned.tsv');owners=table(root/'ownership.tsv');valid=table(root/'valid_endpoints.tsv')
    dist='core/reader/startup/memory_distribution_reg';result=[]
    for ch in range(2):
        adapter=f'core/reader/channels[{ch}].adapter';tracker=adapter+'/tracker';queue=tracker+'/validated_cut'
        name=queue+'/valid_reg_reg';rready=tracker+'/rready_reg_reg';p=lambda obj,key:props[ch,obj,key]
        ends=[r['endpoint'] for r in valid if r['channel']==str(ch)]
        require(ends and len(ends)==len(set(ends)),'Validated input fanout inventory')
        own=[r for r in owners if r['channel']==str(ch)]
        owner_set={queue+'/'+tail for tail in ['count_reg[0]','count_reg[1]','write_slot_reg']}
        require(len(own)==9 and {(r['cell'],r['pin']) for r in own}=={(c,k) for c in owner_set for k in ['CE','D','R']},'Validated ownership inventory')
        for r in own:require(r['valid_dependency']==str(int(r['cell']+'/'+r['pin'] in ends)),'Validated ownership fanout differs')
        for c in owner_set:require(any(r['cell']==c and r['valid_dependency']=='1' for r in own),'Validated ownership lost VALID')
        cells=[r for r in payload if r['channel']==str(ch)]
        absent=[r for r in pruned if r['channel']==str(ch)]
        missing={(int(r['slot']),int(r['bit'])) for r in absent}
        expected={f'{queue}/slot{s}_reg[{b}]' for s in range(2) for b in range(265)}
        actual={r['cell'] for r in cells};missing_names={f'{queue}/slot{s}_reg[{b}]' for s,b in missing}
        require(len(actual)==len(cells) and not actual&missing_names and actual|missing_names==expected,'Validated payload/pruning inventory')
        for s,b in missing:
            require(s in (0,1) and 259<=b<265,'Unproven validated data pruning')
            boundary=[r for r in absent if int(r['slot'])==s and int(r['bit'])==b]
            pins={f'{queue}/m_data[{b}]',f'{tracker}/validated_rid[{b-259}]'}
            if b<263:pins.add(f'{adapter}/protocol_converter/m_axi_rid[{b-259}]')
            require(len(boundary)==len(pins) and {r['port'] for r in boundary}==pins,'Pruned validated ID boundaries')
            for r in boundary:
                if r['primitive']=='UNCONNECTED':require(r['nets']=='0' and not r['driver'],'Pruned ID is still connected')
                else:require(r['primitive'] in ('GND','VCC') and int(r['nets'])>0 and r['driver'],'Pruned ID remains dynamic')
        for r in cells:
            starts=r['startpoints'].split(',')
            require(r['primitive']=='FDRE' and r['clock']=='clk_out2_fabric_clocks' and starts and '' not in starts,'Validated storage primitive/clock')
            require(r['cell']+'/CE' in ends,'Validated payload write lost input VALID')
        driver=p(queue+'/m_valid','DRIVER')
        require(driver==p(tracker+'/p_rvalid','DRIVER')==p(adapter+'/protocol_converter/m_axi_rvalid','DRIVER'),'Validated/protocol binding')
        d_cells=[rready]
        if candidate:
            require(driver==name+'/Q' and p(name,'REF_NAME')=='FDCE' and p(name,'INIT')=="1'b0",'Validated register primitive')
            require(p(name,'ASYNC_REG') not in ('1','true','TRUE') and p(name,'DONT_TOUCH') not in ('1','true','TRUE'),'Validated register attributes')
            require(p(name+'/C','CLOCK')=='clk_out2_fabric_clocks' and p(name+'/CE','DRIVER_PRIMITIVE')=='VCC','Validated clock or CE')
            require(p(name+'/CLR','STARTPOINTS')==dist+'/C','Validated reset source')
            clr=p(name+'/CLR','DRIVER');encoding=p(name,'RESET_ENCODING');inv=p(name+'/CLR','IS_INVERTED')
            if encoding=='PRIMITIVE_INVERTED_CLR':require(clr==dist+'/Q' and inv=='1','Validated direct reset polarity')
            else:
                cell=clr.removesuffix('/O')
                require(encoding=='EXPLICIT_INVERTER' and inv=='0' and p(cell,'REF_NAME')=='LUT1' and p(cell,'INIT')=="2'h1" and p(cell+'/I0','DRIVER')==dist+'/Q','Validated explicit reset polarity')
            d_cells.append(name)
        else:require(p(driver.removesuffix('/O'),'REF_NAME').startswith('LUT'),'Baseline validated gate')
        dependencies={}
        for cell in d_cells:
            starts=p(cell+'/D','STARTPOINTS').split(',')
            require(starts and '' not in starts and len(starts)==len(set(starts)),'Boundary D startpoint inventory')
            observed=dist+'/C' in starts
            require(p(cell+'/D','DISTRIBUTION_DEPENDENCY')==str(int(observed)),'Boundary reset dependency differs')
            dependencies[cell]=observed
        lengths=sorted(x for x in p(rready+'/D','STARTPOINTS').split(',') if x.startswith(tracker+'/h_arlen_reg'))
        require(p(rready+'/D','ARLEN_STARTPOINTS')==','.join(lengths),'Credit length inventory')
        require(not lengths if candidate else bool(lengths),'READY arithmetic dependency')
        item=dict(channel=ch,driver=driver,payload_registers=len(cells),pruned_id_registers=len(missing),
                  ready_length_startpoints=lengths,d_reset_dependencies=dependencies)
        if timed:
            timing={}
            for kind in ['max','min']:
                for category in ['output']+(['input','reset'] if candidate else []):
                    rows=table(root/f'ch{ch}_{category}_{kind}.tsv');require(len(rows)==1,'Validated timing inventory')
                    r=rows[0]
                    target=adapter+'/return_cut/valid_reg_reg/D' if category=='output' else name+('/D' if category=='input' else '/CLR')
                    require(r['ENDPOINT_PIN']==target and r['DELAY_TYPE']==kind,'Validated timing endpoint/type')
                    require(r['STARTPOINT_CLOCK']==r['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks' and not r['EXCEPTION'],'Validated timing clock/exception')
                    require(abs(float(r['REQUIREMENT'])-(2.222 if kind=='max' else 0))<0.002,'Validated timing requirement')
                    if category=='reset':require(r['STARTPOINT_PIN']==dist+'/C','Validated reset timing origin')
                    elif category=='output' and candidate:require(r['STARTPOINT_PIN']==name+'/C','Validated output timing origin')
                    timing[category+'_'+kind]=r
            item['timing']=timing
        result.append(item)
    return dict(candidate=candidate,channels=result,all_predictor_d_reset_free=all(not v for ch in result for v in ch['d_reset_dependencies'].values()),physical_timing_qualified=False)


def synthesis_review(root,reference):
    native=closed(root);out=root/'output';source=root/'source'
    require(native['stage']=='synth' and native['sources_unchanged'],'Synthesis status')
    require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Incomplete synthesis')
    binding=json.loads((source/'BOUNDARY_PHYSICAL_BINDING.json').read_text())
    require(binding['status']=='BOUNDARY_FULL_BOARD_PREPARED','Wrong synthesis binding')
    for n,h in binding['donor_files'].items():require(sha(reference/'rtl-donor'/n)==h,'Frozen donor changed')
    for n,h in binding['prepared_files'].items():require(sha(source/n)==h,'Prepared synthesis source changed')
    require(binding['changed_original_files']==['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv'],'Collateral source change')
    require(binding['tested_unit_binding_sha256']==sha(reference/'unit/PREPARED.json'),'Unit binding differs')
    require((source/'ntrpu_hbm_read_adapter.sv').read_text()==board_adapter(reference/'rtl-donor/ntrpu_hbm_read_adapter.sv',reference/'unit'),'Untested full-board RTL')
    ports=table(out/'ports.tsv');require(len(ports)==len({r['port'] for r in ports})==75,'Physical port inventory')
    require({r['cell'] for r in table(out/'native_instances.tsv')}=={'ntrpu_bridge_i','hbm_binding/memory','clock_generator','core'},'Native composition')
    require('No waivers found.' in (out/'reference_waivers.rpt').read_text(),'Synthesis waivers')
    return dict(status='BOUNDARY_FULL_SYNTHESIS_REVIEWED',seconds=native['seconds'],checkpoint_sha256=sha(out/'synth.dcp'),
                queue=queue_proof(out,False),ready=ready.ready_proof(out,False),controls=control_proof(out,False,True),
                boundary=boundary_proof(out,False,True),full_board_integrated=True,physical_timing_qualified=False)


def review(root,reference,donor,mode):
    result=ready.review(root,reference,donor,mode)
    binding=json.loads((root/'source/CAPTURE_STAGE_BINDING.json').read_text());candidate=binding['control_boundary']
    require(binding['control_predictors'] is True and candidate!=binding['baseline_read_only'] and (candidate or mode=='reopen'),'Wrong boundary stage binding')
    result.update(status='BOUNDARY_FULL_BOARD_REVIEWED',baseline_read_only=not candidate,full_rtl_resynthesis=candidate,
                  controls=control_proof(root/'output',mode!='constrain',True),boundary=boundary_proof(root/'output',mode!='constrain',candidate))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','reference']:p.add_argument(key,type=Path)
    p.add_argument('--donor',type=Path);p.add_argument('--mode',choices=['synth','constrain','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(synthesis_review(a.root,a.reference) if a.mode=='synth' else review(a.root,a.reference,a.donor,a.mode),indent=2))
