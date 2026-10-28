"""Review registered-ready integration plus the unchanged board contracts."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import review_hbm_capture_physical as capture
from review_hbm_capture_physical import closed,sha,require,table
from prepare_hbm_ready_physical import board_adapter


def ready_proof(out,timed):
    root=out/'registered_ready'
    require((root/'STATUS.txt').read_text().strip()=='READY_BOTH_CHANNELS_EXPORTED','Incomplete READY proof')
    rows=table(root/'registers.tsv')
    props={(int(r['channel']),r['object'],r['property']):r['value'] for r in rows}
    require(len(props)==len(rows),'Duplicate READY properties')
    require({key[0] for key in props}=={0,1},'READY channel inventory')
    owners=table(root/'ownership.tsv');valid=table(out/'response_capture/rvalid_endpoints.tsv')
    require(len(owners)==18,'READY ownership inventory')
    dist='core/reader/startup/memory_distribution_reg'
    report=[]
    for ch in range(2):
        tracker=f'core/reader/channels[{ch}].adapter/tracker';name=tracker+'/rready_reg_reg'
        endpoint=f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<{ch}>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST/RREADY'
        p=lambda obj,key:props[ch,obj,key]
        require(p(name,'REF_NAME')=='FDCE' and p(name,'INIT')=="1'b0",'READY primitive')
        require(p(name,'ASYNC_REG') not in ('1','true','TRUE') and p(name,'DONT_TOUCH') not in ('1','true','TRUE'),'READY ordinary register')
        require(p(name+'/C','CLOCK')=='clk_out2_fabric_clocks','READY clock')
        require(p(name+'/CE','DRIVER_PRIMITIVE')=='VCC','READY enable')
        axi=f'hbm_binding/memory/AXI_{ch*16:02d}_RREADY'
        require(p(tracker+'/h_rready','DRIVER')==p(axi,'DRIVER')==name+'/Q','READY output bypass')
        primitive=endpoint.rsplit('/',1)[0];valid_source=p(tracker+'/h_rvalid','DRIVER')
        require(valid_source==primitive+'/RVALID_PIPE','HBM VALID source')
        gate=p(endpoint,'DRIVER').removesuffix('/O')
        require(p(gate,'REF_NAME')=='LUT2','Native READY gate primitive')
        init=p(gate,'INIT');require(init.startswith("4'h"),'Native READY gate INIT')
        truth=int(init[3:],16);drivers=[p(gate+'/I0','DRIVER'),p(gate+'/I1','DRIVER')]
        require((drivers==[name+'/Q',valid_source] and truth==11) or
                (drivers==[valid_source,name+'/Q'] and truth==13),'Native READY gate function/inputs')
        require(set(p(endpoint,'STARTPOINTS').split(','))=={name+'/C',primitive+'/ACLK'},'Native READY extra fan-in')
        require(p(name+'/CLR','STARTPOINTS')==dist+'/C','READY reset bypass')
        driver=p(name+'/CLR','DRIVER');encoding=p(name,'RESET_ENCODING')
        if encoding=='PRIMITIVE_INVERTED_CLR':
            require(driver==dist+'/Q' and p(name+'/CLR','IS_INVERTED')=='1','READY direct reset polarity')
        else:
            inv=driver.removesuffix('/O')
            require(encoding=='EXPLICIT_INVERTER' and p(inv,'REF_NAME')=='LUT1' and p(inv,'INIT')=="2'h1"
                    and p(inv+'/I0','DRIVER')==dist+'/Q' and p(name+'/CLR','IS_INVERTED')=='0','READY reset inverter')
        channel=[r for r in owners if r['channel']==str(ch)]
        expected={(tracker+'/response_cut/'+tail,pin) for tail in ('count_reg[0]','count_reg[1]','write_slot_reg') for pin in ('CE','D','R')}
        require(len(channel)==9 and {(r['cell'],r['pin']) for r in channel}==expected,'READY owner pin inventory')
        ends={r['endpoint'] for r in valid if r['channel']==str(ch)}
        for r in channel:require(r['rvalid_dependency']==str(int(r['cell']+'/'+r['pin'] in ends)),'Ownership fanout mismatch')
        for cell in {r['cell'] for r in channel}:
            require(any(r['rvalid_dependency']=='1' for r in channel if r['cell']==cell),'Ownership lost RVALID')
        item=dict(channel=ch,register=name,driver=name+'/Q',encoding=encoding,
                  native_gate=gate,native_function='AXI_READY || !AXI_VALID')
        if timed:
            timing={}
            for category in ('input','output','reset'):
                for kind in ('max','min'):
                    paths=table(root/f'ch{ch}_{category}_{kind}.tsv')
                    require(len(paths)==1,'READY timing coverage')
                    row=paths[0];to={'input':name+'/D','output':endpoint,'reset':name+'/CLR'}[category]
                    require(row['ENDPOINT_PIN']==to and row['DELAY_TYPE']==kind,'READY timing endpoint/type')
                    require(row['STARTPOINT_CLOCK']==row['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks'
                            and not row['EXCEPTION'],'READY timing clock or exception')
                    if category!='input':require(row['STARTPOINT_PIN']==(name if category=='output' else dist)+'/C','READY timing source')
                    require(abs(float(row['REQUIREMENT'])-(2.222 if kind=='max' else 0))<0.002,'READY timing requirement')
                    timing[category+'_'+kind]=row
            item['timing']=timing
        report.append(item)
    return dict(channels=report,ordinary_registers=2,physical_timing_qualified=False)


def synthesis_review(root,reference,original=None):
    native=closed(root);out=root/'output'
    require(native['sources_unchanged'],'Synthesis sources changed')
    if original is None:
        require(native['stage']=='synth','Synthesis status')
        require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Synthesis incomplete')
        source=root/'source';checkpoint=out/'synth.dcp'
    else:
        checkpoint=original/'output/synth.dcp';source=original/'source'
        require(native['stage']=='inspect' and native['checkpoint_sha256']==sha(checkpoint),'Recovery checkpoint changed')
        require((out/'STATUS.txt').read_text().strip()=='READY_SYNTHESIS_RECOVERY_EXPORTED','Incomplete synthesis recovery')
        recovery=json.loads((root/'source/READY_SYNTH_RECOVERY.json').read_text())
        require(recovery['read_only'] and recovery['checkpoint_sha256']==sha(checkpoint),'Recovery source binding')
        for n,h in recovery['prepared_files'].items():require(sha(root/'source'/n)==h,'Recovery source changed')
        prior=json.loads((original/'RESULT.json').read_text())
        require(prior['stage']=='synth' and prior['status']=='FAILED' and prior['sources_unchanged']
                and not prior['timed_out'] and prior['returncode']==1,'Unexpected original synthesis state')
        require('HBM READY binding changed\nHBM READY binding changed' in (original/'tool.log').read_text(),'Original audit failure differs')
        for n,h in prior['sources'].items():require(sha(source/n)==h,'Original synthesis source changed')
        vendor=(reference/'native_hbm_top.sv').read_text()
        for ch in ('00','16'):require(f'assign rready_{ch} = AXI_{ch}_RREADY || (~AXI_{ch}_RVALID);' in vendor,'Vendor READY function differs')
        generated=original/'project/native_join.gen/sources_1/ip/native_hbm/hdl/rtl/hbm_top.sv'
        require(sha(generated)==sha(reference/'native_hbm_top.sv'),'Vendor source changed')
    binding=json.loads((source/'READY_PHYSICAL_BINDING.json').read_text())
    for n,h in binding['donor_files'].items():require(sha(reference/'rtl-donor'/n)==h,'Frozen donor changed')
    for n,h in binding['prepared_files'].items():require(sha(source/n)==h,'Prepared source changed')
    require(binding['changed_original_files']==['elaborate_hbm_native_join.tcl','ntrpu_hbm_read_adapter.sv'],'Collateral source change')
    require(sha(reference/'unit/PREPARED.json')==binding['tested_unit_binding_sha256'],'Unit binding changed')
    require(board_adapter(reference/'rtl-donor/ntrpu_hbm_read_adapter.sv',reference/'unit')==
            (source/'ntrpu_hbm_read_adapter.sv').read_text(),'Untested full-board RTL delta')
    ports=table(out/'ports.tsv');require(len(ports)==75 and len({p['port'] for p in ports})==75,'Physical ports')
    require({r['cell'] for r in table(out/'native_instances.tsv')}=={'ntrpu_bridge_i','hbm_binding/memory','clock_generator','core'},'Native composition')
    require('No waivers found.' in (out/'reference_waivers.rpt').read_text(),'Synthesis waivers')
    return dict(status='READY_FULL_SYNTHESIS_REVIEWED',seconds=native['seconds'],queue=capture.queue_proof(out,False),
                ready=ready_proof(out,False),checkpoint_sha256=sha(checkpoint),
                recovered_read_only=original is not None,physical_timing_qualified=False)


def binding_probe_review(root,checkpoint):
    native=closed(root)
    require(native['checkpoint_sha256']==sha(checkpoint),'Binding probe input changed')
    require((root/'output/STATUS.txt').read_text().strip()=='READY_BINDING_PROBE_EXPORTED','Binding probe incomplete')
    rows=table(root/'output/binding.tsv');props={(r['object'],r['property']):r['value'] for r in rows}
    require(len(rows)==len(props),'Duplicate binding probe')
    for ch in range(2):
        tracker=f'core/reader/channels[{ch}].adapter/tracker'
        require(props[tracker+'/h_rready','DRIVERS']==tracker+'/rready_reg_reg/Q','Tracker READY not registered')
    return dict(status='READY_BINDING_DIAGNOSTIC_REVIEWED',seconds=native['seconds'],checkpoint_sha256=sha(checkpoint),physical_timing_qualified=False)


def review(root,reference,donor,mode):
    result=capture.review(root,reference,donor,mode)
    binding=json.loads((root/'source/CAPTURE_STAGE_BINDING.json').read_text())
    require(binding.get('registered_ready') is True,'Wrong implementation binding')
    result['status']='READY_FULL_BOARD_REVIEWED';result['ready']=ready_proof(root/'output',mode!='constrain')
    if mode!='constrain':
        result['ready_all_timing_pass']=all(float(row['SLACK'])>=0 for ch in result['ready']['channels'] for row in ch['timing'].values())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference'):p.add_argument(name,type=Path)
    p.add_argument('--donor',type=Path);p.add_argument('--mode',choices=['synth','constrain','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(synthesis_review(a.root,a.reference) if a.mode=='synth' else review(a.root,a.reference,a.donor,a.mode),indent=2))
