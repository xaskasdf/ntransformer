"""Review full-board BREADY integration and all inherited board contracts."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import review_hbm_boundary_physical as boundary
from review_hbm_capture_physical import closed,sha,require,table,queue_proof
from review_hbm_ready_physical import ready_proof
from review_hbm_controls_physical import control_proof
from prepare_hbm_bready_physical import board_loader


def bready_proof(out,timed):
    root=out/'registered_bready'
    require((root/'STATUS.txt').read_text().strip()=='BREADY_BOTH_CHANNELS_EXPORTED','Incomplete BREADY proof')
    rows=table(root/'registers.tsv');props={(int(r['channel']),r['object'],r['property']):r['value'] for r in rows}
    require(len(rows)==len(props) and {k[0] for k in props}=={0,1},'BREADY property inventory')
    dist='core/reader/startup/memory_distribution_reg';channels=[]
    for ch in range(2):
        name=f'weight_loader/bready_reg_reg[{ch}]';p=lambda obj,key:props[ch,obj,key]
        require(p(name,'REF_NAME')=='FDCE' and p(name,'INIT')=="1'b0",'BREADY register primitive')
        require(p(name,'ASYNC_REG') not in ('1','true','TRUE') and p(name,'DONT_TOUCH') not in ('1','true','TRUE'),'BREADY ordinary attributes')
        require(p(name+'/C','CLOCK')=='clk_out2_fabric_clocks' and p(name+'/CE','DRIVER_PRIMITIVE')=='VCC','BREADY clock or enable')
        require(p(f'weight_loader/h_bready[{ch}]','DRIVER')==p(f'hbm_binding/memory/AXI_{ch*16:02d}_BREADY','DRIVER')==name+'/Q','BREADY interface bypass')
        primitive=f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<{ch}>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST'
        endpoint=primitive+'/BREADY';valid=p(f'weight_loader/h_bvalid[{ch}]','DRIVER')
        require(valid==primitive+'/BVALID_PIPE','HBM BVALID binding')
        gate=p(endpoint,'DRIVER').removesuffix('/O');init=p(gate,'INIT')
        require(p(gate,'REF_NAME')=='LUT2' and init.startswith("4'h"),'Native BREADY primitive')
        truth=int(init[3:],16);drivers=[p(gate+'/I0','DRIVER'),p(gate+'/I1','DRIVER')]
        require((drivers==[name+'/Q',valid] and truth==11) or (drivers==[valid,name+'/Q'] and truth==13),'Native BREADY truth table')
        require(set(p(endpoint,'STARTPOINTS').split(','))=={name+'/C',primitive+'/ACLK'},'Native BREADY extra fan-in')
        starts=p(name+'/D','STARTPOINTS').split(',')
        require(starts and '' not in starts and len(starts)==len(set(starts)),'BREADY D fan-in inventory')
        dependency=dist+'/C' in starts
        require(p(name+'/D','DISTRIBUTION_DEPENDENCY')==str(int(dependency)),'BREADY reset dependency inventory')
        require(p(name+'/CLR','STARTPOINTS')==dist+'/C','BREADY clear source')
        driver=p(name+'/CLR','DRIVER');encoding=p(name,'RESET_ENCODING');inv=p(name+'/CLR','IS_INVERTED')
        if encoding=='PRIMITIVE_INVERTED_CLR':require(driver==dist+'/Q' and inv=='1','BREADY primitive reset polarity')
        else:
            cell=driver.removesuffix('/O')
            require(encoding=='EXPLICIT_INVERTER' and inv=='0' and p(cell,'REF_NAME')=='LUT1' and p(cell,'INIT')=="2'h1" and p(cell+'/I0','DRIVER')==dist+'/Q','BREADY reset inverter')
        item=dict(channel=ch,driver=name+'/Q',native_gate=gate,native_function='AXI_BREADY || !AXI_BVALID',
                  d_startpoints=starts,d_reset_dependency=dependency,reset_encoding=encoding)
        if timed:
            item['timing']={}
            for kind in ['max','min']:
                for category in ['input','output','native','reset']:
                    paths=table(root/f'ch{ch}_{category}_{kind}.tsv');require(len(paths)==1,'BREADY timing coverage')
                    r=paths[0];to=name+'/D' if category=='input' else name+'/CLR' if category=='reset' else endpoint
                    require(r['ENDPOINT_PIN']==to and r['DELAY_TYPE']==kind,'BREADY timing endpoint/type')
                    require(r['STARTPOINT_CLOCK']==r['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks' and not r['EXCEPTION'],'BREADY timing clock/exception')
                    require(abs(float(r['REQUIREMENT'])-(2.222 if kind=='max' else 0))<.002,'BREADY timing requirement')
                    if category=='output':require(r['STARTPOINT_PIN']==name+'/C','BREADY output timing origin')
                    elif category=='reset':require(r['STARTPOINT_PIN']==dist+'/C','BREADY reset timing origin')
                    elif category=='native':require(r['STARTPOINT_PIN'] in (name+'/C',primitive+'/ACLK'),'Native BREADY timing origin')
                    else:require(r['STARTPOINT_PIN'] in starts,'BREADY D timing origin')
                    item['timing'][category+'_'+kind]=r
        channels.append(item)
    return dict(channels=channels,all_d_reset_free=not any(c['d_reset_dependency'] for c in channels),physical_timing_qualified=False)


def synthesis_review(root,reference):
    native=closed(root);out=root/'output';source=root/'source'
    require(native['stage']=='synth' and native['sources_unchanged'],'Synthesis status')
    require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_synth_EXPORTED','Incomplete synthesis')
    b=json.loads((source/'BREADY_PHYSICAL_BINDING.json').read_text())
    require(b['status']=='BREADY_FULL_BOARD_PREPARED' and b['changed_original_files']==['elaborate_hbm_native_join.tcl','ntrpu_hbm_weight_loader.sv'],'Synthesis composition')
    for n,h in b['donor_files'].items():require(sha(reference/'rtl-donor'/n)==h,'Frozen board donor changed')
    for n,h in b['prepared_files'].items():require(sha(source/n)==h,'Prepared synthesis source changed')
    require(b['tested_unit_binding_sha256']==sha(reference/'unit/PREPARED.json'),'Unit binding changed')
    require((source/'ntrpu_hbm_weight_loader.sv').read_text()==board_loader(reference/'rtl-donor/ntrpu_hbm_weight_loader.sv',reference/'unit'),'Untested loader delta')
    ports=table(out/'ports.tsv');require(len(ports)==len({r['port'] for r in ports})==75,'Physical port inventory')
    require({r['cell'] for r in table(out/'native_instances.tsv')}=={'ntrpu_bridge_i','hbm_binding/memory','clock_generator','core'},'Native composition')
    require('No waivers found.' in (out/'reference_waivers.rpt').read_text(),'Synthesis waivers')
    return dict(status='BREADY_FULL_SYNTHESIS_REVIEWED',seconds=native['seconds'],checkpoint_sha256=sha(out/'synth.dcp'),
                queue=queue_proof(out,False),ready=ready_proof(out,False),controls=control_proof(out,False,True),
                boundary=boundary.boundary_proof(out,False,True),bready=bready_proof(out,False),
                full_board_integrated=True,physical_timing_qualified=False)


def review(root,reference,donor,mode):
    result=boundary.review(root,reference,donor,mode)
    b=json.loads((root/'source/CAPTURE_STAGE_BINDING.json').read_text())
    require(b['registered_bready'] is True and b['baseline_read_only'] is False,'BREADY stage binding')
    result.update(status='BREADY_FULL_BOARD_REVIEWED',bready=bready_proof(root/'output',mode!='constrain'))
    if mode!='constrain' and b.get('vendor_feedback_probe'):
        result['vendor_feedback']=vendor_proof(root/'output')
    if mode!='constrain':result['bready_all_timing_pass']=all(float(r['SLACK'])>=0 for ch in result['bready']['channels'] for r in ch['timing'].values())
    return result


def vendor_proof(out):
    result=[]
    for ch in range(2):
        primitive=f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<{ch}>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST'
        paths={}
        for kind in ['max','min']:
            rows=table(out/f'registered_bready/ch{ch}_vendor_{kind}.tsv')
            require(len(rows)==1,'Vendor BREADY timing inventory')
            r=rows[0]
            require(r['STARTPOINT_PIN']==primitive+'/ACLK' and r['ENDPOINT_PIN']==primitive+'/BREADY' and r['DELAY_TYPE']==kind,'Vendor BREADY endpoints/type')
            require(r['STARTPOINT_CLOCK']==r['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks' and not r['EXCEPTION'],'Vendor BREADY clock/exception')
            require(abs(float(r['REQUIREMENT'])-(2.222 if kind=='max' else 0))<.002,'Vendor BREADY timing requirement')
            native=table(out/f'registered_bready/ch{ch}_native_{kind}.tsv')
            output=table(out/f'registered_bready/ch{ch}_output_{kind}.tsv')
            require(len(native)==len(output)==1 and float(native[0]['SLACK'])==min(float(output[0]['SLACK']),float(r['SLACK'])),'Native BREADY worst-path consistency')
            paths[kind]=r
        result.append(dict(channel=ch,timing=paths))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','reference']:p.add_argument(n,type=Path)
    p.add_argument('--donor',type=Path);p.add_argument('--mode',choices=['synth','constrain','route','reopen'],required=True)
    a=p.parse_args();print(json.dumps(synthesis_review(a.root,a.reference) if a.mode=='synth' else review(a.root,a.reference,a.donor,a.mode),indent=2))
