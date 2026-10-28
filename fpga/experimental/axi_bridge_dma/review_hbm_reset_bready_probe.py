"""Review physical reset loads and the native write-response READY boundary."""
import collections
import json
from pathlib import Path
import re
from review_hbm_reset_bready import require,sha,table
from review_hbm_capture_physical import complete_timing,route_status


def load_group(pin):
    for prefix in ['weight_loader','hbm_binding','control/snapshot','core/memory_watch',
                   'core/reader/channels[0].adapter','core/reader/channels[1].adapter',
                   'core/reader/channels[0].crossing','core/reader/channels[1].crossing','core/reader/startup']:
        if pin.startswith(prefix+'/'):return prefix
    raise ValueError('Unknown reset consumer '+pin)


def proof(out,reference):
    require((out/'STATUS.txt').read_text().strip()=='HBM_RESET_BREADY_PROBE_EXPORTED','Probe incomplete')
    # write_xdc records its destination in exactly one generated comment.
    # Keep every other byte, including all executable constraints, identical.
    xdc=[]
    for label in ['before','after']:
        text=(out/(label+'.xdc')).read_bytes()
        pattern=rb'(?m)^# Command Used: write_xdc -constraints all ([^\r\n]+/)' + label.encode() + rb'\.xdc(?=\r?$)'
        text,count=re.subn(pattern,rb'# Command Used: write_xdc -constraints all \1DESTINATION.xdc',text)
        require(count==1,'Unexpected write_xdc header')
        xdc.append(text)
    require(xdc[0]==xdc[1],'Probe changed constraints')
    loads=table(out/'loads.tsv');pins=[r['pin'] for r in loads]
    prior=(reference/'distribution_loads.txt').read_text().splitlines()
    require(len(pins)==len(set(pins))==1090 and set(pins)==set(prior),'Reset load inventory changed')
    require(all(r['slr']=='SLR0' and r['location'] and r['cell']==r['pin'].rsplit('/',1)[0] for r in loads),'Reset physical inventory')
    require(collections.Counter(r['clocks'] for r in loads)=={'clk_out2_fabric_clocks':1011,'':78,'clk_out1_fabric_clocks':1},'Reset clock inventory')
    q=table(out/'source.tsv');source={(r['object'],r['property']):r['value'] for r in q}
    require(len(source)==len(q),'Duplicate reset source property')
    dist='core/reader/startup/memory_distribution_reg'
    require(source[dist,'REF_NAME']=='FDCE' and source[dist,'INIT']=="1'b0" and source[dist,'ASYNC_REG'] not in ('1','TRUE','true') and source[dist,'DONT_TOUCH'] not in ('1','TRUE','true'),'Reset source primitive')
    release='core/reader/startup/memory_release_reg[2]/Q'
    require(source[dist+'/CLR','DRIVERS'] in (release,'{'+release+'}') and source[dist,'IS_CLR_INVERTED']=="1'b1",'Canonical reset source or polarity')
    rows=table(out/'bready.tsv');props={}
    for r in rows:
        key=(int(r['channel']),r['object'],r['property'])
        require(key not in props or props[key]==r['value'],'Contradictory boundary properties');props[key]=r['value']
    require({k[0] for k in props}=={0,1},'BREADY channel inventory')
    channels=[]
    old=json.loads((reference/'predecessor_COMPARISON.json').read_text())
    for ch in range(2):
        p=lambda obj,key:props[ch,obj,key]
        port=f'weight_loader/h_bready[{ch}]';driver=p(port,'DRIVER');cell=driver.removesuffix('/O')
        axi=f'hbm_binding/memory/AXI_{ch*16:02d}_BREADY'
        hbm=f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<{ch}>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST'
        require(p(axi,'DRIVER')==driver and p(cell,'REF_NAME')=='LUT4','Loader BREADY binding')
        require(p(cell,'INIT')==( "16'h0004" if ch==0 else "16'h0008"),'Loader BREADY function')
        drivers=['weight_loader/lane_reg/Q',dist+'/Q','weight_loader/FSM_sequential_ms_reg[0]/Q','weight_loader/FSM_sequential_ms_reg[1]/Q']
        require([p(cell+'/I'+str(i),'DRIVER') for i in range(4)]==drivers,'Loader BREADY inputs')
        starts={x.removesuffix('/Q')+'/C' for x in drivers}
        require(set(p(port,'STARTPOINTS').split(','))==set(p(axi,'STARTPOINTS').split(','))==starts,'Loader BREADY fan-in')
        gate=p(hbm+'/BREADY','DRIVER').removesuffix('/O')
        require(p(gate,'REF_NAME')=='LUT2' and p(gate,'INIT')=="4'hB" and p(gate+'/I0','DRIVER')==driver and p(gate+'/I1','DRIVER')==hbm+'/BVALID_PIPE','Native BREADY gate function')
        require(set(p(hbm+'/BREADY','STARTPOINTS').split(','))==starts|{hbm+'/ACLK'},'Native BREADY fan-in')
        timing={}
        for kind in ['max','min']:
            for label in ['bready','validated_clear']:
                paths=table(out/f'ch{ch}_{label}_{kind}.tsv');require(len(paths)==1,'Probe timing inventory')
                row=paths[0];target=hbm+'/BREADY' if label=='bready' else f'core/reader/channels[{ch}].adapter/tracker/validated_cut/valid_reg_reg/CLR'
                require(row['ENDPOINT_PIN']==target and row['DELAY_TYPE']==kind and row['STARTPOINT_CLOCK']==row['ENDPOINT_CLOCK']=='clk_out2_fabric_clocks' and not row['EXCEPTION'],'Probe timing endpoints/clock/exception')
                require(abs(float(row['REQUIREMENT'])-(2.222 if kind=='max' else 0))<.002,'Probe timing requirement')
                if label=='validated_clear':
                    require(row['STARTPOINT_PIN']==dist+'/C' and float(row['SLACK'])==float(old['boundary']['channels'][ch]['timing']['reset_'+kind]['SLACK']),'Validated recovery/removal changed')
                else:require(row['STARTPOINT_PIN'] in starts|{hbm+'/ACLK'},'Unexpected BREADY timing origin')
                timing[label+'_'+kind]=row
        channels.append(dict(channel=ch,gate=gate,native_function='AXI_BREADY || !AXI_BVALID',timing=timing))
    timing=complete_timing(out/'timing.rpt');require(timing==old['candidate_reopened'],'Read-only probe timing changed')
    route=route_status(out/'route_status.rpt',True);require(route['complete'],'Probe route incomplete')
    paths=table(out/'worst_paths.tsv');require(len(paths)==512 and all(r['from']==dist+'/C' for r in paths),'Reset timing sample coverage')
    return dict(loads=len(loads),load_groups=dict(collections.Counter(load_group(r['pin']) for r in loads)),
                slrs=['SLR0'],timing=timing,channels=channels,route=route,reset_path_samples=len(paths),
                failing_in_reset_sample=sum(float(r['slack_ns'])<0 for r in paths),reset_location=source[dist,'LOC'])


def review(root,reference):
    r=json.loads((root/'RESULT.json').read_text());source=root/'source'
    require(r['status']=='HBM_NATIVE_JOIN_EXPORTED' and r['stage']=='inspect' and r['returncode']==0 and not r['timed_out'] and r['sources_unchanged'],'Native probe failed')
    require(r['checkpoint_sha256']==sha(reference/'routed_donor.dcp'),'Probe checkpoint changed')
    binding=json.loads((source/'PREPARED.json').read_text())
    require(binding['status']=='RESET_BREADY_PROBE_PREPARED' and binding['read_only'] and binding['checkpoint_sha256']==r['checkpoint_sha256'],'Probe binding')
    for n,h in r['sources'].items():require(sha(source/n)==h,'Native probe source changed')
    for n,h in binding['files'].items():require(sha(source/n)==h,'Prepared probe changed')
    return dict(status='RESET_BREADY_PROBE_REVIEWED',seconds=r['seconds'],checkpoint_sha256=r['checkpoint_sha256'],
                read_only=True,physical_timing_qualified=False,**proof(root/'output',reference))
