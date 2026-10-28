"""Independently review the reset probe and isolated RTL/synthesis evidence."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
from review_hbm_routed_timing import complete_timing,expand_xdc,require,sha,route_status


def table(path):
    return list(csv.DictReader(path.read_text().splitlines(),delimiter='\t'))


def verified(root):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},
            'Closed file inventory differs')
    for name,digest in files.items():
        require(sha(root/name)==digest,'Evidence hash differs: '+name)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['sources_unchanged'],'Input changed during job')
    for name,digest in result['sources'].items():
        require(sha(root/'source'/name)==digest,'Frozen source differs: '+name)
    return result


def properties(path):
    rows=table(path)
    result={(r['object'],r['property']):r['value'].removeprefix('{').removesuffix('}') for r in rows}
    require(len(result)==len(rows),'Duplicate property')
    return result


def probe(root,donor):
    result=verified(root);out=root/'output'
    require(result['status']=='HBM_NATIVE_JOIN_EXPORTED' and result['returncode']==0 and not result['timed_out'],
            'Native probe failed')
    require(result['checkpoint_sha256']==sha(donor),'Wrong routed donor')
    require((out/'STATUS.txt').read_text().strip()=='HBM_RESET_FANOUT_EXPORTED','Probe incomplete')
    require(expand_xdc((out/'before.xdc').read_text())==expand_xdc((out/'after.xdc').read_text()),
            'Query-only probe changed constraints')
    prop=properties(out/'source.tsv');source='core/reader/startup/memory_release_reg[2]'
    expected={(source,'REF_NAME'):'FDCE',(source,'ASYNC_REG'):'1',(source,'INIT'):"1'b0",
              (source,'IS_CLR_INVERTED'):"1'b1",(source+'/C','DRIVERS'):'clock_generator/inst/clkout2_buf/O',
              (source+'/D','DRIVERS'):'core/reader/startup/memory_release_reg[1]/Q'}
    require(all(prop.get(k)==v for k,v in expected.items()),'Reset source topology differs')
    rows=table(out/'loads.tsv')
    require(len(rows)==len({r['pin'] for r in rows})==1091,'Incomplete leaf input inventory')
    require({r['slr'] for r in rows}=={'SLR0'},'Unexpected SLR crossing')
    readers=[r for r in rows if r['clocks']=='clk_out1_fabric_clocks']
    require(len(readers)==1 and readers[0]['pin']=='core/reader/startup/ready_meta_reg[2]/D',
            'Ready status CDC endpoint changed')
    clocks=dict(Counter(r['clocks'] or 'combinational' for r in rows))
    require(clocks=={'clk_out2_fabric_clocks':999,'combinational':91,'clk_out1_fabric_clocks':1},
            'Load clock classification differs')
    require(sum(r['reference']=='HBM_SNGLBLI_INTF_AXI' for r in rows)==32,'Missing HBM reset consumers')
    timing=complete_timing(out/'timing.rpt')
    require(timing['wns_ns']==-0.413 and timing['tns_ns']==-803.549 and timing['setup_failing']==4555,
            'Routed timing baseline differs')
    return dict(status='HBM_RESET_FANOUT_REVIEWED',donor_sha256=result['checkpoint_sha256'],
                input_pins=len(rows),slr='SLR0',clocks=clocks,references=dict(Counter(r['reference'] for r in rows)),
                constraints_unchanged=True,timing=timing,route_status=route_status(out/'route_status.rpt',True),
                board_qualified=False)


def simulation(root):
    result=verified(root);out=root/'output'
    require(result['status']=='HBM_RESET_DISTRIBUTION_SIM_EXPORTED' and result['error'] is None,'Simulation failed')
    manifest=json.loads((root/'source/PREPARED.json').read_text())
    for name,record in manifest['files'].items():
        require(sha(root/'source'/name)==record['sha256'],'Preparation provenance differs: '+name)
    expected=['compile','positive-elaborate','positive-run','regression-elaborate','regression-run']
    for name in ('bypass','synchronous_assert','fault_reset'):
        expected += [name+'-compile',name+'-elaborate',name+'-run']
    expected += ['synthesis']
    require([s['name'] for s in result['steps']]==expected,'Simulation/synthesis step coverage differs')
    controls=json.loads((root/'NEGATIVE_CONTROLS.json').read_text())
    reasons={'bypass':'distribution release at fourth edge',
             'synchronous_assert':'asynchronous assertion without clocks',
             'fault_reset':'fault must preserve reset epoch'}
    require({r['name']:r['reason'] for r in controls}==reasons,'Negative controls differ')
    for step in result['steps']:
        require(not step['timed_out'],'Timed out step')
        text=(root/(step['name']+'.log')).read_text()
        negative=step['name'].removesuffix('-run')
        if step['name'].endswith('-run') and negative in reasons:
            require(re.search(r'(?im)^Fatal: RD FAIL case=\d+ '+re.escape(reasons[negative]),text),
                    'Missing intended negative rejection')
        else:
            require(step['returncode']==0 and not re.search(r'(?im)^\s*(error|fatal):',text),'Failed positive step')
    cases=re.findall(r'^RD CASE PASS case=(\d+) checks=(\d+) groups=(\d+) popped=(\d+)$',
                     (root/'positive-run.log').read_text(),re.M)
    records=sorted([dict(case=int(a),checks=int(b),groups=int(c),popped=int(d)) for a,b,c,d in cases],key=lambda r:r['case'])
    require([r['case'] for r in records]==[0,1,2,3] and all(r['popped']==6 and r['groups']==19 for r in records),
            'Functional coverage differs')
    require(sorted(result['completed'],key=lambda r:r['case'])==records,'Summary differs from simulation log')
    require('RD PASS cases=4' in (root/'positive-run.log').read_text(),'Missing final simulation marker')
    regression=re.findall(r'^RC CASE PASS case=(\d+) checks=(\d+) groups=(\d+)$',
                          (root/'regression-run.log').read_text(),re.M)
    require(sorted(int(a) for a,_,_ in regression)==[0,1,2,3] and all(int(c)==12 for _,_,c in regression),
            'Original regression coverage differs')
    require('RC PASS cases=4' in (root/'regression-run.log').read_text(),'Missing regression marker')
    require((out/'STATUS.txt').read_text().strip()=='HBM_RESET_DISTRIBUTION_TOPOLOGY_EXPORTED','Missing synthesis completion')
    prop=properties(out/'topology.tsv')
    async_names={f'{name}_reg[{bit}]' for name,count in [('source_release',3),('memory_release',3),
                 ('ready_meta',3),('ready_sync',3),('ready_final',3),('fault_meta',2),('fault_sync',2),('fault_final',2)]
                 for bit in range(count)}
    actual={name for (name,key),value in prop.items() if key=='ASYNC_REG' and value=='1'}
    require(actual==async_names,'Synchronizer attributes changed')
    dist='memory_distribution_reg'
    require(prop[dist,'REF_NAME']=='FDCE' and prop[dist,'INIT']=="1'b0",
            'Wrong post-synchronizer primitive')
    require(prop[dist,'ASYNC_REG'] not in ('1','TRUE') and prop[dist,'DONT_TOUCH'] not in ('1','TRUE'),
            'Distribution replication inhibited')
    for pin,expected_driver in [('memory_release_reg[1]/D','memory_release_reg[0]/Q'),
            ('memory_release_reg[2]/D','memory_release_reg[1]/Q'),
            ('ready_meta_reg[2]/D',dist+'/Q')]:
        require(prop[pin,'DRIVER']==expected_driver,'Primitive connectivity differs: '+pin)
    require(prop[dist+'/C','CLOCK']=='mem_clk','Distribution clock differs')
    require(prop[dist+'/D','DRIVER']==prop[dist+'/CE','DRIVER']=='VCC/P',
            'Distribution constant inputs differ')
    encoding=prop[dist,'RESET_ENCODING']
    clear=prop[dist+'/CLR','DRIVER']
    if encoding=='EXPLICIT_INVERTER':
        inverter=clear.removesuffix('/O')
        require(clear.endswith('/O') and prop[inverter,'REF_NAME']=='LUT1' and prop[inverter,'INIT']=="2'h1" and
                prop[dist+'/CLR','IS_INVERTED']=='0','Reset inverter/polarity differs')
        require(prop[inverter+'/I0','DRIVER']=='memory_release_reg[2]/Q','Distribution reset source differs')
        require(prop[clear,'LOADS']==dist+'/CLR','Reset inverter has extra loads')
        canonical_load=inverter+'/I0'
    else:
        require(encoding=='PRIMITIVE_INVERTED_CLR' and clear=='memory_release_reg[2]/Q' and
                prop[dist+'/CLR','IS_INVERTED']=='1','Primitive reset polarity differs')
        canonical_load=dist+'/CLR'
    require(prop['memory_release_reg[2]/Q','LOADS']==canonical_load,'Canonical fanout not isolated')
    require((out/'distributed_startup.dcp').stat().st_size>0,'Missing synthesized checkpoint')
    for control in controls:
        require(sha(root/('mutation-'+control['name'])/'ntrpu_hbm_startup_distributed.sv')==control['source_sha256'],
                'Negative control provenance differs')
    return dict(status='HBM_RESET_DISTRIBUTION_ISOLATED_REVIEWED',cases=records,
                regression_cases=len(regression),negative_controls=controls,async_registers=21,
                canonical_leaf_loads=1,distribution_primitive='FDCE',reset_encoding=encoding,memory_release_edge=4,
                checkpoint_sha256=sha(out/'distributed_startup.dcp'),sources=result['sources'],
                full_board_integrated=False,physical_timing_qualified=False,native_traffic_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['probe','simulation'])
    p.add_argument('root',type=Path)
    p.add_argument('--donor',type=Path)
    args=p.parse_args()
    print(json.dumps(probe(args.root,args.donor) if args.mode=='probe' else simulation(args.root),indent=2))
