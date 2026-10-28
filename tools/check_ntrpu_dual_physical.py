"""Review isolated four-domain receiver implementation and routed regressions."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_composition import common,digest
from check_ntrpu_multibuffer_credits import require
from check_ntrpu_dual_mig import review as review_mig


def table(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream,delimiter='\t'))


def physical(root):
    record,console=common(root,'NTRPU_DUAL_PHYSICAL_EXPORTED');folder=root/'output/suite'
    metrics=dict(line.split('=',1) for line in (folder/'metrics.txt').read_text().splitlines())
    require(all(metrics[k]==v for k,v in dict(CLOCKS='4',BUFGS='4',DRC_ERRORS='0',BLACKBOXES='0',LATCHES='0').items()),'Physical topology/DRC')
    domains={}
    for label,period in [('SOURCE',5),('MANAGEMENT',10),('MEMORY0',5),('MEMORY1',5)]:
        require(float(metrics[label+'_PERIOD_NS'])==period,'Clock period')
        domains[label]=dict(period_ns=period,setup_ns=float(metrics[label+'_SETUP_NS']),hold_ns=float(metrics[label+'_HOLD_NS']))
    timing=(folder/'timing.rpt').read_text()
    checks={name:int(value) for name,value in re.findall(r'checking (\w+) \((\d+)\)',timing)}
    for name in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops'):
        require(checks.get(name)==0,'Timing coverage: '+name)
    match=re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
    require(match is not None and len(match[1].split())==12,'Timing table')
    values=list(map(float,match[1].split()))
    resources={}
    for key,file in [('core','core_utilization.rpt'),('startup','startup_utilization.rpt')]:
        util=(folder/file).read_text();resources[key]={}
        for label in ('CLB LUTs','LUT as Memory','CLB Registers','Register as Latch','Block RAM Tile','CARRY8','DSPs'):
            found=re.search(r'^\|\s*'+re.escape(label)+r'\*?\s*\|\s*([\d.,]+)',util,re.M)
            require(found is not None,'Missing resource: '+label)
            resources[key][label]=float(found[1].replace(',',''))
        require(resources[key]['Register as Latch']==0,'Inferred latch')
    routing={};text=(folder/'route_status.rpt').read_text()
    for name in ('# of routable nets','# of fully routed nets','# of nets with routing errors'):
        found=re.search(re.escape(name)+r'\.+\s*:\s*(\d+)',text)
        require(found is not None,'Route report');routing[name]=int(found[1])
    require(routing['# of routable nets']==routing['# of fully routed nets'] and routing['# of nets with routing errors']==0,'Routing incomplete')
    closed=all(values[i]>=0 for i in (0,4,8)) and all(values[i]==0 for i in (2,6,10))
    require(not re.search(r'^CRITICAL WARNING:',console,re.M),'Critical tool warning requires investigation')
    return dict(status='FOUR_DOMAIN_TIMING_PASS_CDC_REVIEW_REQUIRED' if closed else 'FOUR_DOMAIN_TIMING_OPEN',
        domains=domains,overall=dict(setup_ns=values[0],hold_ns=values[4],pulse_ns=values[8],
        setup_failures=values[2],hold_failures=values[6],pulse_failures=values[10]),resources=resources,
        routing=routing,timing_checks=checks,seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
        warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
        limitations=['OOC registered measurement boundary, not a deployable AXI adapter',
                    'No DDR PHY/controller placement, pinout, IO timing or physical clock-source selection',
                    'CDC/reset and exported netlist need separate reviews'])


def netlist(root,physical_root):
    require(physical_root is not None,'Physical provenance required')
    result=review_mig(root);folder=root/'output/suite'
    record=json.loads((root/'output/result.json').read_text())
    require(record['returncode']==0 and 'NTRPU_DUAL_ROUTED_EXPORTED' in (root/'output/console.log').read_text(),'Routed export marker')
    hashes={}
    for name in ('routed_core_funcsim.v',):
        hashes[name]=digest(physical_root/'output/suite'/name)
        require(hashes[name]==digest(root/'inputs/evidence'/name),'Routed netlist provenance')
    transformations=json.loads((folder/'bench/transformation.json').read_text())
    expected=[('tb_ntrpu_dual_multibuffer_mig.sv','ntrpu_dual_multibuffer_mig','ntrpu_dual_routed_core'),
              ('tb_ntrpu_dual_mig_cold_start.sv','ntrpu_dual_mig_cold_start','ntrpu_dual_mig_cold_start')]
    require(set(transformations)=={v[0] for v in expected},'Testbench inventory')
    for name,old,new in expected:
        original=root/'inputs/fpga/tb'/name;target=folder/'bench'/name;entry=transformations[name]
        before=old+' dut(.*);';after=new+' dut(.*);'
        require(original.read_text().count(before)==1 and original.read_text().replace(before,after)==target.read_text(),'Only DUT binding changed')
        require(entry==dict(source_sha256=digest(original),generated_sha256=digest(target),before=before,after=after),'TB transformation metadata')
    result.update(status='ROUTED_DUAL_MIG_FUNCTIONAL_PASS_NO_SDF',netlist_hashes=hashes,
        supervisor_unit_scope='13 RTL reference cases; extracted routed cell has no standalone RTL interface',
        integrated_core_scope='56 transactions through the complete routed receiver including startup',
        limitations=result['limitations']+['Routed primitive simulation without SDF; no analog metastability proof',
            'Supervisor unit cases are RTL only; case 12 uses artificial 1 ns source clock, not a timing claim'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['physical','netlist']);parser.add_argument('run',type=Path)
    parser.add_argument('--physical',type=Path);args=parser.parse_args()
    result=physical(args.run) if args.stage=='physical' else netlist(args.run,args.physical)
    result['checker_sha256']=digest(Path(__file__))
    with (args.run/'REVIEW.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
