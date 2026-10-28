"""Verify immutable evidence and independently replay the routed composition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_multibuffer_credits import require, replay, outstanding, negative_traces


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def common(root, marker):
    output = root/'output'
    record = json.loads((output/'result.json').read_text())
    require(record['status']=='EXPORTED_PENDING_REVIEW' and record['returncode']==0
            and not record['timed_out'], 'Incomplete batch')
    require(record['sources']==record['sources_after'], 'Sources changed during batch')
    for name, value in record['sources'].items():
        require(digest(root/'inputs'/name)==value, 'Input hash: '+name)
    for name, value in json.loads((output/'FILES.json').read_text()).items():
        require(digest(output/name)==value, 'Output hash: '+name)
    console = (output/'console.log').read_text()
    require(marker in console, 'Missing export marker')
    require(not re.search(r'^ERROR:', console, re.M), 'Tool error')
    return record, console


def physical(root):
    record, console = common(root, 'NTRPU_COMPOSITION_PHYSICAL_EXPORTED')
    folder = root/'output/suite'
    metrics = dict(line.split('=',1) for line in (folder/'metrics.txt').read_text().splitlines())
    require(metrics['CLOCKS']=='3' and metrics['BUFGS']=='3' and metrics['BLACKBOXES']=='0', 'Clock/implementation topology')
    require(metrics['DRC_ERRORS']=='0', 'DRC errors')
    domains = {}
    for label, period in [('SOURCE',5.0),('MEMORY0',4.0),('MEMORY1',4.3)]:
        require(float(metrics[label+'_PERIOD_NS'])==period, 'Clock period')
        domains[label] = dict(period_ns=period, setup_ns=float(metrics[label+'_SETUP_NS']),
                              hold_ns=float(metrics[label+'_HOLD_NS']))
    timing = (folder/'timing.rpt').read_text()
    checks = {name:int(value) for name,value in re.findall(r'checking (\w+) \((\d+)\)',timing)}
    for name in ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops'):
        require(checks.get(name)==0, 'Timing coverage: '+name)
    table = re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
    require(table and len(table[1].split())==12, 'Timing table')
    values = list(map(float,table[1].split()))
    resources = {}
    util = (folder/'core_utilization.rpt').read_text()
    for label in ('CLB LUTs','LUT as Memory','CLB Registers','Register as Latch','Block RAM Tile','CARRY8','DSPs'):
        match = re.search(r'^\|\s*'+re.escape(label)+r'\*?\s*\|\s*([\d.,]+)',util,re.M)
        require(match is not None, 'Missing resource: '+label)
        resources[label] = float(match[1].replace(',',''))
    require(resources['Register as Latch']==0, 'Inferred latches')
    routing = {}
    text = (folder/'route_status.rpt').read_text()
    for name in ('# of routable nets','# of fully routed nets','# of nets with routing errors'):
        match = re.search(re.escape(name)+r'\.+\s*:\s*(\d+)',text)
        require(match is not None, 'Route status: '+name)
        routing[name] = int(match[1])
    require(routing['# of routable nets']==routing['# of fully routed nets']
            and routing['# of nets with routing errors']==0, 'Incomplete routing')
    closed = all(values[i]>=0 for i in (0,4,8)) and all(values[i]==0 for i in (2,6,10))
    return dict(status='PHYSICAL_TIMING_PASS_CDC_REVIEW_REQUIRED' if closed else 'PHYSICAL_TIMING_OPEN',
        domains=domains, overall=dict(setup_ns=values[0],hold_ns=values[4],pulse_ns=values[8],
        setup_failures=values[2],hold_failures=values[6],pulse_failures=values[10]),
        resources=resources,timing_checks=checks,routing=routing,
        critical_warnings=re.findall(r'^CRITICAL WARNING:.*',console,re.M),
        warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
        seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
        limits=['OOC registered measurement boundary, not a deployable AXI adapter',
                'External IO delays, board clock source and pin constraints are absent',
                'Review CDC, IP waivers and applied/ignored exceptions separately'])


def netlist(root, physical_root):
    record, console = common(root, 'NTRPU_COMPOSITION_NETLIST_EXPORTED')
    require(physical_root is not None, 'Physical provenance required')
    source = physical_root/'output/suite/routed_core_funcsim.v'
    require(digest(source)==digest(root/'inputs/evidence/routed_core_funcsim.v'), 'Netlist provenance')
    folder = root/'output/suite'
    transformation = json.loads((folder/'bench/transformation.json').read_text())
    require(transformation['source_sha256']==digest(root/'inputs/fpga/tb/tb_ntrpu_peer_multibuffer_credits.sv'), 'TB provenance')
    require(transformation['generated_sha256']==digest(folder/'bench/netlist_tb.sv'), 'Generated TB hash')
    profiles = []
    for profile in (0,1,2):
        text = (folder/f'profile-{profile}.log').read_text()
        result = replay(text,profile)
        result['credits'] = outstanding(text,4)
        result['trace_negative_controls'] = negative_traces(text,profile)
        profiles.append(result)
    return dict(status='ROUTED_CORE_FUNCTIONAL_PASS_NO_SDF',profiles=profiles,
        netlist_sha256=digest(source),seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
        warnings=dict(Counter(re.findall(r'^WARNING: \[([^]]+)\]',console,re.M))),
        critical_warnings=re.findall(r'^CRITICAL WARNING:.*',console,re.M),
        limits=['Functional primitive simulation without SDF; no analog metastability model',
                'Synthetic AXI targets; not measured PCIe, HBM or DDR performance'])


def audit(root, physical_root):
    record, console = common(root, 'NTRPU_COMPOSITION_CDC_AUDIT_EXPORTED')
    require(physical_root is not None, 'Physical provenance required')
    original = physical_root/'output/suite'
    folder = root/'output/suite'
    require(digest(root/'inputs/evidence/routed.dcp')==digest(original/'routed.dcp'), 'Checkpoint provenance')
    require(not re.search(r'^CRITICAL WARNING:',console,re.M), 'Critical audit warning')
    text = (folder/'bus_skew.rpt').read_text()
    table = text.rsplit('1. Bus Skew Report Summary',1)[1].split('2. Bus Skew Report Per Constraint')[0]
    rows = re.findall(r'^\s*(Slow|Fast)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s*$',table,re.M)
    require(len(rows)==20 and all(float(r[3])>=0 for r in rows), 'Bus skew coverage/violation')
    cdc = (folder/'cdc_unwaived.rpt').read_text().split('CDC Report',1)[1]
    original_cdc = (original/'cdc.rpt').read_text().split('CDC Report',1)[1]
    def canonical_cdc(text):
        # Reopening a DCP can reorder equal-priority rows. Retain clock pair,
        # multiplicity, rule, depth, exception and reported endpoints.
        result = Counter();source = destination = None
        for line in text.splitlines():
            if line.startswith('Source Clock:'):source=line
            elif line.startswith('Destination Clock:'):destination=line
            elif re.match(r'^\s+\d+\s+CDC-',line):
                result[(source,destination,' '.join(line.split()[1:]))]+=1
        return result
    require(canonical_cdc(cdc)==canonical_cdc(original_cdc)==canonical_cdc(
        (original/'cdc_unwaived.rpt').read_text().split('CDC Report',1)[1]), 'CDC report mismatch')
    summary = re.findall(r'^(CDC-\d+)\s+(\w+)\s+(\d+)\s+',cdc,re.M)
    require(summary==[('CDC-3','Info','20'),('CDC-9','Info','20')], 'Unexpected CDC diagnostics')
    cdc_rows = re.findall(r'^\s+\d+\s+CDC-.*$',cdc,re.M)
    require(len(cdc_rows)==40 and all('/cdc/inst/' in line and 'core/memory_port[' in line for line in cdc_rows), 'CDC outside the IP boundary')
    require('No ignored timing exceptions found.' in (original/'ignored_exceptions.rpt').read_text(), 'Ignored exceptions')
    xdc = (original/'implemented.xdc').read_text()
    context = '';counts = Counter();waivers = Counter()
    for line in xdc.splitlines():
        if line.startswith('current_instance '):context=line
        if line.startswith(('set_false_path','set_max_delay','set_bus_skew')):
            require('/cdc/inst' in context, 'Exception outside vendor CDC: '+context)
            counts[line.split()[0]]+=1
        if line.startswith('create_waiver '):
            require('/cdc/inst' in context, 'Waiver outside vendor CDC')
            waivers[re.search(r'-id \{([^}]+)\}',line)[1]]+=1
    require(counts==dict(set_false_path=44,set_max_delay=20,set_bus_skew=20), 'Exception coverage')
    require(not re.search(r'^set_clock_groups',xdc,re.M), 'Unexpected broad clock exception')
    bindings = (folder/'clock_binding.txt').read_text()
    require(bindings.startswith('CONVERTERS=2\n'), 'CDC instances')
    for g, clock in [(0,'memory0_clk'),(1,'memory1_clk')]:
        prefix=f'core/memory_port[{g}].port/cdc/'
        require(prefix+'s_axi_aclk=source_clk' in bindings and prefix+'m_axi_aclk='+clock in bindings, 'CDC clock binding')
    bufg = [line for line in bindings.splitlines() if line.startswith('BUFG=')]
    require(len(bufg)==3 and all('ROUTE=ROUTED' in line for line in bufg), 'Clock routing')
    return dict(status='CDC_AND_BUS_SKEW_AUDIT_PASS_WITH_IP_CONSTRAINTS',
        bus_skew_constraints=len(rows),min_bus_skew_slack_ns=min(float(r[3]) for r in rows),
        max_bus_skew_ns=max(float(r[2]) for r in rows),cdc_summary=summary,
        exceptions=dict(counts),vendor_waiver_definitions=dict(waivers),clock_bindings=bindings.splitlines(),
        seconds=record['seconds'],cpus=record['cpus'],nice=record['nice'],
        routed_checkpoint_sha256=digest(original/'routed.dcp'),
        limits=['CDC report hides encrypted IP internals; use the supported vendor constraints',
                'Default waiver report omits internal definitions present in implemented.xdc',
                'No MTBF measurement or independent-reset recovery proof'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['physical','netlist','audit'])
    parser.add_argument('run', type=Path)
    parser.add_argument('--physical', type=Path)
    args = parser.parse_args()
    result = physical(args.run) if args.stage=='physical' else (
        netlist(args.run,args.physical) if args.stage=='netlist' else audit(args.run,args.physical))
    result['checker_sha256'] = digest(Path(__file__))
    with (args.run/'REVIEW.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
