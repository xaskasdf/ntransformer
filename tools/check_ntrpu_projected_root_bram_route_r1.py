"""Review reopened OOC routing evidence; preserve failed timing as a measurement."""
from collections import Counter
import copy
import csv
import io
import math
import hashlib
import re


def table(text):
    return list(csv.DictReader(io.StringIO(text), delimiter='\t'))


def review(evidence, checkpoint_sha):
    assert evidence['invocation']['exit_code'] == 0, 'Implementation command failed'
    assert evidence['input_checkpoint_sha256'] == checkpoint_sha, 'Parent checkpoint provenance'
    assert evidence['routed_checkpoint_sha256'] and evidence['netlist_sha256']
    console = evidence['console']
    assert console.splitlines().count('NTRPU_ATTENTION_PROJECTED_ROOT_ROUTE_R1_EXPORTED') == 1
    assert not re.search(r'(?m)^ERROR:', console)
    files = evidence['files']; get = lambda n: files['output/'+n]
    assert get('route_command.txt').startswith('RETURN_CODE 0\n')
    fields = dict(line.split('\t') for line in get('metrics.tsv').splitlines())
    assert fields['top'] == 'ntrpu_attention_projected_root_r2'
    assert fields['part'] == 'xcvu5p-flva2104-2-i' and float(fields['clock_period']) == 5
    assert hashlib.sha256(evidence['netlist'].encode()).hexdigest()==evidence['netlist_sha256']
    buffers=buffer_netlist_review(evidence['netlist'])
    assert fields['blackboxes'] == fields['latches'] == '0' and int(fields['buffers'])==buffers['buffers']
    for name in ('utilization.rpt', 'hierarchy.rpt', 'timing.rpt'):
        assert re.findall(r'^\|\s*Design\s*:\s*(\S+)\s*$', get(name), re.M) == ['ntrpu_attention_projected_root_r2']
        assert re.findall(r'^\|\s*Design State\s*:\s*(\S+)\s*$', get(name), re.M) == ['Physopt'] or re.findall(r'^\|\s*Design State\s*:\s*([^\n]+?)\s*$', get(name), re.M) == ['Physopt postRoute']
    clocks = table(get('reopened_clock.tsv'))
    assert len(clocks) == 1
    clk = clocks[0]
    assert all(clk[k] == v for k,v in dict(instance='ntrpu_projected_root_parent_bufg',ref='BUFGCE',
        loc='BUFGCE_X0Y60',clock='clk',period='5.000',input='clk',ce='VCC').items())
    assert int(clk['loads']) > 1000
    prepared = table(get('prepared_clock.tsv'))
    assert len(prepared) == 1 and int(prepared[0]['loads']) > 1000
    for key in ('instance','ref','loc','clock','period','input','ce'):
        assert prepared[0][key] == clk[key], 'Clock changed after implementation'
    constraints = [get(name) for name in ('constraints_input.xdc','constraints_before.xdc','constraints_after.xdc')]
    commands = lambda text: [line.strip() for line in text.splitlines()
        if re.match(r'^\s*(?:create_clock|set_clock_uncertainty)\s',line)]
    assert commands(constraints[0]) and all(commands(c) == commands(constraints[0]) for c in constraints[1:])
    for text in constraints:
        assert not re.search(r'(?m)^\s*(?:set_false_path|set_multicycle_path|set_clock_groups|set_max_delay|set_min_delay|set_disable_timing|create_waiver)\b',text)
        assert not re.search(r'(?mi)^\s*set_property\s+(?:SEVERITY|HD.CLK_SRC)\b|CLOCK_DEDICATED_ROUTE\s+(?:FALSE|0)',text)
    timing = get('timing.rpt')
    checks = {k:int(v) for k,v in re.findall(r'checking (\w+) \((\d+)\)', timing)}
    required = ('no_clock','constant_clock','unconstrained_internal_endpoints','multiple_clock','loops','latch_loops')
    assert all(k in checks for k in required)
    match = re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',timing,re.M)
    assert match and len(match[1].split()) == 12
    values = list(map(float,match[1].split())); assert all(math.isfinite(v) for v in values)
    internal = {kind:float(fields['internal_'+kind]) for kind in ('setup','hold')}
    for kind,slack in internal.items():
        assert math.isfinite(slack)
        path = get('internal_'+kind+'.rpt')
        reported = re.search(r'Slack\s*\([^)]*\)\s*:\s*(-?[\d.]+)ns',path)
        assert reported and abs(float(reported[1])-slack) < 0.0011, 'Path/summary slack mismatch'
    routes = {}
    for label in ('# of routable nets','# of fully routed nets','# of nets with routing errors'):
        m = re.search(re.escape(label)+r'\.+\s*:\s*(\d+)',get('route.rpt'))
        assert m, 'Missing route counter '+label
        routes[label] = int(m[1])
    inventory = table(get('route_inventory.tsv'))
    assert len(inventory)>1000 and len({row['net'] for row in inventory}) == len(inventory)
    assert all(row['gap'] in ('0','1') and row['status'] for row in inventory)
    drc = table(get('drc_violations.tsv'))
    resources = {}
    for label in ('CLB LUTs','CLB Registers','Block RAM Tile','DSPs'):
        m = re.search(r'(?m)^\|\s*'+re.escape(label)+r'\*?\s*\|\s*([0-9.,]+)\s*\|',get('utilization.rpt'))
        assert m,label; resources[label] = float(m[1].replace(',',''))
    assert all(v>0 for v in resources.values())
    gates = dict(
        complete_routing=routes['# of routable nets']>0 and routes['# of routable nets']==routes['# of fully routed nets'] and routes['# of nets with routing errors']==0,
        no_explicit_gaps=all(row['gap']=='0' for row in inventory),
        clock_routed=clk['route_status']=='ROUTED',
        internal_timing=all(v>=0 for v in internal.values()),
        setup_hold_pulse=all(values[i]>=0 for i in (0,4,8)) and all(values[i]==0 for i in (2,6,10)),
        internal_coverage=all(checks[k]==0 for k in required),
        no_drc_errors=not any(row['severity'] in ('Error','Critical Warning') for row in drc),
        no_critical_console_warnings=not re.search(r'(?m)^CRITICAL WARNING:',console))
    return dict(status='ROUTED_INTERNAL_TIMING_PASS_PARENT_PENDING' if all(gates.values()) else 'ROUTED_INTERNAL_NOT_QUALIFIED',
        gates=gates,internal=internal,resources=resources,routing=routes,timing_checks=checks,
        setup_wns_ns=values[0],hold_whs_ns=values[4],pulse_wpws_ns=values[8],
        route_inventory_rows=len(inventory),route_status_counts=dict(Counter(row['status'] for row in inventory)),
        clock=clk,buffer_netlist_proof=buffers,design_flags={k:fields[k] for k in ('PLACED_FULLY','PARTIALLY_ROUTED','ROUTED_FULLY','ERRORS_IN_ROUTES')},
        drc=drc,input_checkpoint_sha256=checkpoint_sha,routed_checkpoint_sha256=evidence['routed_checkpoint_sha256'],
        netlist_sha256=evidence['netlist_sha256'],physical_board_timing_qualified=False,
        post_route_functional_simulation_qualified=False,
        limitations=['Provisional BUFGCE_X0Y60 placement and unbudgeted external ports.',
                    'No board clock, DDR/PCIe, parent placement or full inference qualification.',
                    'Original RTL simulation belongs to the archived input checkpoint; post-route functional simulation is separate.'])


def controls(evidence, checkpoint_sha):
    rejected = []
    for kind in ('checkpoint','clock_period','clock_location','clock_enable','top','state','missing_marker',
                 'route_command','exception','severity','gap','clock_route','route_error','coverage'):
        e = copy.deepcopy(evidence); f=e['files']; target=None
        if kind=='checkpoint': e['input_checkpoint_sha256']='0'*64
        elif kind=='clock_period': f['output/metrics.tsv']=re.sub(r'(?m)^clock_period\t[^\n]+','clock_period\t6.000',f['output/metrics.tsv'])
        elif kind=='clock_location': f['output/reopened_clock.tsv']=f['output/reopened_clock.tsv'].replace('BUFGCE_X0Y60','BUFGCE_X0Y61')
        elif kind=='clock_enable': f['output/reopened_clock.tsv']=f['output/reopened_clock.tsv'].replace('\tVCC\t','\tLUT1\t')
        elif kind=='top': f['output/metrics.tsv']=f['output/metrics.tsv'].replace('ntrpu_attention_projected_root_r2','wrong_top')
        elif kind=='state': f['output/timing.rpt']=f['output/timing.rpt'].replace('Physopt postRoute','Synthesized').replace('Physopt','Synthesized')
        elif kind=='missing_marker': e['console']=e['console'].replace('NTRPU_ATTENTION_PROJECTED_ROOT_ROUTE_R1_EXPORTED','REMOVED')
        elif kind=='route_command': f['output/route_command.txt']=f['output/route_command.txt'].replace('RETURN_CODE 0','RETURN_CODE 1')
        elif kind=='exception': f['output/constraints_after.xdc']+='\nset_false_path -from [all_registers]\n'
        elif kind=='severity': f['output/constraints_after.xdc']+='\nset_property SEVERITY Warning [get_drc_checks]\n'
        elif kind=='gap':
            f['output/route_inventory.tsv']=re.sub(r'(?m)\t[01]$', '\t1', f['output/route_inventory.tsv'], count=1);target='no_explicit_gaps'
        elif kind=='clock_route':
            f['output/reopened_clock.tsv']=f['output/reopened_clock.tsv'].replace('\tROUTED','\tUNROUTED');target='clock_routed'
        elif kind=='route_error':
            f['output/route.rpt']=re.sub(r'(# of nets with routing errors\.+\s*:\s*)\d+',r'\g<1>99999',f['output/route.rpt']);target='complete_routing'
        elif kind=='coverage':
            f['output/timing.rpt']=re.sub(r'checking no_clock \(\d+\)','checking no_clock (99999)',f['output/timing.rpt']);target='internal_coverage'
        # A failed physical gate may already contain the injected defect.
        if e==evidence:
            assert target and not review(evidence,checkpoint_sha)['gates'][target]
            rejected.append(dict(name=kind,already_failing_gate=target));continue
        try:
            changed = review(e,checkpoint_sha)
        except (AssertionError,ValueError,KeyError):
            rejected.append(dict(name=kind,rejected=True))
        else:
            assert target and not changed['gates'][target], 'Accepted '+kind
            rejected.append(dict(name=kind,rejected_gate=target))
    return rejected

def buffer_netlist_review(netlist):
    # Limited Vivado funcsim parser. Unknown buffers, aliases or loads fail closed.
    modules={};buffers=[]
    pattern=r'(?m)^  (\w+)\s+(?:#\((.*?)\)\s+)?(\\\S+|\w+)\s*\((.*?)\);'
    for module in re.finditer(r'(?ms)^module (\S+).*?^endmodule',netlist):
        assert module[1] not in modules
        cells=[]
        for match in re.finditer(pattern,module[0],re.S):
            ref,params,name,ports=match.groups();pairs=re.findall(r'\.(\w+)\(([^()]*)\)',ports)
            assert len({k for k,v in pairs})==len(pairs)
            cells.append(dict(ref=ref,params=params or '',name=name,pins=dict(pairs)))
        modules[module[1]]=(module[0],cells)
        buffers += [(module[1],c) for c in cells if c['ref'].startswith('BUFG')]
    assert len(buffers)==len(re.findall(r'(?m)^\s*BUFG\w*\b',netlist))>=1
    clock=[c for m,c in buffers if m=='ntrpu_attention_projected_root_r2' and c['name']=='ntrpu_projected_root_parent_bufg']
    assert len(clock)==1;clock=clock[0]
    assert clock['ref']=='BUFGCE' and clock['pins']==dict(CE="1'b1",I='ntrpu_projected_root_parent_clock_input',O='\\^clk ')
    assert re.sub(r'\s+','',clock['params'])=='.CE_TYPE("SYNC"),.SIM_DEVICE("ULTRASCALE"),.STARTUP_SYNC("FALSE")'
    data=[]
    for mod,cell in buffers:
        if cell is clock:continue
        assert cell['ref']=='BUFGCE' and re.sub(r'\s+','',cell['params'])=='.CE_TYPE("ASYNC")'
        assert set(cell['pins'])=={'CE','I','O'} and cell['pins']['CE']=="1'b1"
        body,cells=modules[mod];net=cell['pins']['O'];source=cell['pins']['I']
        loads=[(c,p) for c in cells for p,n in c['pins'].items() if n==net and c is not cell]
        assert loads and all(c['ref'] in ('FDRE','FDSE','FDCE','FDPE') and p=='CE' and c['pins'].get('C') in ('clk','\\^clk ') for c,p in loads)
        count=lambda n:len(re.findall(r'(?<![\w\\])'+re.escape(n)+r'(?!\w)',body))
        assert count(net)==len(loads)+2 and count(source)==3
        drivers=[c for c in cells if c['pins'].get('O')==source]
        assert len(drivers)==1 and re.fullmatch(r'LUT[1-6]',drivers[0]['ref'])
        data.append(dict(module=mod,buffer=cell['name'],loads=len(loads),source=drivers[0]['name'],source_type=drivers[0]['ref'],static_enable=True,asynchronous_passthrough=True,consumer_pin='CE'))
    return dict(buffers=len(buffers),clock_buffers=1,data_buffers=len(data),data=data,all_uses_accounted=True)


def buffer_controls(evidence,checkpoint_sha):
    netlist=evidence['netlist'];proof=buffer_netlist_review(netlist);rejected=[]
    changes={
        'clock_enable_low':("ntrpu_projected_root_parent_bufg\n       (.CE(1'b1)","ntrpu_projected_root_parent_bufg\n       (.CE(1'b0)"),
        'clock_input':('.I(ntrpu_projected_root_parent_clock_input)','.I(rst_n)'),
    }
    if proof['data_buffers']:
        name=proof['data'][0]['buffer']
        changes['data_enable_low']=(name+"\n       (.CE(1'b1)",name+"\n       (.CE(1'b0)")
        changes['data_sync_gating']=('.CE_TYPE("ASYNC")','.CE_TYPE("SYNC")')
    for name,(before,after) in changes.items():
        assert before in netlist,name
        e=copy.deepcopy(evidence);e['netlist']=netlist.replace(before,after,1)
        e['netlist_sha256']=hashlib.sha256(e['netlist'].encode()).hexdigest()
        try:review(e,checkpoint_sha)
        except (AssertionError,KeyError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted buffer mutation '+name)
    e=copy.deepcopy(evidence);e['netlist_sha256']='0'*64
    try:review(e,checkpoint_sha)
    except AssertionError:rejected.append('netlist_hash')
    else:raise AssertionError('Accepted wrong netlist hash')
    return rejected
