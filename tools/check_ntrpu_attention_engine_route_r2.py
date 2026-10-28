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
    assert console.splitlines().count('NTRPU_ATTENTION_ENGINE_ROUTE_R1_EXPORTED') == 1
    assert not re.search(r'(?m)^ERROR:', console)
    files = evidence['files']; get = lambda n: files['output/'+n]
    assert get('route_command.txt').startswith('RETURN_CODE 0\n')
    fields = dict(line.split('\t') for line in get('metrics.tsv').splitlines())
    assert fields['top'] == 'ntrpu_attention_engine_r1'
    assert fields['part'] == 'xcvu5p-flva2104-2-i' and float(fields['clock_period']) == 5
    assert hashlib.sha256(evidence['netlist'].encode()).hexdigest()==evidence['netlist_sha256']
    buffers=buffer_netlist_review(evidence['netlist'])
    assert fields['blackboxes'] == fields['latches'] == '0' and int(fields['buffers'])==buffers['buffers']
    for name in ('utilization.rpt', 'hierarchy.rpt', 'timing.rpt'):
        assert re.findall(r'^\|\s*Design\s*:\s*(\S+)\s*$', get(name), re.M) == ['ntrpu_attention_engine_r1']
        assert re.findall(r'^\|\s*Design State\s*:\s*(\S+)\s*$', get(name), re.M) == ['Routed']
    clocks = table(get('reopened_clock.tsv'))
    assert len(clocks) == 1
    clk = clocks[0]
    assert all(clk[k] == v for k,v in dict(instance='ntrpu_engine_parent_bufg',ref='BUFGCE',
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
        elif kind=='top': f['output/metrics.tsv']=f['output/metrics.tsv'].replace('ntrpu_attention_engine_r1','wrong_top')
        elif kind=='state': f['output/timing.rpt']=f['output/timing.rpt'].replace('Routed','Synthesized')
        elif kind=='missing_marker': e['console']=e['console'].replace('NTRPU_ATTENTION_ENGINE_ROUTE_R1_EXPORTED','REMOVED')
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
    # This recognizer is deliberately limited to the archived Vivado funcsim
    # format and exact two-buffer circuit. Unknown uses fail closed.
    cells_by_module={};buffers=[]
    pattern=r'(?m)^  (\w+)\s+(?:#\((.*?)\)\s+)?(\\\S+|\w+)\s*\((.*?)\);'
    for module in re.finditer(r'(?ms)^module (\S+).*?^endmodule',netlist):
        assert module[1] not in cells_by_module
        cells=[]
        for match in re.finditer(pattern,module[0],re.S):
            ref,params,name,ports=match.groups()
            pairs=re.findall(r'\.(\w+)\(([^()]*)\)',ports)
            assert len({k for k,v in pairs})==len(pairs)
            cells.append(dict(ref=ref,params=params or '',name=name,pins=dict(pairs)))
        cells_by_module[module[1]]=(module[0],cells)
        for cell in cells:
            if cell['ref'].startswith('BUFG'):buffers.append((module[1],cell))
    assert len(buffers)==len(re.findall(r'(?m)^\s*BUFG\w*\b',netlist))==2
    assert {(m,c['name']) for m,c in buffers}=={
        ('ntrpu_attention_engine_r1','ntrpu_engine_parent_bufg'),
        ('ntrpu_attention_context_r2','i___40_bufg_place')}
    data=next(c for m,c in buffers if c['name']=='i___40_bufg_place')
    assert data['ref']=='BUFGCE' and re.sub(r'\s+','',data['params'])=='.CE_TYPE("ASYNC")'
    assert data['pins']==dict(CE="1'b1",I='i___40_n_3_bufg_place',O='i___40_n_3')
    body,cells=cells_by_module['ntrpu_attention_context_r2'];net=data['pins']['O']
    loads=[(cell,pin) for cell in cells for pin,n in cell['pins'].items() if n==net and cell is not data]
    assert len(loads)==2048
    assert all(c['ref']=='FDRE' and p=='CE' and c['pins'].get('C')=='clk' for c,p in loads)
    assert {c['name'] for c,p in loads}=={f'\\query_mem_reg[{i}][{j}]' for i in range(64) for j in range(32)}
    count=lambda n:len(re.findall(r'(?<![\w\\])'+re.escape(n)+r'(?!\w)',body))
    # Exactly declaration, buffer output, and all 2048 CE connections: no
    # alias/assign/hierarchical use may hide an additional consumer.
    assert count(net)==2050 and count(data['pins']['I'])==3
    drivers=[c for c in cells if c['pins'].get('O')==data['pins']['I']]
    assert len(drivers)==1
    driver=drivers[0]
    assert driver['ref']=='LUT2' and driver['name']=='i___40'
    assert re.sub(r'\s+','',driver['params'])==".INIT(4'h8)"
    assert driver['pins']==dict(I0='\\query_position[31]_i_1_n_3 ',I1='rst_n',O=data['pins']['I'])
    clock=next(c for m,c in buffers if c['name']=='ntrpu_engine_parent_bufg')
    assert clock['ref']=='BUFGCE' and clock['pins']==dict(CE="1'b1",I='ntrpu_engine_parent_clock_input',O='\\^clk ')
    assert re.sub(r'\s+','',clock['params'])=='.CE_TYPE("SYNC"),.SIM_DEVICE("ULTRASCALE"),.STARTUP_SYNC("FALSE")'
    return dict(buffers=2,clock_buffers=1,data_buffers=1,data_buffer='ntrpu_attention_context_r2/i___40_bufg_place',
        source_lut='LUT2 INIT=0x8',static_enable=True,asynchronous_passthrough=True,data_consumers=2048,
        consumer_type='FDRE/CE',consumer_clock='clk',all_uses_accounted=True)


def buffer_controls(evidence, checkpoint_sha):
    netlist=evidence['netlist']
    changes={
        'data_enable_low':("i___40_bufg_place\n       (.CE(1'b1)","i___40_bufg_place\n       (.CE(1'b0)"),
        'data_sync_gating':('.CE_TYPE("ASYNC")','.CE_TYPE("SYNC")'),
        'data_clock_pin':('.CE(i___40_n_3)', '.C(i___40_n_3)'),
        'data_missing_load':('.CE(i___40_n_3)', ".CE(1'b0)"),
        'data_extra_load':('wire i___40_n_3;', 'wire i___40_n_3;\n  assign hidden_alias = i___40_n_3;'),
        'data_driver_lut':(".INIT(4'h8)) \n    i___40\n", ".INIT(4'h7)) \n    i___40\n"),
        'clock_enable_low':("ntrpu_engine_parent_bufg\n       (.CE(1'b1)","ntrpu_engine_parent_bufg\n       (.CE(1'b0)"),
        'clock_input':('.I(ntrpu_engine_parent_clock_input)', '.I(rst_n)'),
    }
    rejected=[]
    for name,(before,after) in changes.items():
        assert before in netlist,name
        e=copy.deepcopy(evidence);e['netlist']=netlist.replace(before,after,1)
        # Keep hash internally consistent so the semantic audit must reject it.
        e['netlist_sha256']=hashlib.sha256(e['netlist'].encode()).hexdigest()
        try:review(e,checkpoint_sha)
        except (AssertionError,KeyError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted netlist mutation '+name)
    e=copy.deepcopy(evidence);e['netlist_sha256']='0'*64
    try:review(e,checkpoint_sha)
    except AssertionError:rejected.append('netlist_hash')
    else:raise AssertionError('Accepted wrong netlist hash')
    return rejected

