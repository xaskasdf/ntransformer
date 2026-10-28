"""Review native loader synthesis and scoped constraints, retaining open findings."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_bar_control import (closed, compare_table, exact_connections,
    lock_constants, rows, snapshot_bounds, CSR_WIDTHS, BRIDGE_WIDTHS)
from review_hbm_control_plane import require, sha
from review_hbm_native_constraints import xdc_commands


LOADER_SHA = '14aca9b0b01fbb575671e4399a76a35613010b7a2e16b1e0637b9dbb3bda1103'
HBM_WIDTHS = dict(awaddr=34, awburst=2, awid=6, awlen=4, awsize=3, awvalid=1,
    awready=1, wdata=256, wstrb=32, wlast=1, wvalid=1, wready=1, bid=6,
    bresp=2, bvalid=1, bready=1)


def expand_xdc(text):
    commands = xdc_commands(text)
    aliases = {}
    for line, count in commands.items():
        m = re.fullmatch(r'set (_xlnx_shared_i\d+) (.+)', line)
        if m:
            require(count == 1, 'Duplicate XDC alias')
            aliases[m[1]] = m[2]
    result = Counter()
    for line, count in commands.items():
        if re.match(r'set _xlnx_shared_i\d+ ', line):
            continue
        for _ in range(len(aliases)+1):
            new = re.sub(r'\$(_xlnx_shared_i\d+)\b', lambda m: aliases[m[1]], line)
            if new == line:
                break
            line = new
        require(not re.search(r'\$_xlnx_shared_i', line), 'Unresolved XDC alias')
        result[line] += count
    return result


def loader_paths(table, clocks):
    expected = {}
    def pair(source, dest, sc, dc, bound):
        expected[(source+'/C', dest+'/D')] = (clocks[sc], clocks[dc], bound)
    p = 'weight_loader/'
    for bit in range(34):
        pair(p+f'active_reg[{bit}]', p+(f'address_reg[{bit}]' if bit < 33 else 'lane_reg'), 'pcie_clk', 'mem_clk', 2.222)
    for bit in range(256):
        pair(p+f'active_reg[{bit+128}]', p+f'payload_reg[{bit}]', 'pcie_clk', 'mem_clk', 2.222)
    for bit in range(4):
        pair(p+f'memory_result_reg[{bit}]', p+f'completion_reg[{bit}]', 'mem_clk', 'pcie_clk', 4.)
        pair(p+f'memory_result_reg[{bit}]', p+'failed_reg', 'mem_clk', 'pcie_clk', 4.)
    for src, dst, sc, dc, bound in [('request_reg','req_sync_reg[0]','pcie_clk','mem_clk',2.222),
            ('acknowledge_reg','ack_sync_reg[0]','mem_clk','pcie_clk',4.),
            ('mem_ready_sync_reg[2]','ready_sync_reg[0]','mem_clk','pcie_clk',4.)]:
        pair(p+src, p+dst, sc, dc, bound)
    for src in ('ready_mgmt_reg[0]','ready_mgmt_reg[1]','fault_reg'):
        pair('availability/'+src, p+'mem_ready_sync_reg[0]', 'mgmt_clk', 'mem_clk', 2.222)
    observed = {(r['from'],r['to']):r for r in table}
    require(len(table) == len(observed) == len(expected) == 304 and set(observed) == set(expected), 'Loader path identity/coverage')
    for key, (sc, dc, bound) in expected.items():
        r = observed[key]
        require(r['start_clock'] == sc and r['end_clock'] == dc, 'Loader path clock')
        require(abs(float(r['requirement_ns'])-bound) < .001, 'Loader bound changed')


def loader_sync(sync, inter, clocks):
    expected = {f'weight_loader/{chain}_reg[{bit}]':clocks[domain]
        for chain,domain in [('req_sync','mem_clk'),('mem_ready_sync','mem_clk'),('ack_sync','pcie_clk'),('ready_sync','pcie_clk')]
        for bit in range(3)}
    require(len(sync) == 12 and {r['cell'] for r in sync} == set(expected), 'Loader synchronizer coverage')
    for r in sync:
        require(r['clock'] == expected[r['cell']] and r['async_reg'].upper() in ('1','TRUE'), 'Loader ASYNC_REG/clock')
    pairs = {(f'weight_loader/{chain}_reg[{bit}]', f'weight_loader/{chain}_reg[{bit+1}]/D', type)
             for chain in ('req_sync','mem_ready_sync','ack_sync','ready_sync') for bit in range(2) for type in ('max','min')}
    require(len(inter) == 16 and {(r['from'],r['to'],r['type']) for r in inter} == pairs, 'Loader interstage coverage')
    for r in inter:
        clock = expected[r['from']]
        bound = (2.222222 if clock == clocks['mem_clk'] else 4.) if r['type'] == 'max' else 0.
        require(r['start_clock'] == r['end_clock'] == clock and abs(float(r['requirement_ns'])-bound) < .001, 'Loader interstage timing cut')


def hbm_connections(table):
    expected = {}
    constants = dict(awburst=1, awid=0, awlen=0, awsize=5, wstrb=0xffffffff, wlast=1)
    for lane, pc in enumerate(('00','16')):
        for name,width in HBM_WIDTHS.items():
            for bit in range(width):
                expected[(f'weight_loader/h_{name}[{lane*width+bit}]',
                    f'hbm_binding/memory/AXI_{pc}_{name.upper()}'+(f'[{bit}]' if width>1 else ''))] = (
                    (constants[name]>>bit)&1 if name in constants else -1)
    require(len(table) == len(expected) and {(r['loader'],r['native']) for r in table} == set(expected), 'HBM write connectivity coverage')
    for row in table:
        value = expected[(row['loader'],row['native'])]
        require(int(row['constant']) == value, 'HBM constant classification')
        if value >= 0:
            require(row['native_driver'] == ('weight_loader/VCC/P' if value else 'weight_loader/GND/G'), 'HBM constant value')
            require(row['loader_driver'] in ('',row['native_driver']), 'Loader constant driver')
        else:
            require(row['loader_driver'] == row['native_driver'] != '', 'HBM live driver')


def open_cdc_findings(text):
    summary = [(rule,severity,int(count)) for rule,severity,count in re.findall(
        r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',text,re.M)]
    require([r for r in summary if r[1] == 'Critical'] == [('CDC-10','Critical',1)], 'Expected visible readiness CDC-10')
    details = [line for line in text.splitlines() if re.match(r'^\s*\d+\s+CDC-10\s+',line)]
    require(len(details) == 1 and 'availability/fault_reg/C' in details[0]
        and 'weight_loader/mem_ready_sync_reg[0]/D' in details[0], 'Unreviewed critical CDC location')
    return details[0].strip()


def review(synth, applied, reopened):
    results = [closed(root) for root in (synth,applied,reopened)]
    require(results[0]['stage'] == 'synth' and results[1]['checkpoint_sha256'] == sha(synth/'output/synth.dcp')
            and results[2]['checkpoint_sha256'] == sha(applied/'output/weight_bounded.dcp'), 'Checkpoint chain')
    binding = json.loads((synth/'source/WEIGHT_PHYSICAL_BINDING.json').read_text())
    require(binding['functional_attachment_identical'] and binding['simulation_scaffolding_removed'], 'Composition provenance')
    require(sha(synth/'source/ntrpu_hbm_weight_loader.sv') == binding['loader_sha256'] == LOADER_SHA, 'Frozen loader changed')
    require(sha(synth/'source/ntrpu_axi_bridge_hbm_controlled.sv') == binding['physical_top_sha256'], 'Physical top identity')
    require(len(binding['frozen_rtl']) == 22 and all(sha(synth/'source'/n) == h for n,h in binding['frozen_rtl'].items()), 'Frozen RTL changed')
    def table(name, keys=None):
        a, b = rows(applied,name), rows(reopened,name)
        if keys is None:
            require(a == b, 'Reopened table changed '+name)
            return a
        return compare_table(applied,reopened,name,keys)
    domains = table('domain_clocks.tsv')
    clocks = {r['logical']:r['clock'] for r in domains}
    require(len(domains) == 4 and set(clocks) == {'pcie_clk','reader_clk','mem_clk','mgmt_clk'}, 'Clock domain coverage')
    for r in domains:
        require(abs(float(r['period'])-dict(pcie_clk=4.,reader_clk=5.,mem_clk=1000/450,mgmt_clk=10.)[r['logical']]) < .001
                and r['reference'].startswith('BUFG'), 'Native clock period/buffer')
    csr = table('csr_connections.tsv')
    exact_connections([r for r in csr if r['adapter'].startswith('bar_control/')],CSR_WIDTHS,'adapter','control','bar_control/m_','weight_loader/s_')
    exact_connections([r for r in csr if r['adapter'].startswith('weight_loader/')],CSR_WIDTHS,'adapter','control','weight_loader/m_','control/s_')
    require(len(csr) == 224, 'Loader CSR segments')
    bridge = table('bridge_connections.tsv');exact_connections(bridge,BRIDGE_WIDTHS,'bridge','adapter','ntrpu_bridge_i/m_axib_','bar_control/s_')
    lock_constants(table('bridge_constants.tsv'))
    ports = table('physical_ports.tsv')
    require(len(ports) == 74 and not any('load_' in r['port'] or 'pipe_' in r['port'] for r in ports), 'Unexpected external loader/PIPE ports')
    resets = table('loader_clock_reset.tsv')
    require(len(resets) == 7 and {(r['source'],r['consumer']) for r in resets} == {
        ('core/clk','weight_loader/clk'),('core/mem_clk','weight_loader/mem_clk'),
        ('core/client_cold_rst_n','weight_loader/cold_rst_n'),('core/mem_cold_rst_n','weight_loader/mem_cold_rst_n'),
        ('ntrpu_bridge_i/axi_aresetn','weight_loader/bus_rst_n'),('core/mem_clk','hbm_binding/axi_clk'),
        ('core/mem_cold_rst_n','hbm_binding/axi_cold_reset_n')}, 'Loader reset/clock binding')
    hbm = table('loader_hbm_connections.tsv');hbm_connections(hbm)
    paths = table('loader_bounds.tsv');loader_paths(paths,clocks)
    sync = table('loader_synchronizers.tsv');inter = table('loader_interstage.tsv');loader_sync(sync,inter,clocks)
    groups = table('loader_groups.tsv')
    require(len(groups) == 12 and sum(int(r['endpoints']) for r in groups) == 304, 'Loader constraint group coverage')
    require([(r['group'],float(r['skew_ns'])) for r in groups if float(r['skew_ns'])] == [('payload',1.111),('completion',2.)], 'Loader bus-skew contract')
    old = table('paths.tsv');require(len(old) == 1503, 'Prior CDC endpoint coverage')
    require(len(table('interstage.tsv')) == 180 and len(table('synchronizers.tsv')) == 147, 'Prior synchronizer coverage')
    require(len(table('reset_arcs.tsv')) == 18, 'Prior reset exceptions changed')
    snapshots = table('snapshot_bounds.tsv');snapshot_bounds(snapshots)
    require(len(snapshots) == 390 and len(table('snapshot_interstage.tsv')) == 32, 'Snapshot coverage')
    table('cold_reset_bindings.tsv');table('telemetry_capture_sources.tsv');table('loader_readiness_fanin.tsv')
    for root in (applied,reopened):
        require(float(rows(root,'lock_path.tsv')[-1]['requirement_ns']) == 10., 'Native LOCKED bound')
        props = rows(root,'lock_arc_properties.tsv')
        for key in ('IS_DISABLED','IS_USER_DISABLED'):
            found = [r for r in props if r['property'] == key]
            require(len(found) == 4 and all(r['value'] == ('1' if key == 'IS_DISABLED' else '0') for r in found), 'LOCKED arc altered')
    before = expand_xdc((applied/'output/before.xdc').read_text())
    after = expand_xdc((applied/'output/effective.xdc').read_text())
    require(not before-after, 'Original vendor constraints removed')
    added = after-before
    require(Counter({kind:sum(n for line,n in added.items() if line.startswith(kind+' '))
        for kind in ('set_max_delay','set_bus_skew','set_false_path')}) == dict(set_max_delay=55,set_bus_skew=10,set_false_path=18), 'Scoped constraint counts')
    require(not any(line.startswith(('set_clock_groups ','set_multicycle_path ','set_disable_timing ','create_waiver ')) for line in added), 'Unexpected broad exception/waiver')
    require(all('-datapath_only' in line for line in added if line.startswith('set_max_delay ')), 'Missing datapath-only bound')
    def pin_query(name):
        return '[get_pins -quiet '+('{'+name+'}' if '[' in name else name)+']'
    expected_reset = Counter('set_false_path -from '+pin_query(r['from'])+' -to '+pin_query(r['to']) for r in rows(applied,'reset_arcs.tsv'))
    require(Counter({s:n for s,n in added.items() if s.startswith('set_false_path ')}) == expected_reset, 'Async reset exception scope changed')
    require(after == expand_xdc((reopened/'output/effective.xdc').read_text()) == expand_xdc((reopened/'output/before.xdc').read_text()), 'DCP changed constraint semantics')
    cdc = {}; timing = {}; critical = {}; drc = {}; readiness_finding = {}; slack = {}
    for label,root in zip(('synthesis','apply','reopen'),(synth,applied,reopened)):
        cdc_text = (root/'output/cdc.rpt').read_text()
        readiness_finding[label] = open_cdc_findings(cdc_text)
        cdc[label] = [dict(rule=rule,severity=severity,count=int(count)) for rule,severity,count in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',cdc_text,re.M)]
        coverage = (root/'output/check_timing.rpt').read_text()
        timing[label] = {name:int(count) for name,count in re.findall(r'^\d+\. checking (\w+) \((\d+)\)$',coverage,re.M)}
        require(all(timing[label][key] == 0 for key in ('no_clock','multiple_clock','loops','latch_loops')), 'Native clock/topology failure')
        anchor = '4. checking unconstrained_internal_endpoints'
        start = coverage.index(anchor,coverage.index(anchor)+1)
        pins = [line.strip() for line in coverage[start:coverage.index('5. checking no_input_delay',start)].splitlines() if '/' in line]
        require(len(pins) == 1154 and all(pin.startswith('hbm_binding/memory/') and 'xsdb' in pin.lower() for pin in pins), 'Unexpected unconstrained endpoint')
        require(timing[label]['no_input_delay'] == 2, 'External timing coverage changed')
        timing_text = (root/'output/timing.rpt').read_text()
        section = timing_text[timing_text.index('Design Timing Summary'):]
        values = next(line.split() for line in section.splitlines() if re.match(r'^\s*-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+',line))
        require(len(values) == 12, 'Timing summary format')
        slack[label] = dict(wns_ns=float(values[0]),whs_ns=float(values[4]),wpws_ns=float(values[8]),timing_met=False)
        require('Timing constraints are not met.' in section, 'Unexpected timing qualification')
        critical[label] = re.findall(r'^CRITICAL WARNING:.*$',(root/'tool.log').read_text(errors='replace'),re.M)
        require(all('[Constraints 18-515]' in line and 'clock_generator/inst/mmcme4_adv_inst/LOCKED' in line for line in critical[label]), 'Unreviewed critical tool warning')
        drc[label] = [dict(rule=rule,severity=severity,count=int(count)) for rule,severity,count in re.findall(r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Error)\s*\| [^|]+\| (\d+)\s*\|$',(root/'output/drc.rpt').read_text(),re.M)]
        require(drc[label] == [dict(rule=rule,severity='Critical Warning',count=1) for rule in ('AVAL-326','NSTD-1','UCIO-1')], 'Unreviewed native physical DRC')
    require(cdc['apply'] == cdc['reopen'] and timing['apply'] == timing['reopen'], 'CDC/timing reopen mismatch')
    negative = []
    def rejects(name, fn):
        try:fn()
        except (ValueError,KeyError):negative.append(name)
        else:raise ValueError('Negative control escaped '+name)
    rejects('missing_loader_path',lambda:loader_paths(paths[:-1],clocks))
    bad=copy.deepcopy(paths);bad[0]['requirement_ns']='100'
    rejects('loosened_data_bound',lambda:loader_paths(bad,clocks))
    bad_clock=copy.deepcopy(paths);bad_clock[0]['start_clock']='other_clock'
    rejects('wrong_source_clock',lambda:loader_paths(bad_clock,clocks))
    rejects('missing_failure_reduction',lambda:loader_paths([r for r in paths if r['to']!='weight_loader/failed_reg/D'],clocks))
    rejects('missing_readiness_source',lambda:loader_paths([r for r in paths if r['from']!='availability/fault_reg/C'],clocks))
    rejects('missing_hold_check',lambda:loader_sync(sync,inter[:-1],clocks))
    marked=copy.deepcopy(sync);marked[0]['async_reg']='FALSE'
    rejects('missing_async_reg',lambda:loader_sync(marked,inter,clocks))
    rejects('missing_hbm_wire',lambda:hbm_connections(hbm[:-1]))
    wires=copy.deepcopy(hbm);next(r for r in wires if r['constant']=='-1')['native_driver']='other/Q'
    rejects('wrong_hbm_driver',lambda:hbm_connections(wires))
    constants=copy.deepcopy(hbm);next(r for r in constants if r['constant']=='0')['native_driver']='weight_loader/VCC/P'
    rejects('wrong_hbm_constant',lambda:hbm_connections(constants))
    cdc_text = (applied/'output/cdc.rpt').read_text()
    rejects('hidden_readiness_critical',lambda:open_cdc_findings(cdc_text.replace('CDC-10','REMOVED')))
    rejects('wrong_readiness_sink',lambda:open_cdc_findings(cdc_text.replace('weight_loader/mem_ready_sync_reg[0]/D','unrelated/D')))
    return dict(status='WEIGHT_NATIVE_CONSTRAINTS_REVIEWED_WITH_OPEN_FINDINGS',
        frozen_rtl_files=22,loader_sha256=LOADER_SHA,physical_top_sha256=binding['physical_top_sha256'],
        checkpoint_sha256=sha(applied/'output/weight_bounded.dcp'),physical_port_bits=74,csr_connection_bits=224,
        native_bridge_connection_bits=len(bridge),native_hbm_write_connection_bits=len(hbm),
        loader_constrained_paths=len(paths),loader_unique_endpoints=len({r['to'] for r in paths}),loader_sync_registers=12,
        loader_interstage_pairs=8,loader_max_delay_groups=12,loader_skew_groups=2,
        loader_min_estimated_slack_ns=min(float(r['slack_ns']) for r in paths),
        preserved_prior_bounded_endpoints=1503,preserved_snapshot_bounded_endpoints=390,
        cdc_findings=cdc,unresolved_readiness_cdc=readiness_finding,timing_coverage=timing,timing_estimates=slack,tool_critical_warnings=critical,physical_drc_findings=drc,
        negative_controls=negative,native_integrated_synthesis_completed=True,
        functional_rtl_changed=False,physical_serial_link_tested=False,
        global_cdc_qualified=False,routed_timing_qualified=False,board_qualified=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('synthesis','apply','reopen'):
        p.add_argument(name,type=Path)
    p.add_argument('--output',type=Path)
    a = p.parse_args();text = json.dumps(review(a.synthesis,a.apply,a.reopen),indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(text)
    else:print(text,end='')
