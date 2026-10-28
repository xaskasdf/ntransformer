"""Audit reopened dual-receiver routing; retain the external-reset CDC warning."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

from check_ntrpu_composition import common, digest
from check_ntrpu_dual_physical import table
from check_ntrpu_multibuffer_credits import require


PREFIX = 'core/startup/'


def structural(registers, cones):
    regs = {r['cell']: r for r in registers}
    pins = {(r['cell'], r['pin']): r for r in cones}
    require(len(regs) == len(registers) == 148, 'Startup register inventory')
    groups = {}
    for r in registers:
        if r['async_reg'] == '1':
            match = re.fullmatch(r'(.+)/(release_ff|arststages_ff|syncstages_ff)_reg\[([012])\]', r['cell'])
            require(match is not None, 'Unexpected synchronizer')
            groups.setdefault((match[1], match[2]), {})[int(match[3])] = r
    require(Counter(k[1] for k in groups) == dict(release_ff=15, arststages_ff=4, syncstages_ff=6), 'Synchronizer groups')
    for (base, kind), stages in groups.items():
        require(set(stages) == {0, 1, 2}, 'Three stages required')
        first = stages[0]
        clock = 'mgmt_clk'
        if base.endswith(('source_reset_cdc', 'admission_cdc', 'source_hold/clear_cdc', 'source_probe/outward')):
            clock = 'source_clk'
        for g in range(2):
            if base.endswith((f'memory[{g}].reset_cdc', f'memory[{g}].reset_hold/clear_cdc', f'memory[{g}].probe/outward')):
                clock = f'memory{g}_clk'
        require(len({r['loc'] for r in stages.values()}) == 1 and first['loc'].startswith('SLICE_'), 'Chain co-location')
        for i, r in stages.items():
            require(r['ref'] == {'release_ff': 'FDCE', 'arststages_ff': 'FDPE', 'syncstages_ff': 'FDRE'}[kind]
                    and r['init'] == "1'b0" and r['clock'] == clock, 'Chain type, INIT or clock')
            require(pins[r['cell'], 'CE']['driver_ref'] == 'VCC', 'Unconditional synchronizer enable')
            if i:
                require(r['data_driver'] == stages[i-1]['cell']+'/Q', 'Direct stage-to-stage connection')
            elif kind != 'syncstages_ff':
                require(pins[r['cell'], 'D']['driver_ref'] == ('VCC' if kind == 'release_ff' else 'GND'), 'Chain initial input')
            else:
                expected = base.rsplit('/', 1)[0] + ('/request_reg/Q' if base.endswith('/outward') else '/outward/syncstages_ff_reg[2]/Q')
                require(r['data_driver'] == expected, 'Clock probe toggle loop')
            if kind == 'release_ff':
                c = pins[r['cell'], 'CLR']
                require(c['ports'] == '' and c['driver'].endswith('/Q') and c['driver_ref'].startswith('FD'), 'Registered reset source')
                require(c['driver'] == first['async_drivers'], 'Common chain reset')
                positive = base.endswith(('admission_cdc', 'source_reset_cdc', '.reset_cdc'))
                require(c['pin_inverted'] == ('0' if positive else '1'), 'Reset polarity')
                require(c['driver_clock'] == regs[c['driver'][:-2]]['clock'], 'Reset driver domain')
            elif kind == 'arststages_ff':
                c = pins[r['cell'], 'PRE']
                require(c['pin_inverted'] == '0', 'XPM reset polarity')
                if base.endswith('/management_reset'):
                    require(c['ports'] == 'external_reset_async' and c['driver'] == c['driver_ref'] == '', 'External reset must reach FDPE directly')
                else:
                    require(c['driver'] == PREFIX+'clear_counters_reg/Q' and c['driver_ref'] == 'FDPE', 'Reset hold clear source')
    opened = regs[PREFIX+'opened_reg']
    require(opened['ref'] == 'FDRE' and opened['init'] == "1'b0" and opened['clock'] == 'mgmt_clk'
            and opened['async_pin'] == '', 'Persistent opened register')
    return dict(custom_release_chains=15, xpm_reset_chains=4, xpm_toggle_chains=6,
                synchronizer_registers=75, startup_registers=148, each_chain_in_one_slice=True)


def canonical_cdc(text):
    result = Counter(); source = destination = None
    for line in text.splitlines():
        if line.startswith('Source Clock:'): source = line
        elif line.startswith('Destination Clock:'): destination = line
        elif re.match(r'^\s+\d+\s+CDC-', line):
            result[(source, destination, ' '.join(line.split()[1:]))] += 1
    return result


def constraints(xdc, registers):
    require(not re.search(r'^set_clock_groups', xdc, re.M), 'Broad clock exception')
    context = ''; counts = Counter(); waivers = Counter(); targets = []
    for line in xdc.splitlines():
        if line.startswith('current_instance '): context = line.split(' ', 1)[1].strip('{}')
        if line.startswith(('set_false_path', 'set_max_delay', 'set_bus_skew')):
            owner = 'ip' if '/cdc/inst' in context else 'xpm' if context.startswith(PREFIX) else 'custom'
            counts[owner, line.split()[0]] += 1
            if owner == 'custom':
                require(line.startswith('set_false_path -to [get_pins -of_objects {{') and line.endswith(' -filter REF_PIN_NAME==CLR]'), 'Only CLR pins may be excluded')
                cells = re.findall(r'\{(core/startup/[^{}]+)\}', line)
                require(len(cells) == 3, 'Reset constraint stages'); targets += cells
            elif owner == 'xpm':
                require(line in ('set_false_path -through [get_ports -scoped_to_current_instance -no_traverse src_arst]',
                                'set_false_path -to [get_cells {syncstages_ff_reg[0]}]'), 'XPM constraint shape')
            else:
                require(context.startswith('core/receiver/memory_port['), 'Vendor constraint ownership')
        if line.startswith('create_waiver '):
            require('/cdc/inst' in context, 'No custom CDC waiver')
            waivers[re.search(r'-id \{([^}]+)\}', line)[1]] += 1
    expected = {('ip','set_false_path'):44, ('ip','set_max_delay'):20, ('ip','set_bus_skew'):20,
                ('xpm','set_false_path'):10, ('custom','set_false_path'):15}
    require(counts == expected, 'Constraint inventory')
    require(Counter(targets) == Counter(r['cell'] for r in registers if '/release_ff_reg[' in r['cell']), 'Exact custom reset targets')
    require(waivers == dict(**{'CDC-6':20, 'CDC-10':2, 'CDC-11':2, 'CDC-15':2, 'LUTAR-1':2}), 'Vendor waiver inventory')
    return {'/'.join(k):v for k,v in counts.items()}, dict(waivers)


def review(root, physical):
    record, console = common(root, 'NTRPU_DUAL_PHYSICAL_AUDIT_EXPORTED')
    folder = root/'output/suite'; original = physical/'output/suite'
    require(digest(root/'inputs/evidence/routed.dcp') == digest(original/'routed.dcp'), 'Routed checkpoint provenance')
    require(not re.search(r'^CRITICAL WARNING:', console, re.M), 'Critical audit warning')
    registers = table(folder/'startup_registers.tsv'); cones = table(folder/'startup_input_cones.tsv')
    require(registers == table(original/'startup_registers.tsv'), 'Reopened register structure changed')
    result = structural(registers, cones)
    cdc_reports = [(p/n).read_text() for p in (folder, original) for n in ('cdc.rpt', 'cdc_unwaived.rpt')]
    cdc = canonical_cdc(cdc_reports[0])
    require(sum(cdc.values()) == 65 and all(canonical_cdc(t) == cdc for t in cdc_reports), 'CDC reports agree with and without waivers')
    summary = re.findall(r'^(CDC-\d+)\s+(\w+)\s+(\d+)\s+', cdc_reports[0], re.M)
    require(summary == [('CDC-3','Info','26'), ('CDC-9','Info','38'), ('CDC-26','Warning','1')], 'CDC diagnostics')
    warnings = [row for row in cdc if ' Warning ' in row[2]]
    require(len(warnings) == 1 and warnings[0][2].endswith('external_reset_async core/startup/management_reset/arststages_ff_reg[0]/PRE'), 'External reset warning identity')
    for p in (folder, original):
        require('No ignored timing exceptions found.' in (p/'ignored_exceptions.rpt').read_text(), 'Ignored exception')
        counts, waivers = constraints((p/'implemented.xdc').read_text(), registers)
    skew = (folder/'bus_skew.rpt').read_text().rsplit('1. Bus Skew Report Summary',1)[1].split('2. Bus Skew Report Per Constraint')[0]
    rows = re.findall(r'^\s*(Slow|Fast)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s*$', skew, re.M)
    require(len(rows) == 20 and all(float(r[1]) == 5 and float(r[3]) >= 0 for r in rows), 'Bus skew constraints')
    interaction = (folder/'clock_interaction.rpt').read_text()
    pairs = table(folder/'cross_clock_classification.tsv')
    clocks = {'source_clk','mgmt_clk','memory0_clk','memory1_clk'}
    require({(r['source'],r['destination']) for r in pairs} == {(a,b) for a in clocks for b in clocks if a != b} and len(pairs) == 12, 'Cross-clock inventory')
    for r in pairs:
        a,b = r['source'],r['destination']
        row = re.search(r'^'+a+r'\s+'+b+r'\s+(.+)$', interaction, re.M)
        if a.startswith('memory') and b.startswith('memory'):
            require(r['paths_returned'] == '0' and row is None, 'Independent memories have no direct path')
        elif 'mgmt_clk' in (a,b):
            require(r['paths_returned'] == '1' and r['slack'] == r['requirement'] == ''
                    and row is not None and row[1].strip().endswith('Ignored              False Path'), 'Management crossing classification')
        else:
            require(float(r['slack']) >= 0 and float(r['requirement']) == 5 and '/cdc/inst/' in r['endpoint']
                    and row is not None and 'Max Delay Datapath Only' in row[1], 'Constrained Gray pointer crossing')
    bindings = (folder/'clock_binding.txt').read_text()
    require(bindings.startswith('CONVERTERS=2\n') and bindings.count('ROUTE=ROUTED') == 4, 'Routed clock bindings')
    for g in range(2):
        base = f'core/receiver/memory_port[{g}].port/cdc/'
        require(base+'s_axi_aclk=source_clk' in bindings and base+f'm_axi_aclk=memory{g}_clk' in bindings, 'Converter clocks')
    # The exported subcell is not a reusable module, but its primitive register
    # and LUT provide direct evidence that opened cannot be cleared after set.
    netlist = (original/'routed_startup_funcsim.v').read_text()
    require(re.search(r'opened_reg\s+\(\.C\(mgmt_clk\),\s+\.CE\(1\x27b1\),\s+\.D\([^)]+\),\s+\.Q\(opened\),\s+\.R\(1\x27b0\)\)', netlist), 'Opened has no reset')
    cone = next(r for r in cones if r['cell'] == PREFIX+'opened_reg' and r['pin'] == 'D')
    require(cone['driver_ref'] == 'LUT6' and cone['input_drivers'].startswith(PREFIX+'opened_reg/Q '), 'Opened feedback input')
    value = int(cone['driver_init'].split("'h")[1],16)
    require(all((value >> i) & 1 for i in range(1,64,2)), 'Opened is absorbing when set')
    negatives = []
    for label, field, value in [('initial_value','init',"1'b1"), ('destination_clock','clock','wrong'),
                                ('chain_link','data_driver','unrelated/Q'), ('placement','loc','')]:
        changed = copy.deepcopy(registers)
        item = next(r for r in changed if r['cell'] == PREFIX+'admission_cdc/release_ff_reg[1]');item[field] = value
        try: structural(changed,cones)
        except (ValueError,AssertionError,RuntimeError): negatives.append(label)
        else: raise ValueError('Missed negative control '+label)
    changed = copy.deepcopy(cones)
    next(r for r in changed if r['cell'] == PREFIX+'management_reset/arststages_ff_reg[0]' and r['pin'] == 'PRE')['driver_ref'] = 'LUT1'
    try: structural(registers, changed)
    except (ValueError,AssertionError,RuntimeError): negatives.append('external_reset_logic')
    else: raise ValueError('Missed external reset negative control')
    result.update(status='STRUCTURAL_CDC_PASS_EXTERNAL_RESET_WARNING_RETAINED', cdc_summary=summary,
        external_reset_warning=list(warnings[0]), exceptions=counts, vendor_waiver_definitions=waivers,
        bus_skew_constraints=20, min_bus_skew_slack_ns=min(float(r[3]) for r in rows),
        max_bus_skew_ns=max(float(r[2]) for r in rows), cross_clock_classification=pairs,
        negative_controls=negatives, opened_latch_absorbing=True,
        seconds=record['seconds'], cpus=record['cpus'], nice=record['nice'],
        routed_checkpoint_sha256=digest(original/'routed.dcp'),
        limitations=['CDC-26 external-reset diagnostic retained without waiver; tool classification remains open',
                    'Structural report and digital simulation do not prove analog metastability or MTBF',
                    'OOC external IO delays and physical clock-source placement remain unqualified',
                    'Encrypted converter implementation uses vendor-scoped timing constraints and waivers'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path);parser.add_argument('--physical',type=Path,required=True)
    args = parser.parse_args();result = review(args.run,args.physical)
    result['checker_sha256'] = digest(Path(__file__))
    with (args.run/'REVIEW.json').open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))
