"""Review scoped native CDC constraints; no claim of routing or global closure."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def closed(root, mode):
    manifest = json.loads((root / 'FILES.json').read_text())
    require(all(sha(root/n) == h for n, h in manifest.items()), 'Manifest mismatch')
    r = json.loads((root/'RESULT.json').read_text())
    require(r['status'] == 'HBM_NATIVE_JOIN_EXPORTED' and r['stage'] == 'inspect'
            and r['returncode'] == 0 and not r['timed_out'] and r['sources_unchanged'], 'Incomplete tool run')
    require(all(sha(root/'source'/n) == h for n, h in r['sources'].items()), 'Source mismatch')
    require((root/'output/STATUS.txt').read_text().strip() == f'NATIVE_CONSTRAINTS_{mode}_EXPORTED_PENDING_REVIEW', 'Export gate')
    require(not re.search(r'(?im)^\s*(?:error|fatal):', (root/'tool.log').read_text()), 'Tool error')
    return r


def xdc_commands(text):
    # Vivado's write_xdc emits commands on individual logical lines.
    text = re.sub(r'\\\r?\n\s*', ' ', text)
    return Counter(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#'))


def expanded_capture_aliases(commands):
    # Only the eight capture-query aliases validated by serialized_bindings.
    # DCP reopening writes their expressions inline; no Tcl is evaluated here.
    aliases = {}
    rest = Counter()
    for line,count in commands.items():
        m = re.fullmatch(r'set (_xlnx_shared_i[2-9]) (.+)',line)
        if m:
            require(count == 1 and m[1] not in aliases, 'Duplicate capture alias')
            aliases[m[1]] = m[2]
        else:
            rest[line] = count
    out = Counter()
    def replace(m):
        require(m[1] in aliases, 'Undefined capture alias')
        return aliases[m[1]]
    for line,count in rest.items():
        out[re.sub(r'\$(_xlnx_shared_i[2-9])\b',replace,line)] += count
    return out


def serialized_bindings(e, reset_pairs):
    """Check Vivado's query aliases as well as the actual added constraints."""
    added = e['added_commands']
    commands = Counter(line.split()[0] for line, count in added.items() for _ in range(count))
    require(commands == dict(set_max_delay=33, set_bus_skew=8, set_false_path=18,
                             set=8, current_instance=1), 'Unexpected added constraints')
    require(added['current_instance -quiet'] == 1, 'Unexpected hierarchy scope')
    aliases = {}
    for line, count in added.items():
        if line.startswith('set '):
            m = re.fullmatch(r'set (_xlnx_shared_i[2-9]) (.+)', line)
            require(m is not None and count == 1 and m[1] not in aliases, 'Unexpected XDC alias')
            aliases[m[1]] = m[2]
    expected_skew = Counter()
    for lane in range(2):
        for j, fifo in enumerate(('commands', 'returns')):
            n = 2 + 4*lane + 2*j
            prefix = f'core/transport/lanes[{lane}].{fifo}/'
            pattern = re.escape(prefix) + r'output_data_reg\[[0-9]+\]$'
            target = '[get_pins -of_objects [get_cells -quiet -hier -regexp {^'+pattern+'}] -filter {REF_PIN_NAME == D}]'
            require(aliases[f'_xlnx_shared_i{n+1}'] == target, 'Capture alias target')
            source = aliases[f'_xlnx_shared_i{n}']
            m = re.fullmatch(r'\[get_cells \[list ((?:\{[^{}]+\}\s*)+)\]\]', source)
            require(m is not None, 'Capture alias source syntax')
            cells = re.findall(r'\{([^{}]+)\}', m[1])
            observed = {r['from'].rsplit('/',1)[0] for r in e['captures'] if r['to'].startswith(prefix)}
            require(len(cells) == len(set(cells)) and set(cells) == observed, 'Capture alias source scope')
            for a,b in (('wg','ws1'),('rg','rs1')):
                def gray(reg):
                    return '[get_cells -quiet -hier -regexp {^'+re.escape(prefix+reg)+r'_reg\[[0-2]\]$}]'
                expected_skew['set_bus_skew -from '+gray(a)+' -to [get_pins -of_objects '+gray(b)+' -filter {REF_PIN_NAME == D}] 2.000'] += 1
    require(Counter({s:c for s,c in added.items() if s.startswith('set_bus_skew ')}) == expected_skew, 'Serialized skew scope/value')
    def pin(s):
        return '[get_pins -quiet '+('{'+s+'}' if '[' in s else s)+']'
    expected_reset = Counter('set_false_path -from '+pin(a)+' -to '+pin(b) for a,b in reset_pairs)
    require(Counter({s:c for s,c in added.items() if s.startswith('set_false_path ')}) == expected_reset, 'Serialized reset scope')


CONTROL_PAIRS = [
    ('transport/request_reg', 'transport/echo_r_reg[0]/D', 'mgmt_clk', 'reader_clk'),
    ('transport/echo_r_reg[2]', 'transport/echo_m_reg[0]/D', 'reader_clk', 'mgmt_clk'),
    ('transport/reader_failed_reg', 'transport/failed_m_reg[0]/D', 'reader_clk', 'mgmt_clk'),
    ('transport/armed_reg', 'transport/armed_p_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('transport/management_fault_reg', 'transport/failed_p_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('transport/management_fault_reg', 'transport/failed_r_reg[0]/D', 'mgmt_clk', 'reader_clk'),
    ('transport/stop_held_reg', 'transport/stop_r_reg[0]/D', 'pcie_clk', 'reader_clk'),
    ('reader/startup/started_reg', 'transport/started_m_reg[0]/D', 'reader_clk', 'mgmt_clk'),
    ('endpoint/guard/request_reg', 'endpoint/guard/echo_a_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('endpoint/guard/echo_a_reg[2]', 'endpoint/guard/echo_m_reg[0]/D', 'pcie_clk', 'mgmt_clk'),
    ('endpoint/guard/armed_reg', 'endpoint/guard/armed_a_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('memory_watch/request_reg', 'memory_watch/echo_mem_reg[0]/D', 'mgmt_clk', 'mem_clk'),
    ('memory_watch/echo_mem_reg[2]', 'memory_watch/echo_m_reg[0]/D', 'mem_clk', 'mgmt_clk'),
    ('memory_watch/armed_reg', 'memory_watch/armed_p_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('memory_watch/fault_reg', 'memory_watch/failed_p_reg[0]/D', 'mgmt_clk', 'pcie_clk'),
    ('memory_watch/fault_reg', 'memory_watch/failed_r_reg[0]/D', 'mgmt_clk', 'reader_clk'),
    ('reader/channels[0].adapter/tracker/fault_reg', 'reader/startup/fault_meta_reg[0]/D', 'mem_clk', 'reader_clk'),
    ('reader/channels[1].adapter/tracker/fault_reg', 'reader/startup/fault_meta_reg[1]/D', 'mem_clk', 'reader_clk'),
    ('reader/startup/memory_release_reg[2]', 'reader/startup/ready_meta_reg[2]/D', 'mem_clk', 'reader_clk'),
]


def semantic(e):
    clocks = {r['logical']: r for r in e['clocks']}
    periods = dict(pcie_clk=4., reader_clk=5., mgmt_clk=10., mem_clk=1000/450)
    require(len(e['clocks']) == 4 and set(clocks) == set(periods), 'Clock coverage')
    for k, period in periods.items():
        r = clocks[k]
        require(abs(float(r['period'])-period) <= .001 and r['reference'].startswith('BUFG'), 'Clock period/buffer')
        prefix = dict(pcie_clk='ntrpu_bridge_i/', reader_clk='clock_generator/', mem_clk='clock_generator/', mgmt_clk='management_buffer/')[k]
        require(r['driver'].startswith(prefix), 'Native clock origin')
    expected = {}
    for lane in range(2):
        for fifo, width, wc, rc in [('commands', 136, 'pcie_clk', 'reader_clk'), ('returns', 593, 'reader_clk', 'pcie_clk')]:
            p = f'core/transport/lanes[{lane}].{fifo}/'
            for a, b, sc, dc in [('wg','ws1',wc,rc), ('rg','rs1',rc,wc)]:
                for bit in range(3):
                    expected[p+f'{b}_reg[{bit}]/D'] = ('gray', p+f'{a}_reg[{bit}]/C', sc, dc, 4.)
            for bit in range(width):
                expected[p+f'output_data_reg[{bit}]/D'] = ('capture', p, wc, rc, 4.)
    for src, dst, sc, dc in CONTROL_PAIRS:
        expected['core/'+dst] = ('control', 'core/'+src+'/C', sc, dc, 2.222 if dc == 'mem_clk' else 4.)
    for i in range(2):
        expected[f'core/reader/startup/ready_meta_reg[{i}]/D'] = ('control', f'availability/ready_mgmt_reg[{i}]/C', 'mgmt_clk', 'reader_clk', 4.)
    require(len(expected) == 1503, 'Internal reviewer coverage')
    paths = {r['to']: r for r in e['paths']}
    require(len(e['paths']) == len(paths) == 1503 and set(paths) == set(expected), 'Endpoint coverage')
    for to, (kind, src, sc, dc, bound) in expected.items():
        r = paths[to]
        require(r['kind'] == kind and abs(float(r['requirement_ns'])-bound) <= .001, 'Missing/overridden data bound')
        require(r['start_clock'] == clocks[sc]['clock'] and r['end_clock'] == clocks[dc]['clock'], 'Path clock mismatch')
        require(r['from'].startswith(src) if kind == 'capture' else r['from'] == src, 'Path source mismatch')
    bindings = e['bindings']
    require(len(bindings) == 33 and len({r['name'] for r in bindings}) == 33, 'Binding coverage')
    require(Counter(r['kind'] for r in bindings) == dict(gray=8, capture=4, control=21), 'Binding kinds')
    for r in bindings:
        expected_count = 3 if r['kind'] == 'gray' else 1 if r['kind'] == 'control' else 136 if '_commands_' in r['name'] else 593
        require(int(r['sources']) == int(r['endpoints']) == expected_count, 'Binding cardinality')
        require(float(r['skew_ns']) == (2. if r['kind'] == 'gray' else 0.), 'Gray skew bound')
    captures = {r['to']: r for r in e['captures']}
    capture_targets = {n for n, v in expected.items() if v[0] == 'capture'}
    require(len(e['captures']) == 1458 and set(captures) == capture_targets, 'Capture topology coverage')
    for r in captures.values():
        require(r['leaf_loads'] == '1' and r['driver_ref'] in ('RAMD32','RAMS32','RAMD64E','RAMS64E'), 'Capture bypass/type')
        require(r['from'].rsplit('/',1)[0] == paths[r['to']]['from'].rsplit('/',1)[0], 'Capture source identity')
    sync = {r['cell']: r for r in e['sync']}
    require(len(e['sync']) == len(sync) == 147 and all(r['async_reg'].upper() in ('1','TRUE') for r in sync.values()), 'Synchronizer coverage/attribute')
    pairs = {}
    for r in e['interstage']:
        require(r['from'] in sync and r['to'] in sync and r['start_clock'] == r['end_clock'] == sync[r['from']]['clock'], 'Interstage identity/clock')
        key = (r['from'],r['to']); pairs.setdefault(key,[]).append(r['type'])
        if r['type'] == 'max':
            period = next(float(c['period']) for c in clocks.values() if c['clock'] == r['start_clock'])
            require(abs(float(r['requirement_ns'])-period) <= .001, 'Interstage setup requirement changed')
        else:
            require(r['type'] == 'min' and abs(float(r['requirement_ns'])) <= .001, 'Interstage hold requirement changed')
    require(len(e['interstage']) == 180 and len(pairs) == 90 and all(sorted(v) == ['max','min'] for v in pairs.values()), 'Interstage path coverage')
    reset_pairs = set()
    for i in range(3):
        for n in range(3):
            reset_pairs.add((f'core/endpoint/guard/availability[{n}].release_m_reg[2]/C', f'core/endpoint/guard/availability[{n}].release_a_reg[{i}]/CLR'))
        reset_pairs.add(('core/endpoint/guard/fault_reg/C', f'core/endpoint/guard/stop_a_reg[{i}]/PRE'))
        for n, reg in [(0,'user_lnk_up_reg'),(1,'user_reset_reg')]:
            reset_pairs.add((f'ntrpu_bridge_i/inst/pcie4c_ip_i/inst/{reg}/C', f'core/endpoint/guard/availability[{n}].release_m_reg[{i}]/CLR'))
    require(len(e['resets']) == 18 and {(r['from'],r['to']) for r in e['resets']} == reset_pairs, 'Async control exception scope')
    for r in e['resets']:
        chain = []; inits = []
        m = re.fullmatch(r'(core/endpoint/guard/availability\[([0-2])\]\.release_([am]))_reg\[[0-2]\]/CLR', r['to'])
        if m:
            chain = [m[1]+'[2]_i_1']; inits = ["2'h1"]
            if (m[2], m[3]) == ('1', 'm'):
                chain += ['ntrpu_bridge_i/inst/axi_aresetn_inst', 'ntrpu_bridge_i/inst/axi_aresetn_inferred_i_1']
                inits += ["2'h2", "2'h1"]
        source = r['from'][:-1]+'Q'
        require(r['source_q'] == source and r['direct_driver'] == (chain[0]+'/O' if chain else source), 'Async control driver identity')
        require(r['lut_cells'] == '|'.join(chain) and r['lut_inits'] == '|'.join(inits)
                and int(r['inversions']) == inits.count("2'h1"), 'Async control chain/polarity')
    require(len(e['raw_lock']) == 1 and e['raw_lock'][0]['bounded_by_this_stage'] == '0', 'Raw LOCK scope must remain explicit')
    require(e['raw_lock'][0]['first_stage'] == 'availability/lock_sync_reg[0]/D' and e['raw_lock'][0]['source'].endswith('/LOCKED'), 'Raw LOCK identity')
    require(not e['removed_commands'], 'Original constraints removed')
    serialized_bindings(e, reset_pairs)
    require(all('-datapath_only' in line for line in e['added_commands'] if line.startswith('set_max_delay')), 'Data bound must be datapath-only')
    require(e['cdc'].get('CDC-10',0) == 0, 'Critical CDC regression')


def evidence(root):
    out = root/'output'
    e = {k:table(out/n) for k,n in dict(clocks='domain_clocks.tsv', paths='paths.tsv', bindings='bindings.tsv',
        captures='capture_topology.tsv', sync='synchronizers.tsv', interstage='interstage.tsv', resets='reset_arcs.tsv', raw_lock='raw_lock.tsv').items()}
    before = xdc_commands((out/'before.xdc').read_text()); after = xdc_commands((out/'after.xdc').read_text())
    e.update(added_commands=after-before,removed_commands=before-after,
        cdc={name:int(count) for name,count in re.findall(r'(?m)^(CDC-\d+)\s+\w+\s+(\d+)\s+', (out/'cdc.rpt').read_text())})
    return e


def review(applied, reopened, self_test=False):
    a = closed(applied,'apply'); b = closed(reopened,'reopen')
    require(a['checkpoint_sha256'] == '4613dbb14df1eff881c3e5294825d6398a9af37e3f100f37ab52cd017ab2a89e', 'Native donor identity')
    require(b['checkpoint_sha256'] == sha(applied/'output/constrained.dcp'), 'Reopened checkpoint identity')
    e = evidence(applied); semantic(e)
    r = evidence(reopened)
    require(not r['added_commands'] and not r['removed_commands'], 'Reopen changed constraints')
    applied_xdc = xdc_commands((applied/'output/after.xdc').read_text())
    reopened_xdc = xdc_commands((reopened/'output/after.xdc').read_text())
    require(expanded_capture_aliases(applied_xdc) == expanded_capture_aliases(reopened_xdc), 'Reopened constraint mismatch')
    for key in ('clocks','paths','bindings','captures','sync','interstage','resets','raw_lock','cdc'):
        require(r[key] == e[key], 'Reopened evidence changed: '+key)
    rejected = []
    if self_test:
        def changed_alias(x):
            commands = x['added_commands']
            line = next(s for s in commands if s.startswith('set _xlnx_shared_i2 '))
            count = commands.pop(line)
            commands['set _xlnx_shared_i2 [get_cells *]'] = count
        mutations = dict(
            missing_endpoint=lambda x:x['paths'].pop(),
            wrong_source=lambda x:x['paths'][0].update({'from':'unrelated/C'}),
            overridden_bound=lambda x:x['paths'][0].update(requirement_ns='100.0'),
            wrong_clock=lambda x:x['clocks'][0].update(period='8.0'),
            wrong_skew=lambda x:x['bindings'][0].update(skew_ns='4.0'),
            capture_bypass=lambda x:x['captures'][0].update(leaf_loads='2'),
            missing_synchronizer=lambda x:x['sync'].pop(),
            lost_hold_check=lambda x:x['interstage'].pop(),
            wrong_reset_arc=lambda x:x['resets'][0].update(to='datapath/D'),
            wrong_reset_polarity=lambda x:x['resets'][0].update(inversions='0'),
            wrong_reset_lut=lambda x:x['resets'][-1].update(lut_inits="2'h1|2'h1|2'h1"),
            removed_vendor_constraint=lambda x:x['removed_commands'].update({'vendor':1}),
            broad_clock_cut=lambda x:x['added_commands'].update({'set_clock_groups -asynchronous':1}),
            changed_xdc_alias=changed_alias,
            hidden_raw_lock=lambda x:x['raw_lock'][0].update(bounded_by_this_stage='1'),
            critical_cdc=lambda x:x['cdc'].update({'CDC-10':1}))
        for name, mutate in mutations.items():
            bad=copy.deepcopy(e);mutate(bad)
            try:semantic(bad)
            except ValueError:rejected.append(name)
            else:raise ValueError('Accepted negative control: '+name)
        for name in ('reopened_changed_bound','reopened_missing_constraint'):
            bad = reopened_xdc.copy()
            line = next(s for s in bad if s.startswith('set_max_delay -datapath_only') and s.endswith('4.000'))
            del bad[line]
            if name == 'reopened_changed_bound': bad[line[:-5]+'4.001'] += 1
            require(expanded_capture_aliases(applied_xdc) != expanded_capture_aliases(bad), 'Accepted negative control: '+name)
            rejected.append(name)
    return dict(status='NATIVE_SCOPED_CONSTRAINTS_REOPENED_AND_REVIEWED',
        checkpoint_sha256=b['checkpoint_sha256'], max_delay_groups=33, bounded_endpoints=1503,
        gray_skew_groups=8, async_control_arcs=18, private_synchronizers=147, interstage_pairs=90,
        min_estimated_data_slack_ns=min(float(x['slack_ns']) for x in e['paths']),
        negative_controls_rejected=rejected, remaining_cdc=e['cdc'],
        global_cdc_qualified=False, routed_timing_qualified=False, board_qualified=False,
        raw_lock_route_bounded=False, native_traffic_qualified=False,
        seconds=dict(apply=a['seconds'],reopen=b['seconds']))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('applied',type=Path);p.add_argument('reopened',type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(review(a.applied,a.reopened,a.self_test),indent=2))
