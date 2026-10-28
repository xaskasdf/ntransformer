"""Review a single LOCKED route bound and inventory unresolved external budgets."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import re
import review_hbm_native_constraints as prior

require = prior.require
DONOR = '629bae0ba8bddd48eec2fbdf78224a9f8ac0c149a32299d15bbb82c77f6bf313'
LOCK_SOURCE = 'clock_generator/inst/mmcme4_adv_inst/LOCKED'
LOCK_TARGET = 'availability/lock_sync_reg[0]/D'
LOCK_COMMAND = ('set_max_delay -datapath_only -from [get_pins '+LOCK_SOURCE+
                '] -to [get_pins {'+LOCK_TARGET+'}] 10.000')
SEGMENT_WARNING = ("CRITICAL WARNING: [Constraints 18-515] set_max_delay: Path segmentation by forcing '"
                   +LOCK_SOURCE+"' to be timing startpoint.")
TABLES = dict(clocks='domain_clocks.tsv', paths='paths.tsv', bindings='bindings.tsv',
              captures='capture_topology.tsv', sync='synchronizers.tsv',
              interstage='interstage.tsv', resets='reset_arcs.tsv')
ROLES = {
    'pcie_serial': 'pci_exp_rxn pci_exp_rxp pci_exp_txn pci_exp_txp',
    'reference_clock': 'sys_clk_n sys_clk_p mgmt_clk_in fabric_ref_clk_in hbm_ref0_clk_in hbm_ref1_clk_in',
    'external_sequence': 'sys_rst_n configuration_rst_n fabric_clock_reset fabric_clock_locked',
    'internal_transaction': 'cfg_valid cfg_ready cfg_session cfg_generation cfg_axi_base cfg_pcie_base cfg_rsp_valid cfg_rsp_ready cfg_rsp_error cmd_valid cmd_ready cmd_tag cmd_generation cmd_offset cmd_bytes cmd_source ack_valid ack_ready ack_tag ack_generation ack_bytes reject_valid reject_ready reject_code reject_tag result_valid result_ready result_tag result_bytes',
    'internal_observation': 'adapter_fault adapter_pending availability_armed availability_fault client_cold_rst_n client_link_up client_stop fault fault_code frame_idle guard_armed guard_clock_alive guard_fault guard_fault_code hbm_cattrip hbm_init_complete hbm_read_parity hbm_temperature management_cold_rst_n map_verified mapper_fault mapper_fault_code mem_armed mem_clock_alive mem_cold_rst_n mem_management_fault owned reader_clock_alive reader_cold_rst_n reader_drained reader_fault_raw reader_management_fault reader_started_raw reader_stopped source_fault source_owned window_held',
}


def closed(root, mode):
    manifest = json.loads((root/'FILES.json').read_text())
    require(all(prior.sha(root/n) == h for n,h in manifest.items()), 'Manifest mismatch')
    r = json.loads((root/'RESULT.json').read_text())
    require(r['status'] == 'HBM_NATIVE_JOIN_EXPORTED' and r['stage'] == 'inspect'
            and r['returncode'] == 0 and not r['timed_out'] and r['sources_unchanged'], 'Incomplete tool run')
    require(all(prior.sha(root/'source'/n) == h for n,h in r['sources'].items()), 'Source mismatch')
    require((root/'output/STATUS.txt').read_text().strip() == f'NATIVE_BOUNDARY_{mode}_EXPORTED_PENDING_REVIEW', 'Export gate')
    log = (root/'tool.log').read_text()
    require(not re.search(r'(?im)^\s*(?:error|fatal):',log), 'Tool error')
    critical = re.findall(r'(?m)^CRITICAL WARNING:.*$',log)
    require(critical == [SEGMENT_WARNING] if mode=='apply' else critical in ([],[SEGMENT_WARNING]),
            'Unexpected critical warning; only the exact raw LOCKED segmentation is reviewed')
    return r


def warnings(root):
    text = (root/'output/check_timing.rpt').read_text()
    result = {}
    for index, name in [(4,'unconstrained_internal_endpoints'),(5,'no_input_delay'),(6,'no_output_delay')]:
        matches = list(re.finditer(rf'(?m)^{index}\. checking {name} \((\d+)\)$',text))
        require(len(matches) == 2, 'Missing check_timing section: '+name)
        m = matches[-1]
        end = re.search(r'(?m)^\d+\. checking ',text[m.end():])
        block = text[m.end():m.end()+end.start()] if end else text[m.end():]
        names = [s.strip() for s in block.splitlines() if re.fullmatch(r'[\w/\[\].]+',s.strip())]
        require(len(names) == int(m[1]), 'check_timing endpoint coverage: '+name)
        result[name] = names
    return result


def declared_ports(top):
    header = top.read_text()[:top.read_text().index(');')]
    result = {}
    for line in header.splitlines():
        m = re.fullmatch(r'\s*(input|output)\s+(?:wire\s+)?(?:\[(\d+):(\d+)\]\s*)?([\w,]+),?\s*',line)
        if not m:
            continue
        direction,high,low,names = m.groups()
        for name in names.strip(',').split(','):
            bits = [f'{name}[{i}]' for i in range(int(low),int(high)+1)] if high else [name]
            for bit in bits:
                require(bit not in result,'Duplicate top port')
                result[bit] = 'IN' if direction == 'input' else 'OUT'
    require(len(result) == 1287, 'Native top declaration coverage')
    return result


def boundary(rows, expected, findings):
    observed = {r['port']:r for r in rows}
    require(len(rows) == len(observed) == len(expected) and set(observed) == set(expected), 'Boundary port coverage')
    categories = {name:role for role,names in ROLES.items() for name in names.split()}
    require(len(categories) == 80, 'Boundary role coverage')
    no_in = set(findings['no_input_delay']); no_out = set(findings['no_output_delay'])
    require(no_in <= set(expected) and no_out <= set(expected), 'Unknown unconstrained port')
    groups = defaultdict(list)
    for name,r in observed.items():
        require(r['direction'] == expected[name], 'Port direction changed')
        base = re.sub(r'\[\d+\]$','',name)
        require(base in categories, 'Unclassified boundary: '+base)
        if categories[base] == 'pcie_serial':
            require(re.fullmatch(r'[A-Z]{1,2}[0-9]+',r['package_pin']) is not None, 'Missing native GT package pin')
        else:
            require(r['package_pin'] == '', 'Unexpected physical pin assignment')
        groups[base].append(r)
    pins = [r['package_pin'] for r in rows if r['package_pin']]
    require(len(pins) == len(set(pins)) == 64, 'Native GT pin coverage')
    result = []
    for name,rs in sorted(groups.items()):
        role = categories[name]
        clocks = sorted({c for r in rs for c in r['peer_clocks'].split('|') if c})
        result.append(dict(signal=name,role=role,bits=len(rs),direction=rs[0]['direction'],
            peer_clocks=clocks, no_input_delay=sum(r['port'] in no_in for r in rs),
            no_output_delay=sum(r['port'] in no_out for r in rs),
            observed_package_pins={r['port']:r['package_pin'] for r in rs if r['package_pin']},
            observed_iostandards=sorted({r['iostandard'] for r in rs}),
            numeric_external_budget_ns=None, physical_pinout_approved=False))
    require(len(result) == 80, 'Boundary group coverage')
    return result


def lock_semantic(e):
    require(e['added'] == Counter({LOCK_COMMAND:1}) and not e['removed'], 'LOCKED constraint scope/value')
    expected_arcs = {(LOCK_SOURCE.rsplit('/',1)[0]+'/'+pin,LOCK_SOURCE) for pin in ('RST','PWRDWN')}
    require(len(e['source_arcs']) == 2 and {(r['from'],r['to']) for r in e['source_arcs']} == expected_arcs, 'Incoming LOCKED timing arcs changed')
    props = e['arc_properties']
    before = {(r['from'],r['to'],r['property']):r['value'] for r in props if r['phase']=='before'}
    after = {(r['from'],r['to'],r['property']):r['value'] for r in props if r['phase']=='after'}
    require(before and before == after and len(props) == 2*len(before), 'LOCKED timing arc properties changed')
    for src,dst in expected_arcs:
        require(before[(src,dst,'IS_DISABLED')] == '1' and before[(src,dst,'IS_USER_DISABLED')] == '0'
                and before[(src,dst,'TYPE')] == 'Reg Set/Clr', 'LOCKED primitive arc model changed')
    require(len(e['lock']) == 2 and [r['phase'] for r in e['lock']] == ['before','after'], 'LOCKED path coverage')
    for r in e['lock']:
        require(r['from'] == LOCK_SOURCE and r['to'] == LOCK_TARGET, 'LOCKED endpoint identity')
        require(r['start_clock'] == '' and r['end_clock'] == 'management_refclk', 'LOCKED clock model')
    r = e['lock'][-1]
    require(e['lock'][0]['requirement_ns'] == e['lock'][0]['slack_ns'] == '', 'Original LOCKED path was not unbounded')
    require(abs(float(r['requirement_ns'])-10.) <= .001 and math.isfinite(float(r['slack_ns'])), 'Missing LOCKED budget')
    require(e['interstage'] == e['baseline_interstage'], 'Synchronizer checks changed')
    require(len(e['interstage']) == 180, 'Synchronizer coverage')


def review(baseline, applied, reopened, top, self_test):
    old = prior.closed(baseline,'reopen'); require(old['checkpoint_sha256'] == DONOR,'Baseline identity')
    a = closed(applied,'apply'); b = closed(reopened,'reopen')
    require(a['checkpoint_sha256'] == DONOR, 'Wrong donor')
    require(b['checkpoint_sha256'] == prior.sha(applied/'output/bounded.dcp'),'Wrong reopened DCP')
    require(prior.sha(top) == '8bae8385fcb5f3abd09e390517f0bee49a1c07c643e02f29a59d6971d0178aca', 'Native top identity')
    def commands(root,name): return prior.xdc_commands((root/'output'/name).read_text())
    before = commands(applied,'before.xdc'); after = commands(applied,'after.xdc')
    require(before == commands(baseline,'after.xdc'),'Donor constraints changed')
    require(after == commands(reopened,'before.xdc') == commands(reopened,'after.xdc'),'Reopen changed constraints')
    for name in TABLES.values():
        require(prior.table(baseline/'output'/name) == prior.table(applied/'output'/name)
                == prior.table(reopened/'output'/name),'Prior evidence changed: '+name)
    base_interstage = prior.table(baseline/'output/interstage.tsv')
    e = dict(added=after-before, removed=before-after,
        source_arcs=prior.table(applied/'output/lock_source_arcs.tsv'),
        arc_properties=prior.table(applied/'output/lock_arc_properties.tsv'),
        lock=prior.table(applied/'output/lock_path.tsv'),
        interstage=prior.table(applied/'output/interstage.tsv'),baseline_interstage=base_interstage)
    lock_semantic(e)
    rlock = prior.table(reopened/'output/lock_path.tsv')
    require(len(rlock) == 2 and all({k:v for k,v in r.items() if k!='phase'} == {k:v for k,v in e['lock'][-1].items() if k!='phase'} for r in rlock),'Reopened LOCKED path changed')
    require(prior.table(reopened/'output/lock_source_arcs.tsv') == e['source_arcs'],'Reopened LOCKED arcs changed')
    require(prior.table(reopened/'output/lock_arc_properties.tsv') == e['arc_properties'],'Reopened LOCKED arc properties changed')
    findings = warnings(applied)
    require(findings == warnings(baseline) == warnings(reopened),'External/internal warning coverage changed')
    internal_groups = Counter('/'.join(n.split('/')[:6]) for n in findings['unconstrained_internal_endpoints'])
    expected_internal = {f'hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.u_xsdb_top_{stack}/{unit}':count
        for stack in (0,1) for unit,count in [('U_XSDB_SLAVE',161),('xsdb2adb_u0',416)]}
    require(internal_groups == expected_internal, 'Internal warning provenance')
    expected = declared_ports(top); rows = prior.table(applied/'output/boundary_ports.tsv')
    require(rows == prior.table(reopened/'output/boundary_ports.tsv'),'Reopened port inventory changed')
    inventory = boundary(rows, expected, findings)
    cdc = {n:int(c) for n,c in re.findall(r'(?m)^(CDC-\d+)\s+\w+\s+(\d+)\s+', (applied/'output/cdc.rpt').read_text())}
    require(cdc.get('CDC-10',0)==0,'Critical CDC regression')
    for root in (baseline,reopened):
        other = {n:int(c) for n,c in re.findall(r'(?m)^(CDC-\d+)\s+\w+\s+(\d+)\s+', (root/'output/cdc.rpt').read_text())}
        require(other == cdc,'CDC inventory changed')
    rejected=[]
    if self_test:
        mutations = dict(missing_lock_path=lambda x:x['lock'].pop(),
            wrong_lock_source=lambda x:x['lock'][-1].update({'from':'other/LOCKED'}),
            wrong_lock_budget=lambda x:x['lock'][-1].update(requirement_ns='20.0'),
            wrong_lock_clock=lambda x:x['lock'][-1].update(start_clock='fabric_ref_clk_in'),
            incoming_lock_arc=lambda x:x['source_arcs'].append(dict(from_pin='CLKIN1',to_pin=LOCK_SOURCE)),
            changed_lock_arc=lambda x:x['arc_properties'][0].update(value='changed'),
            removed_prior_constraint=lambda x:x['removed'].update({'old':1}),
            broad_false_path=lambda x:x['added'].update({'set_false_path -to *':1}),
            lost_interstage_check=lambda x:x['interstage'].pop())
        for name,mutate in mutations.items():
            bad=copy.deepcopy(e); mutate(bad)
            try:lock_semantic(bad)
            except ValueError:rejected.append(name)
            else:raise ValueError('Accepted negative control: '+name)
        for name in ('missing_port','wrong_port_direction','invented_pin'):
            bad=copy.deepcopy(rows)
            if name=='missing_port':bad.pop()
            elif name=='wrong_port_direction':bad[0]['direction']='OUT' if bad[0]['direction']=='IN' else 'IN'
            else:bad[0]['package_pin']='A1'
            try:boundary(bad,expected,findings)
            except ValueError:rejected.append(name)
            else:raise ValueError('Accepted negative control: '+name)
    return dict(status='NATIVE_LOCK_BOUND_AND_BOUNDARY_REVIEWED',checkpoint_sha256=b['checkpoint_sha256'],
        lock_max_ns=10.,lock_estimated_slack_ns=float(e['lock'][-1]['slack_ns']),
        intentional_segmented_startpoint=LOCK_SOURCE,critical_warning_retained='Constraints 18-515',
        preserved_bounded_endpoints=1503,preserved_interstage_pairs=90,
        boundary_ports=len(rows),boundary_signals=len(inventory),
        boundary_bits_by_role=dict(Counter({role:sum(r['bits'] for r in inventory if r['role']==role) for role in ROLES})),
        open_findings={k:len(v) for k,v in findings.items()},remaining_cdc=cdc,
        inventory=inventory,negative_controls_rejected=rejected,
        external_budgets_complete=False,physical_pinout_approved=False,routed_timing_qualified=False,
        native_traffic_qualified=False,seconds=dict(apply=a['seconds'],reopen=b['seconds']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','applied','reopened','top'):p.add_argument(n,type=Path)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    print(json.dumps(review(a.baseline,a.applied,a.reopened,a.top,a.self_test),indent=2))
