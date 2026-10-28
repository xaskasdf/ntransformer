"""Review the post-opt AXI bank experiment against exact device-site evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import require, replay, negative_controls
from check_ntrpu_transport_study import physical, performance, netlist, mutations
from check_ntrpu_transport_overlap import extra, negative
from check_ntrpu_overlap_physical import routed_extra
from check_ntrpu_overlap_bankplan import plan_review, rows, sha
from check_ntrpu_overlap_axi_preflight import review as preflight_review
from prepare_ntrpu_overlap_axi import integrated


def physical_regions(out, stage, plan, gate):
    s = out/'suite'; props = rows(s/'bankplan.tsv')
    require(len(props) == 8, 'Region coverage')
    preflight = out/'preflight_evidence'
    cells = {r['cell']: r for r in rows(preflight/'cells.tsv')}
    expected_sites = {int(r['bank']): set(r['sites'].split(';')) for r in rows(preflight/'geometry.tsv')}
    sites = {i: set() for i in range(8)}
    site_rows = rows(s/'bank_sites.tsv')
    for r in site_rows:
        bank = int(r['bank']); require(bank in sites and r['site'] not in sites[bank], 'Duplicate/unknown region site')
        sites[bank].add(r['site'])
    require(sites == expected_sites, 'Applied device site sets changed')
    actual = rows(s/'members_after.tsv')
    names = [r['cell'] for r in actual]; require(len(names) == len(set(names)) == plan['selected_cells'], 'Membership coverage')
    def regions(path):
        return dict(re.findall(r'^resize_pblock \[get_pblocks (ntrpu_bank_[0-7])\] -add \{(SLICE_X\d+Y\d+:SLICE_X\d+Y\d+)\}', path.read_text(), re.M))
    expected_ranges = {'ntrpu_bank_'+str(r['bank']): r['applied'] for r in gate['regions']}
    before = s/'bankplan_before.xdc' if stage == 'place' else out/'inputs/constraints_after.xdc'
    require(regions(before) == regions(s/'constraints_after.xdc') == expected_ranges, 'Region ranges changed')
    result = []
    for g, prop, geometry in zip(plan['groups'], props, gate['regions']):
        bank = g['bank']
        require(prop['bank'] == str(bank) and prop['range'] == geometry['applied'], 'Region order/shape')
        require(prop['soft'] == '1' and prop['exclude'] == prop['contain'] == '0', 'Region behavior')
        require(int(prop['slice_sites']) == len(sites[bank]) >= 80, 'Region site capacity')
        assigned = [r for r in actual if r['bank'] == str(bank)]
        require({r['cell'] for r in assigned} == set(g['members']), 'Changed selected members')
        require(all(r['ref'] == cells[r['cell']]['ref'] for r in assigned), 'Changed member primitive types')
        require(all(re.fullmatch(r'SLICE_X\d+Y\d+', r['loc']) and r['bel'] for r in assigned), 'Unplaced member')
        inside = sum(r['loc'] in sites[bank] for r in assigned)
        result.append(dict(bank=bank, members=len(assigned), inside=inside, outside=len(assigned)-inside,
                           requested_range=g['range'], applied_range=geometry['applied'], slice_sites=len(sites[bank])))
    earlier = rows(s/'selected_before_place.tsv' if stage == 'place' else out/'inputs/members_after.tsv')
    identity = lambda rs: [(r['bank'], r['cell'], r['ref']) for r in rs]
    require(identity(earlier) == identity(actual), 'Implementation changed selected primitives')
    if stage == 'place':
        require(sha(out/'inputs/physical_synth.dcp') == gate['post_opt_checkpoint_sha256'], 'Placement input differs from audited post-opt netlist')
        require(sha(out/'inputs/original_synth.dcp') == gate['synthesis_checkpoint_sha256'], 'Original synthesis identity')
    else:
        require((out/'inputs/bankplan.tsv').read_bytes() == (s/'bankplan.tsv').read_bytes(), 'Route changed regions')
        require((out/'inputs/bank_sites.tsv').read_bytes() == (s/'bank_sites.tsv').read_bytes(), 'Route changed actual sites')
    budget = lambda p: [l.strip() for l in p.read_text().splitlines() if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s', l)]
    paths = [out/'inputs/constraints_after.xdc', preflight/'constraints_before.xdc', preflight/'constraints_after.xdc',
             s/'constraints_before.xdc', s/'constraints_after.xdc']
    require(all(budget(p) == budget(paths[0]) for p in paths), 'Clock budgets changed')
    require(set(re.findall(r'^create_pblock (\S+)', (s/'constraints_after.xdc').read_text(), re.M)) == set(expected_ranges), 'Unexpected Pblocks')
    return dict(status='EXACT_DEVICE_BANK_REGIONS_PASS', regions=result, total_members=len(actual),
                inside=sum(r['inside'] for r in result), outside=sum(r['outside'] for r in result),
                exact_device_site_sets=True, disjoint_initial_regions=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--review', type=Path, required=True); a = ap.parse_args(); out = a.output
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = json.loads((out/'result.json').read_text()); files = json.loads((out/'FILES.json').read_text())
    require(record['experiment'] == 'compact_axi_bankplan', 'Wrong experiment')
    require(all(sha(out/n) == h for n, h in files.items()), 'Evidence hashes')
    require(all(record[k] == record[k+'_after'] for k in ('sources', 'inputs', 'rtl_evidence', 'plan_evidence', 'preflight_evidence')), 'Inputs changed')
    stage = record['stage']; s = out/'suite'; console = (out/'console.log').read_text()
    if record['status'] != 'EXPORTED_PENDING_REVIEW':
        require(record['status'] == 'FAILED_OR_TIMEOUT', 'Invalid result')
        result = dict(status='TIMEOUT_DIAGNOSTIC' if record['timed_out'] else 'TOOL_FAILURE_DIAGNOSTIC',
                      log_tail=console.splitlines()[-25:])
    else:
        marker = 'NTRPU_TRANSPORT_STUDY_COMPLETE ' if stage != 'netlist' else 'NTRPU_AXI_INTEGRATION_COMPLETE '
        require(marker+stage in console and not re.search(r'^ERROR:', console, re.M), 'Tool completion')
        if stage in ('rtl', 'synth', 'netlist'):
            props = dict(l.split('\t', 1) for l in (s/'ip_config.tsv').read_text().splitlines())
            expected = dict(PROTOCOL='AXI4', READ_WRITE_MODE='WRITE_ONLY', ADDR_WIDTH='64', DATA_WIDTH='512', ID_WIDTH='4', REG_AW='1', REG_W='1', REG_B='1')
            require(all(props['CONFIG.'+k] == v for k, v in expected.items()), 'IP configuration')
            fixture = json.loads((s/'generated/AXI_INTEGRATION.json').read_text())
            root = Path(__file__).resolve().parents[1]
            original = (root/'fpga/rtl/ntrpu_candidate_transport.sv').read_text()
            require(sha(root/'fpga/rtl/ntrpu_candidate_transport.sv') == record['sources']['fpga/rtl/ntrpu_candidate_transport.sv'], 'Reviewer reference differs')
            require(hashlib.sha256(original.encode()).hexdigest() == fixture['reference_transport_sha256'], 'Reference transport provenance')
            require((s/'generated/ntrpu_candidate_transport.sv').read_text() == integrated(original), 'Unexpected transport changes')
            require(sha(s/'generated/ntrpu_candidate_transport.sv') == fixture['integrated_transport_sha256'], 'Integrated hash')
            for n in ('ntrpu_destination_candidate', 'ntrpu_peer_multibuffer_offer', 'ntrpu_peer_multibuffer_reserved', 'ntrpu_candidate_dispatch'):
                require(sha(s/'generated'/(n+'.sv')) == sha(out/'rtl_evidence'/(n+'.sv')) == fixture['logic_sha256'][n], 'Compact logic changed')
            if stage == 'synth':
                require(json.loads((out/'inputs/AXI_INTEGRATION.json').read_text()) == fixture, 'Synth differs from functional fixture')
        if stage == 'rtl':
            text = (s/'protocol.log').read_text(); other = (s/'extra.log').read_text(); perf = (s/'perf.log').read_text()
            require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )', x, re.M) for x in (text, other, perf)), 'Simulation failure')
            lines = [l for l in text.splitlines() if l.startswith('TE ')]
            result = replay(lines); result['negative_controls'] = negative_controls(lines)
            result['extra'] = extra(other); result['extra_negative_controls'] = negative(other, extra, ['TE W ', 'TE CPL ', 'OV LAST '])
            result['performance'] = performance(perf); result['performance_negative_controls'] = negative(perf, performance, ['TE IN ', 'TE W ', 'TE CPL ', 'PF RELEASE '])
            result['status'] = 'AXI_RTL_PROTOCOL_PERFORMANCE_PASS'
        elif stage == 'netlist':
            text = (s/'protocol.log').read_text(); other = (s/'extra.log').read_text()
            require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )', x, re.M) for x in (text, other)), 'Netlist simulation failure')
            result = netlist(text); result['extra'] = routed_extra(other)
            result['trace_negative_controls'] = mutations(text, netlist)
            result['extra_negative_controls'] = negative(other, routed_extra, ['TN W ', 'TN FETCH ', 'TN CPL ', 'TN MATCH '])
            comp = fixture['comparison']
            require(comp == dict(netlist_sha256=record['inputs']['physical_funcsim.v'], latency_edges=0, startup_ns=150, sdf=False, force=False, reversible_renaming=True), 'Comparison scope')
        else:
            result = physical(out, stage); util = (s/'utilization.rpt').read_text()
            result['resources'] = {}
            for key in ('CLB LUTs', 'CLB Registers', 'Block RAM Tile', 'RAMB36/FIFO*', 'RAMB18'):
                m = re.search(r'^\|\s*'+re.escape(key)+r'\*?\s*\|\s*([0-9.]+)\s*\|', util, re.M)
                if m: result['resources'][key] = float(m[1])
            gate = preflight_review(out/'preflight_evidence', out/'plan_evidence')
            require(gate['physical_launch_allowed'], 'Plan no longer eligible')
            plan, witness = plan_review(out/'plan_evidence')
            result['bankplan'] = physical_regions(out, stage, plan, gate)
            result['connectivity'] = witness
    result.update(stage=stage, seconds=record['seconds'], checker_sha256=sha(Path(__file__)),
                  scope='Same-clock AXI write slices; external ports, PCIe, HBM and CDC unqualified')
    a.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'timing_checks', 'drc', 'performance', 'log_tail')}))


if __name__ == '__main__':
    main()
