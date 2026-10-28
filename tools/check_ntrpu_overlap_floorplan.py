"""Review the soft control region and established physical/functional gates."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from check_ntrpu_candidate_transport import require


def members(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    require(rows and all(set(r) == {'cell', 'ref', 'loc', 'bel'} for r in rows), 'Membership schema')
    result = {r['cell']: r for r in rows}
    require(len(result) == len(rows), 'Duplicate member')
    return result


def allocator(n):
    return n.startswith('core/core/receiver/core/allocator/')


def generation(n):
    return n.startswith('core/core/receiver/core/active_generation_reg')


def geometry(rows):
    coords = []
    for r in rows:
        m = re.fullmatch(r'SLICE_X(\d+)Y(\d+)', r['loc'])
        require(m, 'Unplaced or non-slice control member')
        coords.append(tuple(map(int, m.groups())))
    require(coords, 'Empty physical group')
    inside = sum(54 <= x <= 71 and 260 <= y <= 289 for x, y in coords)
    return dict(cells=len(coords), inside=inside, outside=len(coords)-inside,
                x_range=[min(x for x, _ in coords), max(x for x, _ in coords)],
                y_range=[min(y for _, y in coords), max(y for _, y in coords)])


def audit(out, stage):
    s = out/'suite'
    props = dict(line.split(' ', 1) for line in (s/'floorplan.txt').read_text().splitlines())
    require(props['NAME'] == 'ntrpu_compact_control' and
            props['GRID_RANGES'] == 'SLICE_X54Y260:SLICE_X71Y289', 'Region identity changed')
    for prop, expected in [('IS_SOFT', True), ('EXCLUDE_PLACEMENT', False), ('CONTAIN_ROUTING', False)]:
        require(props[prop].lower() in (('true', '1') if expected else ('false', '0')), 'Region behavior changed: '+prop)
    budgets = lambda p: [line.strip() for line in p.read_text().splitlines()
                         if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s', line)]
    require(budgets(out/'inputs/constraints_after.xdc') == budgets(s/'constraints_before.xdc') ==
            budgets(s/'constraints_after.xdc'), 'Timing budgets changed')
    xdc = (s/'constraints_after.xdc').read_text()
    require(re.findall(r'^create_pblock (\S+)', xdc, re.M) == ['ntrpu_compact_control'], 'Unexpected Pblock')
    after = members(s/'members_after.tsv')
    require(all(allocator(n) or generation(n) for n in after), 'Unexpected Pblock member')
    require(all(re.fullmatch(r'(LUT[1-6]|CARRY8|FDRE|FDSE)', r['ref']) for r in after.values()), 'Unexpected control resource')
    if stage == 'place':
        before = members(s/'selected_before_place.tsv')
        require(all(allocator(n) or generation(n) for n in before), 'Selection scope changed')
        gens = [r for n, r in before.items() if generation(n)]
        require(len(gens) == 128 and all(r['ref'] == 'FDRE' for r in gens), 'Generation selection changed')
        selected = dict(line.split(' ', 1) for line in (s/'selection.txt').read_text().splitlines())
        require(int(selected['GENERATION']) == 128 and int(selected['TOTAL']) == len(before) and
                int(selected['ALLOCATOR']) == len(before)-128, 'Selection counts differ')
        require(selected['COMMAND'] == 'opt_design; add_control_region; place_design', 'Placement strategy changed')
        require(budgets(s/'floorplan_before.xdc') == budgets(s/'constraints_after.xdc'), 'Floorplan changed timing')
    else:
        before = members(out/'inputs/members_after.tsv')
        require((out/'inputs/floorplan.txt').read_text() == (s/'floorplan.txt').read_text(), 'Region changed during route')
        require({n: r['ref'] for n, r in before.items()} == {n: r['ref'] for n, r in after.items()}, 'Route changed membership')
    return dict(properties=props, before_members=len(before), after_members=len(after),
                added_members=sorted(set(after)-set(before)), removed_members=sorted(set(before)-set(after)),
                types=dict(Counter(r['ref'] for r in after.values())),
                all=geometry(after.values()),
                generation=geometry([r for n, r in after.items() if generation(n)]),
                allocator=geometry([r for n, r in after.items() if allocator(n)]))


def timeout_review(out, record):
    require(record['stage'] == 'route' and record['status'] == 'FAILED_OR_TIMEOUT' and
            record['timed_out'] and record['returncode'] in (-15, -9), 'Not a bounded routing timeout')
    require(record['timeout'] <= record['seconds'] <= record['timeout']+15, 'Unexpected timeout duration')
    require(record['sources'] == record['sources_after'] and record['inputs'] == record['inputs_after'] and
            record['rtl_evidence'] == record['rtl_evidence_after'], 'Inputs changed during timeout')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = json.loads((out/'FILES.json').read_text())
    require(all(sha(out/n) == h for n, h in files.items()), 'Timeout evidence hashes')
    require(record['prior']['stage'] == 'place' and
            sha(out/'inputs/physical_place.dcp') == record['prior']['sha256'], 'Placed checkpoint provenance')
    require(not (out/'suite/physical_route.dcp').exists() and not (out/'suite/physical_funcsim.v').exists(),
            'Partial output requires a separate diagnostic review')
    text = (out/'console.log').read_text()
    require('NTRPU_TRANSPORT_STUDY_COMPLETE route' not in text and
            not re.search(r'^ERROR:', text, re.M), 'Unexpected tool completion/error')
    counts = list(map(int, re.findall(r'Number of Nodes with overlaps = (\d+)', text)))
    summaries = re.findall(r'Intermediate Timing Summary \| WNS=\s*([-\d.]+)\s*\| TNS=\s*([-\d.]+)\s*\| WHS=\s*([-\d.]+)\s*\| THS=\s*([-\d.]+)', text)
    require(counts and summaries, 'Routing progress missing')
    congestion = (out/'iter_120_CongestedCLBsAndNets.txt').read_text()
    return dict(status='ROUTING_TIMEOUT_DIAGNOSTIC', stage='route', variant='compact',
                seconds=record['seconds'], timeout_seconds=record['timeout'],
                last_logged_overlaps=counts[-1], minimum_logged_overlaps=min(counts),
                intermediate_timing=[dict(zip(('setup_ns', 'tns_ns', 'hold_ns', 'ths_ns'), map(float, r))) for r in summaries],
                congestion_report=congestion.splitlines(), final_timing_available=False,
                routed_netlist_available=False, functional_comparison_run=False,
                limitations=['Intermediate timing and zero overlaps in one iteration are not completed route qualification.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path); ap.add_argument('--review', type=Path, required=True)
    a = ap.parse_args()
    record = json.loads((a.output/'result.json').read_text())
    require(record['experiment'] == 'soft_control_floorplan', 'Wrong experiment')
    if record['status'] == 'FAILED_OR_TIMEOUT':
        result = timeout_review(a.output, record)
    else:
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('check_ntrpu_overlap_physical.py')),
                        str(a.output), '--review', str(a.review)], check=True, capture_output=True, text=True)
        result = json.loads(a.review.read_text())
        if record['stage'] in ('place', 'route'):
            result['floorplan'] = audit(a.output, record['stage'])
    result['floorplan_checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'timing_checks', 'drc', 'negative_controls')}))


if __name__ == '__main__':
    main()
