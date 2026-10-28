"""Audit connectivity witnesses, disjoint regions and compact physical gates."""
import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from check_ntrpu_candidate_transport import require
from prepare_ntrpu_overlap_bankplan import derive, PREFIX


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def plan_review(evidence):
    probe = evidence/'probe_evidence'
    plan = json.loads((evidence/'BANK_PLAN.json').read_text())
    require(derive(probe) == plan, 'Plan differs from connectivity reconstruction')
    cells = {r['cell']: r for r in rows(probe/'cells.tsv')}
    deps = {}
    for r in rows(probe/'edges.tsv'):
        name = r['sink'].rsplit('/', 1)[0]; deps.setdefault(name, set())
        if not r['driver'].startswith(('OPEN:', 'PORT:')):
            deps[name].add(r['driver'].rsplit('/', 1)[0])
    @lru_cache(None)
    def leaves(name):
        if name not in deps: return frozenset([name])
        return frozenset().union(*(leaves(p) for p in deps[name]))
    expected = []; witnesses = 0
    for group in plan['groups']:
        bank = group['bank']; seq = 0
        for name in group['members']:
            expected.append(dict(bank=str(bank), cell=name, range=group['range']))
            if cells[name]['ref'] == 'FDRE':
                require(name.startswith(PREFIX+'bank_state['+str(bank)+'].'), 'Foreign bank state')
                seq += 1; continue
            origins = leaves(name); seen_banks = set(); lanes = set()
            for origin in origins:
                require(cells[origin]['ref'] in ('FDRE', 'FDSE', 'GND', 'VCC'), 'Unexamined combinational origin')
                m = re.match(re.escape(PREFIX)+r'bank_state\[([0-7])\]\.', origin)
                if m: seen_banks.add(int(m[1]))
                m = re.match(r'core/core/receiver/core/active_generation_reg\[([01])\]', origin)
                if m: lanes.add(int(m[1]))
            require(seen_banks == {bank} and lanes, 'Comparator connectivity witness failed')
            witnesses += 1
        require(seq == 66, 'Bank state count changed')
    require(rows(evidence/'groups.tsv') == expected, 'Applied plan differs from selected cells')
    require(witnesses == plan['comparator_cells'], 'Comparator coverage')
    for i, a in enumerate(plan['groups']):
        ax, ay, bx, by = a['bounds']
        for b in plan['groups'][i+1:]:
            cx, cy, dx, dy = b['bounds']
            require(bx < cx or dx < ax or by < cy or dy < ay, 'Regions overlap')
    return plan, dict(status='CONNECTIVITY_WITNESSES_PASS', groups=8,
                      comparator_witnesses=witnesses, state_registers=528,
                      shared_active_generation_registers_unassigned=128, selected_cells=plan['selected_cells'])


def physical_regions(out, stage, plan):
    s = out/'suite'; props = rows(s/'bankplan.tsv'); require(len(props) == 8, 'Region count')
    def xdc_regions(path):
        return dict(re.findall(r'^resize_pblock \[get_pblocks (ntrpu_bank_[0-7])\] -add \{(SLICE_X\d+Y\d+:SLICE_X\d+Y\d+)\}', path.read_text(), re.M))
    initial = xdc_regions(s/'bankplan_before.xdc' if stage == 'place' else out/'inputs/constraints_after.xdc')
    final = xdc_regions(s/'constraints_after.xdc')
    require(initial == final and len(final) == 8, 'Applied region geometry changed')
    after = rows(s/'members_after.tsv'); names = [r['cell'] for r in after]
    require(len(names) == len(set(names)), 'Overlapping membership')
    result = []
    for group in plan['groups']:
        bank = group['bank']; prop = props[bank]
        require(prop['bank'] == str(bank) and prop['range'] == final['ntrpu_bank_'+str(bank)], 'Region geometry')
        match = re.fullmatch(r'SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)', prop['range'])
        require(match, 'Unsupported region shape')
        x0, y0, x1, y1 = map(int, match.groups())
        gx0, gy0, gx1, gy1 = group['bounds']
        # Vivado canonicalized four right edges by one column at creation.
        # Accept only that bounded adjustment, unchanged through implementation.
        require((x0, y0, y1) == (gx0, gy0, gy1) and gx1-1 <= x1 <= gx1,
                'Unexpected canonical region adjustment')
        require(prop['soft'] == '1' and prop['exclude'] == prop['contain'] == '0', 'Region behavior')
        require(int(prop['slice_sites']) >= 80, 'Region capacity')
        require(int(prop['slice_sites']) == (x1-x0+1)*(y1-y0+1), 'Canonical region site count')
        assigned = [r for r in after if int(r['bank']) == bank]
        require({r['cell'] for r in assigned} == set(group['members']), 'Changed membership needs explicit review')
        require(dict(Counter(r['ref'] for r in assigned)) == group['types'], 'Resource types changed')
        inside = 0
        for r in assigned:
            m = re.fullmatch(r'SLICE_X(\d+)Y(\d+)', r['loc']); require(m, 'Unplaced bank primitive')
            x, y = map(int, m.groups()); inside += x0 <= x <= x1 and y0 <= y <= y1
        result.append(dict(bank=bank, members=len(assigned), inside=inside, outside=len(assigned)-inside,
                           requested_range=group['range'], applied_range=prop['range'],
                           slice_sites=int(prop['slice_sites'])))
    if stage == 'place':
        before = rows(s/'selected_before_place.tsv')
        require([(r['bank'], r['cell'], r['ref']) for r in before] ==
                [(r['bank'], r['cell'], r['ref']) for r in after], 'Placement changed selected primitives')
    else:
        require((out/'inputs/bankplan.tsv').read_bytes() == (s/'bankplan.tsv').read_bytes(), 'Route changed regions')
    budget = lambda p: [l.strip() for l in p.read_text().splitlines() if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s', l)]
    require(budget(out/'inputs/constraints_after.xdc') == budget(s/'constraints_before.xdc') ==
            budget(s/'constraints_after.xdc'), 'Timing budget changed')
    xdc = (s/'constraints_after.xdc').read_text()
    require(set(re.findall(r'^create_pblock (\S+)', xdc, re.M)) == {'ntrpu_bank_'+str(i) for i in range(8)}, 'Unexpected Pblocks')
    return dict(regions=result, total_members=len(after), inside=sum(r['inside'] for r in result),
                outside=sum(r['outside'] for r in result), disjoint_initial_regions=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path); ap.add_argument('--review', type=Path, required=True)
    a = ap.parse_args(); out = a.output
    record = json.loads((out/'result.json').read_text()); files = json.loads((out/'FILES.json').read_text())
    require(record['experiment'] == 'connectivity_bank_floorplan', 'Wrong experiment')
    require(all(sha(out/n) == h for n, h in files.items()), 'Evidence hashes')
    for key in ('sources', 'inputs', 'rtl_evidence', 'plan_evidence'):
        require(record[key] == record[key+'_after'], 'Changed '+key)
    plan, witness = plan_review(out/'plan_evidence')
    if record['status'] == 'EXPORTED_PENDING_REVIEW':
        subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('check_ntrpu_overlap_physical.py')),
                        str(out), '--review', str(a.review)], check=True, capture_output=True, text=True)
        result = json.loads(a.review.read_text())
        if record['stage'] in ('place', 'route'): result['bankplan'] = physical_regions(out, record['stage'], plan)
    else:
        require(record['status'] == 'FAILED_OR_TIMEOUT', 'Incomplete stage')
        text = (out/'console.log').read_text()
        result = dict(status='TIMEOUT_DIAGNOSTIC' if record['timed_out'] else 'TOOL_FAILURE_DIAGNOSTIC',
                      stage=record['stage'], seconds=record['seconds'], returncode=record['returncode'],
                      final_timing_available=False,
                      log_tail=text.splitlines()[-25:])
    result['connectivity'] = witness
    result['bankplan_checker_sha256'] = sha(Path(__file__))
    a.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'drc', 'timing_checks', 'negative_controls', 'log_tail')}))


if __name__ == '__main__':
    main()
