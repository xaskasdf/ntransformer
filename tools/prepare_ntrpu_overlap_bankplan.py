"""Derive disjoint bank comparison groups from exported leaf-pin connectivity."""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

PREFIX = 'core/core/receiver/core/allocator/'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def derive(probe):
    record = json.loads((probe/'result.json').read_text())
    files = json.loads((probe/'FILES.json').read_text())
    assert record['status'] == 'EXPORTED_PENDING_REVIEW' and record['returncode'] == 0
    assert record['checkpoint_sha256'] == record['checkpoint_sha256_after']
    assert record['script_sha256'] == record['script_sha256_after']
    assert all(sha(probe/n) == h for n, h in files.items())
    assert 'NTRPU_CONNECTIVITY_COMPLETE' in (probe/'console.log').read_text()
    with (probe/'cells.tsv').open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    cells = {r['cell']: r for r in rows}; assert len(cells) == len(rows)
    comb = {n for n, r in cells.items() if n.startswith(PREFIX) and re.fullmatch(r'LUT[1-6]|CARRY8', r['ref'])}
    deps = {n: set() for n in comb}; sink_pins = set(); open_pins = []
    with (probe/'edges.tsv').open(newline='') as f:
        for edge in csv.DictReader(f, delimiter='\t'):
            sink = edge['sink']; cell = sink.rsplit('/', 1)[0]
            assert cell in deps and sink not in sink_pins
            sink_pins.add(sink); driver = edge['driver']
            if driver.startswith('OPEN:'):
                assert not edge['segments']; open_pins.append(sink)
            elif driver.startswith('PORT:'):
                pass
            else:
                source = driver.rsplit('/', 1)[0]; assert source in cells
                deps[cell].add(source)
    assert all(deps.values())
    banks = {}; lanes = {}
    for n, r in cells.items():
        if r['ref'] != 'FDRE': continue
        match = re.fullmatch(re.escape(PREFIX)+r'bank_state\[([0-7])\]\.(?:generation|owner|used)_reg.*', n)
        if match: banks[n] = int(match[1])
        match = re.fullmatch(r'core/core/receiver/core/active_generation_reg\[([01])\]\[\d+\]', n)
        if match: lanes[n] = int(match[1])
    assert Counter(banks.values()) == {i: 66 for i in range(8)}
    assert Counter(lanes.values()) == {0: 64, 1: 64}
    # Mark unexamined combinational inputs as unknown, never bank-exclusive.
    labels = {n: ((1 << banks[n]) if n in banks else 0,
                  (1 << lanes[n]) if n in lanes else 0,
                  r['ref'] not in ('FDRE', 'FDSE', 'GND', 'VCC'))
              for n, r in cells.items() if n not in comb}
    pending = set(comb)
    while pending:
        ready = sorted(n for n in pending if deps[n] <= labels.keys())
        assert ready, 'Combinational dependency cycle or missing source'
        for n in ready:
            bank = lane = 0; unknown = False
            for p in deps[n]:
                b, l, u = labels[p]; bank |= b; lane |= l; unknown |= u
            labels[n] = bank, lane, unknown
        pending.difference_update(ready)
    groups = []
    for bank in range(8):
        compare = sorted(n for n in comb if labels[n][0] == 1 << bank and labels[n][1] and not labels[n][2])
        selected = sorted([n for n, b in banks.items() if b == bank]+compare)
        assert compare, 'Empty bank comparator group'
        positions = [tuple(map(int, re.fullmatch(r'SLICE_X(\d+)Y(\d+)', cells[n]['loc']).groups())) for n in selected]
        groups.append(dict(bank=bank, members=selected, comparator_cells=len(compare), state_registers=66,
                           types=dict(Counter(cells[n]['ref'] for n in selected)),
                           median=[statistics.median(x for x, _ in positions), statistics.median(y for _, y in positions)],
                           original_bounds=[min(x for x, _ in positions), min(y for _, y in positions),
                                            max(x for x, _ in positions), max(y for _, y in positions)]))
    # Eight separate 6x20 initial regions, with two-column/four-row gaps.
    # Assign them by minimum aggregate grid displacement, not predicted delay.
    cx = int(statistics.median(g['median'][0] for g in groups))
    cy = int(statistics.median(g['median'][1] for g in groups))
    x0 = cx-14; y0 = max(240, min(252, cy-22))
    slots = [(x0+8*c, y0+24*r, x0+8*c+5, y0+24*r+19) for r in range(2) for c in range(4)]
    costs = [[sum(abs(int(re.fullmatch(r'SLICE_X(\d+)Y(\d+)', cells[n]['loc'])[1])-(s[0]+s[2])/2)+
                  abs(int(re.fullmatch(r'SLICE_X(\d+)Y(\d+)', cells[n]['loc'])[2])-(s[1]+s[3])/2)
                  for n in g['members']) for s in slots] for g in groups]
    assignment = min(itertools.permutations(range(8)), key=lambda p: sum(costs[i][p[i]] for i in range(8)))
    for i, g in enumerate(groups):
        bounds = slots[assignment[i]]; g['bounds'] = list(bounds)
        g['range'] = 'SLICE_X%dY%d:SLICE_X%dY%d' % bounds
    selected = [n for g in groups for n in g['members']]
    assert len(selected) == len(set(selected)) and not set(selected)&set(lanes)
    return dict(status='CONNECTIVITY_BANK_PLAN', version=1, groups=groups,
                selected_cells=len(selected), comparator_cells=sum(g['comparator_cells'] for g in groups),
                shared_active_generation_registers=len(lanes), shared_registers_assigned=False,
                allocator_combinational_cells=len(comb), examined_input_pins=len(sink_pins),
                disconnected_input_pins=open_pins,
                excluded_unknown_combinational=sum(labels[n][2] for n in comb),
                excluded_multi_bank_combinational=sum(labels[n][0].bit_count()>1 for n in comb),
                timing_budget_ns=5.0, is_soft=True, contain_routing=False, exclude_placement=False,
                checkpoint_sha256=record['checkpoint_sha256'],
                probe_files_sha256={n: sha(probe/n) for n in ('result.json', 'cells.tsv', 'edges.tsv', 'FILES.json')},
                generator_sha256=sha(Path(__file__)),
                method='Propagate bank and lane identities through allocator leaf-cell dependencies. Select one-bank logic reached by a lane generation register, plus that bank state. Unknown external combinational dependencies are excluded.',
                placement_heuristic='Minimum aggregate Manhattan displacement in SLICE coordinates among eight disjoint slots; not a timing prediction.')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('probe', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); plan = derive(a.probe)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'BANK_PLAN.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
    with (a.output/'groups.tsv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n'); writer.writerow(['bank', 'cell', 'range'])
        for group in plan['groups']:
            for cell in group['members']: writer.writerow([group['bank'], cell, group['range']])
    print(json.dumps({k: v for k, v in plan.items() if k not in ('groups', 'disconnected_input_pins')}))
    print(json.dumps([ {k: v for k, v in g.items() if k != 'members'} for g in plan['groups']]))


if __name__ == '__main__':
    main()
