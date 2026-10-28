"""Audit post-opt plan membership, exact driver cones and real SLICE geometry."""
import argparse
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import require
from check_ntrpu_overlap_bankplan import plan_review, rows, sha


def bounds(value):
    m = re.fullmatch(r'SLICE_X(\d+)Y(\d+):SLICE_X(\d+)Y(\d+)', value)
    require(m, 'Unsupported region geometry')
    return tuple(map(int, m.groups()))


def review(out, evidence):
    record = json.loads((out/'result.json').read_text())
    files = json.loads((out/'FILES.json').read_text())
    require(all(sha(out/n) == h for n, h in files.items()), 'Evidence hashes')
    require(record['status'] == 'EXPORTED_PENDING_REVIEW' and record['inputs'] == record['inputs_after'], 'Incomplete/changed probe')
    require('NTRPU_AXI_PREFLIGHT_COMPLETE' in (out/'console.log').read_text(), 'Probe completion')
    require(all(record['inputs']['plan/'+p.relative_to(evidence).as_posix()] == sha(p)
                for p in evidence.rglob('*') if p.is_file()), 'Candidate plan identity')
    plan, witness = plan_review(evidence)
    old_cells = {r['cell']: r for r in rows(evidence/'probe_evidence/cells.tsv')}
    cells = {r['cell']: r for r in rows(out/'cells.tsv')}
    selected = {n for g in plan['groups'] for n in g['members']}
    missing = sorted(selected-cells.keys())
    changed_types = sorted(n for n in selected & cells.keys() if cells[n]['ref'] != old_cells[n]['ref'])

    def cones(path):
        edges = {}
        for r in rows(path): edges.setdefault(r['sink'].rsplit('/', 1)[0], {})[r['sink']] = r['driver']
        visited = set(); pending = list(selected); cone = {}
        while pending:
            name = pending.pop()
            if name in visited: continue
            visited.add(name)
            if name in edges:
                cone.update(edges[name])
                pending.extend(d.rsplit('/', 1)[0] for d in edges[name].values() if not d.startswith(('OPEN:', 'PORT:')))
        return cone, visited

    old_cone, old_nodes = cones(evidence/'probe_evidence/edges.tsv')
    cone, nodes = cones(out/'edges.tsv')
    differences = sorted(k for k in old_cone.keys() | cone.keys() if old_cone.get(k) != cone.get(k))
    node_changes = sorted(n for n in old_nodes | nodes
                          if n not in cells or n not in old_cells or cells[n]['ref'] != old_cells[n]['ref'])
    sites = {}
    for row in rows(out/'device_sites.tsv'):
        m = re.fullmatch(r'SLICE_X(\d+)Y(\d+)', row['site']); require(m, 'Unsupported SLICE site')
        require(row['type'] in ('SLICEL', 'SLICEM') and row['site'] not in sites, 'Duplicate/non-SLICE site')
        sites[row['site']] = tuple(map(int, m.groups()))
    in_region = lambda b: {s for s, (x, y) in sites.items() if b[0] <= x <= b[2] and b[1] <= y <= b[3]}
    geometry = rows(out/'geometry.tsv'); require(len(geometry) == 8, 'Region coverage')
    occupied = set(); regions = []
    for group, row in zip(plan['groups'], geometry):
        require(row['bank'] == str(group['bank']) and row['requested'] == group['range'], 'Wrong requested region')
        requested = in_region(bounds(row['requested'])); applied = set(row['sites'].split(';'))
        require(requested == applied == in_region(bounds(row['applied'])), 'Normalization changed actual site set')
        require(len(applied) >= 80 and not occupied & applied, 'Capacity or disjointness failure')
        occupied |= applied
        regions.append(dict(bank=group['bank'], requested=row['requested'], applied=row['applied'], slice_sites=len(applied)))
    budget = lambda p: [l.strip() for l in p.read_text().splitlines() if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s', l)]
    require(budget(out/'constraints_before.xdc') == budget(out/'constraints_after.xdc'), 'Clock budget changed')
    require(not re.search(r'^\s*(create_pblock|resize_pblock|add_cells_to_pblock)\b',
                          (out/'constraints_after.xdc').read_text(), re.M), 'Temporary regions persisted')
    passed = not (missing or changed_types or differences or node_changes)
    return dict(status='POST_OPT_PLAN_PREFLIGHT_PASS' if passed else 'POST_OPT_PLAN_MISMATCH',
                physical_launch_allowed=passed, physical_validation=False, connectivity=witness,
                selected_cells=len(selected), missing_cells=missing, changed_types=changed_types,
                changed_cone_pins=differences, changed_cone_types=node_changes, checked_cone_pins=len(cone),
                regions=regions, exact_device_site_sets=True, synthesis_checkpoint_sha256=record['inputs']['checkpoint'],
                post_opt_checkpoint_sha256=files['post_opt.dcp'], seconds=record['seconds'])


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--plan', type=Path, required=True); ap.add_argument('--review', type=Path, required=True)
    args = ap.parse_args(); result = review(args.output, args.plan)
    result['checker_sha256'] = sha(Path(__file__))
    args.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
