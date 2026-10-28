"""Conservative structural equivalence for flattened primitive graphs.

This is a zero-delay logical relation, not physical timing or board signoff.
Start with distinct reference cells, merge only identical definitions driven by
already-equivalent signals, then check every implementation cell and output.
No equivalence is inferred from replica names; names only propose a mapping.
"""
import collections
import json
import re
from audit_ntrpu_hier_edif_clock_r2 import PORTS


class CongruenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CongruenceError(message)


# Placement grouping and read-only optimization/origin annotations. Unknown
# properties remain part of the compared definition, including INIT/inversions.
METADATA = frozenset({'SOFT_HLUTNM', 'OPT_MODIFIED', 'PHYS_OPT_MODIFIED', 'ORIG_CELL_NAME'})


def behavior(row):
    return row['kind'], json.dumps({k: v for k, v in row['properties'].items()
                                   if k not in METADATA}, sort_keys=True)


def compare(reference, implementation):
    ref, impl = reference['cells'], implementation['cells']
    require(ref and impl, 'Empty primitive graph')
    for graph in (reference, implementation):
        require(set(graph['inputs']) and len(graph['inputs']) == len(set(graph['inputs'])), 'Invalid top inputs')
        for name, row in graph['cells'].items():
            require(row['kind'] in PORTS and name != 'TOP_INPUT', 'Unsupported primitive')
            interface = PORTS[row['kind']]
            expected = {p + ':' + str(i) for p, (d, w) in interface.items()
                        if d == 'INPUT' for i in range(w)}
            require(row['inputs'].keys() <= expected, 'Unsupported primitive input')
            if row['kind'] == 'FDRE':
                require(row['inputs'].keys() == expected and 'INIT' in row['properties'],
                        'Incomplete register definition')
        for driver in [d for row in graph['cells'].values() for d in row['inputs'].values()] + list(graph['outputs'].values()):
            require(isinstance(driver, list) and len(driver) == 3, 'Malformed driver')
            cell, port, bit = driver
            require(isinstance(port, str) and type(bit) is int and bit >= 0, 'Malformed pin')
            if cell == 'TOP_INPUT':
                require(port + ':' + str(bit) in graph['inputs'], 'Unknown top input')
            else:
                require(cell in graph['cells'], 'Unknown driver: ' + cell)
                interface = PORTS[graph['cells'][cell]['kind']]
                require(port in interface and interface[port][0] == 'OUTPUT' and bit < interface[port][1],
                        'Invalid primitive driver pin')
    require(set(reference['inputs']) == set(implementation['inputs']), 'Top input inventory changed')
    parents = {name: name for name in ref}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    def signal(driver, mapping):
        require(isinstance(driver, list) and len(driver) == 3, 'Malformed driver')
        cell, port, bit = driver
        require(isinstance(port, str) and isinstance(bit, int) and bit >= 0, 'Malformed pin')
        if cell == 'TOP_INPUT':
            return cell, port, bit
        require(cell in mapping, 'Unknown driver: ' + cell)
        return find(mapping[cell]), port, bit

    def definition(row, mapping):
        return behavior(row), tuple((pin, signal(driver, mapping))
                                    for pin, driver in sorted(row['inputs'].items()))

    identity = {name: name for name in ref}
    rounds = []
    while True:
        groups = {}
        merges = 0
        for name, row in ref.items():
            signature = definition(row, identity)
            other = groups.setdefault(signature, name)
            a, b = find(name), find(other)
            if a != b:
                # A stable root makes proof outputs reproducible.
                parents[max(a, b)] = min(a, b)
                merges += 1
        rounds.append(merges)
        if not merges:
            break

    mapping = {}
    added = {}
    for name, row in impl.items():
        origin = name if name in ref else re.sub(r'_replica(?:_\d+)?$', '', name)
        require(origin in ref, 'Unknown implementation primitive: ' + name)
        if name not in ref:
            require(origin != name and row['kind'] == ref[origin]['kind'] == 'FDRE',
                    'Unsupported added primitive: ' + name)
            added[name] = origin
        mapping[name] = origin

    for name, row in impl.items():
        require(definition(row, mapping) == definition(ref[mapping[name]], identity),
                'Definition/input mismatch: ' + name)

    represented = collections.defaultdict(list)
    for name, origin in mapping.items():
        represented[find(origin)].append(name)
    require({find(name) for name in ref} == set(represented),
            'Reference equivalence class entirely removed')
    require(set(reference['outputs']) == set(implementation['outputs']), 'Top output inventory changed')
    for pin, driver in reference['outputs'].items():
        require(signal(driver, identity) == signal(implementation['outputs'][pin], mapping),
                'Top output driver changed: ' + pin)

    removed = {name: dict(kind=ref[name]['kind'], representative=find(name),
                         implemented_equivalents=sorted(represented[find(name)]))
               for name in ref if name not in impl}
    return dict(status='PASS_PRIMITIVE_CONGRUENCE_R1', reference_cells=len(ref),
                implemented_cells=len(impl), equivalence_classes=len(represented),
                merge_rounds=rounds, removed=removed, added_mapping=added,
                checked_outputs=len(reference['outputs']), metadata_exclusions=sorted(METADATA),
                all_implementation_definitions_checked=True,
                all_reference_classes_represented=True, timing_qualified=False,
                routed_replay_qualified=False, board_qualified=False)
