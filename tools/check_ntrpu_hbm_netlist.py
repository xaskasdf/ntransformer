"""Independent replay and completeness checks for routed HBM-side experiments."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_hbm_cdc import need, replay, cdc_checks, controls, cdc_controls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base(out):
    files = json.loads((out/'FILES.json').read_text())
    need(all(sha(out/n) == h for n, h in files.items()), 'Evidence hash mismatch')
    r = json.loads((out/'result.json').read_text())
    need(r['experiment'] == 'hbm_netlist' and r['status'] == 'EXPORTED_PENDING_REVIEW', 'Tool failure')
    need(r['sources'] == r['sources_after'] and r['generated'] == r['generated_after'], 'Changed sources')
    need(r['input_sha256'] == r['input_after'] == r['donor_after'], 'Changed routed input')
    return r


def routed_trace(text):
    expected = Counter(); observed = Counter(); samples = {'SOURCE': [], 'MEMORY': []}
    case = 0; finish = []; resets = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 2:
            continue
        prefix, kind = f[:2]
        if f[:2] == ['TE', 'CASE']:
            case = int(f[3])
        if prefix == 'HT' and kind in ('AW', 'W', 'B'):
            time, lane = map(int, f[2:4])
            if kind == 'AW':
                values = (int(f[5], 16), int(f[6]))
            elif kind == 'W':
                values = (int(f[6], 16), int(f[7], 16), int(f[8]))
            else:
                values = tuple(map(int, f[5:7]))
            expected[(kind, time, case, lane, *values)] += 1
        elif prefix == 'TE' and kind in ('FETCH', 'CPL'):
            time, lane = map(int, f[2:4])
            values = (int(f[5], 16),) if kind == 'FETCH' else tuple(map(int, f[4:6]))
            expected[(kind, time, case, lane, *values)] += 1
        elif prefix == 'HN':
            if kind in samples:
                samples[kind].append(tuple(map(int, f[2:])))
            elif kind == 'ALL_PASS':
                finish.append(dict(s.split('=') for s in f[2:]))
            elif kind == 'RESET':
                resets.append(tuple(map(int, f[2:])))
            elif kind in ('AW', 'W', 'B', 'FETCH', 'CPL'):
                time, number, lane = map(int, f[2:5])
                if kind == 'AW':
                    values = (int(f[5], 16), int(f[6]))
                elif kind == 'W':
                    values = (int(f[5], 16), int(f[6], 16), int(f[7]))
                elif kind == 'FETCH':
                    values = (int(f[5], 16),)
                else:
                    values = tuple(map(int, f[5:7]))
                observed[(kind, time, number, lane, *values)] += 1
    need(len(finish) == 1, 'Missing/duplicate routed suite completion')
    need([v[1] for v in resets] == list(range(21)), 'Missing/duplicate settled cold-reset checks')
    need(expected == observed and expected, 'Routed event mismatch, duplicate or removal')
    coverage = {}
    for domain, records in samples.items():
        need(records and [v[2] for v in records] == list(range(1, len(records)+1)), 'Routed sample sequence')
        need(int(finish[0][domain.lower()]) == len(records), 'Routed sample total')
        need(all(a[0] < b[0] for a, b in zip(records, records[1:])), 'Routed sample time ordering')
        by_case = Counter(v[1] for v in records)
        need(set(by_case) == set(range(1, 22)), 'Routed scenario coverage')
        coverage[domain] = dict(samples=len(records), by_case=dict(by_case))
    return dict(domains=coverage, events=dict(Counter(k[0] for k in observed.elements())), reset_checks=len(resets),
                shared_inputs=True, cycle_alignment=0, formal_equivalence=False, sdf=False)


def routed_controls(text):
    result = []
    for prefix in ('HN AW ', 'HN W ', 'HN B ', 'HN FETCH ', 'HN CPL ', 'HN SOURCE ', 'HN MEMORY ', 'HN RESET ', 'HN ALL_PASS '):
        lines = text.splitlines(); index = next(i for i, s in enumerate(lines) if s.startswith(prefix)); del lines[index]
        try:
            routed_trace('\n'.join(lines))
        except ValueError:
            result.append('remove_'+prefix.strip().replace(' ', '_'))
        else:
            raise ValueError('Undetected removal '+prefix)
    lines = text.splitlines(); i = next(i for i, s in enumerate(lines) if s.startswith('HN W '))
    fields = lines[i].split(); fields[5] = format(int(fields[5], 16)^1, 'x'); lines[i] = ' '.join(fields)
    try:
        routed_trace('\n'.join(lines))
    except ValueError:
        result.append('corrupt_HN_W_payload')
    else:
        raise ValueError('Undetected routed payload corruption')
    return result


def simulation(out, r):
    text = (out/'suite/simulation.log').read_text()
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )', text, re.M), 'Simulation failure')
    normalized = text.replace('CD ALL_PASS cases=21 ', 'HT ALL_PASS cases=21 ')
    proof = json.loads((out/'suite/NETLIST_NAMESPACE.json').read_text())
    need(proof['status'] == 'BYTE_IDENTICAL_LIBRARY_ISOLATION' and proof['library'] == 'hbm_routed', 'Namespace isolation')
    need(proof['original_sha256'] == proof['isolated_sha256'] == proof['restored_sha256'] == r['input_sha256'], 'Netlist identity')
    need(sha(out/'suite/generated/route_isolated.v') == proof['isolated_sha256'], 'Isolated circuit identity')
    need(sha(out/'suite/generated/route_library.v') == proof['library_wrapper_sha256'], 'Scoped library binding')
    elaboration = list((out/'tool_logs').rglob('elaborate.log'))
    need(len(elaboration) == 1, 'Missing elaboration evidence')
    binding = elaboration[0].read_text()
    for library in ('xil_defaultlib', 'hbm_routed'):
        for module in ('ntrpu_hbm_startup', 'hbm_cdc'):
            need('Compiling module '+library+'.'+module in binding, 'Reference/routed library binding')
    expected = {'PROTOCOL': 'AXI4', 'READ_WRITE_MODE': 'WRITE_ONLY', 'DATA_WIDTH': '512',
                'ADDR_WIDTH': '64', 'ID_WIDTH': '4', 'ACLK_ASYNC': '1', 'SYNCHRONIZATION_STAGES': '3'}
    config = dict(line.split('\t', 1) for line in (out/'suite/hbm_cdc_config.tsv').read_text().splitlines() if '\t' in line)
    need(all(config['CONFIG.'+k] == v for k, v in expected.items()), 'Changed CDC IP parameters')
    negatives = {}
    for kind, signal in (('PAYLOAD', 'h_wdata'), ('READY', 'submit_ready')):
        failed = (out/f'suite/negative_{kind}.log').read_text()
        need('HN_MISMATCH '+signal in failed and 'HN ALL_PASS' not in failed, 'Undetected output injection '+kind)
        negatives[kind] = re.search(r'Fatal:.*HN_MISMATCH[^\n]*', failed)[0]
    return dict(status='ROUTED_FUNCTIONAL_REGRESSION_PASS', reference_replay=replay(normalized),
                cdc=cdc_checks(text), routed=routed_trace(text),
                negative_trace_controls=controls(normalized)+cdc_controls(text)+routed_controls(text),
                negative_simulations=negatives, netlist_sha256=r['input_sha256'],
                native_hbm_traffic=False, board_timing_qualified=False)


def rows(out, name):
    with (out/name).open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def lut_used(ref, init, pin):
    if not re.fullmatch(r'LUT[1-6]', ref) or not re.fullmatch(r'I[0-5]', pin):
        return None
    width = int(ref[-1]); index = int(pin[1]); need(index < width, 'LUT pin index')
    match = re.fullmatch(r"(\d+)'h([0-9a-fA-F]+)", init)
    need(match is not None and int(match[1]) == 2**width, 'LUT INIT format')
    value = int(match[2], 16)
    return any(((value >> j) ^ (value >> (j ^ (1 << index)))) & 1 for j in range(2**width))


def drc(out, r):
    # Complete object inventories, not the report's first-15-object abbreviation.
    violations = defaultdict(dict)
    for row in rows(out, 'violations.tsv'):
        violations[row['violation']][row['property']] = row['value']
    need(len(violations) == 13, 'Unexpected DRC count')
    need(all(v['SEVERITY'] == 'Warning' and v['IS_WAIVED'] == '0' for v in violations.values()), 'Unexpected severity or waiver')
    inventory = Counter(v.split('#')[0] for v in violations)
    need(inventory == {'PDCN-1569': 12, 'RTSTAT-10': 1}, 'Unexpected DRC rules')
    properties = defaultdict(dict)
    for row in rows(out, 'cells.tsv'):
        properties[(row['violation'], row['cell'])][row['property']] = row['value']
    pins = rows(out, 'pins.tsv'); lut_evidence = []
    for name, values in violations.items():
        if not name.startswith('PDCN-1569'):
            continue
        description = ' '.join(values.values())
        found = []
        for (violation, cell), props in properties.items():
            if violation != name:
                continue
            for pin in pins:
                if pin['violation'] == name and pin['cell'] == cell and '(pin '+pin['pin']+')' in description:
                    logical = pin['pin'].rsplit('/', 1)[1]
                    used = lut_used(props['REF_NAME'], props['INIT'], logical)
                    need(used is False, 'Reported unused LUT input affects INIT')
                    found.append(dict(cell=cell, pin=logical, ref=props['REF_NAME'], init=props['INIT'],
                                      net=pin['net'], physical_pins=pin['bel_pins']))
        need(len(found) == 1, 'Missing/ambiguous DRC pin association')
        lut_evidence.append(dict(violation=name, **found[0]))
    nets = rows(out, 'nets.tsv'); loads = rows(out, 'loads.tsv')
    rt = [n for n in nets if n['violation'].startswith('RTSTAT-10')]
    reported = int(re.match(r'(\d+) net\(s\)', violations['RTSTAT-10#1']['DESCRIPTION'])[1])
    need(len(rt) == len({n['net'] for n in rt}) == reported == 1218, 'Truncated RTSTAT net inventory')
    classification = Counter(); unresolved = []; net_evidence = []
    for net in rt:
        inputs = [p for p in loads if p['violation'] == net['violation'] and p['net'] == net['net'] and p['direction'] == 'IN']
        used = [lut_used(p['ref_name'], p['init'], p['pin'].rsplit('/', 1)[1]) for p in inputs]
        if not inputs and net['ports'] and net['route_status'] == 'HIERPORT':
            kind = 'external_ooc_port_without_internal_load'
        elif not inputs and not net['ports'] and net['route_status'] == 'NOLOADS':
            kind = 'no_logical_leaf_loads'
        elif inputs and all(v is False for v in used) and not net['ports']:
            kind = 'only_functionally_unused_lut_inputs'
        else:
            kind = 'requires_further_review'; unresolved.append(dict(net=net, inputs=inputs, functional_use=used))
        classification[kind] += 1
        net_evidence.append(dict(net=net['net'], route_status=net['route_status'], classification=kind, loads=len(inputs)))
    report = (out/'route_status.rpt').read_text()
    need(re.search(r'# of nets with routing errors[.\s]*:\s*0', report, re.I), 'Routing errors')
    return dict(status='ROUTED_DRC_OBJECTS_REVIEWED', warnings=13, rules=dict(inventory),
                unused_lut_pins=lut_evidence, rtstat_classification=dict(classification),
                rtstat_nets=net_evidence, unresolved=unresolved,
                warnings_waived=False, drc_clean=False, native_hbm_traffic=False,
                checkpoint_sha256=r['input_sha256'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path); ap.add_argument('--review', type=Path, required=True)
    a = ap.parse_args(); r = base(a.output)
    result = simulation(a.output, r) if r['stage'] == 'sim' else drc(a.output, r)
    result.update(seconds=r['seconds'], checker_sha256=sha(Path(__file__)))
    a.review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rtstat_nets', 'unused_lut_pins', 'unresolved', 'reference_replay')}))


if __name__ == '__main__':
    main()
