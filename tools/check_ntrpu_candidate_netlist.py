"""Replay reference protocol and independently compare actual routed port traces."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import check_ntrpu_candidate_dispatch as candidate
import check_ntrpu_registered_dispatch as reference
from prepare_ntrpu_candidate_netlist import ALWAYS, SCALAR_DATA, CHANNEL_DATA

need = reference.need


def routed_trace(text):
    expected, actual = defaultdict(list), defaultdict(list)
    samples, matches = Counter(), Counter()
    case = None
    count = 0
    for line in text.splitlines():
        f = line.split()
        if line.startswith('RS S '):
            case = int(f[3]);samples[case] += 1
        elif line.startswith('RS FETCH '):
            expected[case, 'FETCH', int(f[3])].append((int(f[5], 16),))
        elif line.startswith('RS W '):
            expected[case, 'W', int(f[3])].append((int(f[7], 16), (1 << 64)-1, int(f[6], 2)))
        elif line.startswith('RS CPL '):
            expected[case, 'CPL', int(f[3])].append((int(f[4]), int(f[5])))
        elif line.startswith('CN MATCH '):
            count += 1
            need(int(f[3]) == count, 'every routed comparison occurs exactly once')
            matches[int(f[2])] += 1
        elif line.startswith('CN FETCH '):
            actual[int(f[2]), 'FETCH', int(f[3])].append((int(f[4], 16),))
        elif line.startswith('CN W '):
            actual[int(f[2]), 'W', int(f[3])].append((int(f[4], 16), int(f[5], 16), int(f[6], 2)))
        elif line.startswith('CN CPL '):
            actual[int(f[2]), 'CPL', int(f[3])].append((int(f[4]), int(f[5])))
    need(count > 0 and samples == matches, 'routed comparison covers every active reference sample, including final flush')
    need(re.findall(r'CN ALL_PASS checks=(\d+)', text) == [str(count)], 'one final routed comparison marker')
    need(expected == actual, 'routed payload, metadata, completion status and order match reference')
    return dict(samples=count, by_case=dict(matches),
                fetches=sum(len(v) for (c, kind, ch), v in actual.items() if kind == 'FETCH'),
                completions=sum(len(v) for (c, kind, ch), v in actual.items() if kind == 'CPL'),
                bytes=64*sum(len(v) for (c, kind, ch), v in actual.items() if kind == 'W'))


def negatives(text):
    original = text.splitlines()
    results = []
    for mutation in ('routed_data_bit_flip', 'routed_beat_lost', 'routed_comparison_lost', 'routed_bad_completion'):
        lines = list(original)
        prefix = 'CN MATCH ' if mutation == 'routed_comparison_lost' else 'CN CPL ' if mutation == 'routed_bad_completion' else 'CN W '
        i = next(i for i, line in enumerate(lines) if line.startswith(prefix))
        if mutation.endswith('_lost'):
            del lines[i]
        else:
            f = lines[i].split()
            if mutation == 'routed_data_bit_flip':
                f[4] = format(int(f[4], 16) ^ 1, '0128x')
            else:
                f[5] = str(int(f[5]) ^ 2)
            lines[i] = ' '.join(f)
        try:
            routed_trace('\n'.join(lines))
        except ValueError as exc:
            results.append(dict(mutation=mutation, rejection=str(exc)))
        else:
            raise ValueError('Missed negative control '+mutation)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    out = args.run/'output'
    record = json.loads((out/'result.json').read_text())
    need(record['status'] == 'EXPORTED_PENDING_REVIEW' and record['returncode'] == 0 and not record['timed_out'], 'completed simulation')
    need(record['sources'] == record['sources_after'], 'sources unchanged')
    netlist = out/'inputs/candidate_fixture_funcsim.v'
    net_hash = hashlib.sha256(netlist.read_bytes()).hexdigest()
    need(net_hash == record['netlist_sha256'] == record['netlist_sha256_after'], 'unchanged routed netlist')
    prior = json.loads((out/'inputs/FILES.json').read_text())
    need(prior['candidate_fixture_funcsim.v'] == net_hash, 'route batch netlist provenance')
    manifest = json.loads((out/'suite/generated/NETLIST_COMPARISON.json').read_text())
    need(manifest['netlist_sha256'] == net_hash and manifest['latency_edges'] == 2 and manifest['startup_ns'] == 150 and
         not manifest['sdf'] and not manifest['force'] and not record['sdf'], 'explicit sampler latency, startup and functional-only scope')
    need(not set(manifest['modules'].values()) & set(manifest['netlist_modules']), 'reference and netlist module names disjoint')
    need(all(v['reversible'] for v in manifest['sources'].values()), 'reference identifier transformation reversible')
    outputs = {name for direction, width, name in manifest['ports'] if direction == 'output'}
    need(outputs == set(ALWAYS) | set(SCALAR_DATA) | set(CHANNEL_DATA), 'all outputs checked with valid-qualified payload')
    text = (out/'suite/simulation.log').read_text()
    extra = (out/'suite/extra.log').read_text()
    need(not re.search(r'(?mi)^\s*(Fatal:|ERROR:|FAIL )', text+'\n'+extra), 'simulation diagnostics clean')
    result = reference.replay(text)
    result.update(registered_heads=reference.check_heads(text), candidates=candidate.candidates(text),
                  extra=candidate.extra(extra), routed=routed_trace(text), routed_extra=routed_trace(extra),
                  negative_controls=reference.negatives(text)+reference.head_negatives(text)+candidate.candidate_negatives(text)+negatives(text))
    need(result['candidates']['committed'] == 85 and result['routed']['fetches'] == result['routed']['completions'] == 85,
         'all eight-case transactions reproduced')
    need(result['routed']['bytes'] == 138944 and result['routed_extra']['bytes'] == 4096 and
         result['routed_extra']['completions'] == 1, 'exact routed payload total')
    result.update(status='ROUTED_FUNCTIONAL_EQUIVALENCE_PASS_INTEGRATION_PENDING', sdf=False,
                  netlist_sha256=net_hash, output_ports=len(outputs), reference_internal_monitors=True,
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (args.run/'REVIEW.json').open('x') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
