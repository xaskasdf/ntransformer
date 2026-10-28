"""Preserve the physical review and verify explicit parent routing preflight."""
import copy
import re
import check_ntrpu_attention_projected_root_physopt_r1 as original


def review(evidence, checkpoint_sha):
    result = original.review(evidence, checkpoint_sha)
    files = evidence['files']
    identity = dict(line.split('\t') for line in files['output/parent_identity.tsv'].splitlines())
    assert identity['top'] == 'ntrpu_attention_projected_root_r1'
    assert identity['part'] == 'xcvu5p-flva2104-2-i'
    assert float(identity['clock_period']) == 5 and identity['blackboxes'] == '0'
    counts = {}
    for label, expected in [('# of routable nets', 86884), ('# of fully routed nets', 86884), ('# of nets with routing errors', 0)]:
        found = re.findall(re.escape(label)+r'\.+\s*:\s*(\d+)', files['output/parent_route.rpt'])
        assert len(found) == 1 and int(found[0]) == expected, 'Parent route count '+label
        counts[label] = int(found[0])
    result['parent_route_preflight'] = dict(identity=identity, routing=counts, property_flag_used_as_gate=False)
    return result


def controls(evidence, checkpoint_sha):
    rejected = original.controls(evidence, checkpoint_sha)
    for kind in ('parent_route_total', 'parent_route_incomplete', 'parent_route_errors', 'parent_identity'):
        changed = copy.deepcopy(evidence)
        files = changed['files']
        if kind == 'parent_identity':
            files['output/parent_identity.tsv'] = files['output/parent_identity.tsv'].replace('ntrpu_attention_projected_root_r1', 'wrong_top')
        else:
            label, value = {'parent_route_total': ('# of routable nets', 86885),
                            'parent_route_incomplete': ('# of fully routed nets', 86883),
                            'parent_route_errors': ('# of nets with routing errors', 1)}[kind]
            files['output/parent_route.rpt'] = re.sub('('+re.escape(label)+r'\.+\s*:\s*)\d+', lambda m:m[1]+str(value), files['output/parent_route.rpt'])
        assert changed != evidence
        try:
            review(changed, checkpoint_sha)
        except (AssertionError, KeyError, ValueError):
            rejected.append(dict(name=kind, rejected=True))
        else:
            raise AssertionError('Accepted '+kind)
    return rejected


buffer_controls = original.buffer_controls
