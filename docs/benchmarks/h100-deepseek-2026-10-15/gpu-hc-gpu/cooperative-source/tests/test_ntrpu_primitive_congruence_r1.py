"""Adversarial tests for the sequential relation, not name-based acceptance."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_ntrpu_primitive_congruence_r1 import compare, CongruenceError
from extract_ntrpu_primitive_graph_r1 import flatten


def verify():
    def top(p):
        return ['TOP_INPUT', p, 0]
    ff = dict(kind='FDRE', properties={'INIT': ['string', '"1\'b0"']},
              inputs={'C:0': top('clk'), 'CE:0': top('ce'), 'R:0': top('rst'), 'D:0': top('din')})
    lut = dict(kind='LUT1', properties={'INIT': ['string', '"2\'h2"']}, inputs={'I0:0': ['a', 'Q', 0]})
    ref = dict(cells={'a': ff, 'b': copy.deepcopy(ff), 'l': lut, 'm': copy.deepcopy(lut)},
               inputs=['clk:0', 'ce:0', 'rst:0', 'din:0'], outputs={'out:0': ['m', 'O', 0]})
    ref['cells']['m']['inputs']['I0:0'] = ['b', 'Q', 0]
    impl = copy.deepcopy(ref)
    del impl['cells']['b'], impl['cells']['m']
    impl['cells']['a_replica'] = copy.deepcopy(ff)
    impl['cells']['l']['inputs']['I0:0'] = ['a_replica', 'Q', 0]
    impl['outputs']['out:0'] = ['l', 'O', 0]
    positive = compare(ref, impl)
    assert set(positive['removed']) == {'b', 'm'}
    assert positive['added_mapping'] == {'a_replica': 'a'}
    controls = []

    def reject(name, mutate):
        r, i = copy.deepcopy(ref), copy.deepcopy(impl)
        mutate(r, i)
        try:
            compare(r, i)
        except CongruenceError as e:
            controls.append(dict(name=name, rejected=True, reason=str(e)))
        else:
            raise AssertionError('Accepted ' + name)

    for pin in ['C', 'CE', 'R', 'D']:
        reject('replica_wrong_' + pin, lambda r, i, p=pin:
               i['cells']['a_replica']['inputs'].__setitem__(p + ':0', top('rst' if p != 'R' else 'ce')))
    reject('replica_wrong_init', lambda r, i: i['cells']['a_replica']['properties'].__setitem__('INIT', ['string', '"1\'b1"']))
    reject('replica_wrong_polarity', lambda r, i: i['cells']['a_replica']['properties'].__setitem__('IS_R_INVERTED', ['integer', '1']))
    reject('lut_wrong_init', lambda r, i: i['cells']['l']['properties'].__setitem__('INIT', ['string', '"2\'h1"']))
    reject('wrong_top_output', lambda r, i: i['outputs'].__setitem__('out:0', top('din')))
    reject('missing_input', lambda r, i: i['cells']['a_replica']['inputs'].pop('CE:0'))
    reject('missing_initial_state', lambda r, i: i['cells']['a_replica']['properties'].pop('INIT'))
    reject('unknown_added_primitive', lambda r, i: i['cells'].__setitem__('unknown', copy.deepcopy(ff)))
    reject('removed_unmatched_register', lambda r, i: r['cells']['b']['properties'].__setitem__('INIT', ['string', '"1\'b1"']))
    reject('wrong_driver_pin', lambda r, i: i['cells']['l']['inputs'].__setitem__('I0:0', ['a', 'D', 0]))
    reject('wrong_driver_bit', lambda r, i: i['cells']['l']['inputs'].__setitem__('I0:0', ['a', 'Q', 1]))
    reject('unknown_property', lambda r, i: i['cells']['l']['properties'].__setitem__('UNKNOWN', ['integer', '1']))
    reject('missing_output', lambda r, i: i['outputs'].clear())
    reject('undeclared_top_input', lambda r, i: i['inputs'].remove('din:0'))
    # Equal shapes in a feedback loop are not a proof of equal state.
    reject('unproved_mutual_feedback', lambda r, i: (
        r['cells']['a']['inputs'].__setitem__('D:0', ['b', 'Q', 0]),
        r['cells']['b']['inputs'].__setitem__('D:0', ['a', 'Q', 0]),
        i['cells']['a']['inputs'].__setitem__('D:0', ['a', 'Q', 0])))

    fixture = '''(edif f (library work
      (cell child (view v (interface (port i (direction INPUT)) (port o (direction OUTPUT)))
       (contents (instance (rename l "lut[0]") (viewref v (cellref LUT1 (libraryref hdi_primitives)))
                  (property INIT (string "2'h2")))
        (net i (joined (portref i) (portref I0 (instanceref l))))
        (net o (joined (portref O (instanceref l)) (portref o))))))
      (cell top (view v (interface (port i (direction INPUT)) (port o (direction OUTPUT)))
       (contents (instance c (viewref v (cellref child (libraryref work))))
        (net i (joined (portref i) (portref i (instanceref c))))
        (net o (joined (portref o (instanceref c)) (portref o)))))))
      (design top (cellref top (libraryref work))))'''
    graph = flatten(fixture)
    assert graph['cells']['c/lut[0]']['inputs'] == {'I0:0': top('i')}
    assert graph['outputs'] == {'o:0': ['c/lut[0]', 'O', 0]}
    assert compare(graph, graph)['checked_outputs'] == 1
    for name, bad in [
        ('undriven_output', fixture.replace('(portref O (instanceref l))', '')),
        ('multiple_drivers', fixture.replace('(portref I0 (instanceref l))', '(portref I0 (instanceref l)) (portref O (instanceref l))')),
        ('unsupported_primitive', fixture.replace('cellref LUT1', 'cellref UNKNOWN')),
    ]:
        try:
            flatten(bad)
        except ValueError as e:
            controls.append(dict(name=name, rejected=True, reason=str(e)))
        else:
            raise AssertionError('Accepted ' + name)
    return dict(status='PASS_PRIMITIVE_CONGRUENCE_CONTROLS_R1', controls=controls,
                positive_removed=sorted(positive['removed']), native_hierarchy_and_ports_checked=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
