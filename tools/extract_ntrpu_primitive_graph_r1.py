"""Flatten reviewed EDIF syntax while preserving primitive parameters and pins.

Reuse the immutable hierarchy parser prefix; the existing complete DAG/clock
audit is a separate prerequisite. Never infer a connection from a cell name.
"""
import ast
from collections import defaultdict
import json
from pathlib import Path
import audit_ntrpu_hier_edif_clock_r2 as parser


def flatten(text):
    module = ast.parse(Path(parser.__file__).read_text())
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    points = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == 'connections' for t in n.targets)]
    if len(points) != 1:
        raise ValueError('Hierarchy parser boundary changed')
    fn.body = fn.body[:points[0]] + ast.parse(
        'return dict(cells=cells, interfaces=interfaces, root=root, leafs=leafs, properties=leaf_properties, '
        'pins=pins, top_inputs=top_inputs, parents=parents, find=find)').body
    ns = dict(vars(parser))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<reviewed-hierarchy-parser-prefix>', 'exec'), ns)
    data = ns['audit'](text)
    native = {}

    def display(n):
        return json.loads(n[2]) if isinstance(n, list) else n

    def names(key, internal=(), actual=()):
        for inst in parser.children(parser.one(data['cells'][key], 'contents'), 'instance'):
            label = parser.name(inst[1])
            target = parser.cell_reference(parser.one(parser.one(inst, 'viewref'), 'cellref'))
            path, actual_path = internal + (label,), actual + (display(inst[1]),)
            if target[0] == 'hdi_primitives':
                native[path] = '/'.join(actual_path)
            else:
                names(target, path, actual_path)

    names(data['root'])
    if len(native) != len(data['leafs']) or len(set(native.values())) != len(native):
        raise ValueError('Ambiguous native primitive names')
    roots = defaultdict(list)
    for ep, (inst, direction) in data['pins'].items():
        if direction == 'OUTPUT':
            roots[data['find'](ep)].append([native[inst], ep[2], ep[3]])
    for ep in data['top_inputs']:
        roots[data['find'](ep)].append(['TOP_INPUT', ep[2], ep[3]])
    if any(len(v) != 1 for v in roots.values()):
        raise ValueError('Multiple drivers')

    def driver(ep):
        found = roots.get(data['find'](ep))
        if not found:
            raise ValueError('Undriven input/output: ' + str(ep))
        return found[0]

    cells = {native[p]: dict(kind=k, properties=data['properties'][p], inputs={})
             for p, k in data['leafs'].items()}
    for ep, (inst, direction) in data['pins'].items():
        if direction == 'INPUT':
            cells[native[inst]]['inputs'][ep[2] + ':' + str(ep[3])] = driver(ep)
    outputs, inputs = {}, []
    for port, (direction, width) in data['interfaces'][data['root']].items():
        for bit in range(width):
            ep = ('port', (), port, bit)
            key = port + ':' + str(bit)
            if direction == 'OUTPUT':
                outputs[key] = driver(ep)
            else:
                inputs.append(key)
    return dict(cells=cells, outputs=outputs, inputs=sorted(inputs))
