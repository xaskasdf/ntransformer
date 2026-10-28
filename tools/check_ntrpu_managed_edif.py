"""Strict comparison of the observed Vivado EDIF export variation.

This is intentionally not a general EDIF equivalence engine. Accept only a
timestamp in the export header and permutations of complete, uniquely named
scalar properties directly inside one simple instance. Every other byte,
including primitive parameters, connections and instance identity, must match.
Unknown syntax/differences fail closed. No protected data is decrypted.
"""
import datetime
import hashlib
import re


PROPERTY = re.compile(
    rb' +\(property ([A-Za-z_][A-Za-z_0-9]*) '
    rb'\((?:string "[^"\r\n]*"|integer -?[0-9]+)\)\)\n'
)
INSTANCE = re.compile(
    rb' +\(instance ([A-Za-z_][A-Za-z_0-9]*) \(viewref netlist '
    rb'\(cellref ([A-Za-z_][A-Za-z_0-9]*) \(libraryref hdi_primitives\)\)\)\n'
)
TIMESTAMP = re.compile(rb'  \(timeStamp ([0-9 ]+)\)\n')


def compare(original, constrained):
    lhs, rhs = original.splitlines(keepends=True), constrained.splitlines(keepends=True)
    assert len(lhs) == len(rhs), 'EDIF line count differs'
    assert lhs[:6] == rhs[:6] and lhs[0].startswith(b'(edif ')
    assert lhs[4:6] == [b'(status\n', b' (written\n'], 'Unexpected timestamp context'
    stamps = []
    for lines in (lhs, rhs):
        m = TIMESTAMP.fullmatch(lines[6])
        assert m, 'Unexpected timestamp syntax'
        fields = list(map(int, m[1].split()))
        assert len(fields) == 6
        stamps.append(datetime.datetime(*fields).isoformat())
    differences = sum(a != z for a, z in zip(lhs, rhs))
    lhs[6] = rhs[6] = b'  (timeStamp EXPORT_PROVENANCE)\n'
    groups = []
    i = 0
    while i < len(lhs):
        if lhs[i] == rhs[i]:
            i += 1
            continue
        start = i
        # Include a same-valued leading property, but never cross an object.
        while start > 0 and PROPERTY.fullmatch(lhs[start - 1]) and PROPERTY.fullmatch(rhs[start - 1]):
            start -= 1
        assert start > 0 and lhs[start - 1] == rhs[start - 1], 'Instance context changed'
        parent = INSTANCE.fullmatch(lhs[start - 1])
        assert parent, 'Difference is not a direct scalar property of an instance'
        ends, names = [], []
        for lines in (lhs, rhs):
            end = start
            keys = []
            while end < len(lines) and (match := PROPERTY.fullmatch(lines[end])):
                keys.append(match[1])
                end += 1
            assert len(set(keys)) == len(keys), 'Duplicate property name'
            ends.append(end)
            names.append(keys)
        assert ends[0] == ends[1] and ends[0] > start, 'Property block size differs'
        end = ends[0]
        assert lhs[end] == rhs[end] and lhs[end].strip() == b')', 'Instance boundary changed'
        assert sorted(lhs[start:end]) == sorted(rhs[start:end]), 'Property name or value changed'
        groups.append(dict(first_line=start + 1, last_line=end,
                           instance=parent[1].decode(), primitive=parent[2].decode(),
                           properties=len(names[0]),
                           original_order=[x.decode() for x in names[0]],
                           constrained_order=[x.decode() for x in names[1]]))
        lhs[start:end] = rhs[start:end] = sorted(lhs[start:end])
        i = end
    assert lhs == rhs, 'Unexplained EDIF difference'
    return dict(identical_except_export_provenance_and_local_property_order=True,
                bytes=len(original), lines=len(lhs), differing_lines=differences,
                export_timestamps=stamps, reordered_property_groups=groups,
                canonical_sha256=hashlib.sha256(b''.join(lhs)).hexdigest())


def mutation_checks(original, constrained):
    """Use actual full exports; altered values/connections must be rejected."""
    rejected = []
    def reject(name, changed):
        assert changed != constrained, name
        try:
            compare(original, changed)
        except (AssertionError, ValueError):
            rejected.append(name)
        else:
            raise AssertionError('Accepted corruption: ' + name)

    for letter in 'ABCDEFGH':
        old = ('(property INIT_' + letter + ' (string "64\'h0000000000000000"))').encode()
        assert old in constrained
        reject('INIT_' + letter, constrained.replace(old, old.replace(b'0000000000000000', b'0000000000000001'), 1))
    match = re.search(rb'\(portref [^\s()]+', constrained)
    assert match
    reject('net_port_connection', constrained[:match.end()] + b'_ALTERED' + constrained[match.end():])
    old = b'(instance address_reg_0_3_0_13 '
    assert old in constrained
    reject('instance_identity', constrained.replace(old, old.replace(b'_0_13', b'_0_14'), 1))
    line = next(x for x in constrained.splitlines(keepends=True) if b'(property RTL_RAM_BITS (integer 148))' in x)
    reject('ram_geometry', constrained.replace(line, line.replace(b'148', b'149'), 1))
    reject('duplicate_property', constrained.replace(line, line + line, 1))
    reject('missing_property', constrained.replace(line, b'', 1))
    reject('primitive_type', constrained.replace(b'(cellref RAM32M16 ', b'(cellref RAM32M4 ', 1))
    return rejected
