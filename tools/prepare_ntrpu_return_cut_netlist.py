"""Isolate structural module names without changing the routed circuit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

TOKEN = re.compile(rb'//[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\\[^\s]+|[A-Za-z_$][A-Za-z0-9_$]*', re.S)
KEEP = {b'ntrpu_transport_physical', b'glbl'}
PREFIX = b'routeiso_'


def rename(data, mapping):
    previous = b''
    counts = {n: 0 for n in mapping}

    def replace(match):
        nonlocal previous
        token = match[0]
        if token.startswith((b'//', b'/*', b'"')):
            return token
        if token in mapping:
            # Only module declarations and instantiated types may change.
            assert previous == b'module' or re.match(
                rb'\s+(?:\\\S+|[A-Za-z_$][A-Za-z0-9_$]*)\s*\(', data[match.end():match.end()+512])
            counts[token] += 1
        previous = token
        return mapping.get(token, token)

    return TOKEN.sub(replace, data), counts


def isolate(data):
    modules = re.findall(rb'^module (\w+)\b', data, re.M)
    assert len(modules) == len(set(modules)) and KEEP <= set(modules)
    assert len(modules) == len(re.findall(rb'^\s*module\b', data, re.M))
    assert PREFIX not in data
    mapping = {n: PREFIX+n for n in modules if n not in KEEP}
    changed, counts = rename(data, mapping)
    assert all(n >= 1 for n in counts.values())
    restored, _ = rename(changed, {v: k for k, v in mapping.items()})
    assert restored == data
    glbl = lambda raw: re.search(rb'^module glbl\b[\s\S]*?^endmodule', raw, re.M)[0]
    assert glbl(data) == glbl(changed)
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    proof = dict(status='REVERSIBLE_MODULE_NAMESPACE_PASS', original_sha256=sha(data),
                 isolated_sha256=sha(changed), restored_sha256=sha(restored),
                 glbl_sha256=sha(glbl(data)), retained_modules=sorted(n.decode() for n in KEEP),
                 mapping={k.decode(): v.decode() for k, v in mapping.items()},
                 replacements={k.decode(): v for k, v in counts.items()},
                 comments_strings_escaped_identifiers_unchanged=True)
    return changed, proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path); ap.add_argument('--netlist', type=Path, required=True)
    a = ap.parse_args()
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('prepare_ntrpu_return_cut.py')),
                    str(a.output), '--netlist', str(a.netlist)], check=True)
    changed, proof = isolate(a.netlist.read_bytes())
    (a.output/'route_isolated.v').write_bytes(changed)
    (a.output/'NETLIST_NAMESPACE.json').write_text(json.dumps(proof, indent=2)+'\n')


if __name__ == '__main__':
    main()
