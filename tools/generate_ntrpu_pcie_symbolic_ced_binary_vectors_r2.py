"""Select whole cold-isolated binary scenarios for exact C/E/D netlist replay."""
import hashlib
import json
from pathlib import Path
import re
import sys

RAW = dict(main='pcie_request_inventory_r7.txt', ownership='pcie_rc_ownership_integration.txt',
           rq='pcie_rq_admission_integration.txt', prefix='pcie_rc_prefix_integration.txt',
           lanes='pcie_rc_lane_integration_rtl.txt', context='pcie_symbolic_value_context.txt')
KNOWN = dict(main='pcie_request_inventory_binary_r5.txt', lanes='pcie_rc_lane_integration_binary.txt',
             context='pcie_symbolic_value_context_binary.txt')


def select(lines):
    groups = {}
    prior = None
    for index, line in enumerate(lines):
        row = line.split()
        assert len(row) == 28
        scenario = row[0]
        if scenario != prior:
            assert scenario not in groups and row[2] == '0'
        groups.setdefault(scenario, []).append((index, line))
        prior = scenario
    excluded = [key for key, rows in groups.items()
                if any(re.search('[xz]', ' '.join(line.split()[2:16]), re.I) for _, line in rows)]
    output = [line for key, rows in groups.items() if key not in excluded for _, line in rows]
    assert output and not any(re.search('[xz]', ' '.join(line.split()[2:16]), re.I) for line in output)
    return output, [dict(scenario=k, first_row=groups[k][0][0], rows=len(groups[k])) for k in excluded]


if __name__ == '__main__':
    root = Path(sys.argv[1])
    manifest_path = root/'pcie_symbolic_ced_binary_r1.json'
    assert not manifest_path.exists()
    specs = {**{'raw_'+k: v for k, v in RAW.items()},
             **{'drained_'+k: 'pcie_symbolic_ced_drained_'+k+'_r1.txt' for k in RAW},
             'directed': 'pcie_symbolic_ced_directed_r1.txt'}
    # Also preserve the entire previously reviewed binary main corpus. It is
    # supplemental; it is not represented as a byte-identical raw-r7 subset.
    from generate_ntrpu_pcie_symbolic_ced_drained_vectors import drained
    supplemental_source = root/'pcie_request_inventory_binary_r5.txt'
    supplemental_rows, supplemental_insertions = drained(supplemental_source.read_text().splitlines())
    supplemental_path = root/'pcie_symbolic_ced_established_main_drained_r1.txt'
    with supplemental_path.open('x', newline='\n') as f:
        f.write('\n'.join(supplemental_rows)+'\n')
    specs['established_main'] = supplemental_source.name
    specs['established_main_drained'] = supplemental_path.name
    banks = {}
    for name, filename in specs.items():
        source = root/filename
        original = source.read_bytes()
        lines = original.decode().splitlines()
        selected, excluded = select(lines)
        raw = ('\n'.join(selected)+'\n').encode()
        if not excluded:
            assert raw == original
            target = source
        elif name.startswith('raw_') and name[4:] in KNOWN and (root/KNOWN[name[4:]]).read_bytes() == raw:
            target = root/KNOWN[name[4:]]
        else:
            target = root/('pcie_symbolic_ced_binary_'+name+'_r1.txt')
            with target.open('xb') as f:
                f.write(raw)
        assert target.read_bytes() == raw
        banks[name] = dict(source=source.name, source_sha256=hashlib.sha256(original).hexdigest(),
                           path=target.name, sha256=hashlib.sha256(raw).hexdigest(),
                           source_rows=len(lines), rows=len(selected), excluded_whole_scenarios=excluded,
                           scenarios=len({v.split()[0] for v in selected}), unchanged=not excluded)
    smoke_source = root/banks['directed']['path']
    smoke = b'\n'.join(smoke_source.read_bytes().splitlines()[:64])+b'\n'
    smoke_path = root/'pcie_symbolic_ced_binary_smoke_r1.txt'
    with smoke_path.open('xb') as f:
        f.write(smoke)
    record = dict(banks=banks,
                  supplemental_main=dict(source=supplemental_source.name,
                      source_sha256=hashlib.sha256(supplemental_source.read_bytes()).hexdigest(),
                      drained_path=supplemental_path.name, insertions=supplemental_insertions,
                      generator='generate_ntrpu_pcie_symbolic_ced_drained_vectors.py',
                      generator_sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_symbolic_ced_drained_vectors.py').read_bytes()).hexdigest(),
                      scope='Previously reviewed binary corpus, retained as a separate supplemental bank; not an exact subset of raw r7.'),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  smoke=dict(path=smoke_path.name, sha256=hashlib.sha256(smoke).hexdigest(), rows=64,
                             source=smoke_source.name, source_sha256=hashlib.sha256(smoke_source.read_bytes()).hexdigest()),
                  scope='Netlist replay uses complete cold-isolated scenarios with known 0/1 inputs. Four-state raw RTL banks remain mandatory and unchanged. No post-fault or malformed binary traffic is filtered.')
    with manifest_path.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(json.dumps({k: dict(rows=v['rows'], excluded=v['excluded_whole_scenarios']) for k, v in banks.items()}))
