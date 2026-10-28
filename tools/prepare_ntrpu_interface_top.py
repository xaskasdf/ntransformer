"""Build a structural DDR/PCIe wrapper from exported checkpoint interfaces."""
import argparse
import csv
import hashlib
import json
import pathlib
import re


def prepare(inventory, output):
    output.mkdir(exist_ok=False)
    declarations, instances, all_rows, hashes = [], [], [], {}
    balls = set()
    for name, expected_count, expected_module in [('ddr', 154, 'example_top'),
                                                   ('pcie', 67, 'xilinx_dma_pcie_ep')]:
        ports = inventory / (name + '_ports.tsv')
        stub = inventory / (name + '_stub.v')
        rows = list(csv.DictReader(ports.open(), delimiter='\t'))
        if len(rows) != expected_count or len({r['port'] for r in rows}) != expected_count:
            raise ValueError('Unexpected checkpoint interface size: ' + name)
        modules = re.findall(r'^module\s+(\w+)\s*\(', stub.read_text(), re.M)
        if modules != [expected_module]:
            raise ValueError('Unexpected generated stub module')
        groups = {}
        for row in rows:
            match = re.fullmatch(r'(\w+)(?:\[(\d+)\])?', row['port'])
            if not match or not row['ball'] or row['ball'] in balls:
                raise ValueError('Unsupported port name or duplicate/missing package ball')
            balls.add(row['ball'])
            base, bit = match.groups()
            group = groups.setdefault(base, {'direction': row['direction'], 'bits': []})
            if group['direction'] != row['direction']:
                raise ValueError('Mixed bus directions')
            group['bits'].append(None if bit is None else int(bit))
            all_rows.append(dict(row, port=name+'_'+row['port'], original=row['port'], interface=name))
        connections = []
        for base, group in sorted(groups.items()):
            bits = group['bits']
            width = ''
            if bits != [None]:
                if None in bits or sorted(bits) != list(range(max(bits)+1)):
                    raise ValueError('Only complete zero-based buses are supported')
                width = f'[{max(bits)}:0] '
            direction = {'IN': 'input', 'OUT': 'output', 'INOUT': 'inout'}[group['direction']]
            declarations.append(f'    {direction} wire {width}{name}_{base}')
            connections.append(f'        .{base}({name}_{base})')
        instances.append(f'    (* DONT_TOUCH = "yes" *) {expected_module} u_{name} (\n' +
                         ',\n'.join(connections) + '\n    );')
        (output / (name+'_stub.v')).write_bytes(stub.read_bytes())
        for path in (ports, stub):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'coexist_top.v').write_text(
        '// Structural coexistence probe only: independent example traffic paths.\n'
        'module ntrpu_ddr_pcie_coexist (\n'+',\n'.join(declarations)+'\n);\n'+
        '\n'.join(instances)+'\nendmodule\n')
    with (output / 'ports.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(all_rows)
    (output / 'inputs.json').write_text(json.dumps({'sha256': hashes, 'scalar_ports': len(all_rows),
        'unique_balls': len(balls), 'scope': 'One generic DDR channel and independent serial PCIe example; not a DMA-to-DDR datapath'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    prepare(args.inventory, args.output)
