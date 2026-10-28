"""Check an isolated bus-number tap against independently reviewed seed exports.

This establishes exported connectivity only, not sampled/native correctness.
"""
from check_ntrpu_xdma_boundary_segments import BOUNDARY, WIDTHS, table, unique


def review(reports, seed, segments, hard):
    cells_text = reports['before_cells.tsv']
    if cells_text != reports['after_cells.tsv'] or cells_text != seed['cells.tsv']:
        raise ValueError('Cell inventory differs from the reviewed seed')
    cells = unique(table(cells_text, ['cell', 'reference', 'primitive', 'blackbox']), 'cell')
    if reports['before_transport.tsv'] != reports['after_transport.tsv']:
        raise ValueError('Transport endpoints changed')
    columns = ['boundary_pin', 'kind', 'endpoint', 'direction', 'reference']
    current = table(reports['before_transport.tsv'], columns)
    previous = table(segments['boundary_endpoints.tsv'], columns[:-1] + ['cell', 'reference'])
    # Both exports enumerate leaf pins and top ports, without segment duplication.
    actual = [tuple(row[key] for key in columns) for row in current]
    expected = {tuple(row[key] for key in columns) for row in previous}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Transport export differs from the reviewed segments')
    bits = {BOUNDARY + '/' + name + (f'[{i}]' if width > 1 else '')
            for name, width in WIDTHS.items() for i in range(width)}
    if {row['boundary_pin'] for row in current} != bits:
        raise ValueError('Incomplete boundary')
    taps = unique(table(reports['bus_taps.tsv'], [
        'bit', 'driver', 'primitive', 'original_net', 'new_port',
        'leaves_before', 'leaves_after', 'ports_after']), 'bit')
    if set(taps) != {str(i) for i in range(8)}:
        raise ValueError('Expected exactly eight bus taps')
    for bit, row in taps.items():
        driver = hard['hard_cell'] + '/CFGBUSNUMBER[' + bit + ']'
        port = 'ntrpu_observe_cfg_bus_number_' + bit
        prior = hard['pins'][driver]
        if (row['driver'] != driver or row['primitive'] != 'PCIE40E4' or
                cells[hard['hard_cell']]['reference'] != 'PCIE40E4' or
                prior['unconnected'] or prior['direct_nets'] != [row['original_net']]):
            raise ValueError('Bus source does not match reviewed hard-block output')
        if any(row[key] != driver for key in ['leaves_before', 'leaves_after']):
            raise ValueError('Primitive leaf endpoints changed')
        if row['new_port'] != port or row['ports_after'] != port:
            raise ValueError('Unexpected observer output sink')
        if any(e['kind'] != 'pin' or e['endpoint'] != driver or
               e['direction'] != 'OUT' or e['reference'] != 'PCIE40E4'
               for e in prior['endpoint_rows']):
            raise ValueError('Seed bus net already has other endpoints')
    return dict(cells=len(cells), transport_bits=len(bits),
                transport_endpoint_rows=len(current), output_sinks=len(taps),
                original_cells_unchanged=True, transport_endpoints_unchanged=True,
                primitive_endpoints_unchanged=True, taps=list(taps.values()))
