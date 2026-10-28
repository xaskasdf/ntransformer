"""Compare the persisted observation DCP with its reviewed in-memory export."""
from check_ntrpu_xdma_boundary_segments import nets, table, unique


def review(reports, previous, measurements):
    if reports['cells.tsv'] != previous['after_cells.tsv']:
        raise ValueError('Serialized cells differ')
    if reports['transport.tsv'] != previous['after_transport.tsv']:
        raise ValueError('Serialized transport endpoints differ')
    ports = unique(table(reports['ports.tsv'], ['port', 'direction', 'nets']), 'port')
    taps = unique(table(reports['persisted_bus_taps.tsv'], [
        'bit', 'driver', 'primitive', 'driver_direction', 'new_port',
        'port_direction', 'leaves', 'ports', 'segments']), 'bit')
    expected = {row['bit']: row for row in measurements['taps']}
    if set(taps) != set(expected) or len(taps) != 8:
        raise ValueError('Serialized bus bits differ')
    expected_ports = {row['new_port'] for row in expected.values()}
    if {name for name in ports if name.startswith('ntrpu_observe_')} != expected_ports:
        raise ValueError('Unexpected observation port')
    for bit, row in taps.items():
        old = expected[bit]
        if any(row[key] != old[key] for key in ['driver', 'primitive', 'new_port']):
            raise ValueError('Serialized bus identity differs')
        if row['driver_direction'] != 'OUT' or row['port_direction'] != 'OUT':
            raise ValueError('Observation must be output only')
        if row['leaves'] != old['driver'] or row['ports'] != old['new_port']:
            raise ValueError('Unexpected bus endpoint')
        segments = nets(row['segments'])
        port = ports[row['new_port']]
        if old['original_net'] not in segments or port['direction'] != 'OUT':
            raise ValueError('Original bus net or output direction lost')
        direct = nets(port['nets'])
        if len(direct) != 1 or not direct.issubset(segments):
            raise ValueError('Output port is not on the reported bus net')
    return dict(original_cells_identical=True, transport_endpoints_identical=True,
                output_sinks=8, total_top_level_port_bits=len(ports),
                taps=list(taps.values()))
