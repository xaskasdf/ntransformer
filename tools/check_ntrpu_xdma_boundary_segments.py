"""Validate the completeness of a read-only synthesized XDMA connectivity export.

This reports observation gaps. It does not prove that a passive tap is supported
or that any reported constant/alias is a valid replacement for a profile signal.
"""
import csv
import io


BOUNDARY = 'inst/pcie4_ip_i'
WIDTHS = {
    's_axis_rq_tdata': 512, 's_axis_rq_tuser': 137,
    's_axis_rq_tvalid': 1, 's_axis_rq_tready': 4,
    'm_axis_rc_tdata': 512, 'm_axis_rc_tuser': 161,
    'm_axis_rc_tvalid': 1, 'm_axis_rc_tready': 1,
    'm_axis_rc_tkeep': 16, 'm_axis_rc_tlast': 1,
    'cfg_bus_number': 8, 'user_clk': 1, 'user_reset': 1, 'user_lnk_up': 1,
}


def table(text, columns):
    reader = csv.DictReader(io.StringIO(text), delimiter='\t')
    if reader.fieldnames != columns:
        raise ValueError('Unexpected export columns')
    rows = list(reader)
    if any(set(row) != set(columns) or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Malformed export row')
    return rows


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError('Duplicate ' + key + ': ' + value)
        result[value] = row
    return result


def nets(text):
    values = text.split(',') if text else []
    if values != sorted(set(values)) or any(not value for value in values):
        raise ValueError('Nets must be sorted, unique and nonempty')
    return set(values)


def review(reports, prior_reports):
    if reports['cells.tsv'] != prior_reports['cells.tsv']:
        raise ValueError('Cell inventory changed from reviewed seed')
    cells = unique(table(reports['cells.tsv'], ['cell', 'reference', 'primitive', 'blackbox']), 'cell')
    previous = table(prior_reports['pcie_boundary_pins.tsv'], ['cell', 'pin', 'direction', 'nets'])
    previous = unique([row for row in previous if row['cell'] == BOUNDARY], 'pin')
    current = unique(table(reports['boundary_segments.tsv'], ['pin', 'direction', 'upper', 'lower', 'segments']), 'pin')
    expected = {
        BOUNDARY + '/' + signal + (f'[{bit}]' if width > 1 else '')
        for signal, width in WIDTHS.items() for bit in range(width)
    }
    if set(current) != expected or not expected.issubset(previous):
        raise ValueError('Incomplete or unexpected boundary bits')
    endpoints = table(reports['boundary_endpoints.tsv'], ['boundary_pin', 'kind', 'endpoint', 'direction', 'cell', 'reference'])
    seen = set()
    per_pin = {pin: [] for pin in expected}
    for row in endpoints:
        key = row['boundary_pin'], row['kind'], row['endpoint']
        if key in seen or row['boundary_pin'] not in expected:
            raise ValueError('Duplicate or unattributed endpoint')
        seen.add(key)
        if row['direction'] not in {'IN', 'OUT', 'INOUT'}:
            raise ValueError('Unknown endpoint direction')
        if row['kind'] == 'pin':
            owner = cells.get(row['cell'])
            if (owner is None or owner['reference'] != row['reference'] or
                    (owner['primitive'], owner['blackbox']) == ('0', '0') or
                    row['endpoint'].rsplit('/', 1)[0] != row['cell']):
                raise ValueError('Leaf endpoint owner/reference mismatch')
        elif row['kind'] == 'port':
            if row['cell'] or row['reference'] or '/' in row['endpoint']:
                raise ValueError('Malformed top-level port')
        else:
            raise ValueError('Unknown endpoint kind')
        per_pin[row['boundary_pin']].append(row)
    details = {}
    for pin, row in sorted(current.items()):
        if row['direction'] != previous[pin]['direction'] or row['upper'] != previous[pin]['nets']:
            raise ValueError('Boundary direction/upper connection differs from seed')
        upper, lower, segments = (nets(row[key]) for key in ('upper', 'lower', 'segments'))
        if not (upper | lower).issubset(segments):
            raise ValueError('Connected segment omitted')
        linked = per_pin[pin]
        if linked and not segments:
            raise ValueError('Endpoint reported without any net segment')
        drivers = [e for e in linked if
                   (e['kind'] == 'pin' and e['direction'] == 'OUT') or
                   (e['kind'] == 'port' and e['direction'] == 'IN')]
        inouts = [e for e in linked if e['direction'] == 'INOUT']
        details[pin] = dict(
            upper=sorted(upper), lower=sorted(lower), segments=sorted(segments),
            endpoint_count=len(linked), drivers=drivers, inout_endpoints=inouts,
            single_unambiguous_driver=len(drivers) == 1 and not inouts,
            constant_driver=any(e['reference'] in {'GND', 'VCC'} for e in drivers),
        )
    return dict(
        cells=len(cells), boundary_bits=len(expected), endpoint_rows=len(endpoints),
        parent_connections_unchanged=True, cells_unchanged=True,
        no_parent_net=[p for p, d in details.items() if not d['upper']],
        no_lower_net=[p for p, d in details.items() if not d['lower']],
        no_segments=[p for p, d in details.items() if not d['segments']],
        driver_gaps=[p for p, d in details.items() if not d['single_unambiguous_driver']],
        constant_driven=[p for p, d in details.items() if d['constant_driver']],
        pins=details,
    )
