"""Observe DDR4 command encodings at public component pins in a VCD.

No model internals, timing-rule qualification, or power approval. A representative
device per rank cannot establish coverage of all chips. Same-timestamp pin/clock
changes are ambiguous and never counted as an observed command.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

PINS = dict(CK=2, RESET_n=1, CKE=1, CS_n=1, ACT_n=1, RAS_n_A16=1,
            CAS_n_A15=1, WE_n_A14=1, BG=2, BA=2, ADDR=14, TEN=1)
EXPECTED_RANKS = ('slot0/rank0', 'slot0/rank1', 'slot1/rank0', 'slot1/rank1')
COMMANDS = {'000': 'MRS', '001': 'REF', '010': 'PRE', '011': 'RESERVED',
            '100': 'WR', '101': 'RD', '110': 'ZQ', '111': 'NOP'}
SCOPE = re.compile(r'dimms\[(\d+)\]\.memory/.*rank_instances\[(\d+)\].*'
                   r'instance_of_sdram_devices\[0\]\.iDDR4$')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized(value, width):
    value = value.lower()
    require(bool(re.fullmatch('[01xz]+', value)) and len(value) <= width, 'Invalid VCD value width')
    return value.rjust(width, value[0] if value[0] in 'xz' else '0')


def read_vcd(path):
    """Retain only representative command interfaces; preserve same-time changes."""
    paths, widths, scopes, ranks = {}, {}, [], {}
    events = defaultdict(list)
    with Path(path).open(encoding='utf-8') as f:
        header, header_bytes = [], 0
        for line in f:
            header.append(line)
            header_bytes += len(line)
            require(header_bytes < 2 * 1024 * 1024, 'Oversized VCD header')
            if '$enddefinitions' in line:
                break
        else:
            raise ValueError('Missing VCD definitions')
        text = ''.join(header)
        scales = re.findall(r'\$timescale\s+(\d+)\s*(\w+)\s+\$end', text)
        require(scales == [('1', 'ps')], 'Expected exactly one 1 ps timescale')
        for command in re.findall(r'\$\w+\b.*?\$end', text, re.S):
            words = command.split()
            if words[0] == '$scope':
                require(len(words) == 4, 'Malformed scope')
                scopes.append(words[2])
            elif words[0] == '$upscope':
                require(bool(scopes), 'Unbalanced VCD scope')
                scopes.pop()
            elif words[0] == '$var':
                require(len(words) in (6, 7), 'Malformed VCD variable')
                width, code, name = int(words[2]), words[3], words[4]
                require(width > 0, 'Invalid VCD width')
                require(code not in widths or widths[code] == width, 'Conflicting VCD alias width')
                widths[code] = width
                prefix = '/'.join(scopes[1:])
                match = SCOPE.fullmatch(prefix)
                if match and name in PINS:
                    rank = 'slot%s/rank%s' % match.groups()
                    require(rank in EXPECTED_RANKS, 'Unexpected slot/rank')
                    require(width == PINS[name], 'Unexpected pin width: '+name)
                    pins = ranks.setdefault(rank, dict(scope=prefix, pins={}))
                    require(pins['scope'] == prefix and name not in pins['pins'], 'Duplicate representative pin')
                    pins['pins'][name] = code
                    paths[prefix+'/'+name] = code
        require(not scopes, 'Unclosed VCD scope')
        require(bool(ranks), 'No representative component interfaces')
        codes = set(paths.values())
        first = now = None
        in_comment = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_comment:
                in_comment = '$end' not in line
                continue
            if line.startswith('$comment'):
                in_comment = '$end' not in line
                continue
            if line in ('$dumpvars', '$end'):
                continue
            require(not line.startswith('$'), 'Unsupported VCD directive: '+line)
            if line.startswith('#'):
                require(line[1:].isdigit(), 'Invalid VCD timestamp')
                timestamp = int(line[1:])
                require(now is None or timestamp >= now, 'VCD time moved backward')
                now = timestamp
                if first is None: first = now
                continue
            require(now is not None, 'Value before first timestamp')
            if line.startswith(('b', 'B')):
                row = line[1:].split()
                require(len(row) == 2, 'Malformed vector value')
                value, code = row
            else:
                value, code = line[0], line[1:]
            require(code in widths, 'Undeclared VCD identifier')
            if code in codes:
                events[code].append((now, normalized(value, widths[code])))
        require(not in_comment and now is not None and first < now, 'Incomplete VCD extent')
    return ranks, events, first, now


def decode(pins, previous_cke):
    """Classify encodings, never infer completion of a command or self refresh."""
    if pins['RESET_n'] != '1':
        return 'RESET' if pins['RESET_n'] == '0' else 'UNKNOWN_RESET'
    if pins['TEN'] != '0':
        return 'TEST_MODE_OR_UNKNOWN'
    if any(v not in ('0', '1') for v in (pins['CKE'], previous_cke, pins['CS_n'])):
        return 'UNKNOWN_CONTROL'
    cke, cs = pins['CKE'], pins['CS_n']
    if cke == '0':
        # REF encoding with CKE falling is self-refresh entry, never a REF count.
        bits = pins['ACT_n']+pins['RAS_n_A16']+pins['CAS_n_A15']+pins['WE_n_A14']
        if previous_cke == '1' and cs == '0' and bits == '1001':
            return 'SRE_ENCODING'
        return 'CKE_LOW'
    if previous_cke == '0':
        return 'CKE_EXIT'
    if cs == '1':
        return 'DES'
    if any(re.search('[xz]', value) for value in pins.values()):
        return 'UNKNOWN_SELECTED'
    if pins['ACT_n'] == '0':
        return 'ACT'
    command = COMMANDS[pins['RAS_n_A16']+pins['CAS_n_A15']+pins['WE_n_A14']]
    a10 = pins['ADDR'][-11]
    if command == 'PRE' and a10 == '1': return 'PREA'
    if command in ('RD', 'WR') and a10 == '1': return command+'A'
    if command == 'ZQ': return 'ZQCL' if a10 == '1' else 'ZQCS'
    return command


def rank_samples(pin_codes, events):
    by_time = defaultdict(lambda: defaultdict(list))
    for pin, code in pin_codes.items():
        for timestamp, value in events[code]:
            by_time[timestamp][pin].append(value)
    current = {pin: 'x'*width for pin, width in PINS.items()}
    previous_cke = 'x'
    samples, clock_problems = [], []
    for timestamp, changes in sorted(by_time.items()):
        old_clock = current['CK']
        new_clock = changes.get('CK', [old_clock])[-1]
        if 'CK' in changes:
            if len(changes['CK']) != 1 or re.search('[xz]', old_clock+new_clock):
                clock_problems.append(timestamp)
                previous_cke = 'x'
            elif old_clock == '01' and new_clock == '10':
                ambiguous = [pin for pin, values in changes.items() if pin != 'CK'
                             and any(value != current[pin] for value in values)]
                label = 'AMBIGUOUS_EDGE' if ambiguous else decode(current, previous_cke)
                samples.append(dict(time_ps=timestamp, command=label,
                                    ambiguous_pins=sorted(ambiguous)))
                previous_cke = 'x' if 'CKE' in ambiguous else current['CKE']
            elif (old_clock, new_clock) != ('10', '01') and old_clock != new_clock:
                clock_problems.append(timestamp)
                previous_cke = 'x'
        for pin, values in changes.items(): current[pin] = values[-1]
    return samples, clock_problems


def review(path, windows=None):
    ranks, events, first, last = read_vcd(path)
    windows = windows if windows is not None else [dict(name='trace', start_ps=first, end_ps=last)]
    require(bool(windows), 'No windows')
    seen = set()
    for window in windows:
        require(set(window) == {'name', 'start_ps', 'end_ps'}, 'Unexpected window fields')
        require(isinstance(window['name'], str) and window['name'] and window['name'] not in seen, 'Duplicate/invalid window')
        seen.add(window['name'])
        require(type(window['start_ps']) is int and type(window['end_ps']) is int, 'Integer picoseconds required')
        require(first <= window['start_ps'] < window['end_ps'] <= last, 'Window outside VCD extent')
    output = {}
    for rank in EXPECTED_RANKS:
        if rank not in ranks:
            output[rank] = dict(observed=False, missing_pins=sorted(PINS))
            continue
        row = ranks[rank]
        missing = sorted(set(PINS)-row['pins'].keys())
        if missing:
            output[rank] = dict(observed=False, scope=row['scope'], missing_pins=missing)
            continue
        samples, bad_clock = rank_samples(row['pins'], events)
        reports = []
        for window in windows:
            selected = [s for s in samples if window['start_ps'] <= s['time_ps'] < window['end_ps']]
            counts = dict(sorted(Counter(s['command'] for s in selected).items()))
            refs = [s['time_ps'] for s in selected if s['command'] == 'REF']
            gaps = [b-a for a,b in zip(refs, refs[1:])]
            reports.append(dict(**window, rising_edges=len(selected), counts=counts,
                refresh_times_ps=refs, maximum_observed_refresh_gap_ps=max(gaps, default=None),
                clock_problem_events=sum(window['start_ps'] <= t < window['end_ps'] for t in bad_clock),
                read_encoding_observed=counts.get('RD', 0)+counts.get('RDA', 0)>0,
                write_encoding_observed=counts.get('WR', 0)+counts.get('WRA', 0)>0,
                refresh_encoding_observed=bool(refs)))
        output[rank] = dict(observed=True, scope=row['scope'], missing_pins=[], windows=reports)
    return dict(status='DDR_RANK_COMMAND_OBSERVATIONS', vcd_sha256=digest(path),
        trace_first_ps=first, trace_last_ps=last, windows=windows, ranks=output,
        all_four_rank_interfaces_present=all(r['observed'] for r in output.values()),
        coverage_is_one_component_per_rank=True, protocol_timing_qualified=False,
        refresh_schedule_qualified=False, native_activity_r1_qualified=False,
        ready_for_power_estimate=False, board_power_qualified=False,
        scope='Pin encodings on stable 01-to-10 differential CK transitions. '
              'Half-open windows; same-time control changes are ambiguous. '
              'No inference about command acceptance, all chips, or refresh compliance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vcd', type=Path)
    parser.add_argument('--windows', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() != args.vcd.resolve(), 'Output is input')
    windows = json.loads(args.windows.read_text()) if args.windows else None
    result = review(args.vcd, windows)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
