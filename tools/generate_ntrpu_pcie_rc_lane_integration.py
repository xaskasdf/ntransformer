"""Prepare raw EOP corner cases; only later RTL runs can validate hardware."""
import argparse
import hashlib
import json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq, rc, rc_header
from generate_ntrpu_pcie_rc_prefix_integration import quad


def finish_fault(v):
    v.step(rq_beat=rq(180, 0x700))
    v.step(rc_beat=rc(180, address=0x700))
    v.end()


def generate():
    v = Vectors()
    patterns = [(3, 7, 11, 15), (3, 3, 11, 15), (15, 11, 7, 3),
                (3, 3, 3, 3), (0, 4, 8, 12), (15, 15, 15, 15)]
    for pattern_id, ends in enumerate(patterns):
        for terminal in (False, True):
            for offset in range(4):
                v.case_begin(f'lane_eop_{pattern_id}_terminal_{int(terminal)}_offset_{offset}')
                count = 4 if terminal else 8
                v.step(rq_beat=rq(45, count=count, first=(15 << offset) & 15))
                v.step(rc_beat=quad(45, count * 4 - offset, 0x100 + offset,
                                   terminal_last=terminal, ends=ends), expect_error=pattern_id != 0)
                if v.fault:
                    finish_fault(v)
                else:
                    if not terminal:
                        v.step(rc_beat=rc(45, bytes(16), address=0x110, remaining=16))
                    v.end()
    # Retained context followed by three headers. Repeat malformed EOP maps
    # as well as the legal map, preserving post-fault cycles.
    for ends in ((2, 7, 11, 15), (2, 2, 11, 15), (15, 11, 7, 2)):
        for offset in range(4):
            v.case_begin(f'lane_retained_{ends}_{offset}')
            v.step(rq_beat=rq(43, count=20, first=(15 << offset) & 15))
            raw = rc_header(43, 16, 80 - offset, 0x100 + offset, terminal=False) + bytes(64)
            v.step(rc_beat=(place((0, raw[:64])), ((1 << (52 - offset)) - 1) << (12 + offset) | 1 << 64))
            pieces = [(0, raw[64:])]
            user = 0xfff | 7 << 64 | 0x39 << 68 | 15 << 76
            for slot, pointer in enumerate(ends):
                user |= pointer << (80 + 4 * slot)
            for i in range(3):
                pieces.append((16 * (i + 1), rc_header(43, 1, 16 - 4 * i, 0x140 + 4 * i, terminal=False) + bytes(4)))
                user |= 15 << (28 + 16 * i)
            v.step(rc_beat=(place(*pieces), user), expect_error=ends != (2, 7, 11, 15))
            if v.fault:
                finish_fault(v)
            else:
                v.step(rc_beat=rc(43, address=0x14c, remaining=4))
                v.end()
    binary_cases = len(v.scenarios)
    binary_rows = len(v.rows)
    # These cases intentionally begin with malformed duplicate pointers.
    # Python stops checking state after the fault; the immutable R7 RTL
    # reference must check every subsequent row, including unknown values.
    for kind in ('pointer', 'terminal', 'tag', 'count'):
        for slot in range(4):
            for unknown in 'xz':
                v.case_begin(f'lane_four_state_{kind}_{slot}_{unknown}')
                v.step(rq_beat=rq(45, count=8))
                v.step(rc_beat=quad(45, 32, 0x100, ends=(3, 3, 11, 15)), expect_error=True)
                fields = v.rows[-1].split()
                assert fields[17] == '1'
                fields[18] = '0'
                if kind == 'pointer':
                    user = list(fields[15].zfill(41))
                    user[-1 - (80 // 4 + slot)] = unknown
                    fields[15] = ''.join(user)
                else:
                    bit = slot * 128 + {'terminal': 30, 'tag': 64, 'count': 32}[kind]
                    data = list(fields[14])
                    data[-1 - bit] = unknown
                    fields[14] = ''.join(data)
                v.rows[-1] = ' '.join(fields)
                v.untrusted = True
                v.case['four_state_only'] = True
                finish_fault(v)
    return v, binary_cases, binary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    v, binary_cases, binary_rows = generate()
    for mode, rows, scenarios in [('rtl', v.rows, v.scenarios),
                                  ('binary', v.rows[:binary_rows], v.scenarios[:binary_cases])]:
        out = args.output.with_name(args.output.stem + '_' + mode + args.output.suffix)
        text = '\n'.join(rows) + '\n'
        if mode == 'binary':
            assert not any('x' in row or 'z' in row for row in rows)
        result = dict(status='PREPARED_RAW_INPUTS_NOT_RTL_QUALIFIED', rows=len(rows), scenarios=scenarios,
            binary_scenarios=binary_cases, four_state_scenarios=len(scenarios) - binary_cases,
            excluded_four_state_scenarios=0 if mode == 'rtl' else len(v.scenarios) - binary_cases,
            vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),
            generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_rows=0,
            scope='Raw inputs only. Independent software expectations before faults; immutable R7 RTL must compare all post-fault and four-state rows. Binary bank preserves the exact complete known-input prefix, without concretizing unknowns.')
        with out.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        with out.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(path=out.as_posix(), rows=len(rows), scenarios=len(scenarios), sha256=result['vectors_sha256'])))


if __name__ == '__main__':
    main()
