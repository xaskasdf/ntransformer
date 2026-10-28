"""Binary metadata-effect composition study; no decoded traffic or RTL qualification."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time


@dataclass(frozen=True)
class Effect:
    # An RQ replacement discards preceding metadata effects for this row.
    # This says nothing about ownership or the separate retained RC payload.
    replacement: tuple[int, int] | None = None
    has_consume: bool = False
    first_words: int = 0
    total_words: int = 0


def compose(earlier, later):
    if later.replacement is not None:
        return later
    return Effect(
        earlier.replacement,
        earlier.has_consume or later.has_consume,
        earlier.first_words if earlier.has_consume else later.first_words,
        earlier.total_words + later.total_words,
    )


def apply(effect, state):
    remaining, address = effect.replacement if effect.replacement is not None else state
    if effect.has_consume:
        if effect.first_words == 0 and (address & 3):
            delta = remaining
        else:
            delta = min(remaining, 4 * effect.total_words - (address & 3))
        remaining -= delta
        address = (address + delta) & 4095
    return remaining, address


def reference(events, state):
    # Scalar recurrence from the stored R13/A12 and W11 values in R7.
    # Signed available[14:0] is compared with unsigned remaining[12:0].
    remaining, address = state
    for event in events:
        if event[0] == 'replace':
            remaining, address = event[1:]
        elif event[0] == 'consume':
            available_bits = (event[1] * 4 - (address & 3)) & 0x7fff
            delivered_bits = available_bits if available_bits < remaining else remaining
            remaining = (remaining - delivered_bits) & 0x1fff
            address = (address + delivered_bits) & 0xfff
        else:
            assert event == ('identity',)
    return remaining, address


def summarize(events):
    value = Effect()
    for event in events:
        if event[0] == 'replace':
            unit = Effect(replacement=tuple(event[1:]))
        elif event[0] == 'consume':
            unit = Effect(has_consume=True, first_words=event[1], total_words=event[1])
        else:
            assert event == ('identity',)
            unit = Effect()
        value = compose(value, unit)
    return value


def check(events, state):
    expected = reference(events, state)
    summary = summarize(events)
    assert apply(summary, state) == expected, (events, state, expected, summary)
    # Exercise grouping boundaries representing independently prepared windows.
    n = len(events)
    boundaries = sorted({(0, n), (n // 3, 2 * n // 3), (n // 2, n // 2), (n, n)})
    for a, b in boundaries:
        x, y, z = [summarize(part) for part in (events[:a], events[a:b], events[b:])]
        left, right = compose(compose(x, y), z), compose(x, compose(y, z))
        assert left == right == summary
        assert apply(left, state) == expected
    return len(boundaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.monotonic()
    cases = grouped_checks = replacement_cases = 0
    lengths = set()
    # Include wrap, a zero-count/misaligned consume and replacement boundaries.
    directed = [
        [('consume', 0), ('consume', 1)],
        [('consume', 1024), ('consume', 1024)],
        [('consume', 2047)] * 16,
        [('consume', 0), ('replace', 13, 4095), ('consume', 0)],
        [('replace', 8191, 4095), ('consume', 1), ('replace', 7, 3), ('consume', 0)],
        [('identity',), ('replace', 0, 4095), ('identity',)],
    ]
    for remaining in (0, 1, 2, 3, 4, 7, 13, 4095, 4096, 8190, 8191):
        for address in (0, 1, 2, 3, 4092, 4093, 4094, 4095):
            for events in directed:
                grouped_checks += check(events, (remaining, address))
                cases += 1
                replacement_cases += any(e[0] == 'replace' for e in events)
                lengths.add(len(events))
    seed = 0x4E54525014
    rng = random.Random(seed)
    for _ in range(8192):
        state = (rng.randrange(8192), rng.randrange(4096))
        events = []
        for length in range(1, 17):
            mode = rng.randrange(10)
            if mode < 2:
                event = ('replace', rng.randrange(8192), rng.randrange(4096))
            elif mode == 2:
                event = ('identity',)
            else:
                event = ('consume', 0 if mode == 3 else rng.randrange(2048))
            events.append(event)
            grouped_checks += check(events, state)
            cases += 1
            replacement_cases += any(e[0] == 'replace' for e in events)
            lengths.add(length)
    controls = []
    def reject(name, events, state, incorrect):
        expected = reference(events, state)
        assert incorrect != expected, name
        controls.append(dict(name=name, events=events, initial=state, expected=expected, incorrect=incorrect, rejected=True))
    reject('omit_first_zero_misaligned_rule', [('consume', 0), ('consume', 1)], (10, 1), (7, 4))
    reject('truncate_word_sum_to_11_bits', [('consume', 1024), ('consume', 1024)], (8191, 0), (8191, 0))
    events = [('consume', 1), ('replace', 12, 1)]
    reject('commute_consume_and_replace', events, (9, 0), reference(list(reversed(events)), (9, 0)))
    events = [('replace', 13, 4095), ('consume', 0)]
    reject('keep_old_seed_after_replace', events, (4, 0), reference([('consume', 0)], (4, 0)))
    reject('omit_address_wrap', [('consume', 1)], (13, 4095), (12, 4096))
    assert cases == 131600 and len(controls) == 5 and lengths == set(range(1, 17))
    root = Path(__file__).resolve().parents[1]
    sources = ['fpga/rtl/ntrpu_pcie_request_inventory_r7.sv', 'fpga/rtl/ntrpu_pcie_inventory_scheduler_r14.sv',
               'tools/check_ntrpu_pcie_rc_prefix_math.py', 'docs/fpga/board/PCIE_INVENTORY_REGISTERED_SUM_FOLLOWUP.md']
    result = dict(status='PASS_BINARY_METADATA_EFFECT_COMPOSITION_MODEL', cases=cases,
        grouping_checks=grouped_checks, cases_including_replacement=replacement_cases, lengths=sorted(lengths),
        random_seed=seed, controls=controls, duration_seconds=round(time.monotonic()-started, 3),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        exact_rtl_widths=dict(remaining=13, address=12, words=11, available=15),
        aggregate_words_width='Unbounded Python integer; a hardware window and width are not selected.',
        scope='Finite binary study of already-selected per-row consume/replace/identity effects, including boundaries and grouping. Consume uses canonical seed metadata, not received byte count. Not raw traffic/event selection, tag ownership/generation wrap, retained terminal payload, XZ, stage reset/empty/latency, RTL, timing, or whole-pipeline qualification.',
        rtl_qualified=False,full_pipeline_qualified=False,physical_design_qualified=False)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
