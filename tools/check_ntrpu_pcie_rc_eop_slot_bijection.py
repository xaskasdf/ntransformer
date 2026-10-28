"""Check finite EOP placement models; this does not prove an RTL rewrite."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def decode_forward(pointers):
    lanes = {}
    for slot, pointer in enumerate(pointers):
        if pointer is not None:
            lanes[pointer] = slot
    return lanes


def decode_reverse(pointers):
    return {lane: next(slot for slot in reversed(range(len(pointers)))
                      if pointers[slot] == lane)
            for lane in range(16) if lane in pointers}


def check(decoder):
    cases = 0
    duplicate_cases = 0
    unknown_pointer_cases = 0
    histogram = [0] * 5
    for count in range(5):
        for pointers in itertools.product([*range(16), None], repeat=count):
            actual = decoder(pointers)
            expected = decode_reverse(pointers)
            assert actual == expected, (pointers, actual, expected)
            slots = list(actual.values())
            assert len(slots) == len(set(slots)) <= count <= 4
            assert all(0 <= slot < count for slot in slots)
            # Every eligible subset still has unique lane/slot destinations.
            # Therefore moving its values between these destinations preserves
            # the multiset of addends, including arbitrary unknown values.
            assert all(pointers[slot] == lane for lane, slot in actual.items())
            known = [p for p in pointers if p is not None]
            duplicate_cases += len(known) != len(set(known))
            unknown_pointer_cases += None in pointers
            histogram[len(actual)] += 1
            cases += 1
    return dict(cases=cases, duplicate_pointer_cases=duplicate_cases,
                unknown_pointer_class_cases=unknown_pointer_cases,
                effective_eop_count_histogram=histogram)


def first_wins(pointers):
    result = {}
    for slot, pointer in enumerate(pointers):
        if pointer is not None:
            result.setdefault(pointer, slot)
    return result


def unknown_aliases_zero(pointers):
    return decode_forward([0 if p is None else p for p in pointers])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('producer', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.producer.read_text()
    binding = [
        'for (slot=0;slot<4;slot=slot+1)',
        'if (slot<ends)',
        'ptr=rc_user[80+slot*4+:4];',
        'eop_map[ptr]=1; rc_end_slot[ptr]=slot; prev_ptr=ptr;',
        'for (lane=0;lane<16;lane=lane+1)',
        'rc_removed_span[rc_end_slot[lane]]=coriginal;',
        'rc_removed_span[rc_end_slot[lane]]=cdelivered;',
    ]
    assert all(piece in source for piece in binding)
    result = check(decode_forward)
    assert result['cases'] == 88741
    controls = []
    for name, decoder in [('first_duplicate_wins', first_wins),
                          ('unknown_pointer_aliases_zero', unknown_aliases_zero)]:
        try:
            check(decoder)
        except AssertionError:
            controls.append(dict(name=name, rejected=True))
        else:
            raise AssertionError('Accepted bad decoder: ' + name)
    thermometer = {'0000': 0, '0001': 1, '0011': 2, '0111': 3, '1111': 4}
    masks = [''.join(bits) for bits in itertools.product('01xz', repeat=4)]
    invalid = [mask for mask in masks if mask not in thermometer]
    assert len(invalid) == 251
    result.update(
        status='PASS_FINITE_DECODED_EOP_LANE_SLOT_BIJECTION',
        producer=args.producer.as_posix(),
        producer_sha256=hashlib.sha256(args.producer.read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pointer_classes=17, active_slot_counts=list(range(5)),
        four_state_thermometer_patterns=256, invalid_thermometer_patterns=251,
        controls=controls, source_binding=binding,
        assumptions=[
            'Unknown indexed writes leave decoded map/table entries unchanged, as in the existing RTL; the model groups these pointer values as None.',
            'The source thermometer accepts only 0,1,3,7,15 and otherwise returns -1.',
            'A future lane reduction must retain the same procedural eligibility, 13-bit values and 15-bit unsigned sum width.',
        ],
        scope='Finite decoder placement model, including duplicates and one unknown-pointer class. Injectivity preserves the multiset of eligible addends. It does not simulate four-state RTL, prove physical timing or qualify a new producer.',
        new_rtl_qualified=False)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
