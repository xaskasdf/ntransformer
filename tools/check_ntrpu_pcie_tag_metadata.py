"""Compare every registered result with ordered software snapshots of the table."""
import re


def unpack(value, width, count):
    return [(value >> (index*width)) & ((1 << width)-1) for index in range(count)]


def review(log, vectors, manifest):
    if re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|FAIL )', log, re.M):
        raise ValueError('Simulation diagnostic')
    scope = 'META_SCOPE tags=256 width=30 lookups=4 updates=6 latency=1 native=0'
    if log.splitlines().count(scope) != 1:
        raise ValueError('Missing or changed scope')
    lines = [line for line in log.splitlines() if line.startswith('META_ROW ')]
    inputs = vectors.splitlines()
    if len(lines) != len(inputs) or len(lines) != manifest['rows']:
        raise ValueError('Missing or duplicate rows')
    if log.splitlines().count(f'META_SUITE_PASS rows={len(inputs)}') != 1:
        raise ValueError('Missing complete suite marker')
    memory = [0]*256
    phases = {p['id']: 0 for p in manifest['phases']}
    read_tags, written_tags, prefixes = set(), set(), set()
    all_four = all_six = same_tag_six = errors = resets = partial_after_full = 0
    expected_rows = []
    for index, (line, vector) in enumerate(zip(lines, inputs)):
        fields = vector.split()
        phase = int(fields[0])
        rst, valid, tags, cutoffs, enables, full, write_tags, write_data = [int(v, 16) for v in fields[1:]]
        if phase not in phases:
            raise ValueError('Unknown phase')
        phases[phase] += 1
        qtags, cuts = unpack(tags, 8, 4), unpack(cutoffs, 3, 4)
        wtags, words = unpack(write_tags, 8, 6), unpack(write_data, 30, 6)
        expected_valid = expected_error = 0
        result = [0]*4
        if not rst:
            memory = [0]*256
            resets += 1
        else:
            # Each snapshot represents the complete table after an ordered prefix.
            snapshots = [memory.copy()]
            prior_full = set()
            for slot, (tag, word) in enumerate(zip(wtags, words)):
                if enables & (1 << slot):
                    written_tags.add(tag)
                    if full & (1 << slot):
                        memory[tag] = word
                        prior_full.add(tag)
                    else:
                        partial_after_full += int(tag in prior_full)
                        memory[tag] = (memory[tag] & ~((1 << 25)-1)) | (word & ((1 << 25)-1))
                snapshots.append(memory.copy())
            for q, (tag, cutoff) in enumerate(zip(qtags, cuts)):
                if valid & (1 << q):
                    read_tags.add(tag)
                    prefixes.add(cutoff)
                    if cutoff == 7:
                        expected_error |= 1 << q
                        errors += 1
                    else:
                        expected_valid |= 1 << q
                        result[q] = snapshots[cutoff][tag]
            all_four += int(valid == 15)
            all_six += int(enables == 63)
            same_tag_six += int(enables == 63 and len(set(wtags)) == 1)
        expected_data = sum(word << (q*30) for q, word in enumerate(result))
        expected = dict(row=index, phase=phase, valid=expected_valid,
                        error=expected_error, data=expected_data)
        match = re.fullmatch(r'META_ROW row=(\d+) phase=(\d+) valid=([0-9a-f]+) error=([0-9a-f]+) data=([0-9a-f]+)', line)
        if not match:
            raise ValueError('Malformed or unknown result')
        actual = dict(zip(expected, [int(value, 10 if pos < 2 else 16) for pos, value in enumerate(match.groups())]))
        if actual != expected:
            raise ValueError(f'Metadata mismatch at row {index}: {actual} != {expected}')
        expected_rows.append(expected)
    if phases != {p['id']: p['rows'] for p in manifest['phases']}:
        raise ValueError('Incomplete phases')
    if read_tags != set(range(256)) or written_tags != set(range(256)) or prefixes != set(range(8)):
        raise ValueError('Incomplete address/cutoff coverage')
    if min(all_four, all_six, same_tag_six) < 4096 or not partial_after_full or errors == 0 or resets < 2:
        raise ValueError('Contention/error/reset coverage missing')
    return dict(status='PASS_PCIE_METADATA_REGISTERED_LOOKUP_REFERENCE', rows=len(inputs),
        phases=phases, all_four_query_cycles=all_four, all_six_write_cycles=all_six,
        all_six_same_tag_cycles=same_tag_six, invalid_queries=errors, reset_cycles=resets,
        partial_after_full_updates=partial_after_full, read_tags=len(read_tags), written_tags=len(written_tags),
        latency_cycles=1, known_binary_inputs_only=True, native_endpoint_qualified=False)
