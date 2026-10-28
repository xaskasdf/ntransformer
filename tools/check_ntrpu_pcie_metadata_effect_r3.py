"""Review finite RTL evidence, with an independent binary raw-event recurrence."""
import re

ROW = re.compile(r'EFFECT_ROW index=(\d+) reset=([01]) accept=([01]) cut=(\d+) raw=([0-9a-fxz]+) valid=([01]) effect=([0-9a-fxz]+) metadata=([0-9a-fxz]+) reference=([0-9a-fxz]+)')


def binary_reference(raw):
    """Sequential assignments, explicitly retaining the original unsigned comparison."""
    writes = raw & ((1 << 480) - 1)
    counts = (raw >> 480) & ((1 << 176) - 1)
    replace = (raw >> 656) & 65535
    valid = (raw >> 672) & 65535
    seed = raw >> 688
    metadata = seed
    replacement = None
    used_counts = []
    for n in range(16):
        if not (valid >> n) & 1:
            continue
        if (replace >> n) & 1:
            metadata = (writes >> (30 * n)) & ((1 << 30) - 1)
            replacement = metadata
            used_counts = []
        else:
            count = (counts >> (11 * n)) & 2047
            used_counts.append(count)
            remaining, address = metadata & 8191, (metadata >> 13) & 4095
            available = (4 * count - (address & 3)) & 32767
            captured = min(available, remaining) & 8191
            metadata = (metadata & (31 << 25)) | (((address + captured) & 4095) << 13) | ((remaining - captured) & 8191)
    # Independently encode the suffix after the last replacement, without composing summaries.
    effect = 0 if replacement is None else (1 << 56) | (replacement << 26)
    if used_counts:
        effect |= min(len(used_counts), 2) << 24
        effect |= used_counts[0] << 13
        effect |= min(sum(used_counts), 2049) << 1
    return f'{metadata:08x}', f'{effect:015x}'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def review(normal, mutant):
    require(normal.splitlines().count('EFFECT_MODE mutant=0') == 1, 'normal mode')
    require(normal.splitlines().count('EFFECT_PASS checks=17114 unknown_injections=4308') == 1, 'normal completion')
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|FAIL )', normal, re.M), 'normal failure')
    require(mutant.splitlines().count('EFFECT_MODE mutant=1') == 1, 'mutant mode')
    require(len(re.findall(r'Fatal: EFFECT_METADATA_MISMATCH check=0 actual=22008009 expected=2200200c', mutant)) == 1, 'actual order mutant failure')
    require('EFFECT_PASS' not in mutant and 'EFFECT_ROW' not in mutant, 'mutant incorrectly completed')
    rows = [line for line in normal.splitlines() if line.startswith('EFFECT_ROW ')]
    require(len(rows) == 17114, 'row count')
    previous_metadata, previous_effect = '00000000', '000000000000000'
    binary_checked = held = reset_checks = 0
    for index, line in enumerate(rows):
        match = ROW.fullmatch(line.replace('X', 'x').replace('Z', 'z'))
        require(match is not None, 'row format')
        n, reset, accept, cut, raw, valid, effect, metadata, reference = match.groups()
        require(int(n) == index and 0 <= int(cut) <= 16, 'row sequence/cut')
        require(len(raw) == 180 and len(effect) == 15 and len(metadata) == len(reference) == 8, 'trace width')
        require(metadata == reference, 'RTL sequential reference differs')
        require(valid == str(int(reset) & int(accept)), 'valid/reset alignment')
        if reset == '0':
            require(metadata == '00000000' and effect == '000000000000000', 'reset data')
            reset_checks += 1
        elif accept == '0':
            require(metadata == previous_metadata and effect == previous_effect, 'bubble data hold')
            held += 1
        elif not re.search('[xz]', raw):
            want_metadata, want_effect = binary_reference(int(raw, 16))
            require(metadata == want_metadata and effect == want_effect, 'independent binary recurrence')
            binary_checked += 1
        previous_metadata, previous_effect = metadata, effect
    coverage = re.findall(r'^EFFECT_COVERAGE capped=(\d+) unknown_results=(\d+) replacement=(\d+) single=(\d+) many=(\d+)$', normal, re.M)
    require(len(coverage) == 1 and all(int(n) > 0 for n in coverage[0]), 'coverage')
    require(binary_checked > 12000 and held > 200 and reset_checks > 100, 'binary/clock coverage')
    return dict(checks=len(rows), unknown_injections=4308, binary_rows_independently_checked=binary_checked,
        bubble_holds=held, reset_rows=reset_checks,
        coverage=dict(zip(['capped', 'unknown_results', 'replacement', 'single', 'many'], map(int, coverage[0]))),
        actual_order_mutant_rejected=True, four_state_reference='SystemVerilog sequential raw-event recurrence; masked and active injections included')
