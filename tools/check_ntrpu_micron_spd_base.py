"""Decode only the verified DDR4 base block of a public Micron SPD table.

The input is a saved Micron product JSON response, not an EEPROM dump.
Sparse/truncated module-specific fields are never padded. Geometry and time
field encodings were cross-checked with upstream i2c-tools 4.4 decode-dimms;
this report does not qualify the DIMM, its RCD, its model or the MIG profile.
"""
import argparse
import binascii
import hashlib
import json
import math
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def base_block(rows):
    known = {}
    incomplete = []
    for row in rows:
        match = re.fullmatch(r'(\d+)(?:\s*-\s*(\d+))?', row['byte-number'])
        require(match is not None, 'Unrecognized byte address')
        start = int(match[1])
        end = int(match[2]) if match[2] else start + 1
        require(end > start, 'Invalid byte range')
        value = row['byte-value']
        if start == 329:
            # Micron presents the module name as text, not a raw hex field.
            incomplete.append(dict(range=row['byte-number'], reason='Textual part number, no raw EEPROM bytes inferred'))
            continue
        is_hex = re.fullmatch(r'(?:[0-9a-fA-F]{2})+', value) is not None
        if start >= 128:
            if not is_hex or len(value) // 2 != end - start:
                incomplete.append(dict(range=row['byte-number'], reason='Export length differs from address span',
                                       declared_bytes=end-start, supplied_hex_bytes=len(value)//2 if is_hex else None))
            continue
        require(is_hex and len(value) // 2 == end - start and end <= 128,
                'Incomplete or ambiguous base-block field')
        for address, byte in enumerate(bytes.fromhex(value), start):
            require(address not in known, 'Overlapping base-block bytes')
            known[address] = byte
    require(set(known) == set(range(128)), 'Missing base-block bytes')
    raw = bytes(known[i] for i in range(128))
    require(binascii.crc_hqx(raw[:126], 0) == int.from_bytes(raw[126:128], 'little'),
            'Base-block CRC mismatch')
    return raw, incomplete


def decode_base(raw, clock_ps=1250):
    require(len(raw) == 128, 'Expected exactly 128 verified bytes')
    require(raw[2] == 0x0c and raw[3] & 15 == 1, 'Only DDR4 RDIMM supported')
    require(raw[1] >> 4 == 1, 'Unsupported SPD major revision')
    require(raw[6] == 0 and raw[10] == 0 and not raw[12] & 0x40,
            'Only symmetric monolithic packages supported')
    require(raw[17] == 0, 'Unsupported or reserved timebase')
    density_code = raw[4] & 15
    device_width_code = raw[12] & 7
    bus_width_code = raw[13] & 7
    extension_code = (raw[13] >> 3) & 3
    require(density_code <= 7 and device_width_code <= 3 and bus_width_code <= 3 and extension_code <= 1,
            'Unsupported geometry encoding')
    density_bits = (256 << density_code) * (1 << 20)
    device_width = 4 << device_width_code
    primary_width = 8 << bus_width_code
    ranks = ((raw[12] >> 3) & 7) + 1
    row_bits = ((raw[5] >> 3) & 7) + 12
    column_bits = (raw[5] & 7) + 9
    bank_bits = ((raw[4] >> 4) & 3) + 2
    bank_group_bits = (raw[4] >> 6) & 3
    geometry_device_bits = (1 << (row_bits + column_bits + bank_bits + bank_group_bits)) * device_width
    require(density_bits == geometry_device_bits, 'Density and addressing disagree')
    require(primary_width % device_width == 0, 'Fractional devices per rank')
    module_bytes = density_bits // 8 * (primary_width // device_width) * ranks

    def fine(medium, offset):
        signed = raw[offset] if raw[offset] < 128 else raw[offset] - 256
        return medium * 125 + signed

    def word(lo):
        return int.from_bytes(raw[lo:lo+2], 'little') * 125

    timings = dict(tCKmin=fine(raw[18], 125), tCKmax=fine(raw[19], 124),
                   tAA=fine(raw[24], 123), tRCD=fine(raw[25], 122), tRP=fine(raw[26], 121),
                   tRAS=((raw[27] & 15) * 256 + raw[28]) * 125,
                   tRC=fine((raw[27] >> 4) * 256 + raw[29], 120),
                   tRFC1=word(30), tRFC2=word(32), tRFC4=word(34),
                   tFAW=((raw[36] & 15) * 256 + raw[37]) * 125,
                   tRRD_S=fine(raw[38], 119), tRRD_L=fine(raw[39], 118), tCCD_L=fine(raw[40], 117),
                   tWR=((raw[41] & 15) * 256 + raw[42]) * 125,
                   tWTR_S=((raw[43] & 15) * 256 + raw[44]) * 125,
                   tWTR_L=(raw[43] >> 4) * 256 * 125 + raw[45] * 125)
    require(all(v > 0 for v in timings.values()), 'Zero or negative timing')
    cas_base = 23 if raw[23] & 0x80 else 7
    cas_mask = int.from_bytes(raw[20:24], 'little') & ((1 << 30) - 1)
    cas = [cas_base + bit for bit in range(30) if cas_mask & (1 << bit)]
    require(clock_ps > 0, 'Nonpositive clock period')
    eligible = [cl for cl in cas if cl * clock_ps >= timings['tAA']]
    return dict(module_type='DDR4 RDIMM', module_bytes=module_bytes, module_gib=module_bytes / (1 << 30),
                device_density_bits=density_bits, device_width_bits=device_width, ranks=ranks,
                primary_bus_bits=primary_width, extension_bus_bits=8 if extension_code else 0,
                row_bits=row_bits, column_bits=column_bits, bank_bits=bank_bits, bank_group_bits=bank_group_bits,
                timings_ps=timings, supported_cas_latencies=cas,
                analytical_target=dict(clock_ps=clock_ps, within_raw_spd_clock_range=timings['tCKmin'] <= clock_ps <= timings['tCKmax'],
                                       minimum_supported_cas_satisfying_tAA=min(eligible) if eligible else None,
                                       tRCD_cycles_ceiling=math.ceil(timings['tRCD']/clock_ps),
                                       tRP_cycles_ceiling=math.ceil(timings['tRP']/clock_ps),
                                       tRAS_cycles_ceiling=math.ceil(timings['tRAS']/clock_ps),
                                       two_modules_per_channel_bytes=2*module_bytes, four_channel_bytes=8*module_bytes,
                                       electrical_topology_or_MIG_qualification=False))


def review(data):
    variants = []
    for process, rows in data['process-codes'].items():
        raw, incomplete = base_block(rows)
        variants.append(dict(process_code=process.strip(), base_128_sha256=hashlib.sha256(raw).hexdigest(),
                             base_crc=hex(int.from_bytes(raw[126:128], 'little')), base_crc_valid=True,
                             decoded=decode_base(raw), unconverted_export_fields=incomplete))
    require(bool(variants), 'No SPD process variants')
    return dict(status='BASE_FIELDS_DECODED_NOT_DIMM_QUALIFIED', part=data['part-number'], variants=variants,
                full_eeprom_image_produced=False, full_spd_crc_validated=False,
                scope='Only bytes0..127 verified and decoded; no RCD, full SPD, functional model, two-slot electrical or hardware qualification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = review(json.loads(args.input.read_text(encoding='utf-8')))
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], part=result['part'], processes=len(result['variants']))))
