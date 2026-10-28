"""Exhaustively compare the known-bit byte-bound formulas used by R3 and R5.

This is an arithmetic lemma, not RTL equivalence or four-state qualification.
The existing XSIM banks must still check selection, context and clocked behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def check():
    started = time.monotonic()
    delivered = np.arange(8192, dtype=np.int32)[None, :]
    comparisons = 0
    for low in range(4):
        end = delivered + low  # Fourteen bits; never truncate end>>2 to eleven.
        for start in range(0, 2048, 128):
            seen = np.arange(start, start + 128, dtype=np.int32)[:, None]
            dword = (seen - 3) & 0x7ff
            for byte in range(4):
                # R3's signed 15-bit temporary is nonnegative whenever seen>=3.
                index = ((seen - 3) * 4 + byte) & 0x7fff
                original = (seen >= 3) & (index >= low) & (index < low + delivered)
                bounded = ((seen >= 3) & ((dword != 0) | (byte >= low))
                           & ((dword < (end >> 2))
                              | ((dword == (end >> 2)) & (byte < (end & 3)))))
                mismatch = np.argwhere(original != bounded)
                if mismatch.size:
                    row, column = map(int, mismatch[0])
                    raise ValueError((start + row, low, column, byte))
                comparisons += original.size
    assert comparisons == 268435456
    return {
        'status': 'PASS_EXHAUSTIVE_KNOWN_BIT_PAYLOAD_BOUND_LEMMA',
        'seen_range': [0, 2047], 'low_range': [0, 3],
        'delivered_range': [0, 8191], 'tuples': 67108864,
        'byte_comparisons': comparisons, 'mismatches': 0,
        'duration_seconds': time.monotonic() - started,
        'rtl_equivalence_proven': False, 'four_state_equivalence_proven': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    result = check()
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / 'fpga/rtl/ntrpu_pcie_rc_event_stage_r3.sv',
             root / 'fpga/rtl/ntrpu_pcie_rc_event_stage_r5.sv']
    result['source_hashes'] = {
        str(path.relative_to(root)).replace('\\', '/'):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
