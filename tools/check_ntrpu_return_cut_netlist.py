"""Require reversible namespace isolation and the complete frozen comparison."""
import hashlib
import json
from pathlib import Path
import re
import sys

import check_ntrpu_return_cut as frozen
from prepare_ntrpu_return_cut_netlist import isolate, rename
from run_ntrpu_return_cut import SOURCES as FROZEN_SOURCES


def main():
    out = Path(sys.argv[1]); review = Path(sys.argv[sys.argv.index('--review')+1])
    record = json.loads((out/'result.json').read_text())
    assert record['stage'] == 'netlist'
    old = json.loads((out/'inputs/result.json').read_text())
    assert old['sources'] == old['sources_after'] == {n: record['sources'][n] for n in FROZEN_SOURCES}
    original = (out/'inputs/physical_funcsim.v').read_bytes()
    expected, proof = isolate(original)
    gen = out/'suite/generated'
    assert (gen/'route_isolated.v').read_bytes() == expected
    assert json.loads((gen/'NETLIST_NAMESPACE.json').read_text()) == proof
    assert proof['original_sha256'] == record['inputs']['physical_funcsim.v']
    inverse = {v.encode(): k.encode() for k, v in proof['mapping'].items()}
    # A modified circuit must fail the byte-recovery criterion.
    damaged = expected.replace(b'1\'b0', b'1\'b1', 1)
    assert damaged != expected and rename(damaged, inverse)[0] != original
    if record['status'] == 'EXPORTED_PENDING_REVIEW':
        logs = [out/'console.log', *(out/'tool_logs').rglob('*.log')]
        assert not any(re.search(r'VRFC 10-2861|does not have a parameter named', p.read_text(errors='replace')) for p in logs)
    frozen.main()
    result = json.loads(review.read_text())
    result['namespace_isolation'] = proof
    result['namespace_negative_controls'] = ['changed_constant_rejected']
    result['frozen_source_bridge_verified'] = True
    result['namespace_checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    review.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
