"""Freeze the reviewed mapped native sources for a separate OOC implementation."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]
TOOLS = tuple('tools/' + name + '_ntrpu_peer_regions_physical.py'
              for name in ('prepare', 'run', 'check'))


def prepare(selected, output):
    result = json.loads((selected / 'output/result.json').read_text(encoding='utf-8'))
    review = json.loads((selected / 'REVIEW.json').read_text(encoding='utf-8'))
    assert review['status'] == 'REGIONS_NATIVE_TRAFFIC_REVIEWED'
    assert review['native_hbm_controller'] and review['ownership']['counters']['successful_retire'] == 16
    assert review['result_sha256'] == sha(selected / 'output/result.json')
    assert result['status'] == 'EXPORTED_PENDING_REVIEW' and result['sources'] == result['sources_after']
    assert all(sha(selected / 'repo' / n) == h for n, h in result['sources'].items())
    assert all(sha(selected / 'output' / n) == h for n, h in
               json.loads((selected / 'output/FILES.json').read_text(encoding='utf-8')).items())
    shutil.copytree(selected / 'repo', output)
    for name in TOOLS:
        assert not (output / name).exists()
        shutil.copyfile(ROOT / name, output / name)
    save(output / 'REGIONS_PHYSICAL_PROFILE.json', dict(
        native_result_sha256=review['result_sha256'],
        native_review_sha256=sha(selected / 'REVIEW.json'),
        native_sources=result['sources'], part='xcvu47p-fsvh2892-2-e',
        transport_period_ns=5.0, memory_period_ns=2.222,
        scope='Mapped shared-reader logic, OOC; native HBM and PCIe excluded',
        native_hbm_in_design=False, external_io_qualified=False, board_qualified=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('selected', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); prepare(args.selected.resolve(), args.output.resolve())
