"""Freeze the sequence-index variant without changing its native donor profile."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

LANE = 'fpga/experimental/ntrpu_peer_read_lane.sv'
CANDIDATE = 'fpga/experimental/ntrpu_peer_read_lane_seq_selector.sv'
REFERENCE = 'fpga/experimental/ntrpu_peer_read_lane_reference.sv'
ROOT = Path(__file__).resolve().parents[1]


def variant(text, alias=False):
    changes = [('context1.identity==dma_done.identity;',
                'context1.identity.sequence_id==dma_done.identity.sequence_id;'),
               ('context1.identity==ack.identity;',
                'context1.identity.sequence_id==ack.identity.sequence_id;')]
    if not alias:
        changes.append(('module ntrpu_peer_read_lane #(', 'module ntrpu_peer_read_lane_seq_selector #('))
    for before, after in changes:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    return text


def prepare(donor, output):
    assert not output.exists()
    profile = json.loads((donor / 'NATIVE_PROFILE.json').read_text())
    before = {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}
    assert all(before[n] == h for n, h in profile['selected_rtl'].items())
    original = (donor / LANE).read_text()
    assert (ROOT / CANDIDATE).read_text() == variant(original)
    shutil.copytree(donor, output)
    shutil.copyfile(donor / LANE, output / 'SELECTOR_REFERENCE.sv')
    for name in [CANDIDATE, 'fpga/tb/tb_ntrpu_peer_selector.sv',
                 'tools/prepare_ntrpu_peer_selector.py', 'tools/run_ntrpu_peer_selector.py',
                 'tools/check_ntrpu_peer_selector.py']:
        shutil.copyfile(ROOT / name, output / name)
    (output / REFERENCE).write_text(original.replace('module ntrpu_peer_read_lane #(',
                                                    'module ntrpu_peer_read_lane_reference #('))
    (output / LANE).write_text(variant(original, alias=True))
    save(output / 'SELECTOR_PROFILE.json', dict(
        scope='Sequence-only context selection; full event authentication retained',
        reference_profile_sha256=sha(donor / 'NATIVE_PROFILE.json'),
        reference_lane_sha256=before[LANE], reference_sources=before,
        selected_lane_sha256=sha(output / LANE), candidate_sha256=sha(output / CANDIDATE),
        native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False))
    assert before == {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); prepare(args.donor.resolve(), args.output.resolve())
