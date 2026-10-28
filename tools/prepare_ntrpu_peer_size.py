"""Freeze size-bound comparisons against the reviewed interval-guard donor."""
import argparse
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]
GUARD = 'fpga/experimental/ntrpu_peer_read_admission.sv'
CANDIDATE = 'fpga/experimental/ntrpu_peer_read_admission_size.sv'
REFERENCE = 'fpga/experimental/ntrpu_peer_size_reference.sv'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def variant(text, alias=False):
    if not alias:
        text = once(text, 'module ntrpu_peer_read_admission #(',
                    'module ntrpu_peer_read_admission_size #(')
    text = once(text, '''req[l].source_offset>lease.source_size ||
                req[l].bytes>lease.source_size-req[l].source_offset''',
                "request_source_end[l]>{1'b0,lease.source_size}")
    text = once(text, '''req[l].target_offset>lease.target_size || req[l].bytes>lease.target_size-req[l].target_offset''',
                "request_target_end[l]>{1'b0,lease.target_size}")
    return text


def prepare(donor, output):
    from check_ntrpu_peer_interval import lineage
    lineage(donor)
    assert not output.exists() and not output.is_relative_to(donor) and not output.is_relative_to(ROOT)
    before = {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}
    original = (donor / GUARD).read_text()
    assert (ROOT / CANDIDATE).read_text() == variant(original)
    shutil.copytree(donor, output)
    shutil.copyfile(donor / GUARD, output / 'SIZE_REFERENCE.sv')
    for name in [CANDIDATE, 'fpga/tb/tb_ntrpu_peer_size.sv',
                 'tools/prepare_ntrpu_peer_size.py', 'tools/run_ntrpu_peer_size.py',
                 'tools/check_ntrpu_peer_size.py']:
        shutil.copyfile(ROOT / name, output / name)
    (output / REFERENCE).write_text(once(original, 'module ntrpu_peer_read_admission #(',
                                      'module ntrpu_peer_size_reference #('))
    (output / GUARD).write_text(variant(original, alias=True))
    save(output / 'SIZE_PROFILE.json', dict(
        scope='Reuse 65-bit request endpoints for source_size and target_size; no new state or cycles',
        reference_sources=before, reference_guard_sha256=before[GUARD],
        selected_guard_sha256=sha(output / GUARD), candidate_sha256=sha(output / CANDIDATE),
        native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False))
    assert before == {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); prepare(a.donor.resolve(), a.output.resolve())
