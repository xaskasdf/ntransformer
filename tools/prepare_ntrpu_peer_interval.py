"""Freeze an interval-bound guard against the reviewed parallel-lane donor."""
import argparse
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]
GUARD = 'fpga/experimental/ntrpu_peer_read_admission.sv'
CANDIDATE = 'fpga/experimental/ntrpu_peer_read_admission_interval.sv'
REFERENCE = 'fpga/experimental/ntrpu_peer_admission_reference.sv'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def variant(text, alias=False):
    if not alias:
        text = once(text, 'module ntrpu_peer_read_admission #(',
                    'module ntrpu_peer_read_admission_interval #(')
    text = once(text, '    lease_t lease;', '''    lease_t lease;
    // Inclusive lower bounds remain explicit; these exclusive ends never wrap.
    reg [64:0] window_end, published_end;
    wire [64:0] request_source_end[0:1],request_target_end[0:1];''')
    text = once(text, '    for(genvar l=0;l<2;l++)begin: channels', '''    for(genvar l=0;l<2;l++)begin: channels
        assign request_source_end[l]={1'b0,req[l].source_offset}+{1'b0,req[l].bytes};
        assign request_target_end[l]={1'b0,req[l].target_offset}+{1'b0,req[l].bytes};''')
    text = once(text, '''req[l].target_offset<lease.window_offset || req[l].target_offset-lease.window_offset>lease.window_bytes ||
                req[l].bytes>lease.window_bytes-(req[l].target_offset-lease.window_offset)''',
                '''req[l].target_offset<lease.window_offset || request_target_end[l]>window_end''')
    text = once(text, '''req[l].source_offset-lease.published_offset>lease.published_bytes ||
                req[l].bytes>lease.published_bytes-(req[l].source_offset-lease.published_offset)''',
                '''request_source_end[l]>published_end''')
    text = once(text, "            lease<='0;configured<=0;", "            window_end<=0;published_end<=0;\n            lease<='0;configured<=0;")
    text = once(text, '                if(!cfg_bad)begin lease<=cfg;configured<=1;revoked<=revoke;end', '''                if(!cfg_bad)begin
                    lease<=cfg;configured<=1;revoked<=revoke;
                    window_end<={1'b0,cfg.window_offset}+{1'b0,cfg.window_bytes};
                    published_end<={1'b0,cfg.published_offset}+{1'b0,cfg.published_bytes};
                end''')
    return text


def prepare(donor, output):
    from check_ntrpu_peer_parallel import lineage
    lineage(donor)
    assert not output.exists() and not output.is_relative_to(donor) and not output.is_relative_to(ROOT)
    before = {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}
    original = (donor / GUARD).read_text()
    assert (ROOT / CANDIDATE).read_text() == variant(original)
    shutil.copytree(donor, output)
    shutil.copyfile(donor / GUARD, output / 'INTERVAL_REFERENCE.sv')
    for name in [CANDIDATE, 'fpga/tb/tb_ntrpu_peer_interval.sv',
                 'tools/prepare_ntrpu_peer_interval.py', 'tools/run_ntrpu_peer_interval.py',
                 'tools/check_ntrpu_peer_interval.py']:
        shutil.copyfile(ROOT / name, output / name)
    (output / REFERENCE).write_text(once(original, 'module ntrpu_peer_read_admission #(',
                                      'module ntrpu_peer_admission_reference #('))
    (output / GUARD).write_text(variant(original, alias=True))
    save(output / 'INTERVAL_PROFILE.json', dict(
        scope='65-bit interval endpoints cached only on accepted valid configuration',
        reference_sources=before, reference_guard_sha256=before[GUARD],
        selected_guard_sha256=sha(output / GUARD), candidate_sha256=sha(output / CANDIDATE),
        native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False))
    assert before == {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); prepare(a.donor.resolve(), a.output.resolve())
