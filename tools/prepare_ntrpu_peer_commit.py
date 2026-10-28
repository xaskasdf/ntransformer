"""Freeze a lane context-commit stage against the reviewed size-bound donor."""
import argparse
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]
LANE = 'fpga/experimental/ntrpu_peer_read_lane.sv'
CANDIDATE = 'fpga/experimental/ntrpu_peer_read_lane_commit.sv'
REFERENCE = 'fpga/experimental/ntrpu_peer_commit_reference.sv'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def variant(text, alias=False):
    if not alias:
        text = once(text, 'module ntrpu_peer_read_lane #(',
                    'module ntrpu_peer_read_lane_commit #(')
    text = once(text, '    identity_t saved_reject;', '''    // Reserve at the input handshake; commit the wide context one edge later.
    // The reserved slot cannot stall commit, so replacement can occur each edge.
    reg stage_valid, stage_slot;
    request_t stage_request;
    reg [63:0] stage_source, stage_target;
    request_t reserved0, reserved1;
    always @* begin
        reserved0=(stage_valid && !stage_slot)?stage_request:context0;
        reserved1=(stage_valid && stage_slot)?stage_request:context1;
    end
    identity_t saved_reject;''')
    # Only overlap sees the not-yet-written context. Commands and results wait.
    old = '''        else if((active[0] && request.target_offset<context0.target_offset+context0.bytes &&
                 context0.target_offset<request.target_offset+request.bytes) ||
                (active[1] && request.target_offset<context1.target_offset+context1.bytes &&
                 context1.target_offset<request.target_offset+request.bytes)) admission_error=OVERLAP;'''
    text = once(text, old, old.replace('context0', 'reserved0').replace('context1', 'reserved1'))
    text = once(text, 'assign cmd_valid=active[issue_ptr] && !issued[issue_ptr];',
                'assign cmd_valid=active[issue_ptr] && !issued[issue_ptr] && !(stage_valid && stage_slot==issue_ptr);')
    text = once(text, "            configured<=0;revoked<=0;watermark<=0;lease<='0;",
                "            stage_valid<=0;stage_slot<=0;stage_request<='0;stage_source<=0;stage_target<=0;\n            configured<=0;revoked<=0;watermark<=0;lease<='0;")
    text = once(text, '''        end else begin
            cfg_error<=0;''', '''        end else begin
            stage_valid<=0;
            if(stage_valid)begin
                if(stage_slot)context1<=stage_request;else context0<=stage_request;
                addresses[stage_slot]<=stage_source;destinations[stage_slot]<=stage_target;
            end
            cfg_error<=0;''')
    text = once(text, '''            if(request_valid && request_ready)begin
                if(admission_error!=OK)begin''', '''            if(request_valid && request_ready)begin
                // These wide enables depend on the handshake, not on admission arithmetic.
                stage_request<=request;stage_slot<=alloc_ptr;
                stage_source<=start_address[63:0];stage_target<=target_address[63:0];
                if(admission_error!=OK)begin''')
    text = once(text, '                    if(alloc_ptr)context1<=request;else context0<=request;',
                '                    stage_valid<=1;')
    text = once(text, 'sent[alloc_ptr]<=0;addresses[alloc_ptr]<=start_address[63:0];destinations[alloc_ptr]<=target_address[63:0];',
                'sent[alloc_ptr]<=0;')
    text = once(text, 'if(!result_valid && active[retire_ptr] && !reported[retire_ptr])begin',
                'if(!result_valid && active[retire_ptr] && !reported[retire_ptr] && !(stage_valid && stage_slot==retire_ptr))begin')
    return text


def prepare(donor, output):
    from check_ntrpu_peer_size import lineage
    lineage(donor)
    assert not output.exists() and not output.is_relative_to(donor) and not output.is_relative_to(ROOT)
    before = {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}
    original = (donor / LANE).read_text()
    assert (ROOT / CANDIDATE).read_text() == variant(original)
    shutil.copytree(donor, output)
    shutil.copyfile(donor / LANE, output / 'COMMIT_REFERENCE.sv')
    for name in [CANDIDATE, 'fpga/tb/tb_ntrpu_peer_commit.sv',
                 'tools/prepare_ntrpu_peer_commit.py', 'tools/run_ntrpu_peer_commit.py',
                 'tools/check_ntrpu_peer_commit.py']:
        shutil.copyfile(ROOT / name, output / name)
    (output / REFERENCE).write_text(once(original, 'module ntrpu_peer_read_lane #(',
                                      'module ntrpu_peer_commit_reference #('))
    (output / LANE).write_text(variant(original, alias=True))
    save(output / 'COMMIT_PROFILE.json', dict(scope='Reserve at handshake, commit context/addresses one cycle later; no extra reservations',
        reference_sources=before, reference_lane_sha256=before[LANE],
        selected_lane_sha256=sha(output / LANE), candidate_sha256=sha(output / CANDIDATE),
        native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False))
    assert before == {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    a = ap.parse_args(); prepare(a.donor.resolve(), a.output.resolve())
