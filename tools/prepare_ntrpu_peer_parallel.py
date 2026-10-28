"""Freeze parallel per-context event validation against the unchanged donor."""
import argparse
import json
from pathlib import Path
import shutil
from run_ntrpu_peer_read import sha, save

ROOT = Path(__file__).resolve().parents[1]
LANE = 'fpga/experimental/ntrpu_peer_read_lane.sv'
CANDIDATE = 'fpga/experimental/ntrpu_peer_read_lane_parallel.sv'
REFERENCE = 'fpga/experimental/ntrpu_peer_read_lane_reference.sv'


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def variant(text, alias=False):
    if not alias:
        text = once(text, 'module ntrpu_peer_read_lane #(', 'module ntrpu_peer_read_lane_parallel #(')
    text = once(text, 'selected_cpl, selected_dma, selected_ack, selected_issue', 'selected_cpl, selected_issue')
    for name, value in [('dma', 'dma_done'), ('ack', 'ack')]:
        text = once(text, f'    wire {name}_index=active[1] && context1.identity=={value}.identity;\n', '')
        text = once(text, f'        selected_{name}={name}_index?context1:context0;\n', '')
    start = text.index('    wire dma_match=')
    end = text.index('    wire data_error=', start)
    replacement = '''    // No wide context multiplexer precedes event authentication.
    // Admission guarantees distinct sequence IDs for live contexts.
    wire [1:0] dma_accept, ack_accept;
    for(genvar e=0;e<2;e++)begin: event_validation
        request_t event_context;
        assign event_context=contexts[e];
        assign dma_accept[e]=active[e] && issued[e] && !dma_complete[e] &&
            event_context.identity==dma_done.identity && event_context.target==dma_done.target &&
            event_context.window_generation==dma_done.window_generation && dma_done.status==0 &&
            dma_done.bytes==event_context.bytes && sent[e]==event_context.bytes;
        assign ack_accept[e]=active[e] && issued[e] && !ack_complete[e] &&
            event_context.identity==ack.identity && event_context.target==ack.target &&
            event_context.window_generation==ack.window_generation && ack.status==0 &&
            ack.bytes==event_context.bytes && sent[e]==event_context.bytes;
    end
'''
    text = text[:start] + replacement + text[end:]
    start = text.index('    wire dma_error=')
    end = text.index('    wire fail_now=', start)
    text = text[:start] + '''    wire dma_error=dma_done_valid && !(|dma_accept);
    wire ack_error=ack_valid && !(|ack_accept);
''' + text[end:]
    text = once(text, '''            if(dma_done_valid && dma_match && !dma_error)dma_complete[dma_index]<=1;
            if(ack_valid && ack_match && !ack_error)ack_complete[ack_index]<=1;
''', '''            for(integer i=0;i<2;i++)begin
                if(dma_done_valid && dma_accept[i])dma_complete[i]<=1;
                if(ack_valid && ack_accept[i])ack_complete[i]<=1;
            end
''')
    return text


def prepare(donor, output):
    assert not output.exists() and not output.is_relative_to(donor) and not output.is_relative_to(ROOT)
    before = {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}
    native = json.loads((donor / 'NATIVE_PROFILE.json').read_text())
    assert all(before[n] == h for n, h in native['selected_rtl'].items())
    original = (donor / LANE).read_text()
    assert (ROOT / CANDIDATE).read_text() == variant(original)
    shutil.copytree(donor, output); shutil.copyfile(donor / LANE, output / 'PARALLEL_REFERENCE.sv')
    for name in [CANDIDATE, 'fpga/tb/tb_ntrpu_peer_parallel.sv',
                 'tools/prepare_ntrpu_peer_parallel.py', 'tools/run_ntrpu_peer_parallel.py',
                 'tools/check_ntrpu_peer_parallel.py']:
        shutil.copyfile(ROOT / name, output / name)
    (output / REFERENCE).write_text(once(original, 'module ntrpu_peer_read_lane #(',
                                      'module ntrpu_peer_read_lane_reference #('))
    (output / LANE).write_text(variant(original, alias=True))
    save(output / 'PARALLEL_PROFILE.json', dict(
        scope='Full DMA/ACK validation per context in parallel; no new sequential stage',
        reference_profile_sha256=sha(donor / 'NATIVE_PROFILE.json'), reference_lane_sha256=before[LANE],
        reference_sources=before, selected_lane_sha256=sha(output / LANE), candidate_sha256=sha(output / CANDIDATE),
        native_hbm_qualified=False, physical_timing_qualified=False, board_qualified=False))
    assert before == {p.relative_to(donor).as_posix(): sha(p) for p in donor.rglob('*') if p.is_file()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('donor', type=Path); ap.add_argument('output', type=Path)
    args = ap.parse_args(); prepare(args.donor.resolve(), args.output.resolve())
