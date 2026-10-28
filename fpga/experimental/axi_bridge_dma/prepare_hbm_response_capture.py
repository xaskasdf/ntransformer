"""Freeze the response-only free-slot candidate against an explicit board donor."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tokens(text):
    return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', text))


def module(text, name):
    matches = re.findall(r'\bmodule\s+' + name + r'\b.*?\bendmodule\b', text, re.S)
    if len(matches) != 1:
        raise ValueError('Expected one module: ' + name)
    return '`timescale 1ns/1ps\n' + matches[0] + '\n'


def prepare(donor, output):
    if output.exists():
        raise ValueError('Fresh output required')
    here = Path(__file__).resolve().parent
    adapter = donor / 'ntrpu_hbm_read_adapter.sv'
    frozen = adapter.read_text()
    queue = module(frozen, 'ntrpu_hbm_read_response_queue')
    tracker = module(frozen, 'ntrpu_hbm_read_tracker')
    candidate = (here / 'ntrpu_hbm_response_capture_queue.sv').read_text()
    anchor = '''if(push)begin
                if(write_slot)slot1<=s_data;
                else slot0<=s_data;
                write_slot<=!write_slot;
            end'''
    replacement = '''if(s_ready)begin
                if(write_slot)slot1<=s_data;
                else slot0<=s_data;
            end
            if(push)write_slot<=!write_slot;'''
    if queue.count(anchor) != 1:
        raise ValueError('Queue transformation anchor changed')
    expected = queue.replace(anchor, replacement).replace(
        'module ntrpu_hbm_read_response_queue ', 'module ntrpu_hbm_response_capture_queue ', 1)
    if tokens(expected) != tokens(candidate):
        raise ValueError('Candidate has collateral changes')
    binding = 'ntrpu_hbm_read_response_queue #(.WIDTH(265)) response_cut ('
    if tracker.count(binding) != 1:
        raise ValueError('Tracker response binding changed')
    modified = tracker.replace('module ntrpu_hbm_read_tracker ',
                               'module ntrpu_hbm_read_capture_tracker ', 1).replace(
        binding, 'ntrpu_hbm_response_capture_queue #(.WIDTH(265)) response_cut (', 1)
    output.mkdir(parents=True)
    origins = {}
    for name, text in [('reference_queue.sv', queue), ('reference_tracker.sv', tracker),
                       ('candidate_tracker.sv', modified)]:
        (output / name).write_text(text, newline='\n')
        origins[name] = dict(origin=str(adapter), origin_sha256=sha(adapter), transformed=True)
    for name in ('ntrpu_hbm_response_capture_queue.sv', 'tb_hbm_response_capture.sv',
                 'tb_hbm_capture_tracker.sv', 'synthesize_hbm_response_capture.tcl',
                 'run_hbm_response_capture.py', 'prepare_hbm_response_capture.py',
                 'review_hbm_response_capture.py'):
        shutil.copy2(here / name, output / name)
        origins[name] = dict(origin=str(here / name), origin_sha256=sha(here / name), transformed=False)
    for name in origins:
        origins[name]['sha256'] = sha(output / name)
    manifest = dict(status='HBM_RESPONSE_CAPTURE_PREPARED', files=origins,
                    donor_adapter_sha256=sha(adapter), response_only=True,
                    validated_queue_changed=False, full_board_integrated=False,
                    functional_delta='Unowned slot capture on s_ready; ownership still on push')
    (output / 'PREPARED.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), files=len(origins))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--donor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepare(a.donor.resolve(), a.output.resolve())
