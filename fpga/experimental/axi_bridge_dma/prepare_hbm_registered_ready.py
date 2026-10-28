"""Freeze predictive RREADY against the exact physically measured capture RTL."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def module(text,name):
    found=re.findall(r'\bmodule\s+'+name+r'\b.*?\bendmodule\b',text,re.S)
    if len(found)!=1:raise ValueError('Expected one module: '+name)
    return '`timescale 1ns/1ps\n'+found[0]+'\n'


def replace_once(text,anchor,new):
    if text.count(anchor)!=1:raise ValueError('Changed transformation anchor: '+anchor)
    return text.replace(anchor,new,1)


def transform(tracker):
    text=replace_once(tracker,'module ntrpu_hbm_read_tracker ',
                      'module ntrpu_hbm_registered_ready_tracker ')
    text=replace_once(text,'    assign h_rready=q_space&&capture_enable;', '''    // Predict post-edge admission, not a one-cycle delay of current READY.
    wire q_space_next;
    wire [7:0] capture_remaining_next=capture_remaining+
        (ar_fire?({4'b0,h_arlen}+8'd1):8'd0)-(capture_fire?8'd1:8'd0);
    wire bad_request=reserve_fire&&(p_arid!=0||p_arsize!=5||p_arburst!=1);
    wire fault_next=fault||bad_response||unexpected_offer||bad_request;
    reg rready_reg;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)rready_reg<=0;
        else rready_reg<=q_space_next&&(capture_remaining_next!=0)&&!fault_next;
    end
    assign h_rready=rready_reg;''')
    text=replace_once(text,'ntrpu_hbm_response_capture_queue #(.WIDTH(265)) response_cut (',
                     'ntrpu_hbm_ready_capture_queue #(.WIDTH(265)) response_cut (')
    text=replace_once(text,'.m_data({q_rid,q_resp,q_last,q_data}),.empty(q_empty));',
                     '.m_data({q_rid,q_resp,q_last,q_data}),.empty(q_empty),.next_space(q_space_next));')
    return text


def prepare(donor,output):
    if output.exists():raise ValueError('Fresh output required')
    here=Path(__file__).resolve().parent
    frozen=donor.read_text()
    tracker=module(frozen,'ntrpu_hbm_read_tracker')
    candidate=transform(tracker)
    output.mkdir(parents=True)
    generated={'reference_tracker.sv':tracker,'candidate_tracker.sv':candidate,
               'reference_queue.sv':module(frozen,'ntrpu_hbm_read_response_queue'),
               'reference_capture_queue.sv':module(frozen,'ntrpu_hbm_response_capture_queue')}
    for name,text in generated.items():(output/name).write_text(text,newline='\n')
    names=['ntrpu_hbm_ready_capture_queue.sv','tb_hbm_registered_ready_queue.sv',
           'tb_hbm_registered_ready_tracker.sv','synthesize_hbm_registered_ready.tcl',
           'run_hbm_registered_ready.py','prepare_hbm_registered_ready.py','review_hbm_registered_ready.py']
    for name in names:shutil.copy2(here/name,output/name)
    binding=dict(status='HBM_REGISTERED_READY_PREPARED',donor_adapter_sha256=sha(donor),
                 files={p.name:sha(p) for p in output.iterdir()},
                 response_depth=2,registered_hbm_rready=True,predicts_occupancy_credit_and_fault=True,
                 full_board_integrated=False,physical_timing_qualified=False)
    (output/'PREPARED.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),files=len(binding['files']),donor_sha256=sha(donor))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('donor',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();prepare(a.donor.resolve(),a.output.resolve())
