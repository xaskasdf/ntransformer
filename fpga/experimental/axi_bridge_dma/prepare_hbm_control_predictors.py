"""Freeze isolated READY-input and return-VALID experiments from a routed donor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
from prepare_hbm_registered_ready import module, replace_once


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracker_variant(text):
    text = replace_once(text, 'module ntrpu_hbm_read_tracker ',
                        'module ntrpu_hbm_registered_ready_tracker ')
    text = replace_once(text,
        '    wire fault_next=fault||bad_response||unexpected_offer||bad_request;',
        '''    // These look-ahead terms are evaluated only by an asynchronously
    // cleared register at an active clock edge with rst_n high. Keep reset
    // masking on all actual interfaces and ownership updates, but exclude it
    // from the D cone. Raw queue state is not an external handshake.
    wire validated_raw_space;
    wire predict_bad_response=!q_empty&&(!can_validate||q_rid!=0||q_resp==2'b01||q_last!=terminal);
    wire predict_qready=validated_raw_space&&can_validate&&!fault&&!predict_bad_response;
    wire predict_bad_request=p_arvalid&&!fault&&!offer_valid&&reserved_count<MAX_PENDING&&
        (p_arid!=0||p_arsize!=5||p_arburst!=1);
    wire fault_next=fault||predict_bad_response||unexpected_offer||predict_bad_request;''')
    text = replace_once(text, 'ntrpu_hbm_ready_capture_queue #(.WIDTH(265)) response_cut (',
                        'ntrpu_hbm_predict_capture_queue #(.WIDTH(265)) response_cut (')
    text = replace_once(text, '.empty(q_empty),.next_space(q_space_next));',
                        '.empty(q_empty),.next_space(q_space_next),.predict_ready(predict_qready));')
    text = replace_once(text, 'ntrpu_hbm_read_response_queue #(.WIDTH(265)) validated_cut (',
                        'ntrpu_hbm_predict_validated_queue #(.WIDTH(265)) validated_cut (')
    return replace_once(text, '.empty(validated_empty));',
                        '.empty(validated_empty),.raw_space(validated_raw_space));')


def capture_variant(text):
    text = replace_once(text, 'module ntrpu_hbm_ready_capture_queue ',
                        'module ntrpu_hbm_predict_capture_queue ')
    text = replace_once(text, '    output wire next_space',
                        '    input wire predict_ready,\n    output wire next_space')
    return replace_once(text, '    assign next_space=next_count<2;', '''    // Reset-free prediction; actual push/pop and storage remain reset-masked.
    wire predict_push=s_valid&&count<2;
    wire predict_pop=count!=0&&predict_ready;
    wire [1:0] predict_count=(predict_push&&!predict_pop)?count+2'd1:
                             (!predict_push&&predict_pop)?count-2'd1:count;
    assign next_space=predict_count<2;''')


def validated_variant(text):
    text = replace_once(text, 'module ntrpu_hbm_read_response_queue ',
                        'module ntrpu_hbm_predict_validated_queue ')
    text = replace_once(text, '    output wire empty',
                        '    output wire raw_space,\n    output wire empty')
    return replace_once(text, '    assign empty=count==0;',
                        '    assign empty=count==0;\n    assign raw_space=count<2;')


def return_variant(text):
    text = replace_once(text, 'module ntrpu_hbm_read_return_cut ',
                        'module ntrpu_hbm_return_valid_queue ')
    return replace_once(text, '    assign m_valid=rst_n&&count!=0;', '''    // Predict occupancy after this edge. No extra slot or cycle of latency.
    // Reset assertion clears external VALID immediately; coordinated cold
    // reset still must span a clock edge to clear synchronous ownership.
    wire predict_push=s_valid&&count<2;
    wire predict_pop=count!=0&&m_ready;
    wire [1:0] predict_count=(predict_push&&!predict_pop)?count+2'd1:
                             (!predict_push&&predict_pop)?count-2'd1:count;
    reg valid_reg;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)valid_reg<=0;
        else valid_reg<=predict_count!=0;
    end
    assign m_valid=valid_reg;''')


def generated(donor):
    text = donor.read_text()
    refs = {'reference_tracker.sv': module(text, 'ntrpu_hbm_read_tracker'),
            'reference_ready_queue.sv': module(text, 'ntrpu_hbm_ready_capture_queue'),
            'reference_queue.sv': module(text, 'ntrpu_hbm_read_response_queue'),
            'reference_return.sv': module(text, 'ntrpu_hbm_read_return_cut')}
    refs.update({'candidate_tracker.sv': tracker_variant(refs['reference_tracker.sv']),
                 'candidate_capture.sv': capture_variant(refs['reference_ready_queue.sv']),
                 'candidate_validated.sv': validated_variant(refs['reference_queue.sv']),
                 'candidate_return.sv': return_variant(refs['reference_return.sv'])})
    return refs


def prepare(donor, output):
    if output.exists():
        raise ValueError('Fresh output required')
    here = Path(__file__).resolve().parent
    contents = generated(donor)
    queue = (here/'tb_hbm_registered_ready_queue.sv').read_text()
    queue = replace_once(queue, 'ntrpu_hbm_ready_capture_queue #', 'ntrpu_hbm_predict_capture_queue #')
    queue = replace_once(queue, '.empty(empty),.next_space(next_space));',
                         '.empty(empty),.next_space(next_space),.predict_ready(m_ready));')
    contents['tb_hbm_registered_ready_queue.sv'] = queue
    tracker = (here/'tb_hbm_registered_ready_tracker.sv').read_text()
    tracker = replace_once(tracker, '            if(rst_n)begin', '''            if(rst_n)begin
                check(dut.q_space_next===reference.q_space_next,"active-edge space predictor equivalence");
                check(dut.fault_next===reference.fault_next,"active-edge fault predictor equivalence");
                check(dut.predict_qready===dut.q_ready,"active-edge validate predictor equivalence");
                check(dut.predict_bad_response===dut.bad_response,"active-edge response fault equivalence");
                check(dut.predict_bad_request===dut.bad_request,"active-edge request fault equivalence");''')
    contents['tb_hbm_registered_ready_tracker.sv'] = tracker
    output.mkdir(parents=True)
    for name, text in contents.items():
        (output/name).write_text(text, newline='\n')
    names = ['tb_hbm_return_valid_queue.sv', 'synthesize_hbm_control_predictors.tcl',
             'run_hbm_control_predictors.py', 'prepare_hbm_control_predictors.py',
             'prepare_hbm_registered_ready.py', 'review_hbm_control_predictors.py']
    for name in names:
        shutil.copy2(here/name, output/name)
    receipt = dict(status='HBM_CONTROL_PREDICTORS_PREPARED', donor_adapter_sha256=sha(donor),
                   files={p.name: sha(p) for p in sorted(output.iterdir())},
                   response_depth=2, return_depth=2, full_board_integrated=False,
                   physical_timing_qualified=False)
    (output/'PREPARED.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('donor', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.donor.resolve(), a.output.resolve()), indent=2))
