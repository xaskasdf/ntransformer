"""Freeze credit look-ahead and the validated-response control boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode = True
from prepare_hbm_registered_ready import module, replace_once


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def generated(donor):
    text = donor.read_text()
    tracker = module(text, 'ntrpu_hbm_read_tracker')
    capture = module(text, 'ntrpu_hbm_predict_capture_queue')
    validated = module(text, 'ntrpu_hbm_predict_validated_queue')
    candidate = replace_once(tracker, 'module ntrpu_hbm_read_tracker ', 'module ntrpu_hbm_boundary_tracker ')
    candidate = replace_once(candidate, '    reg rready_reg;', '''    wire capture_credit_next;
    ntrpu_hbm_capture_credit_predictor credit_predictor(
        .remaining(capture_remaining),.ar_fire(ar_fire),.capture_fire(capture_fire),
        .next_available(capture_credit_next));
    wire predict_validated=!q_empty&&can_validate&&!fault&&!predict_bad_response;
    reg rready_reg;''')
    candidate = replace_once(candidate, 'q_space_next&&(capture_remaining_next!=0)&&!fault_next;',
                              'q_space_next&&capture_credit_next&&!fault_next;')
    candidate = replace_once(candidate, 'ntrpu_hbm_predict_validated_queue #', 'ntrpu_hbm_registered_validated_queue #')
    candidate = replace_once(candidate, '.empty(validated_empty),.raw_space(validated_raw_space));',
                              '.empty(validated_empty),.raw_space(validated_raw_space),.predict_valid(predict_validated));')
    queue = replace_once(validated, 'module ntrpu_hbm_predict_validated_queue ',
                         'module ntrpu_hbm_registered_validated_queue ')
    queue = replace_once(queue, '    input wire s_valid,', '    input wire s_valid,\n    input wire predict_valid,')
    queue = replace_once(queue, '    assign m_valid=rst_n&&count!=0;', '''    // Only this next-state prediction uses unmasked admission. Actual payload
    // and ownership updates retain their original VALID/reset handling.
    wire predict_push=predict_valid&&count<2;
    wire predict_pop=count!=0&&m_ready;
    wire [1:0] predict_count=(predict_push&&!predict_pop)?count+2'd1:
                             (!predict_push&&predict_pop)?count-2'd1:count;
    reg valid_reg;
    always @(posedge clk or negedge rst_n)begin
        if(!rst_n)valid_reg<=0;
        else valid_reg<=predict_count!=0;
    end
    assign m_valid=valid_reg;''')
    credit = '''`timescale 1ns/1ps
// Reachable state: remaining <= 128, and a capture consumes an existing credit.
// Any accepted AR adds 1..16 credits. Under that contract it guarantees that
// next occupancy is nonzero, including a simultaneous capture. Keep the real
// credit counter unchanged; this output is only the READY predictor.
module ntrpu_hbm_capture_credit_predictor(
    input wire [7:0] remaining,
    input wire ar_fire,capture_fire,
    output wire next_available
);
    assign next_available=ar_fire||(|remaining[7:1])||(remaining[0]&&!capture_fire);
endmodule
'''
    return {'reference_tracker.sv':tracker, 'reference_capture.sv':capture,
            'reference_validated.sv':validated, 'candidate_tracker.sv':candidate,
            'candidate_validated.sv':queue, 'candidate_credit.sv':credit}


def prepare(donor, benches, output):
    if output.exists():raise ValueError('Fresh output required')
    here=Path(__file__).resolve().parent
    contents=generated(donor)
    tracker=(benches/'tb_hbm_registered_ready_tracker.sv').read_text()
    tracker=replace_once(tracker,'ntrpu_hbm_registered_ready_tracker #','ntrpu_hbm_boundary_tracker #')
    tracker=replace_once(tracker,'            if(rst_n)begin','''            if(rst_n)begin
                check(dut.capture_remaining<=128,"bounded capture credit");
                check(dut.capture_credit_next===(dut.capture_remaining_next!=0),"credit predictor arithmetic equivalence");
                check(dut.predict_validated===dut.validated_cut.s_valid,"validated predictor active-edge equivalence");''')
    tracker=replace_once(tracker,'    always @(h_rready)if(rst_n&&$realtime!=last_clock_edge)',
                        '''    always @(p_rvalid)if(rst_n&&$realtime!=last_clock_edge)
        $fatal(1,"REGISTERED_READY_TRACKER pending=%0d P_RVALID changed between clock edges",MAX_PENDING);
    always @(h_rready)if(rst_n&&$realtime!=last_clock_edge)''')
    contents['tb_hbm_registered_ready_tracker.sv']=tracker
    queue=(benches/'tb_hbm_return_valid_queue.sv').read_text()
    queue=queue.replace('RETURN_VALID_QUEUE','VALIDATED_BOUNDARY').replace('return_valid_case','validated_boundary_case').replace('tb_hbm_return_valid_queue','tb_hbm_validated_boundary')
    queue=replace_once(queue,'    ntrpu_hbm_return_valid_queue #(.WIDTH(WIDTH)) dut(.*);','''    wire raw_space,r_raw_space;
    wire predict_valid=rst_n?s_valid:!s_valid;
    ntrpu_hbm_registered_validated_queue #(.WIDTH(WIDTH)) dut(.*);''')
    queue=replace_once(queue,'ntrpu_hbm_read_return_cut #','ntrpu_hbm_predict_validated_queue #')
    queue=replace_once(queue,'.m_valid(r_valid),.m_ready(m_ready),.m_data(r_data),.empty(r_empty));',
                        '.m_valid(r_valid),.m_ready(m_ready),.m_data(r_data),.empty(r_empty),.raw_space(r_raw_space));')
    queue=replace_once(queue,'            if(!reset_n||prior==2)check(dut.slot0===old0&&dut.slot1===old1,"full/reset payload retained");',
                        '            if(!take)check(dut.slot0===old0&&dut.slot1===old1,"unaccepted payload retained");')
    queue=replace_once(queue,'            check(empty===(occupied==0),"pending empty model");',
                        '            check(empty===(occupied==0),"pending empty model");\n            check(raw_space===(occupied<2)&&raw_space===r_raw_space,"raw space model");')
    contents['tb_hbm_validated_boundary.sv']=queue
    output.mkdir(parents=True)
    for name,value in contents.items():(output/name).write_text(value,newline='\n')
    for name in ['prepare_hbm_control_boundaries.py','prepare_hbm_registered_ready.py',
                 'tb_hbm_credit_predicate.sv','synthesize_hbm_control_boundaries.tcl',
                 'run_hbm_control_boundaries.py','review_hbm_control_boundaries.py']:
        shutil.copy2(here/name,output/name)
    binding=dict(status='HBM_CONTROL_BOUNDARIES_PREPARED',donor_adapter_sha256=sha(donor),
                 bench_sources={n:sha(benches/n) for n in ['tb_hbm_registered_ready_tracker.sv','tb_hbm_return_valid_queue.sv']},
                 files={p.name:sha(p) for p in sorted(output.iterdir())},full_board_integrated=False,
                 physical_timing_qualified=False)
    (output/'PREPARED.json').write_text(json.dumps(binding,indent=2)+'\n')
    return binding


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('donor','benches','output'):p.add_argument(name,type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.donor,a.benches,a.output),indent=2))
