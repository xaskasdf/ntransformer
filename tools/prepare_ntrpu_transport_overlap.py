"""Generate isolated overlap experiments; leave the qualified reference untouched."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_ntrpu_registered_dispatch import one
from prepare_ntrpu_transport_study import performance

ROOT = Path(__file__).resolve().parents[1]


def dispatcher(text, variant):
    text = one(text, '    always @(posedge clk) begin\n        if (!rst_n) begin\n            candidate_valid<=0;',
               '''    // Metadata may be prepared without a credit. Only reserve_take owns
    // resources; its IDLE/live-offer/fault checks remain unchanged.
    wire [1:0] prepare_mask=(|eligible) ? eligible : head_valid;
    always @(posedge clk) begin
        if (!rst_n) begin
            candidate_valid<=0;''')
    text = one(text, 'fault || state!=IDLE || (|reserve_take)', 'fault || (|reserve_take)')
    text = one(text, 'if (eligible[next_channel]) begin', 'if (prepare_mask[next_channel]) begin')
    text = one(text, 'else if (eligible[!next_channel]) begin', 'else if (prepare_mask[!next_channel]) begin')
    if variant == 'compact':
        text = one(text, 'assign cmd_valid = rst_n && state==COMMAND;', 'assign cmd_valid = rst_n && state==FETCH;')
        text = one(text, 'state<=COMMAND;', 'state<=FETCH;')
    return text


def adapter(text, variant):
    if variant == 'ahead':
        return text
    text = one(text, '    assign fault=adapter_fault', '''    // Echo acknowledges the already captured lease, never a second command.
    wire command_accepted=command_seen || (command_valid && command_ready);
    // Completion refers to acceptance by the fill engine, not the outer queue.
    wire finishing=payload_fire && received+1'b1==expected_beats;
    wire frame_complete=done_pending || finishing;
    wire [8:0] completed_beats=done_pending ? received : received+1'b1;

    assign fault=adapter_fault''')
    assert text.count('&& command_seen &&') == 2
    text = text.replace('&& command_seen &&', '&& command_accepted &&')
    text = one(text, '.frame_done_valid(done_pending)', '.frame_done_valid(frame_complete)')
    text = one(text, '.frame_done_beats(received)', '.frame_done_beats(completed_beats)')
    text = one(text, 'if(done_pending && done_ready)begin', 'if(frame_complete && done_ready)begin')
    return text


OBSERVE = '''    always @(posedge clk)begin
        if(rst_n)begin
            need(!(dut.core.lease_valid && (|dut.core.reserve_take)),"NO_SECOND_ACTIVE_LEASE");
            if(dut.core.candidate_valid && dut.core.lease_valid)
                $display("OV AHEAD %0d %0d %0d %0d",cycle,scenario,
                    dut.core.lease_desc[199:136],dut.core.selected_desc[199:136]);
            if(dut.core.command_valid && dut.core.command_ready &&
               dut.core.fetch_valid && dut.core.dispatch_fetch_ready)
                $display("OV ECHO_FETCH %0d %0d",cycle,scenario);
            if(dut.core.payload_fire && dut.core.received+1'b1==dut.core.expected_beats)
                $display("OV LAST %0d %0d %0d %b %b",cycle,scenario,
                    dut.core.lease_desc[199:136],fault_in,dut.core.dispatch.frame_done_valid);
        end
    end
'''


EXTRA = '''    task wait_ahead;
        begin: waiting
            for(integer j=0;j<100;j++)begin step;
                if(dut.core.lease_valid && dut.core.payload_active && dut.core.candidate_valid)
                    disable waiting;
            end
            need(0,"AHEAD_TIMEOUT");
        end
        need(dut.core.selected_desc[199:136]==1001 && dut.core.lease_desc[199:136]==1000,
            "SEPARATE_ACTIVE_AND_PREPARED_METADATA");
        need(dut.core.reserve_take==0,"AHEAD_OWNS_NO_CREDIT");
    endtask
    initial begin
        reset_case(1);hold_payload=1;push_one(0,1000,8);push_one(1,1001,8);wait_ahead;
        $display("OV WINDOW 1 active_candidate_fault");fault_in=1;step;fault_in=0;hold_payload=0;
        wait_cpl(0,1);repeat(12)step;
        need(fault&&!drained&&cpls[1]==0&&queued==8'h10&&banks_used==0,"AHEAD_FAULT_QUARANTINE");pass_case;

        reset_case(2);hold_payload=1;hold_cpl=3;
        push_one(0,1000,8);push_one(1,1001,8);wait_ahead;
        $display("OV WINDOW 2 active_candidate_stop_fence");stop=1;send_fence(202);hold_payload=0;
        repeat(150)step;need(cpl_valid==3&&!fence_done_valid&&!drained,"FENCE_WAITS_EXTERNAL_RESULTS");
        hold_cpl=0;wait_fence;fence_done_ready=1;step;fence_done_ready=0;drain_all;
        need(submit_ready==0,"STOP_RETAINS_CLOSURE");pass_case;

        reset_case(3);hold_payload=1;push_one(0,1000,8);push_one(1,1001,8);wait_ahead;
        hold_payload=0;
        begin: last_wait
            for(integer j=0;j<100;j++)begin step;
                if(dut.core.payload_fire && dut.core.received+1'b1==dut.core.expected_beats)
                    disable last_wait;
            end
            need(0,"LAST_BEAT_TIMEOUT");
        end
        $display("OV WINDOW 3 last_engine_beat_fault");fault_in=1;
        #0.1;need(dut.core.payload_fire,"FAULT_DOES_NOT_CANCEL_COMMITTED_LAST_BEAT");
        step;fault_in=0;wait_cpl(0,1);repeat(12)step;
        need(fault&&!drained&&cpls[1]==0&&queued==8'h10&&banks_used==0,"LAST_FAULT_QUARANTINE");pass_case;
        $display("OV EXTRA_PASS cases=3");$finish;
    end
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--variant', choices=('ahead', 'compact'), required=True)
    args = ap.parse_args(); out = args.output.resolve()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_candidate_dispatch.py'), str(out)], check=True)
    manifest = {}
    for name, transform in [('ntrpu_destination_candidate.sv', dispatcher), ('ntrpu_candidate_dispatch.sv', adapter)]:
        path = out/name; before = path.read_text(); after = transform(before, args.variant)
        path.write_text(after)
        (out/(name+'.overlap.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))))
        manifest[name] = dict(before=hashlib.sha256(before.encode()).hexdigest(), after=hashlib.sha256(after.encode()).hexdigest())
    original = (ROOT/'fpga/tb/tb_ntrpu_candidate_transport.sv').read_text()
    original = one(original, 'endmodule', OBSERVE+'endmodule')
    (out/'tb_ntrpu_candidate_transport.sv').write_text(original)
    (out/'tb_ntrpu_transport_perf.sv').write_text(performance(original))
    begin = original.index('    initial begin\n        reset_case(1);')
    end = original.index('    initial begin #1000000;', begin)
    extra = one(original[:begin]+EXTRA+original[end:], 'module tb_ntrpu_candidate_transport;', 'module tb_ntrpu_transport_overlap_extra;')
    (out/'tb_ntrpu_transport_overlap_extra.sv').write_text(extra)
    (out/'OVERLAP.json').write_text(json.dumps(dict(variant=args.variant, transforms=manifest,
        reference_modified=False, credit_commit_unchanged=True, physical_qualification=False), indent=2)+'\n')
    (out/'OVERLAP_FILES.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()}, indent=2)+'\n')


if __name__ == '__main__':main()
