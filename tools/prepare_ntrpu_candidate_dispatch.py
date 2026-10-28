"""Generate isolated candidate staging and registered physical timing fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_registered_dispatch import one, emit

ROOT = Path(__file__).resolve().parents[1]


def dispatcher(text):
    text = one(text, 'module ntrpu_destination_registered #(', 'module ntrpu_destination_candidate #(')
    text = one(text, '    output reg [1:0] reserve_take,', '''    output reg [1:0] reserve_take,
    output reg candidate_valid, candidate_channel,
    output reg [231:0] candidate_desc,''')
    text = one(text, '''        if (rst_n && !fault && state==IDLE) begin
            if (eligible[next_channel]) reserve_take[next_channel]=1;
            else if (eligible[!next_channel]) reserve_take[!next_channel]=1;
        end''', '''        // A candidate owns no credit. Recheck the live offer at commit.
        if (rst_n && !fault && state==IDLE && candidate_valid &&
            eligible[candidate_channel]) reserve_take[candidate_channel]=1;''')
    text = one(text, '    always @(posedge clk) begin\n        if (!rst_n) begin\n            state<=IDLE;', '''    always @(posedge clk) begin
        if (!rst_n) begin
            candidate_valid<=0;candidate_channel<=0;candidate_desc<=0;
        end else if (fault || state!=IDLE || (|reserve_take)) begin
            candidate_valid<=0;
        end else begin
            // If an offer disappeared, select a currently eligible head again.
            // No FIFO pop or bank allocation occurs at this preparation edge.
            candidate_valid<=0;
            if (eligible[next_channel]) begin
                candidate_valid<=1;candidate_channel<=next_channel;
                candidate_desc<=head_desc[next_channel*232+:232];
            end else if (eligible[!next_channel]) begin
                candidate_valid<=1;candidate_channel<=!next_channel;
                candidate_desc<=head_desc[(!next_channel)*232+:232];
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state<=IDLE;''')
    text = one(text, 'selected_desc<=head_desc[reserve_take[1]*232+:232];',
               'selected_desc<=candidate_desc;')
    return text


def adapter(text):
    text = one(text, 'module ntrpu_registered_dispatch (', 'module ntrpu_candidate_dispatch (')
    text = one(text, '    wire selected_channel=reserve_take[1];\n    wire [231:0] selected_desc=head_desc[selected_channel*232+:232];',
               '    wire candidate_valid,selected_channel;\n    wire [231:0] selected_desc;')
    text = one(text, '    ntrpu_destination_registered dispatch (', '    ntrpu_destination_candidate dispatch (')
    text = one(text, '        .reserve_can(reserve_can),.reserve_take(reserve_take),', '''        .reserve_can(reserve_can),.reserve_take(reserve_take),
        .candidate_valid(candidate_valid),.candidate_channel(selected_channel),.candidate_desc(selected_desc),''')
    return text


def bench(text):
    text = one(text, 'module tb_ntrpu_registered_dispatch;', 'module tb_ntrpu_candidate_dispatch;')
    text = one(text, 'ntrpu_registered_dispatch dut', 'ntrpu_candidate_dispatch dut')
    text = one(text, '            $display("RH HEAD', '''            $display("CP CAND %0d %b %b %h %0d %b %b",cycle,dut.candidate_valid,
                dut.selected_channel,dut.selected_desc,dut.dispatch.state,
                dut.dispatch.next_channel,dut.dispatch.eligible);
            $display("RH HEAD''')
    return text


EXTRA = '''    initial begin
        // Fault in the new interval between preparation and atomic allocation.
        reset_case(9);push_one(0,500,8);
        begin: candidate_wait
            for(integer tries=0;tries<100;tries++)begin step;if(dut.candidate_valid)disable candidate_wait;end
            need(0,"CANDIDATE_TIMEOUT");
        end
        need(banks_used==0&&!dut.lease_valid&&issued[0]==0,"CANDIDATE_IS_NOT_RESERVATION");
        fault_in=1;step;fault_in=0;repeat(8)step;
        need(fault&&banks_used==0&&!dut.lease_valid&&!dut.candidate_valid&&queued==1&&
            issued[0]==0&&issued[1]==0&&!fetch_valid,"PRECOMMIT_FAULT_QUARANTINES");pass_case;

        // Stop closes publication, but does not cancel queued work.
        reset_case(10);push_one(1,501,64);
        begin: stop_wait
            for(integer tries=0;tries<100;tries++)begin step;if(dut.candidate_valid)disable stop_wait;end
            need(0,"CANDIDATE_STOP_TIMEOUT");
        end
        need(banks_used==0&&!dut.lease_valid,"STOP_BEFORE_COMMIT");
        stop=1;drain_all;need(submit_ready==0&&cpls[1]==1,"STOP_CANDIDATE_DRAINS");pass_case;
        $display("CP EXTRA_PASS cases=2 cycle=%0d",cycle);$finish;
    end
'''


def extra_bench(text):
    text = one(text, 'module tb_ntrpu_candidate_dispatch;', 'module tb_ntrpu_candidate_extra;')
    start = text.index('    initial begin\n        reset_case(1);')
    end = text.index('    initial begin #500000;', start)
    return text[:start] + EXTRA + text[end:]


def ports(source):
    header = source.split('module ntrpu_registered_dispatch (', 1)[1].split('\n);', 1)[0]
    result = []
    for field in header.split(','):
        match = re.fullmatch(r'\s*(?:(input|output) wire\s*(\[[^\]]+\])?\s*)?(\w+)\s*', field)
        if not match:
            raise ValueError('Unsupported port declaration: ' + field)
        direct, bits, name = match.groups()
        if direct:
            direction, width = direct, bits or ''
        result.append((direction, width, name))
    assert len(result) == 46 and result[0] == ('input', '', 'clk'), result
    return result


def fixture(source, variant):
    """Timing sampler, NOT a ready/valid adapter; functional tests use the core."""
    signals = ports(source)
    top = 'ntrpu_' + variant + '_fixture'
    lines = ['`timescale 1ns/1ps', '// Diagnostic sampler only: not a transport integration wrapper.',
             '// Every core data/control port terminates at a same-clock register.',
             '// External sampler ports have no board I/O timing qualification.', 'module ' + top + ' (']
    lines += ['    ' + d + ' wire ' + w + ' ' + n + (',' if i < len(signals)-1 else '')
              for i, (d, w, n) in enumerate(signals)]
    lines += [');', '    wire core_clk;',
              '    (* DONT_TOUCH = "yes" *) BUFGCE parent_clock (.I(clk),.CE(1\'b1),.O(core_clk));']
    for direction, width, name in signals[1:]:
        lines.append('    (* DONT_TOUCH = "yes" *) reg ' + width + ' boundary_' + name + ';')
        if direction == 'output':
            lines += ['    wire ' + width + ' core_' + name + ';',
                      '    assign ' + name + ' = boundary_' + name + ';']
        lines += ['    always @(posedge core_clk) boundary_' + name + ' <= ' +
                  (name if direction == 'input' else 'core_' + name) + ';']
    lines.append('    ntrpu_' + variant + '_dispatch core (')
    lines += ['        .' + n + '(' + ('core_clk' if n == 'clk' else
               ('boundary_' if d == 'input' else 'core_') + n) + ')' +
              (',' if i < len(signals)-1 else '') for i, (d, w, n) in enumerate(signals)]
    lines += ['    );', 'endmodule', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    out = ap.parse_args().output.resolve()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_registered_dispatch.py'), str(out)], check=True)
    manifest = []
    for original, target, transform in (
        ('ntrpu_destination_registered.sv', 'ntrpu_destination_candidate.sv', dispatcher),
        ('tb_ntrpu_registered_dispatch.sv', 'tb_ntrpu_candidate_dispatch.sv', bench),
        ('tb_ntrpu_candidate_dispatch.sv', 'tb_ntrpu_candidate_extra.sv', extra_bench)):
        # emit accepts absolute source paths, preserving hashes and readable diffs.
        manifest.append(emit(out, str(out/original), target, transform))
    manifest.append(emit(out, 'fpga/rtl/ntrpu_registered_dispatch.sv', 'ntrpu_candidate_dispatch.sv', adapter))
    source = (ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text()
    for variant in ('registered', 'candidate'):
        name = 'ntrpu_' + variant + '_fixture.sv'
        (out/name).write_text(fixture(source, variant))
    (out/'CANDIDATE_TRANSFORMATION.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'GENERATED_FILES.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
