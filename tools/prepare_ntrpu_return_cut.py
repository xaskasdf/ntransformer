"""Generate an isolated registered-return transport with the same AMD AXI slices."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_registered_dispatch import one
from prepare_ntrpu_candidate_dispatch import ports
from prepare_ntrpu_transport_study import DIRECT, GENERATED, comparison

ROOT = Path(__file__).resolve().parents[1]


def allocator(text):
    core = one(text, 'module ntrpu_buffer_pool_allocator_decoded #(',
               'module ntrpu_buffer_pool_allocator_cut_core #(')
    core = one(core, '(!used[g] || returned[g]) &&', '!used[g] &&')
    core = one(core, 'used[b] && !returned[b] && owner[b] == g', 'used[b] && owner[b] == g')
    core = one(core, '!(|return_error)) begin', '!(|return_valid)) begin')
    header = text[:text.index('    localparam integer CHANNEL_BITS')]
    header = one(header, 'module ntrpu_buffer_pool_allocator_decoded #(',
                 'module ntrpu_buffer_pool_allocator_return_cut #(')
    header = one(header, '    output reg fault,', '    output wire fault, return_busy,')
    header = one(header, '    output reg [CHANNELS-1:0] alloc_ready,',
                 '    output wire [CHANNELS-1:0] alloc_ready,')
    wrapper = header + '''    // A return remains owned through capture and validation. New grants
    // are closed in BOTH phases, before a delayed error can become sticky.
    reg [CHANNELS-1:0] pending;
    reg [CHANNELS*BANK_BITS-1:0] pending_bank;
    reg [CHANNELS*GENERATION_BITS-1:0] pending_generation;
    wire inner_drained;
    assign return_busy = (|return_valid) || (|pending);
    assign return_ready = {CHANNELS{rst_n}};
    assign drained = inner_drained && !(|pending);
    always @(posedge clk) begin
        if (!rst_n) begin
            pending <= 0; pending_bank <= 0; pending_generation <= 0;
        end else begin
            pending <= return_valid;
            for (integer p=0; p<CHANNELS; p=p+1) begin
                if (return_valid[p]) begin
                    pending_bank[p*BANK_BITS+:BANK_BITS] <= return_bank[p*BANK_BITS+:BANK_BITS];
                    pending_generation[p*GENERATION_BITS+:GENERATION_BITS] <= return_generation[p*GENERATION_BITS+:GENERATION_BITS];
                end
            end
        end
    end
    ntrpu_buffer_pool_allocator_cut_core #(
        .CHANNELS(CHANNELS), .DEDICATED_BANKS(DEDICATED_BANKS),
        .RESERVE_BANKS(RESERVE_BANKS), .RESERVE_QUOTA(RESERVE_QUOTA),
        .GENERATION_BITS(GENERATION_BITS), .BANKS(BANKS), .BANK_BITS(BANK_BITS)
    ) core (
        .clk(clk), .rst_n(rst_n), .stop(stop || return_busy),
        .alloc_valid(alloc_valid), .alloc_ready(alloc_ready), .alloc_bank(alloc_bank),
        .alloc_generation(alloc_generation), .return_valid(pending), .return_ready(),
        .return_bank(pending_bank), .return_generation(pending_generation),
        .return_error(return_error), .fault(fault), .used_mask(used_mask),
        .retired_mask(retired_mask), .drained(inner_drained)
    );
endmodule
'''
    return wrapper + '\n' + core


def receiver(text):
    text = one(text, 'wire allocator_fault, allocator_drained;', 'wire allocator_fault, allocator_drained, return_busy;')
    text = one(text, 'ntrpu_buffer_pool_allocator_decoded #(', 'ntrpu_buffer_pool_allocator_return_cut #(')
    text = one(text, '.return_generation(return_generation), .return_error(return_error),',
               '.return_generation(return_generation), .return_error(return_error), .return_busy(return_busy),')
    text = one(text, 'allocator_fault || (|return_error) || (|response_error) || bad_cmd',
               'allocator_fault || (|response_error) || bad_cmd')
    text = one(text, '!(|return_error) && !(|response_error)', '!return_busy && !(|response_error)')
    return text


def unit_bench(text):
    text = one(text, 'ntrpu_buffer_pool_allocator #(', 'ntrpu_buffer_pool_allocator_return_cut #(')
    text = one(text, 'wire fault, drained;', 'wire fault, drained, return_busy;')
    text = one(text, '    task step;', '    task edge_step;')
    anchor = '    task prepare_returns;'
    text = one(text, anchor, '''    task step;
        input reset_value, stop_value;
        input [CHANNELS-1:0] requests, releases;
        input [CHANNELS*BANK_BITS-1:0] banks;
        input [CHANNELS*GENERATION_BITS-1:0] generations;
        begin
            edge_step(reset_value, stop_value, requests, releases, banks, generations);
            if (reset_value && (|releases)) begin
                if (pre_ready != 0) $fatal(1,"grant during capture");
                edge_step(1, stop_value, requests, 0, 0, 0);
                if (pre_ready != 0) $fatal(1,"grant during validation");
            end
        end
    endtask

''' + anchor)
    text = one(text, '        if (!pre_ready[0] || pre_bank != saved_bank || pre_generation != saved_generation + 1\'b1)',
               '''        if (pre_ready != 0) $fatal(1,"early reuse");
        step(1, 0, 1, 0, 0, 0);
        if (!pre_ready[0] || pre_bank != saved_bank || pre_generation != saved_generation + 1'b1)''')
    text = text.replace('simultaneous_return_reuse', 'staged_return_reuse').replace('same-cycle reuse failed', 'staged reuse failed')
    text = one(text, '        $display("CASE final_drain");', '''        $display("CASE consecutive_returns");
        cold_reset;
        for (i=0; i<BANKS; i=i+1) step(1,0,{CHANNELS{1'b1}},0,0,0);
        // No settling helper: exercise capture and validation on adjacent edges.
        for (i=0; i<BANKS+2; i=i+1) begin
            prepare_returns;
            edge_step(1,0,{CHANNELS{1'b1}},release_bits,release_banks,release_generations);
        end
        edge_step(1,1,0,0,0,0);
        if (fault) $fatal(1,"consecutive legal returns faulted");
        $display("CASE final_drain");''')
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    ap.add_argument('--netlist', type=Path); a = ap.parse_args(); out = a.output.resolve()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_overlap_axi.py'), str(out)], check=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    original = (ROOT/'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv').read_text()
    (out/'ntrpu_buffer_pool_allocator_return_cut.sv').write_text(allocator(original))
    path = out/'ntrpu_peer_multibuffer_offer.sv'; before = path.read_text(); after = receiver(before)
    path.write_text(after)
    (out/'RETURN_CUT.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True))))
    monitor = '''    always @(posedge clk) if (rst_n) begin
        need(!(dut.core.receiver.core.return_busy && (|dut.core.receiver.core.alloc_ready)),
             "RETURN_BARRIER_BLOCKS_GRANTS");
        need(!(dut.core.receiver.core.return_busy && (|dut.core.reserve_take)),
             "RETURN_BARRIER_BLOCKS_RESERVATIONS");
    end
'''
    for name in ('tb_ntrpu_candidate_transport', 'tb_ntrpu_transport_overlap_extra', 'tb_ntrpu_transport_perf'):
        p = out/(name+'.sv'); p.write_text(one(p.read_text(), 'endmodule', monitor+'endmodule'))
    unit = unit_bench((ROOT/'fpga/tb/tb_ntrpu_buffer_pool_allocator.sv').read_text())
    profiles = dict(pool=(2, 2, 4, 4, 64), bounds=(2, 1, 1, 1, 64), wrap=(1, 1, 0, 0, 3))
    for name, profile in profiles.items():
        text = unit.replace('module tb_ntrpu_buffer_pool_allocator;', 'module tb_ntrpu_return_cut_'+name+';')
        for key, value in zip(('CHANNELS', 'DEDICATED_BANKS', 'RESERVE_BANKS', 'RESERVE_QUOTA', 'GENERATION_BITS'), profile):
            text, count = re.subn(r'(parameter integer '+key+r' = )[^;]+;', r'\g<1>'+str(value)+';', text); assert count == 1
        (out/('tb_ntrpu_return_cut_'+name+'.sv')).write_text(text)
    paths = {n: out/(n+'.sv') if n == 'ntrpu_candidate_transport' or n in GENERATED else ROOT/'fpga/rtl'/(n+'.sv') for n in DIRECT+GENERATED}
    paths['ntrpu_buffer_pool_allocator_return_cut'] = out/'ntrpu_buffer_pool_allocator_return_cut.sv'
    fixture = dict(variant='registered_return_axi', original_allocator_sha256=sha(ROOT/'fpga/rtl/ntrpu_buffer_pool_allocator_decoded.sv'),
                   original_receiver_sha256=hashlib.sha256(before.encode()).hexdigest(), logic_sha256={n: sha(p) for n,p in paths.items()},
                   return_capture_cycles=1, block_admission_during_capture=True, block_admission_during_validation=True,
                   same_edge_reuse=False, profiles=profiles, startup_ns=150)
    if a.netlist:
        signals = ports((ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text())
        net = a.netlist.read_text(); top = net.split('module ntrpu_transport_physical\n', 1)[1].split('\n  wire', 1)[0]
        assert re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);', top, re.M) == signals
        sources = {n:p.read_text() for n,p in paths.items()}
        modules = {m for t in sources.values() for m in re.findall(r'^module (\w+)',t,re.M)} | {'ntrpu_mem_regslice'}
        mapping = {n:'ref_'+n for n in modules}; inverse = {v:k for k,v in mapping.items()}
        assert not set(mapping.values()) & set(re.findall(r'^module (\w+)',net,re.M))
        regex = re.compile(r'\b('+'|'.join(sorted(mapping))+r')\b'); reverse = re.compile(r'\b('+'|'.join(sorted(inverse))+r')\b')
        for name, text in sources.items():
            renamed = regex.sub(lambda m:mapping[m[0]],text)
            assert reverse.sub(lambda m:inverse[m[0]],renamed) == text
            (out/('ref_'+name+'.sv')).write_text(renamed)
        for source, target in [('tb_ntrpu_candidate_transport','tb_ntrpu_transport_netlist'), ('tb_ntrpu_transport_overlap_extra','tb_ntrpu_overlap_netlist_extra')]:
            text = (out/(source+'.sv')).read_text().replace('        #150;\n','')
            if source != 'tb_ntrpu_candidate_transport': text = one(text,'module '+source+';','module tb_ntrpu_candidate_transport;')
            text = comparison(text,signals)
            if target != 'tb_ntrpu_transport_netlist': text = one(text,'module tb_ntrpu_transport_netlist;','module '+target+';')
            (out/(target+'.sv')).write_text(text)
        fixture['comparison'] = dict(netlist_sha256=sha(a.netlist), latency_edges=0, startup_ns=150, sdf=False, force=False, reversible_renaming=True)
    (out/'RETURN_CUT.json').write_text(json.dumps(fixture,indent=2)+'\n')


if __name__ == '__main__':
    main()
