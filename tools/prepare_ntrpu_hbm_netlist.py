"""Add a routed-output miter to the immutable, accepted 21-case CDC fixture."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ALWAYS = ('rst_n', 'started', 'inflight', 'submit_ready', 'fetch_valid', 'payload_ready',
                 'fence_ready', 'fence_done_valid', 'cpl_valid', 'fault', 'drained',
                 'banks_used', 'queued', 'pending_counts')
MEMORY_ALWAYS = ('mem_rst_n', 'h_awvalid', 'h_wvalid', 'h_bready', 'adapter_idle', 'adapter_fault')
SCALAR = {'fetch_channel': 'fetch_valid', 'fetch_desc': 'fetch_valid', 'fence_done_id': 'fence_done_valid'}
SOURCE_LANES = {'cpl_tag': ('cpl_valid', 64), 'cpl_status': ('cpl_valid', 2)}
MEMORY_LANES = {'h_awaddr': ('h_awvalid', 34), 'h_awlen': ('h_awvalid', 4),
                'h_awid': ('h_awvalid', 6), 'h_awsize': ('h_awvalid', 3),
                'h_awburst': ('h_awvalid', 2), 'h_wdata': ('h_wvalid', 256),
                'h_wstrb': ('h_wvalid', 32), 'h_wlast': ('h_wvalid', 1)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def one(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def signals(text):
    header = text.split('module ntrpu_hbm_transport_async (', 1)[1].split(');', 1)[0]
    result = []
    for direction, width, names in re.findall(r'\b(input|output)\s+wire\s*(\[[^\]]+\])?\s*(.*?)(?=\b(?:input|output)\b|\Z)', header, re.S):
        for name in names.strip().rstrip(',').split(','):
            assert re.fullmatch(r'\w+', name.strip())
            result.append((direction, width, name.strip()))
    return result


def comparison(original, ports):
    outputs = {n for d, w, n in ports if d == 'output'}
    assert outputs == set(SOURCE_ALWAYS+MEMORY_ALWAYS) | set(SCALAR) | set(SOURCE_LANES) | set(MEMORY_LANES)
    assert '#150;' in original
    lines = ['    integer hn_source=0,hn_memory=0;',
             '    wire hn_bad_payload=$test$plusargs("HN_BAD_PAYLOAD");',
             '    wire hn_bad_ready=$test$plusargs("HN_BAD_READY");']
    lines += [f'    wire {w} route_{n};' for d, w, n in ports if d == 'output']
    lines += ['    wire [511:0] raw_h_wdata;', '    wire [1:0] raw_submit_ready;',
              "    assign route_h_wdata=raw_h_wdata ^ ((hn_bad_payload&&scenario==1&&route_h_wvalid[0])?512'b1:512'b0);",
              "    assign route_submit_ready=raw_submit_ready ^ ((hn_bad_ready&&scenario==1&&started)?2'b1:2'b0);",
              '    ntrpu_hbm_physical routed_dut (']
    for i, (d, w, n) in enumerate(ports):
        prefix = ('raw_' if n in ('h_wdata', 'submit_ready') else 'route_') if d == 'output' else ''
        lines.append(f'        .{n}({prefix}{n})'+(',' if i+1 < len(ports) else ''))
    lines += ['    );',
              '    // AMD FDCE functional model has assign #100 Q (1 ps units).',
              '    // Cold reset is coordinated; validate assertion after settling,',
              '    // and never compare active transactions while it is asserted.',
              '    task automatic hn_reset_check;',
              '        if({rst_n,route_rst_n,mem_rst_n,route_mem_rst_n} !== 4\'b0)$fatal(1,"HN_COLD_RESET_NOT_ASSERTED");',
              '        $display("HN RESET %0d %0d",stamp(),scenario);',
              '    endtask',
              '    initial begin #150.2;hn_reset_check;end',
              '    always @(negedge cold_reset_n) if($realtime>150)begin #0.2;hn_reset_check;end']
    for clock, reset, always, lanes, domain in (
            ('clk', 'rst_n', SOURCE_ALWAYS, SOURCE_LANES, 'SOURCE'),
            ('mem_clk', 'mem_rst_n', MEMORY_ALWAYS, MEMORY_LANES, 'MEMORY')):
        lines += [f'    always @(posedge {clock})begin', '        if($realtime>150 && cold_reset_n)begin',
                  f'            if(route_{reset} !== {reset})$fatal(1,"HN_MISMATCH {reset} case=%0d t=%0d",scenario,stamp());',
                  f'            if({reset})begin']
        for n in always:
            if n != reset:
                lines.append(f'                if(route_{n} !== {n})$fatal(1,"HN_MISMATCH {n} case=%0d t=%0d ref=%h route=%h",scenario,stamp(),{n},route_{n});')
        if domain == 'SOURCE':
            for n, v in SCALAR.items():
                lines.append(f'                if({v} && route_{n} !== {n})$fatal(1,"HN_MISMATCH {n} case=%0d",scenario);')
        lines += ['                for(integer lane=0;lane<2;lane++)begin']
        for n, (v, width) in lanes.items():
            part = f'[lane*{width}+:{width}]'
            lines.append(f'                    if({v}[lane] && route_{n}{part} !== {n}{part})$fatal(1,"HN_MISMATCH {n} case=%0d t=%0d lane=%0d",scenario,stamp(),lane);')
        if domain == 'SOURCE':
            lines += ['                    if(route_cpl_valid[lane]&&cpl_ready[lane])$display("HN CPL %0d %0d %0d %0d %0d",stamp(),scenario,lane,route_cpl_tag[lane*64+:64],route_cpl_status[lane*2+:2]);']
        else:
            lines += ['                    if(route_h_awvalid[lane]&&h_awready[lane])$display("HN AW %0d %0d %0d %h %0d",stamp(),scenario,lane,route_h_awaddr[lane*34+:34],int\'(route_h_awlen[lane*4+:4])+1);',
                      '                    if(route_h_wvalid[lane]&&h_wready[lane])$display("HN W %0d %0d %0d %h %h %b",stamp(),scenario,lane,route_h_wdata[lane*256+:256],route_h_wstrb[lane*32+:32],route_h_wlast[lane]);',
                      '                    if(route_h_bready[lane]&&h_bvalid[lane])$display("HN B %0d %0d %0d %0d %0d",stamp(),scenario,lane,h_bid[lane*6+:6],h_bresp[lane*2+:2]);']
        lines += ['                end']
        if domain == 'SOURCE':
            lines += ['                if(route_fetch_valid&&fetch_ready)$display("HN FETCH %0d %0d %b %h",stamp(),scenario,route_fetch_channel,route_fetch_desc);']
        lines += [f'                hn_{domain.lower()}++;$display("HN {domain} %0d %0d %0d",stamp(),scenario,hn_{domain.lower()});',
                  '            end', '        end', '    end']
    result = one(original, '$finish;', '$display("HN ALL_PASS source=%0d memory=%0d",hn_source,hn_memory);$finish;')
    return one(result, 'endmodule', '\n'.join(lines)+'\nendmodule')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path); ap.add_argument('--netlist', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--mutant', choices=('none', 'payload', 'ready'), default='none')
    a = ap.parse_args(); out = a.output.resolve(); gen = out/'generated'; gen.mkdir(parents=True)
    hashes = json.loads((a.reference/'FILES.json').read_text())
    reference_result = json.loads((a.reference/'result.json').read_text())
    assert reference_result['status'] == 'EXPORTED_PENDING_REVIEW'
    assert reference_result['sources'] == reference_result['sources_after']
    assert all(sha((ROOT/n).read_bytes()) == h for n, h in reference_result['sources'].items())
    copied = {}
    for p in sorted((a.reference/'suite/generated').glob('*.sv')):
        raw = p.read_bytes(); assert sha(raw) == hashes[p.relative_to(a.reference).as_posix()]
        (gen/p.name).write_bytes(raw); copied[p.name] = sha(raw)
    original = (gen/'tb_ntrpu_hbm_cdc.sv').read_text()
    ports = signals((ROOT/'fpga/rtl/ntrpu_hbm_transport_async.sv').read_text())
    raw = a.netlist.read_bytes()
    # Protected AMD blocks retain references to structural XPM module names.
    # A separate Verilog library preserves both visible and encrypted content.
    proof = dict(status='BYTE_IDENTICAL_LIBRARY_ISOLATION', library='hbm_routed',
                 original_sha256=sha(raw), isolated_sha256=sha(raw), restored_sha256=sha(raw),
                 renamed_modules=0, encrypted_blocks_preserved=True)
    top = raw.decode().split('module ntrpu_hbm_physical\n', 1)[1].split('\n  wire', 1)[0]
    netports = re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);', top, re.M)
    assert set(netports) == set(ports), 'Routed port mismatch'
    bench = comparison(original, ports)
    (gen/'tb_ntrpu_hbm_cdc.sv').write_text(bench)
    (gen/'route_isolated.v').write_bytes(raw)
    # xelab's -L list precedes the parent module's library. Scope every
    # structural binding explicitly, including references in protected blocks.
    wrapper = '`uselib lib=hbm_routed lib=unisims_ver lib=secureip\n`include "route_isolated.v"\n`uselib\n'
    (gen/'route_library.v').write_text(wrapper)
    proof['library_wrapper_sha256'] = sha(wrapper.encode())
    (out/'BENCH.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), bench.splitlines(True))))
    (out/'NETLIST_NAMESPACE.json').write_text(json.dumps(proof, indent=2)+'\n')
    (out/'COMPARISON.json').write_text(json.dumps(dict(ports=ports, reference_generated=copied,
        original_bench_sha256=sha(original.encode()), netlist_sha256=sha(raw), sdf=False, force=False,
        latency_edges=0, sample='pre-update rising edge in each domain outside coordinated cold reset',
        cold_reset_assertion_settle_ps=200, idle_data_masked=True,
        negative_controls=['HN_BAD_PAYLOAD', 'HN_BAD_READY']), indent=2)+'\n')
    flow = (ROOT/'tools/ntrpu_hbm_cdc_async.tcl').read_text()
    flow = one(flow, '    launch_simulation;run all;close_sim',
               '    add_files -fileset sim_1 [file join $output generated route_library.v]\n'
               '    set_property include_dirs [list [file join $output generated]] [get_filesets sim_1]\n'
               '    set_property library hbm_routed [get_files [file join $output generated route_library.v]]\n'
               '    launch_simulation;run all;close_sim')
    if a.mutant != 'none':
        flow = one(flow, '    launch_simulation;run all;close_sim',
                   f'    set_property xsim.simulate.xsim.more_options {{-testplusarg HN_BAD_{a.mutant.upper()}}} [get_filesets sim_1]\n'
                   '    launch_simulation;run all;close_sim')
    else:
        # Reuse this batch's compiled snapshot for two comparator fault injections.
        flow = one(flow, '    close_project', '''    if {[string first {HN ALL_PASS source=} $text]<0} {error "Missing routed comparison"}
    set here [pwd]
    set simdir [file join $output project hbm_adapter.sim sim_1 behav xsim]
    set f [open [file join $output negative.tcl] w];puts $f {run all;quit};close $f
    cd $simdir
    foreach kind {PAYLOAD READY} {
        set log [file join $output negative_${kind}.log]
        set failed [catch {exec xsim tb_ntrpu_hbm_cdc_behav -testplusarg HN_BAD_$kind -tclbatch [file join $output negative.tcl] -log $log 2>@1} message]
        set f [open [file join $output negative_${kind}_console.log] w];puts $f $message;close $f
        set ntext [read_text $log]
        set needle [expr {$kind eq "PAYLOAD" ? "HN_MISMATCH h_wdata" : "HN_MISMATCH submit_ready"}]
        if {[string first $needle $ntext]<0 || [string first {HN ALL_PASS} $ntext]>=0} {error "Undetected comparator injection $kind"}
        puts "HN_NEGATIVE_REJECTED $kind"
    }
    cd $here
    close_project''')
    flow += '\nif {[catch {run_async {'+str(ROOT)+'} {'+str(out)+'}} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}\nexit 0\n'
    (out/'run.tcl').write_text(flow)


if __name__ == '__main__':
    main()
