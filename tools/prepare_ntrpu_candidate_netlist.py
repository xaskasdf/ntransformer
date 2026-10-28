"""Compare the unchanged routed sampler with the original core, two cycles later.

The sampler is not a ready/valid adapter. Identical input history reaches its
core one edge later, and its output registers add one further observation edge.
Only reference module identifiers are renamed; routed Verilog is immutable.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_candidate_dispatch import ports
from prepare_ntrpu_registered_dispatch import one

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ['ntrpu_buffer_pool_allocator', 'ntrpu_buffer_pool_allocator_decoded', 'ntrpu_rv_elastic']
GENERATED = ['ntrpu_destination_candidate', 'ntrpu_peer_multibuffer_offer',
             'ntrpu_peer_multibuffer_reserved', 'ntrpu_candidate_dispatch']
ALWAYS = 'submit_ready fetch_valid payload_ready fence_ready fence_done_valid cpl_valid fault drained banks_used queued pending_counts m_awvalid m_wvalid m_bready'.split()
SCALAR_DATA = {'fetch_channel': 'fetch_valid', 'fetch_desc': 'fetch_valid', 'fence_done_id': 'fence_done_valid'}
CHANNEL_DATA = {'cpl_tag': ('cpl_valid', 64), 'cpl_status': ('cpl_valid', 2),
                'm_awaddr': ('m_awvalid', 64), 'm_awlen': ('m_awvalid', 8),
                'm_awid': ('m_awvalid', 4), 'm_awcache': ('m_awvalid', 4),
                'm_awsize': ('m_awvalid', 3), 'm_awburst': ('m_awvalid', 2),
                'm_wdata': ('m_wvalid', 512), 'm_wstrb': ('m_wvalid', 64), 'm_wlast': ('m_wvalid', 1)}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def instrument(text, signals):
    text = one(text, 'ntrpu_candidate_dispatch dut', 'ref_ntrpu_candidate_dispatch dut')
    text = one(text, '    initial begin\n', '    initial begin\n        #150; // Allow the unmodified UNISIM global startup reset to finish.\n')
    text = one(text, '        end else begin\n            $display("RS S',
               '        end else if (!cn_done) begin\n            $display("RS S')
    text = one(text, '$finish;', 'cn_done=1;repeat(3)step;\n        $display("CN ALL_PASS checks=%0d",cn_checks);$finish;')
    out = [(d, w, n) for d, w, n in signals if d == 'output']
    assert {n for _, _, n in out} == set(ALWAYS) | set(SCALAR_DATA) | set(CHANNEL_DATA)
    delayed = out + [('input', '', 'fetch_ready'), ('input', '[1:0]', 'm_wready'),
                     ('input', '[1:0]', 'cpl_ready')]
    lines = ['    // Two observation edges: one input sampler, one output sampler.',
             '    reg cn_done=0,cn_active0=0,cn_active1=0;',
             '    integer cn_case0=0,cn_case1=0,cn_checks=0;']
    for _, width, name in out:
        lines.append(f'    wire {width} route_{name};')
    for _, width, name in delayed:
        lines.append(f'    reg {width} cn0_{name},cn1_{name};')
    lines.append('    ntrpu_candidate_fixture routed_dut (')
    lines += [f'        .{n}({"route_" if d == "output" else ""}{n})' + (',' if i+1 < len(signals) else '')
              for i, (d, w, n) in enumerate(signals)]
    lines += ['    );', '    always @(posedge clk) begin',
              '        cn_active0<=rst_n&&!cn_done;cn_active1<=cn_active0;',
              '        cn_case0<=case_id;cn_case1<=cn_case0;']
    lines += [f'        cn0_{n}<={n};cn1_{n}<=cn0_{n};' for _, _, n in delayed]
    lines += ['        if(cn_active1)begin']
    for name in ALWAYS:
        lines.append(f'            if(route_{name} !== cn1_{name}) $fatal(1,"CN_MISMATCH {name} case=%0d expected=%h actual=%h",cn_case1,cn1_{name},route_{name});')
    for name, valid in SCALAR_DATA.items():
        lines.append(f'            if(cn1_{valid} && route_{name} !== cn1_{name}) $fatal(1,"CN_MISMATCH {name} case=%0d",cn_case1);')
    lines += ['            for(integer ch=0;ch<2;ch++)begin']
    for name, (valid, width) in CHANNEL_DATA.items():
        a = f'[ch*{width}+:{width}]'
        lines.append(f'                if(cn1_{valid}[ch] && route_{name}{a} !== cn1_{name}{a}) $fatal(1,"CN_MISMATCH {name} case=%0d channel=%0d",cn_case1,ch);')
    lines += ['                if(route_m_wvalid[ch]&&cn1_m_wready[ch])',
              '                    $display("CN W %0d %0d %h %h %b",cn_case1,ch,route_m_wdata[ch*512+:512],route_m_wstrb[ch*64+:64],route_m_wlast[ch]);',
              '                if(route_cpl_valid[ch]&&cn1_cpl_ready[ch])',
              '                    $display("CN CPL %0d %0d %0d %0d",cn_case1,ch,route_cpl_tag[ch*64+:64],route_cpl_status[ch*2+:2]);',
              '            end',
              '            if(route_fetch_valid&&cn1_fetch_ready)',
              '                $display("CN FETCH %0d %b %h",cn_case1,route_fetch_channel,route_fetch_desc);',
              '            cn_checks++;$display("CN MATCH %0d %0d",cn_case1,cn_checks);',
              '        end', '    end', '']
    return one(text, 'endmodule', '\n'.join(lines) + 'endmodule')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--netlist', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    subprocess.run([sys.executable, '-B', str(ROOT/'tools/prepare_ntrpu_candidate_dispatch.py'), str(out)], check=True)
    signals = ports((ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text())
    net = args.netlist.read_text()
    top = net.split('module ntrpu_candidate_fixture\n', 1)[1].split('\n  wire', 1)[0]
    declared = re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);', top, re.M)
    assert declared == signals, 'Routed sampler interface differs'
    net_modules = set(re.findall(r'^module (\w+)', net, re.M))
    sources = {name: ((ROOT/'fpga/rtl' if name in DIRECT else out)/(name+'.sv')).read_text()
               for name in DIRECT + GENERATED}
    modules = {m for text in sources.values() for m in re.findall(r'^module (\w+)', text, re.M)}
    mapping = {m: 'ref_'+m for m in modules}
    assert not set(mapping.values()) & net_modules
    regex = re.compile(r'\b('+'|'.join(sorted(modules))+r')\b')
    reverse = re.compile(r'\b('+'|'.join(sorted(mapping.values()))+r')\b')
    reverse_map = {v: k for k, v in mapping.items()}
    manifest = dict(netlist_sha256=hashlib.sha256(args.netlist.read_bytes()).hexdigest(),
                    latency_edges=2, startup_ns=150, sdf=False, force=False,
                    modules=mapping, netlist_modules=sorted(net_modules), ports=signals,
                    always=ALWAYS, scalar_data=SCALAR_DATA, channel_data=CHANNEL_DATA, sources={})
    for name, source in sources.items():
        target = regex.sub(lambda m: mapping[m[0]], source)
        assert reverse.sub(lambda m: reverse_map[m[0]], target) == source
        (out/('ref_'+name+'.sv')).write_text(target)
        manifest['sources'][name] = dict(normalized_source_sha256=digest(source),
                                        normalized_reference_sha256=digest(target), reversible=True)
    for name in ('tb_ntrpu_candidate_dispatch', 'tb_ntrpu_candidate_extra'):
        before = (out/(name+'.sv')).read_text()
        after = instrument(before, signals)
        target = name.replace('candidate_', 'candidate_netlist_')
        after = one(after, 'module '+name+';', 'module '+target+';')
        (out/(target+'.sv')).write_text(after)
        (out/(target+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=name, tofile=target)))
    (out/'NETLIST_COMPARISON.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
